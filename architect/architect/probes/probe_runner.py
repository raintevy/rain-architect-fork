"""Counterfactual probe execution for robot programs.

Given a program and a base scene, the probe runner generates K perturbed
scenes via :mod:`architect.probes.scene_mutator`, executes the program against each
in dry-run mode, and reports per-perturbation pass/fail plus the static
audit flags from :mod:`architect.probes.audit`. The output is the substrate for both
the C1 active-sampling ranker and the C2 graduation gate; in P2 it
provides the primitive "run K probes, get a pass-rate" capability the
exit criterion calls for.

Probes never invoke an LLM and never re-execute physics. Unknown names
called by the program (LLM-synthesized helpers not present in the
namespace) are stubbed with a permissive object that absorbs any access
pattern, so the probe does not crash on programs whose helpers haven't
been graduated into the library yet. This keeps probes cheap and
deterministic.

P3 will plumb in scorers and invariants. P5 will call this module with
the in-session scene to gate skill graduation.
"""

from __future__ import annotations

import ast
import builtins
import io
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Union

from architect.probes.audit import RiskFlag, audit_program
from architect.probes.scene_mutator import Scene, bind_scene, default_scene, generate_probe_suite
from architect.corrections.llm_scorer import ScorerArtifact, load_artifact


# ---------------------------------------------------------------------------
# Permissive stub for undefined helpers
# ---------------------------------------------------------------------------


class _PermissiveStub:
    """Absorbs arbitrary access patterns without raising.

    A program may subscript, attribute-access, call, iterate, len, bool, or
    coerce-to-number any return value of an undefined helper; this stub
    behaves plausibly for all of those so the probe doesn't crash before
    we've measured anything. ``__call__`` returns ``self`` so chains like
    ``foo(...)["bar"][0]`` work.
    """

    __slots__ = ()

    def __getitem__(self, _key): return self
    def __setitem__(self, _key, _value): return None
    def __getattr__(self, _name): return self
    def __call__(self, *args, **kwargs): return self
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __bool__(self): return True
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __repr__(self): return "<probe stub>"

    # Numeric coercion paths some programs hit when subscripting into a stub
    # then doing arithmetic on the result.
    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __sub__(self, other): return self
    def __rsub__(self, other): return self
    def __mul__(self, other): return self
    def __rmul__(self, other): return self


_STUB = _PermissiveStub()


# ---------------------------------------------------------------------------
# Call log capture
# ---------------------------------------------------------------------------


# Functions whose invocations the call-log tracks. These are the program's
# *actions* on the world — anything the scorer / invariant might want to
# reason about. Perception calls are excluded; scenes already capture those.
_LOGGED_FUNCS = frozenset({
    "move_ee_to_pose",
    "move_ee_to_rel_pose",
    "set_gripper_width",
    "set_arm_joints",
    "set_camera_pose",
    "move_base",
    "move_base_to_rel",
    "reset_robot",
    # Library-graduated motion helpers commonly emitted by Claude — logging
    # them lets scorers reason about composed motions even when the helper
    # itself isn't introspectable.
    "move_arm_to_config",
    "rotate_wrist_to_config",
    "adjust_gripper_width",
    # Franka extensions (commit set ported from franka_entire_program):
    # guarded contact moves, in-place wrist rotation, and ReKep-style
    # waypoint trajectories. Logging them keeps the scorer's view of
    # the program's motion structure complete when the LLM uses these
    # control primitives.
    "move_ee_guarded",
    "rotate_wrist",
    "execute_waypoint_trajectory",
    # Verification: post-grasp VQA queries are explicit signals in the
    # call_log that the program checked whether the grasp / placement
    # actually succeeded. Scorers (e.g. pick_place_cycles with
    # require_vqa_verification) inspect the recorded VQA response to
    # accept a cycle only when the program affirmed the outcome instead
    # of blindly proceeding from a gripper-close event.
    "get_vqa_response",
    "verify_grasp",
})


def _wrap_for_call_log(namespace: dict, log: list[dict]) -> None:
    """Replace each ``_LOGGED_FUNCS`` entry in ``namespace`` with a wrapper
    that records ``{"name", "args", "kwargs"}`` to ``log`` before delegating.

    Wrappers only fire when the program *calls* the wrapped name; an unused
    primitive contributes nothing to the log. Args / kwargs are stored as
    Python values (dicts, floats, lists) — not stringified — so scorers can
    do arithmetic over them.
    """
    for name in list(namespace):
        if name not in _LOGGED_FUNCS:
            continue
        original = namespace[name]
        namespace[name] = _make_logger(name, original, log)


def _make_logger(name: str, original: Callable[..., Any], log: list[dict]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Reserve the entry *before* the call so its position in the log
        # reflects when it was issued (some scorers walk the log in order
        # and would be confused if a slow call landed out of sequence).
        # Result is filled in after the call returns so scorers that
        # inspect VQA / perception responses (e.g. pick_place_cycles
        # checking whether a post-grasp get_vqa_response was affirmative)
        # see what the program actually saw. Existing scorers that only
        # look at name/args/kwargs are unaffected — the new key is
        # additive.
        entry: dict[str, Any] = {"name": name, "args": list(args), "kwargs": dict(kwargs)}
        log.append(entry)
        result = original(*args, **kwargs)
        entry["result"] = result
        return result
    return wrapped


def _called_names(code: str) -> set[str]:
    """Return the set of *bare* function names called in ``code``.

    Only ``ast.Call(func=ast.Name(...))`` is collected; attribute and
    subscript callees are out of scope (they're presumed bound to perception
    return values via the namespace).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _install_stubs(namespace: dict, code: str, log: list[dict]) -> list[str]:
    """Install permissive stubs for names called in ``code`` but not present
    in ``namespace`` and not Python builtins. Stubbed callables that match
    a logged primitive name still record to ``log`` so call-log-based scorers
    work even when the helper hasn't been graduated into the dry-run namespace.
    Returns the names stubbed (useful for diagnostics)."""
    builtin_names = set(dir(builtins))
    stubbed: list[str] = []
    for name in _called_names(code):
        if name in namespace or name in builtin_names:
            continue
        if name in _LOGGED_FUNCS:
            namespace[name] = _make_logger(name, lambda *a, **kw: _STUB, log)
        else:
            namespace[name] = lambda *a, **kw: _STUB
        stubbed.append(name)
    return stubbed


# ---------------------------------------------------------------------------
# Per-perturbation result + aggregate ProbeReport
# ---------------------------------------------------------------------------


@dataclass
class ProbeOutcome:
    """One probe's result, preserving per-artifact identity.

    Previously this stored a single ``scorer_score`` mean and a single
    ``invariant_pass`` boolean, which made it impossible to tell *which*
    scorer or invariant was responsible for a candidate's ranking. The
    per-artifact dicts (``scorer_scores`` / ``invariant_results``) preserve
    the audit trail through the entire pipeline so the slate UI and
    episode log can show which artifact moved the needle. The rolled-up
    ``scorer_score`` and ``invariant_pass`` are kept as read-only
    properties for the few back-compat consumers.
    """

    axis: str                                # mutation axis name (or "identity")
    passed: bool                             # exec ran AND scorer_pass AND invariant_pass
    error: str | None                        # exception message; None if passed
    error_type: str | None
    scorer_scores: dict[str, float]          # name → score in [0,1]; empty if no scorers
    scorer_pass: bool                        # all values ≥ scorer_threshold
    invariant_results: dict[str, bool]       # name → True (held) / False (violated)
    call_log: list[dict] = field(default_factory=list)

    @property
    def scorer_score(self) -> float | None:
        """Rolled-up mean over the per-scorer dict; ``None`` if no scorers ran."""
        if not self.scorer_scores:
            return None
        return sum(self.scorer_scores.values()) / len(self.scorer_scores)

    @property
    def invariant_pass(self) -> bool:
        """True iff every invariant held (or none ran)."""
        return all(self.invariant_results.values()) if self.invariant_results else True


@dataclass
class ProbeReport:
    audit_flags: list[RiskFlag]
    outcomes: list[ProbeOutcome]
    stubbed_names: list[str]
    scorer_threshold: float = 0.5  # minimum mean score to count as scorer_pass

    @property
    def k(self) -> int:
        return len(self.outcomes)

    @property
    def n_passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def pass_rate(self) -> float:
        return (self.n_passed / self.k) if self.k else 0.0

    @property
    def invariant_pass_rate(self) -> float:
        """Fraction of (probe, invariant) checks that held.

        ``1.0`` when no invariants were supplied or every check held;
        ``0.0`` when every check failed (typical of a stale invariant
        leaked through the relevance filter). Reserved for the C2
        graduation gate; C1 ranking deliberately doesn't gate on this.
        """
        total = 0
        held = 0
        for o in self.outcomes:
            for v in o.invariant_results.values():
                total += 1
                if v:
                    held += 1
        return (held / total) if total else 1.0

    @property
    def all_invariants_hold(self) -> bool:
        """True iff every invariant held on every probe — the C2 hard gate."""
        return all(
            all(o.invariant_results.values())
            for o in self.outcomes
        )

    @property
    def per_axis_pass_rate(self) -> dict[str, tuple[int, int]]:
        """``{axis: (passed, total)}`` for diagnosing which axis breaks the program."""
        agg: dict[str, list[int]] = {}
        for o in self.outcomes:
            cell = agg.setdefault(o.axis, [0, 0])
            cell[1] += 1
            if o.passed:
                cell[0] += 1
        return {k: (v[0], v[1]) for k, v in agg.items()}

    def audit_severity(self) -> dict[str, int]:
        from architect.probes.audit import severity_summary
        return severity_summary(self.audit_flags)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_ScorerInput = Union[Callable[[dict, dict, list[dict]], Any], ScorerArtifact]


def _resolve_artifacts(
    items: tuple[_ScorerInput, ...],
    expected_kind: str,
) -> list[tuple[str, Callable[..., Any]]]:
    """Materialize ``ScorerArtifact``s into ``(name, callable)`` pairs.

    The name is what the probe runner stamps into ``ProbeOutcome.scorer_scores``
    / ``invariant_results`` so downstream consumers (the slate UI, the
    episode log) can attribute outcomes to a specific artifact. Bare
    callables (used by tests) fall back to ``__name__`` or a positional
    placeholder.

    Artifacts whose kind doesn't match ``expected_kind`` and artifacts that
    fail to load are silently dropped — a probe that can't run a scorer
    should fall back to "no opinion" rather than crash.
    """
    out: list[tuple[str, Callable[..., Any]]] = []
    for i, item in enumerate(items):
        if isinstance(item, ScorerArtifact):
            if item.kind != expected_kind:
                continue
            fn = load_artifact(item)
            if fn is not None:
                out.append((item.name, fn))
        else:
            name = getattr(item, "__name__", f"{expected_kind}_{i}")
            out.append((name, item))
    return out


def run_probes(
    program: str,
    *,
    base_scene: Scene | None = None,
    robot_name: str = "franka",
    k: int = 16,
    seed: int | None = None,
    namespace_factory: Callable[[], dict] | None = None,
    scorers: tuple[_ScorerInput, ...] = (),
    invariants: tuple[_ScorerInput, ...] = (),
    scorer_threshold: float = 0.5,
    suppress_program_stdout: bool = True,
) -> ProbeReport:
    """Run ``program`` against ``k`` perturbed scenes and return a report.

    Parameters
    ----------
    program:
        Python source to probe.
    base_scene:
        Scene to perturb. If ``None``, ``default_scene(robot_name)`` is used.
    robot_name:
        Used only when ``base_scene`` is ``None`` (and reserved for future
        per-robot stub variants).
    k:
        Number of perturbations to run, including the unperturbed identity
        scene as the first probe.
    seed:
        RNG seed for reproducible suites.
    namespace_factory:
        Callable returning a fresh execution namespace per probe. Defaults
        to ``dict`` (no API stubs), in which case unknown names are
        permissive-stubbed. When called from the agent, pass a closure
        that builds the per-robot dry-run namespace so primitive API
        no-ops match production behavior.
    scorers / invariants:
        Reserved for P3. Currently each probe records ``scorer_pass=True``
        and ``invariant_pass=True`` unconditionally so the report shape
        is stable across phases.
    suppress_program_stdout:
        Programs (and their stubs) tend to print extensively. Capture
        their stdout per probe so the runner stays quiet on the console.
    """
    audit_flags = audit_program(program)

    if base_scene is None:
        base_scene = default_scene(robot_name)

    suite = generate_probe_suite(base_scene, k=k, seed=seed)
    scorer_pairs = _resolve_artifacts(scorers, expected_kind="scorer")
    invariant_pairs = _resolve_artifacts(invariants, expected_kind="invariant")

    outcomes: list[ProbeOutcome] = []
    stubbed_seen: set[str] = set()
    factory = namespace_factory or (lambda: {})

    for axis, scene in suite:
        namespace = factory()
        bind_scene(namespace, scene)
        call_log: list[dict] = []
        # Wrap dry-run primitive APIs first, then install stubs for unknown
        # helpers (which may also be _LOGGED_FUNCS — the stub installer
        # routes them through the same logger).
        _wrap_for_call_log(namespace, call_log)
        stubbed = _install_stubs(namespace, program, call_log)
        stubbed_seen.update(stubbed)

        scene_before = asdict(scene)

        exec_ok = True
        err_msg: str | None = None
        err_type: str | None = None
        try:
            if suppress_program_stdout:
                with redirect_stdout(io.StringIO()):
                    exec(program, namespace)  # noqa: S102
            else:
                exec(program, namespace)  # noqa: S102
        except Exception as exc:
            exec_ok = False
            err_type = type(exc).__name__
            err_msg = str(exc)

        # In dry-run, scene_after equals scene_before (perception is mocked,
        # not re-read). Real-robot eval will pass an actually-observed
        # post-execution scene here. Scorers should not assume motion.
        scene_after = asdict(scene)

        scorer_scores, scorer_pass = _evaluate_scorers(
            scorer_pairs, scene_before, scene_after, call_log,
            threshold=scorer_threshold, exec_ok=exec_ok,
        )
        if exec_ok:
            invariant_results = _evaluate_invariants(
                invariant_pairs, scene_before, scene_after, call_log,
            )
        else:
            # Crashed exec → every invariant counts as violated so the
            # diagnostic invariant_violation_rate doesn't paper over dead
            # trajectories.
            invariant_results = {name: False for name, _ in invariant_pairs}

        # ``passed`` gates C1 ranking via ProbeReport.pass_rate. Invariants
        # are *not* AND-gated here — they're computed and exposed for
        # diagnostics + the future C2 graduation gate, but they no longer
        # zero a candidate's pass_rate just because a stale invariant
        # (built for an unrelated correction) violated. The relevance
        # filter in architect.library.relevance already removes most cross-talk; this
        # change makes the residual cross-talk soft rather than fatal.
        outcomes.append(ProbeOutcome(
            axis=axis,
            passed=exec_ok and scorer_pass,
            error=err_msg,
            error_type=err_type,
            scorer_scores=scorer_scores,
            scorer_pass=scorer_pass,
            invariant_results=invariant_results,
            call_log=call_log,
        ))

    return ProbeReport(
        audit_flags=audit_flags,
        outcomes=outcomes,
        stubbed_names=sorted(stubbed_seen),
        scorer_threshold=scorer_threshold,
    )


def _evaluate_scorers(
    pairs: list[tuple[str, Callable[..., Any]]],
    scene_before: dict,
    scene_after: dict,
    call_log: list[dict],
    *,
    threshold: float,
    exec_ok: bool,
) -> tuple[dict[str, float], bool]:
    """Run scorers; return ``(per_scorer_scores, all_above_threshold)``.

    When the program crashed, every scorer is treated as 0.0 (failed). This
    means a probe that didn't even run can never satisfy a scorer-based
    likelihood — desirable: the agent shouldn't get reward credit for a
    dead trajectory. With zero scorers, the dict is empty and ``passed``
    is True (no constraints to violate).
    """
    if not pairs:
        return {}, True
    if not exec_ok:
        return ({name: 0.0 for name, _ in pairs}, False)
    scores: dict[str, float] = {}
    for name, fn in pairs:
        try:
            v = float(fn(scene_before, scene_after, call_log))
        except Exception:
            v = 0.0
        scores[name] = max(0.0, min(1.0, v))
    passed = all(s >= threshold for s in scores.values())
    return scores, passed


def _evaluate_invariants(
    pairs: list[tuple[str, Callable[..., Any]]],
    scene_before: dict,
    scene_after: dict,
    call_log: list[dict],
) -> dict[str, bool]:
    """Run every invariant; return ``{name: held?}``.

    Exceptions inside an invariant are treated as a violation (``False``) so
    a buggy invariant fails closed. Callers that need the legacy "all held"
    boolean can use ``all(results.values())`` or the
    :attr:`ProbeOutcome.invariant_pass` property.
    """
    results: dict[str, bool] = {}
    for name, fn in pairs:
        try:
            results[name] = bool(fn(scene_before, scene_after, call_log))
        except Exception:
            results[name] = False
    return results


