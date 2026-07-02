"""Graduation gate for newly-approved primitives (P5 / C2).

A primitive approved via ``write_primitive`` is initially a *candidate*: it
sits in the SkillStore with ``graduation_status='candidate'`` and is callable
in-session, but won't yet be retrieved into future-session prompts. The
graduation gate decides whether to promote it to ``'graduated'`` by:

  1. Synthesizing a minimal test program that calls the primitive with
     permissive stub arguments.
  2. Running ``run_probes`` against that test program across ``k``
     counterfactual perturbations of the current scene.
  3. Admitting the primitive only if the probe pass-rate ≥ ``threshold``
     AND every invariant supplied to the gate held on every probe.

The probe suite is persisted as a row in ``probe_suites``; the skill row's
``probe_suite_id`` is updated to point at it so a future retrieval-time
``recompile_check`` can replay the exact same perturbations and verify
the skill still passes.

This is the C2 paper claim made executable: skills carry their generalization
evidence with them. Failing the gate is not destructive — the primitive
stays in the library as a candidate, surfaced via ``/library`` and
force-promotable via ``/graduate <name>`` if the user disagrees with the
verdict.
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from architect.probes.probe_runner import ProbeReport, _PermissiveStub, run_probes
from architect.probes.scene_mutator import Scene, default_scene
from architect.corrections.llm_scorer import ScorerArtifact


def _json_fallback(obj: Any) -> str:
    """JSON encoder fallback for stub objects captured in call_log entries.

    The probe runner wraps API primitives so every call records its args +
    kwargs verbatim. When the primitive under test threads its own
    arguments through those wrapped functions, permissive-stub instances
    end up in the log. They serialise to ``"<_PermissiveStub>"`` rather
    than blowing up the entire suite-row INSERT.
    """
    return f"<{type(obj).__name__}>"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GraduationResult:
    """Outcome of one graduation attempt.

    Persistence happens *outside* this dataclass: ``try_graduate`` returns
    the result and the caller (typically ``agent_tools._write_primitive``)
    decides whether to call ``store.register_probe_suite`` +
    ``store.admit`` based on the verdict. Keeping the side-effects out
    of the pure function makes it easy to unit-test without a SkillStore.
    """

    name: str
    admitted: bool
    pass_rate: float
    invariant_pass_rate: float
    all_invariants_hold: bool
    threshold: float
    seed: int
    k: int
    robot_name: str
    probe_report: ProbeReport | None = None
    reason: str = ""

    def report_json(self) -> str:
        """A compact JSON snapshot of the probe report — what `probe_suites.report_json` stores.

        The call_log entries inside each ProbeOutcome can carry permissive-
        stub objects as argument values when the primitive under test
        forwards its own arguments to logged API calls (e.g. ``def
        grasp_object_at_pose(grasp_pose): move_ee_to_pose(grasp_pose)``
        called with ``_GRADUATION_STUB`` as ``grasp_pose``). Those stub
        instances aren't JSON-serialisable, so we attach a default fallback
        that stringifies anything :mod:`json` can't otherwise handle.
        Without this, the entire suite-row INSERT used to fail silently and
        the skill stayed at ``candidate`` despite passing the gate.
        """
        if self.probe_report is None:
            return ""
        outcomes = [asdict(o) for o in self.probe_report.outcomes]
        flags = [asdict(f) for f in self.probe_report.audit_flags]
        return json.dumps({
            "audit_flags": flags,
            "outcomes": outcomes,
            "stubbed_names": list(self.probe_report.stubbed_names),
            "scorer_threshold": self.probe_report.scorer_threshold,
        }, default=_json_fallback)


# ---------------------------------------------------------------------------
# Test program synthesis
# ---------------------------------------------------------------------------


_STUB_NAME = "_GRADUATION_STUB"


def _count_required_positional_args(code: str, name: str) -> int:
    """Inspect the def for ``name`` in ``code`` and count its required positional args.

    The graduation test program passes the same permissive stub for every
    required positional — works for primitives that subscript / call / iterate
    their args because the stub absorbs all of those patterns. Variadic args
    (``*args`` / ``**kwargs``) are not counted; the call site won't supply them.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            n_pos = len(args.args)
            n_default = len(args.defaults)
            return max(0, n_pos - n_default)
    return 0


def _build_test_program(name: str, code: str) -> str:
    """Synthesize ``<def>\\n<name>(<stub>, <stub>, ...)``.

    The primitive's full source is included so that ``run_probes`` exec's it
    into the same namespace that calls it — same arrangement as a real
    submitted program. The stub sentinel is bound by the namespace factory
    via :func:`_make_factory` so the probe runner doesn't need to know
    about it specifically.
    """
    n = _count_required_positional_args(code, name)
    call_args = ", ".join([_STUB_NAME] * n)
    return f"{code.rstrip()}\n\n# graduation gate test call\n{name}({call_args})\n"


def _make_factory() -> Callable[[], dict]:
    """Namespace factory that pre-binds the permissive graduation stub.

    All other API calls (``move_ee_to_pose``, ``detect_objects``, etc.) are
    handled by the probe runner — :func:`bind_scene` injects perception
    functions, and any remaining names get auto-stubbed.
    """
    def factory() -> dict:
        return {_STUB_NAME: _PermissiveStub()}
    return factory


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


# Default graduation threshold. ``0.7`` matches the value cited in §9 P5 of
# the implementation plan and §8.4 of the design doc. Real-robot eval can
# override per experiment.
DEFAULT_THRESHOLD: float = 0.7
DEFAULT_K: int = 8
DEFAULT_SEED: int = 42


def try_graduate(
    name: str,
    code: str,
    *,
    base_scene: Scene | None = None,
    robot_name: str = "franka",
    scorers: tuple[ScorerArtifact, ...] = (),
    invariants: tuple[ScorerArtifact, ...] = (),
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
    require_invariants: bool = True,
) -> GraduationResult:
    """Probe-test a primitive; decide whether to admit it to the library.

    The decision rule:

      admitted = pass_rate ≥ threshold
                 AND (no invariants supplied OR every invariant held on every probe)

    Scorers are off by default for the graduation gate — they're
    correction-specific and have known cross-talk on out-of-context
    candidates (see architect.library.relevance for the C1-side mitigation). When the
    caller does pass scorers, they participate in the per-probe ``passed``
    flag the way they would for C1 ranking.

    Returns a :class:`GraduationResult` describing the outcome. Persistence
    (writing a ``probe_suites`` row + calling ``store.admit``) is the
    caller's responsibility.
    """
    if base_scene is None:
        base_scene = default_scene(robot_name)

    test_program = _build_test_program(name, code)

    probe_report = run_probes(
        test_program,
        base_scene=base_scene,
        robot_name=robot_name,
        k=k,
        seed=seed,
        scorers=scorers,
        invariants=invariants,
        namespace_factory=_make_factory(),
    )

    pass_rate = probe_report.pass_rate
    inv_rate = probe_report.invariant_pass_rate
    all_inv = probe_report.all_invariants_hold

    if pass_rate < threshold:
        reason = (
            f"probe pass-rate {pass_rate:.2f} < threshold {threshold:.2f} "
            f"({probe_report.n_passed}/{probe_report.k} probes passed)"
        )
        admitted = False
    elif require_invariants and not all_inv:
        reason = (
            f"probe pass-rate {pass_rate:.2f} OK, but invariants violated "
            f"on {sum(1 for o in probe_report.outcomes if not o.invariant_pass)}/"
            f"{probe_report.k} probes"
        )
        admitted = False
    else:
        reason = (
            f"probe pass-rate {pass_rate:.2f} ≥ {threshold:.2f}"
            + (f", invariants {inv_rate:.0%} hold" if invariants else "")
        )
        admitted = True

    return GraduationResult(
        name=name,
        admitted=admitted,
        pass_rate=pass_rate,
        invariant_pass_rate=inv_rate,
        all_invariants_hold=all_inv,
        threshold=threshold,
        seed=seed,
        k=k,
        robot_name=robot_name,
        probe_report=probe_report,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Persistence helper (used by agent_tools._write_primitive)
# ---------------------------------------------------------------------------


def persist_and_admit(
    store,  # SkillStore — typed loosely to avoid the import cycle
    result: GraduationResult,
    *,
    console=None,
) -> int | None:
    """Write a probe_suites row for ``result``; admit the skill on pass.

    Returns the new ``probe_suite_id`` (always written, even for failed
    graduation attempts — failed attempts are valuable audit data) or
    ``None`` if persistence failed.

    Failed graduations leave the skill at ``status='candidate'`` but
    still record a probe_suite row so the user can inspect what went
    wrong via ``/library``.

    ``console`` is optional; when supplied, persistence failures are
    surfaced as dim warnings instead of swallowed silently. Without
    this, a real bug in the persistence path (e.g. a non-JSON-serialisable
    object in the report) lets ``try_graduate`` print "✓ graduated" while
    the skill silently stays at candidate.
    """
    if result.probe_report is None:
        return None
    try:
        suite_id = store.register_probe_suite(
            seed=result.seed,
            k=result.k,
            robot_name=result.robot_name,
            pass_rate=result.pass_rate,
            n_passed=result.probe_report.n_passed,
            n_total=result.probe_report.k,
            invariant_pass_rate=result.invariant_pass_rate,
            all_invariants_hold=result.all_invariants_hold,
            report_json=result.report_json(),
        )
    except Exception as exc:
        if console is not None:
            console.print(
                f"  [dim]\\[graduation] persist failed for {result.name!r}: "
                f"{type(exc).__name__}: {exc}[/dim]"
            )
        return None

    if result.admitted:
        try:
            store.admit(result.name, probe_suite_id=suite_id)
        except Exception as exc:
            if console is not None:
                console.print(
                    f"  [dim]\\[graduation] admit failed for {result.name!r}: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
    else:
        # Stash the suite id on the candidate row so ``/library`` can show
        # "last attempt: pass-rate X". Status stays at 'candidate'.
        try:
            store.set_probe_suite_id(result.name, suite_id)
        except Exception as exc:
            if console is not None:
                console.print(
                    f"  [dim]\\[graduation] set_probe_suite_id failed for "
                    f"{result.name!r}: {type(exc).__name__}: {exc}[/dim]"
                )

    return suite_id
