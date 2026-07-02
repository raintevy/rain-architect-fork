"""Scorer + invariant artifacts emitted from a single user correction (P3).

A correction like *"approach from the side, not above"* is asked of the LLM
to be operationalized as two artifacts:

  * a single ``score(scene_before, scene_after, call_log) -> float`` returning
    a value in ``[0, 1]`` measuring how well a candidate program satisfies
    the correction, and
  * zero or more ``invariant_<n>(scene_before, scene_after, call_log) -> bool``
    predicates that should hold for any program *fully* satisfying it.

Scorers feed C1's active-sampling ranker (per-candidate likelihood); invariants
feed both the C1 ranker and the C2 graduation gate (skill admitted only if all
invariants hold across the perturbation suite). They are persisted in the
``scorers`` table of the per-robot library DB so they survive sessions and
serve as regression tests on future programs.

Runtime contract for both artifacts (same signature, different return type):

    def score(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float
    def invariant_0(scene_before, scene_after, call_log) -> bool

  * ``scene_before`` / ``scene_after`` — ``dataclasses.asdict`` of
    :class:`architect.scene_mutator.Scene`. Plain dicts so the artifact can be
    pickled / persisted / shipped without importing project modules.
  * ``call_log`` — list of ``{"name": str, "args": list, "kwargs": dict}``
    capturing the program's robot-API calls in execution order. The probe
    runner builds this by wrapping the namespace's primitive functions.

This module is responsible for *emission* (LLM call → artifact) and
*loading* (artifact → callable). Probe-time evaluation lives in
:mod:`architect.probes.probe_runner`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from architect.probes.scene_mutator import Scene


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


@dataclass
class ScorerArtifact:
    """A persisted scorer or invariant.

    ``code`` is a complete top-level ``def <name>(...): ...``. ``preamble``
    holds any imports, helper functions, and module-level statements emitted
    alongside the artifact in the same LLM response. At load time the
    runtime exec's ``preamble + "\\n" + code`` into a fresh sandbox so the
    named callable can resolve helpers and stdlib imports the LLM relied on.

    Loading is lazy: the store keeps source, callers materialize via
    :func:`load_artifact` on demand. Rationale — keeping artifacts as source
    means they're inspectable, hand-editable by the user, and don't carry
    pickled closures across Python versions.
    """

    name: str
    kind: str  # 'scorer' | 'invariant'
    code: str
    description: str
    preamble: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing the LLM response
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_python_blocks(text: str) -> list[str]:
    """Return the content of every ```python ...``` fenced block in order.

    Falls back to treating the entire ``text`` as a single block when no
    fences are present — some LLM outputs skip them when only one block is
    requested.
    """
    blocks = [b.strip() for b in _FENCE_RE.findall(text) if b.strip()]
    if blocks:
        return blocks
    stripped = text.strip()
    return [stripped] if stripped else []


def _functions_in(code: str) -> list[str]:
    """Names of top-level ``def`` functions in ``code``. ``[]`` if it doesn't parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]


def _is_target_def(node: ast.AST) -> bool:
    """True if ``node`` is the scorer or an invariant function definition."""
    return (
        isinstance(node, ast.FunctionDef)
        and (node.name == "score" or re.fullmatch(r"invariant_\d+", node.name) is not None)
    )


def parse_response(
    text: str,
) -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
    """Extract scorer + invariant defs along with their shared preamble.

    Returns ``(scorer_pair_or_None, [(invariant_code, preamble), ...])``.
    Each pair is ``(code, preamble)`` where:

    * ``code`` is the single ``def score`` / ``def invariant_<n>`` source.
    * ``preamble`` is the concatenation of every module-level statement in the
      LLM response that is NOT a target def: imports, helper ``def``s,
      constants, class definitions. The same preamble is attached to every
      artifact in the response so the loader can resolve cross-references
      (``import math``, ``def _is_side_approach``, etc.) regardless of which
      fence block the helper appeared in.

    Multiple defs in one fence are split apart; defs across separate fences
    accumulate. The classifier matches *exactly* ``score`` for the scorer
    and an ``invariant_<digit>`` prefix for invariants so the LLM has only
    one valid naming convention to obey.
    """
    blocks = _extract_python_blocks(text)

    preamble_parts: list[str] = []
    scorer_node: ast.FunctionDef | None = None
    invariant_nodes: list[ast.FunctionDef] = []

    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if _is_target_def(node):
                if node.name == "score":
                    if scorer_node is None:
                        scorer_node = node
                    else:
                        # A duplicate ``def score`` from the LLM is treated as
                        # a helper — we keep the first definition canonical.
                        preamble_parts.append(ast.unparse(node))
                else:
                    invariant_nodes.append(node)
            else:
                preamble_parts.append(ast.unparse(node))

    preamble = "\n\n".join(preamble_parts)

    scorer_pair: tuple[str, str] | None = None
    if scorer_node is not None:
        scorer_pair = (ast.unparse(scorer_node), preamble)

    invariant_pairs: list[tuple[str, str]] = [
        (ast.unparse(node), preamble) for node in invariant_nodes
    ]

    return scorer_pair, invariant_pairs


# ---------------------------------------------------------------------------
# Validation + loading
# ---------------------------------------------------------------------------


def _combine(preamble: str, code: str) -> str:
    """Join preamble and code with a separator, omitting the join if either is empty."""
    if not preamble:
        return code
    if not code:
        return preamble
    return preamble + "\n\n" + code


def validate(
    code: str,
    expected_name: str,
    *,
    preamble: str = "",
) -> tuple[bool, str | None]:
    """Sandbox-exec ``preamble + code``; check it defines a callable named
    ``expected_name`` with the 3-arg ``(scene_before, scene_after, call_log)``
    signature.

    Returns ``(ok, error_message)``. The ``error_message`` is suitable for
    display in a CLI / regression test. ``preamble`` carries imports and
    helper defs the artifact relies on; passing an empty string preserves
    the legacy "single-def" validation.
    """
    full = _combine(preamble, code)
    try:
        tree = ast.parse(full)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == expected_name for f in fns):
        names = [f.name for f in fns]
        return False, f"missing def {expected_name}; defined: {names!r}"

    sandbox: dict[str, Any] = {}
    try:
        exec(full, sandbox)  # noqa: S102
    except Exception as exc:
        return False, f"exec failed: {type(exc).__name__}: {exc}"

    fn = sandbox.get(expected_name)
    if not callable(fn):
        return False, f"{expected_name} is not callable"

    target = next(f for f in fns if f.name == expected_name)
    n_pos = len(target.args.args)
    if n_pos != 3:
        return False, (
            f"{expected_name} takes {n_pos} positional args; expected 3 "
            f"(scene_before, scene_after, call_log)"
        )

    return True, None


def load_artifact(artifact: ScorerArtifact) -> Callable[..., Any] | None:
    """Exec the artifact's preamble + code and return the live callable, or
    ``None`` on failure. The returned callable closes over the sandbox dict
    so it keeps any helper functions and imports defined in the preamble.

    The artifact's stored ``name`` may carry a session/correction prefix
    (e.g. ``c1_score``) but the ``def`` inside ``code`` is always the bare
    canonical name (``score`` or ``invariant_<n>``). The actual lookup name
    is therefore parsed out of ``code`` rather than read off ``artifact.name``.
    """
    sandbox: dict[str, Any] = {}
    try:
        exec(_combine(artifact.preamble, artifact.code), sandbox)  # noqa: S102
    except Exception:
        return None
    try:
        tree = ast.parse(artifact.code)
    except SyntaxError:
        return None
    fn_name = next(
        (n.name for n in tree.body if isinstance(n, ast.FunctionDef)),
        None,
    )
    if fn_name is None:
        return None
    fn = sandbox.get(fn_name)
    return fn if callable(fn) else None


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


# Default model used when the caller hasn't pinned one. Imported lazily so
# unit tests can inject a stub LLM without dragging in the real client.
def _default_llm_call(messages: list[dict], model: str) -> str:
    from architect.llm.client import call_claude
    return call_claude(messages, model=model)


def emit_scorer_and_invariants(
    correction: str,
    current_program: str,
    *,
    scene: Scene | None = None,
    llm_call: Callable[..., str] | None = None,
    model: str | None = None,
    name_prefix: str = "",
) -> tuple[ScorerArtifact | None, list[ScorerArtifact]]:
    """Run the LLM once to produce a scorer + invariant artifacts for ``correction``.

    Returns ``(scorer_artifact_or_None, [invariant_artifact, ...])``. The
    scorer is ``None`` if the response did not contain a parseable +
    validate-able ``def score(...)``; invariants that fail validation are
    silently dropped (and not stored). The caller decides what to do —
    usually the agent loop logs a warning and moves on.

    ``llm_call`` is dependency-injected so tests can stub the LLM. Its
    contract matches :func:`architect.llm_client.call_claude`:
    ``llm_call(messages, model) -> str``.

    ``name_prefix`` lets the caller scope artifact names to a session or
    correction id, avoiding ``UNIQUE`` violations when multiple corrections
    in one run produce defs with the same canonical name.
    """
    from architect.llm.prompts import build_scorer_prompt

    llm = llm_call if llm_call is not None else _default_llm_call
    if model is None:
        from config import ANTHROPIC_MODEL  # late import — avoids circular import
        model = ANTHROPIC_MODEL

    messages = build_scorer_prompt(correction, current_program, scene=scene)
    response = llm(messages, model=model)

    scorer_pair, invariant_pairs = parse_response(response)
    provenance = {
        "correction": correction,
        "program_excerpt": current_program[:500],
        "scene": asdict(scene) if scene is not None else None,
    }

    scorer_artifact: ScorerArtifact | None = None
    if scorer_pair is not None:
        scorer_src, scorer_preamble = scorer_pair
        ok, _err = validate(scorer_src, "score", preamble=scorer_preamble)
        if ok:
            scorer_artifact = ScorerArtifact(
                name=f"{name_prefix}score" if name_prefix else "score",
                kind="scorer",
                code=scorer_src,
                description=correction,
                preamble=scorer_preamble,
                provenance=provenance,
            )

    invariant_artifacts: list[ScorerArtifact] = []
    for i, (src, preamble) in enumerate(invariant_pairs):
        # The parser already verified each block is a single FunctionDef
        # named ``invariant_<digit>``; pull the name out for validation.
        try:
            tree = ast.parse(src)
            inv_name = next(
                n.name for n in tree.body if isinstance(n, ast.FunctionDef)
            )
        except (SyntaxError, StopIteration):
            continue
        ok, _err = validate(src, inv_name, preamble=preamble)
        if not ok:
            continue
        invariant_artifacts.append(ScorerArtifact(
            name=f"{name_prefix}{inv_name}",
            kind="invariant",
            code=src,
            description=correction,
            preamble=preamble,
            provenance={**provenance, "index": i},
        ))

    return scorer_artifact, invariant_artifacts
