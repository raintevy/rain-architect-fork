"""Active counterfactual program sampling (C1).

Given a single correction, generate a slate of ``n`` candidate programs that
each commit to a *different* design choice on some under-specified axis, then
rank them by:

  - **Probe pass-rate** (does the program survive counterfactual scenes?)
  - **Mean scorer score** in [0, 1] over stored scorers
  - **Audit penalty**       (high-severity audit flags pull the score down)

Behavioral fingerprints (the AST sequence of API call names) are used to
*deduplicate* before ranking — two candidates with identical call sequences
are not "different" no matter how the LLM labelled them.

The output is a :class:`Slate` of :class:`Candidate` objects sorted by score
descending, plus a ``dominant`` flag the agent uses to decide whether to
auto-apply or surface to the user. The user's pick (when surfaced) feeds the
episode log via :mod:`architect.library.episode_log`, which records the *un-picked* axis
labels as multi-axis supervision per §4.3 of the design doc.

Paper claim C1 lives here: candidates are sampled to disagree on what the
correction left under-specified; the user's choice between visibly different
programs is the highest-information signal we can extract from a single
correction.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from architect.probes.audit import RiskFlag, audit_program, severity_summary
from architect.probes.probe_runner import ProbeReport, run_probes
from architect.probes.scene_mutator import Scene, default_scene
from architect.corrections.llm_scorer import ScorerArtifact


# ---------------------------------------------------------------------------
# Candidate + Slate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One LLM-emitted alternative refinement of a correction.

    ``axis_label`` is the LLM's own description of the design choice this
    candidate commits to (e.g. *"approach from above with a 10cm offset"*).
    It surfaces in the diff UI and is the supervision signal recorded against
    rejected candidates in the episode log.
    """

    code: str
    axis_label: str
    fingerprint: tuple[str, ...]
    audit_flags: list[RiskFlag] = field(default_factory=list)
    probe_report: ProbeReport | None = None
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class Slate:
    correction: str
    candidates: list[Candidate]
    dominant: bool                # if True, top candidate is unambiguously best
    auto_apply_threshold: float   # the threshold used to compute ``dominant``

    def __len__(self) -> int:
        return len(self.candidates)

    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


# ---------------------------------------------------------------------------
# Parse the LLM response into Candidate objects
# ---------------------------------------------------------------------------


# Matches:  # Candidate <label>: <description>\n```python\n<code>\n```
# The label is anything up to ":" or "\n"; description is the rest of the line.
_CANDIDATE_RE = re.compile(
    r"#\s*Candidate\s+([^:\n]+?)\s*:\s*([^\n]+)\s*\n+"
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_slate_response(text: str) -> list[Candidate]:
    """Extract every ``# Candidate X: <label>\\n```python ...``` block in order.

    Dedup key is the *normalized code text* (whitespace + comments collapsed),
    not the behavioral fingerprint. Two programs with the same call sequence
    but different arguments — e.g. one moving above the object and one
    moving to its side — share a fingerprint but have different behavior, so
    we keep both. The fingerprint is reserved for MMR distance in
    :func:`select_diverse`. Returns an empty list if the response had no
    parseable candidates — the caller decides whether to retry or fall back.
    """
    out: list[Candidate] = []
    seen_codes: set[str] = set()
    for match in _CANDIDATE_RE.finditer(text):
        _label_id, description, code = match.group(1), match.group(2), match.group(3)
        code = code.strip()
        if not code:
            continue
        key = _normalize_code(code)
        if key in seen_codes:
            continue
        seen_codes.add(key)
        out.append(Candidate(
            code=code,
            axis_label=description.strip(),
            fingerprint=behavioral_fingerprint(code),
        ))
    return out


def _normalize_code(code: str) -> str:
    """Strip comments, blank lines, and extra whitespace.

    Used as the dedup key for slate candidates. Robust to identical programs
    with different formatting; preserves all argument values so candidates
    differing only in numeric literals (offset magnitudes, etc.) survive
    deduplication.
    """
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except SyntaxError:
        # Programs that don't parse keep their raw text — they will fail at
        # exec time but we still surface them so the user sees the LLM's
        # mistake rather than silently dropping it.
        return " ".join(code.split())


# ---------------------------------------------------------------------------
# Behavioral fingerprint
# ---------------------------------------------------------------------------


def behavioral_fingerprint(code: str) -> tuple[str, ...]:
    """Sequence of ``Name(...)`` call expressions in source order.

    Two programs with identical fingerprints make the same robot-API calls
    in the same order; argument differences (literals, variables) are
    normalized away. This is the dedup key used by :func:`parse_slate_response`
    and the diversity signal used by :func:`select_diverse`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Programs that don't parse get a unique singleton fingerprint so
        # they can't be deduplicated against valid programs.
        return ("__syntax_error__",)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return tuple(names)


def fingerprint_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard-style distance over the call-sequence multisets, in ``[0, 1]``.

    Used by :func:`select_diverse` for MMR. Identical sequences score ``0``;
    completely disjoint sequences score ``1``. Length differences are folded
    in: a 5-call program vs a 10-call program with the same set of names
    returns ``0.5``.
    """
    if not a and not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    set_dist = 1.0 - (len(intersection) / len(union))
    # Length asymmetry: 0 when same length, → 1 as one shrinks to nothing.
    if max(len(a), len(b)) == 0:
        len_dist = 0.0
    else:
        len_dist = 1.0 - (min(len(a), len(b)) / max(len(a), len(b)))
    return 0.7 * set_dist + 0.3 * len_dist


# ---------------------------------------------------------------------------
# Subset selection (MMR over fingerprints)
# ---------------------------------------------------------------------------


def select_diverse(
    candidates: list[Candidate],
    k: int,
    *,
    relevance: Callable[[Candidate], float] | None = None,
    lambda_: float = 0.5,
) -> list[Candidate]:
    """Pick ``k`` candidates by Maximal Marginal Relevance.

    With ``relevance=None``, the relevance term defaults to ``candidate.score``,
    so pre-ranking by :func:`rank_candidates` makes this select the
    score-and-distance-balanced subset. ``lambda_=0`` is pure diversity;
    ``lambda_=1`` recovers the top-k by relevance.
    """
    if k <= 0 or not candidates:
        return []
    rel: Callable[[Candidate], float] = relevance or (lambda c: c.score)
    pool = list(candidates)
    pool.sort(key=rel, reverse=True)
    picked: list[Candidate] = [pool.pop(0)]

    while pool and len(picked) < k:
        best_idx = 0
        best_score = float("-inf")
        for i, c in enumerate(pool):
            min_dist = min(
                fingerprint_distance(c.fingerprint, p.fingerprint) for p in picked
            )
            mmr = lambda_ * rel(c) + (1.0 - lambda_) * min_dist
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        picked.append(pool.pop(best_idx))

    return picked


# ---------------------------------------------------------------------------
# Sampling (one LLM call → list[Candidate])
# ---------------------------------------------------------------------------


def _default_llm_call(messages: list[dict], model: str) -> str:
    from architect.llm.client import call_claude
    return call_claude(messages, model=model, max_tokens=8192, temperature=0.7)


def sample_slate(
    correction: str,
    current_program: str,
    *,
    api_spec: str,
    robot_label: str,
    n: int = 3,
    current_ee_pose: dict | None = None,
    execution_trace=None,
    fault_summary: str | None = None,
    skill_docs: list[tuple[str, str]] | None = None,
    llm_call: Callable[..., str] | None = None,
    model: str | None = None,
) -> list[Candidate]:
    """Single LLM call → ``n``-ish behaviorally-different candidate programs.

    The LLM is asked to emit candidates each committing to a different
    design choice, with explicit axis labels. Duplicates by behavioral
    fingerprint are dropped during parsing; the returned list may have
    fewer than ``n`` entries if the LLM was unable to produce that many
    behaviorally-distinct candidates (we surface that honestly rather than
    padding with near-duplicates).

    ``execution_trace`` and ``fault_summary`` are passed straight through
    to :func:`architect.prompt_builder.build_slate_correction_prompt`. See its
    docstring for what each contributes; both are optional and the slate
    sampler works without them (just with less context per candidate).

    ``temperature`` is set to ``0.7`` in the default LLM call so the LLM
    actually explores the design-choice space; the standard correction
    flow uses ``0.2``.
    """
    from architect.llm.prompts import build_slate_correction_prompt

    llm = llm_call if llm_call is not None else _default_llm_call
    if model is None:
        from config import ANTHROPIC_MODEL
        model = ANTHROPIC_MODEL

    messages = build_slate_correction_prompt(
        current_program, correction,
        api_spec=api_spec, robot_label=robot_label,
        n=n, current_ee_pose=current_ee_pose,
        execution_trace=execution_trace,
        fault_summary=fault_summary,
        skill_docs=skill_docs,
    )
    response = llm(messages, model=model)
    return parse_slate_response(response)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _aggregate_per_scorer(outcomes: list[ProbeOutcome]) -> dict[str, float]:
    """Mean of each scorer's per-probe scores across ``outcomes``.

    Returns ``{}`` if no probes carried a per-scorer score (e.g. no scorers
    were supplied to ``run_probes``). The keys are the artifact names from
    the SkillStore, so the diff UI can label them with their description
    by joining against the store.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for o in outcomes:
        for name, score in o.scorer_scores.items():
            sums[name] = sums.get(name, 0.0) + score
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def _aggregate_invariant_violations(outcomes: list[ProbeOutcome]) -> dict[str, float]:
    """Fraction of probes in which each invariant was violated.

    A value of ``0.0`` means the invariant held on every probe; ``1.0``
    means it was violated on every probe — the latter usually signals
    that the invariant is irrelevant to this correction (cross-talk from
    a stale invariant built for a different correction).
    """
    violated: dict[str, int] = {}
    total: dict[str, int] = {}
    for o in outcomes:
        for name, held in o.invariant_results.items():
            total[name] = total.get(name, 0) + 1
            if not held:
                violated[name] = violated.get(name, 0) + 1
    return {
        name: violated.get(name, 0) / total[name] for name in total
    }


def rank_candidates(
    candidates: list[Candidate],
    *,
    base_scene: Scene | None = None,
    robot_name: str = "franka",
    scorers: tuple[ScorerArtifact, ...] = (),
    invariants: tuple[ScorerArtifact, ...] = (),
    k_probes: int = 8,
    seed: int | None = None,
) -> list[Candidate]:
    """Score each candidate via :func:`run_probes` and sort descending.

    Score formula (all components in ``[0, 1]``):

        score = probe_pass_rate
              * (mean_scorer_score if scorers else 1.0)
              - 0.05 * #high_severity_audit_flags
              - 0.02 * #medium_severity_audit_flags

    The audit penalty is small enough that a clean program with a slightly
    lower probe pass-rate can still outrank a high-passing program with
    a hardcoded pose, but large enough that flagged programs get nudged
    down the slate. Negative scores are floored at zero.

    Mutates ``candidates`` in place (sets each candidate's score / probe
    report / audit flags) and returns them sorted.
    """
    base = base_scene if base_scene is not None else default_scene(robot_name)

    for c in candidates:
        c.audit_flags = audit_program(c.code)
        c.probe_report = run_probes(
            c.code,
            base_scene=base,
            robot_name=robot_name,
            k=k_probes,
            seed=seed,
            scorers=scorers,
            invariants=invariants,
        )
        pass_rate = c.probe_report.pass_rate
        scorer_scores = [
            o.scorer_score for o in c.probe_report.outcomes
            if o.scorer_score is not None
        ]
        scorer_mean = (
            sum(scorer_scores) / len(scorer_scores)
            if scorer_scores else 1.0
        )
        sev = severity_summary(c.audit_flags)
        audit_penalty = 0.05 * sev.get("high", 0) + 0.02 * sev.get("medium", 0)

        # Per-artifact aggregation across probes. Preserves attribution so the
        # slate UI and episode log can show which scorer or invariant moved
        # the candidate's score, not just the rolled-up mean.
        per_scorer = _aggregate_per_scorer(c.probe_report.outcomes)
        per_invariant_violation_rate = _aggregate_invariant_violations(
            c.probe_report.outcomes
        )

        raw = pass_rate * scorer_mean - audit_penalty
        c.score = max(0.0, min(1.0, raw))
        c.score_breakdown = {
            "pass_rate": pass_rate,
            "scorer_mean": scorer_mean,
            "audit_penalty": audit_penalty,
            "per_scorer": per_scorer,
            "per_invariant_violation_rate": per_invariant_violation_rate,
        }

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_slate(
    correction: str,
    current_program: str,
    *,
    api_spec: str,
    robot_label: str,
    n: int = 3,
    current_ee_pose: dict | None = None,
    execution_trace=None,
    fault_summary: str | None = None,
    skill_docs: list[tuple[str, str]] | None = None,
    base_scene: Scene | None = None,
    robot_name: str = "franka",
    scorers: tuple[ScorerArtifact, ...] = (),
    invariants: tuple[ScorerArtifact, ...] = (),
    k_probes: int = 8,
    seed: int | None = None,
    auto_apply_threshold: float = 0.85,
    auto_apply_margin: float = 0.20,
    llm_call: Callable[..., str] | None = None,
    model: str | None = None,
) -> Slate:
    """One-shot: sample → rank → diversify → return a :class:`Slate`.

    ``execution_trace`` (the most recent live-run trace) and
    ``fault_summary`` (the pre-correction LLM digest of what likely went
    wrong) are forwarded to :func:`sample_slate` and end up in the slate
    generator's user message — see
    :func:`architect.prompt_builder.build_slate_correction_prompt` for the
    rendering. Both are optional; sampling works without them.

    ``dominant`` is set when the top candidate's score is at least
    ``auto_apply_threshold`` AND its margin over the runner-up is at least
    ``auto_apply_margin``. The agent uses this flag to decide auto-apply vs.
    surface-to-user.
    """
    candidates = sample_slate(
        correction, current_program,
        api_spec=api_spec, robot_label=robot_label,
        n=n, current_ee_pose=current_ee_pose,
        execution_trace=execution_trace,
        fault_summary=fault_summary,
        skill_docs=skill_docs,
        llm_call=llm_call, model=model,
    )
    if not candidates:
        return Slate(
            correction=correction, candidates=[],
            dominant=False, auto_apply_threshold=auto_apply_threshold,
        )

    rank_candidates(
        candidates,
        base_scene=base_scene, robot_name=robot_name,
        scorers=scorers, invariants=invariants,
        k_probes=k_probes, seed=seed,
    )

    diverse = select_diverse(candidates, k=min(n, len(candidates)))

    top = diverse[0]
    second = diverse[1].score if len(diverse) > 1 else 0.0
    dominant = (top.score >= auto_apply_threshold) and (top.score - second >= auto_apply_margin)

    return Slate(
        correction=correction,
        candidates=diverse,
        dominant=dominant,
        auto_apply_threshold=auto_apply_threshold,
    )
