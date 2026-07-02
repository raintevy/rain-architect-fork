"""Pre-registered task-level success scorers for the P7 eval harness.

The probe runner already supports per-probe scorers (P3). For the headline
OOD pass-rate measurement, we need *task-level* scorers that ask "did this
program semantically accomplish the task?" rather than just "did exec()
succeed?".

Without these, OOD pass-rate saturates at 100% because the probe runner's
permissive stubs let any non-crashing program pass — even a one-line
``reset_robot()``. The first --llm claude sweep showed exactly that: 120/120
trials at 100% across ICL / FuncReuse / SlateGated despite the conditions producing
visibly different programs.

These scorers are *registered* in :mod:`scripts.eval.tasks_json`-style
config and *invoked* by ``scripts.eval.run_trial._evaluate_ood``. They
inspect the probe's ``call_log`` for the canonical robot-API patterns:

    detect_objects(...)
       → move_ee_to_pose(grasp_pose)        # approach
       → set_gripper_width(w < 0.05)        # close  — start of cycle
       → get_vqa_response("did it grasp?")  # affirmative — verification gate
       → move_ee_to_pose(place_pose)        # transport
       → set_gripper_width(w >= 0.05)       # release — cycle complete

A cycle counts toward the score only when the *verification gate* fires:
between close and release the program must call ``get_vqa_response`` and
get an affirmative answer (word-token check, see :func:`_vqa_affirmative`).
Without verification, a program that ceremoniously closes-moves-opens on
empty space passes — that's the saturation pattern the previous sweep
showed and the reason this scorer was tightened. The verification
requirement defaults ON; pass ``require_vqa_verification=False`` to fall
back to the structural-only version for ablation.

For tasks requiring N successful cycles (pickup tasks → 1, stack 2 blocks
→ 2, cabinet+bottle → 2 distinct grasp/place phases), the scorer returns
``count / N`` clamped to [0, 1]. With 4 probes per perturbation and 3
perturbations per task, a program that completes 1 of 2 expected cycles
scores 0.5 — well below the 0.70 early-exit threshold, so corrections 2/3
get a chance to fire.

This file is deliberately tiny and stdlib-only. Adding a new task means
adding one line to tasks.json's ``success_scorer`` field referencing one
of these named scorers; no per-task code maintenance.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Pick-place cycle counter (parametrised by min_cycles)
# ---------------------------------------------------------------------------


def _gripper_width(call: dict) -> float | None:
    """Pull the ``width`` argument off a ``set_gripper_width`` call_log entry."""
    args = call.get("args") or []
    kwargs = call.get("kwargs") or {}
    raw = args[0] if args else kwargs.get("width")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# Word tokens used by :func:`_vqa_affirmative` to recognise / reject answers.
# Tokenised lookups so substrings inside other words ("yesterday", "snowed")
# don't trigger false matches. Negation tokens take precedence over
# affirmative ones — "yes, but no" counts as negative.
_VQA_AFFIRMATIVE_WORDS = frozenset({"yes", "true", "affirmative", "correct"})
_VQA_NEGATIVE_WORDS = frozenset({
    "no", "not", "cannot", "empty", "fail", "failed", "unable",
})
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _extract_vqa_answer(result: Any) -> str | None:
    """Pull the answer string out of a ``get_vqa_response`` return value.

    Handles the two robot-specific shapes plus a bare-string fallback:
      * Franka live + dry-run: ``{"data": {"answer": "<string>"}}``
      * Stretch live: bare string
      * Generic ``{"answer": "<string>"}``

    Returns ``None`` when the response shape is unrecognised or carries no
    string answer. Mirrors :func:`architect.execution_trace._extract_vqa_answer`
    so the scorer reads the same answers the trace renderer does.
    """
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"]
        ans = result.get("answer")
        if isinstance(ans, str):
            return ans
    if isinstance(result, str):
        return result
    return None


def _vqa_affirmative(result: Any) -> bool:
    """Heuristic word-token classifier for VQA responses.

    Affirmative iff the answer contains a positive token (``yes``, ``true``,
    ``affirmative``, ``correct``) AND no negation token (``no``, ``not``,
    ``cannot``, ``empty``, ``fail``, ``failed``, ``unable``). Tokenisation is
    on ASCII word boundaries so embedded substrings ("yesterday", "snowed")
    don't match. Returns ``False`` for unrecognisable / non-string responses
    so a malformed VQA backend can't be silently treated as success.
    """
    answer = _extract_vqa_answer(result)
    if not answer:
        return False
    tokens = {w.lower() for w in _WORD_RE.findall(answer)}
    if tokens & _VQA_NEGATIVE_WORDS:
        return False
    return bool(tokens & _VQA_AFFIRMATIVE_WORDS)


def _walk_cycles(
    call_log: list[dict],
    require_vqa_verification: bool = True,
) -> dict[str, int]:
    """Walk the call_log's cycle state machine and report rich diagnostics.

    Used by both :func:`pick_place_cycles` (which only needs the count)
    and :func:`explain_failure` (which needs to tell the agent *why*
    the cycle count was too low — was there no VQA in any cycle?
    were there close/release pairs that didn't have motion between
    them? etc.).

    Returns a dict with:
      * ``verified_cycles``: count of full cycles that satisfied the
        gating logic (close → motion → optional VQA → motion → release,
        with VQA required when require_vqa_verification=True)
      * ``close_release_pairs``: count of close→release pairs regardless
        of whether VQA was satisfied — i.e. the structural cycle count
      * ``cycles_missing_vqa``: pairs that had motion but lacked an
        affirmative VQA in the close-release window
      * ``cycles_missing_motion``: pairs that had affirmative VQA but
        no motion in the close-release window
      * ``any_close``: 1 iff at least one set_gripper_width(close) fired
    """
    verified_cycles = 0
    close_release_pairs = 0
    cycles_missing_vqa = 0
    cycles_missing_motion = 0
    any_close = 0

    state = "awaiting_close"
    moves_after_close = 0
    grasp_verified = False

    for c in call_log:
        name = c.get("name", "")
        if name == "set_gripper_width":
            w = _gripper_width(c)
            if w is None:
                continue
            if w < 0.05:
                any_close = 1
                state = "closed"
                moves_after_close = 0
                grasp_verified = False
            elif state == "closed" and w >= 0.05:
                close_release_pairs += 1
                enough_motion = moves_after_close >= 1
                verification_ok = grasp_verified or not require_vqa_verification
                if enough_motion and verification_ok:
                    verified_cycles += 1
                elif enough_motion and not verification_ok:
                    cycles_missing_vqa += 1
                elif not enough_motion and verification_ok:
                    cycles_missing_motion += 1
                state = "awaiting_close"
        elif state == "closed":
            if name.startswith("move_ee_"):
                moves_after_close += 1
            elif name == "get_vqa_response":
                if _vqa_affirmative(c.get("result")):
                    grasp_verified = True

    return {
        "verified_cycles":       verified_cycles,
        "close_release_pairs":   close_release_pairs,
        "cycles_missing_vqa":    cycles_missing_vqa,
        "cycles_missing_motion": cycles_missing_motion,
        "any_close":             any_close,
    }


def pick_place_cycles(
    min_cycles: int = 1,
    require_vqa_verification: bool = True,
) -> Callable[..., float]:
    """Build a scorer that counts completed pick-place cycles in ``call_log``.

    One cycle (with ``require_vqa_verification=True``, the default) =
        detect_*(...)
        → (some) move_ee_*(...)
        → set_gripper_width(w < 0.05)         # grasp/close
        → get_vqa_response(...) → affirmative  # verification gate
        → (some) move_ee_*(...)
        → set_gripper_width(w >= 0.05)        # release/open

    With ``require_vqa_verification=False`` the verification gate is
    removed — close → motion → release is enough.

    Score is *binary*: ``1.0`` iff the call_log contains at least
    ``min_cycles`` complete cycles, ``0.0`` otherwise. State machine
    factored into :func:`_walk_cycles` so :func:`explain_failure` can
    introspect *why* a cycle count came up short.
    """

    def score(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float:
        if min_cycles <= 0:
            return 1.0
        diag = _walk_cycles(call_log, require_vqa_verification=require_vqa_verification)
        return 1.0 if diag["verified_cycles"] >= min_cycles else 0.0

    # Carry the parametrisation forward so ``ScorerArtifact``-like consumers
    # can introspect what was actually measured.
    suffix = "_verified" if require_vqa_verification else ""
    score.__name__ = f"pick_place_cycles_min_{min_cycles}{suffix}"
    return score


# ---------------------------------------------------------------------------
# Final-state VQA scorer (goal-state check)
# ---------------------------------------------------------------------------


def final_state_vqa(success_keywords: tuple[str, ...] = ()) -> Callable[..., float]:
    """Build a scorer that returns 1.0 iff a VQA call in the call_log whose
    prompt contains any of ``success_keywords`` had an affirmative answer.

    Walks the call_log and locates the **last** matching VQA call — the
    program's final-state check. ``_vqa_affirmative`` parses the recorded
    response. If no matching VQA call exists, the scorer returns 0.0:
    the program didn't actually check it succeeded.

    Why this exists alongside :func:`pick_place_cycles`: the cycle counter
    rewards a structural property ("did N pick-place cycles happen?")
    which can be optimised away by a clever LLM (e.g. "the red block is
    already in place, no need to move it"). A final-state VQA asks the
    LLM-as-its-own-judge whether the goal was actually met. Use it via
    :func:`task_complete` for the AND of both — programs need to do the
    cycles AND end in the goal state.
    """
    keywords_lower = tuple(k.lower() for k in success_keywords)

    def score(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float:
        last_match: dict | None = None
        for c in call_log:
            if c.get("name") != "get_vqa_response":
                continue
            args = c.get("args") or []
            kwargs = c.get("kwargs") or {}
            prompt = args[0] if args else kwargs.get("prompt", "")
            if not isinstance(prompt, str):
                continue
            prompt_lower = prompt.lower()
            if not keywords_lower or any(kw in prompt_lower for kw in keywords_lower):
                last_match = c
        if last_match is None:
            return 0.0
        return 1.0 if _vqa_affirmative(last_match.get("result")) else 0.0

    suffix = "_".join(k.replace(" ", "_") for k in success_keywords[:2]) or "any"
    score.__name__ = f"final_state_vqa_{suffix}"
    return score


# ---------------------------------------------------------------------------
# Combined scorer: cycle count AND final-state VQA
# ---------------------------------------------------------------------------


def task_complete(
    min_cycles: int = 1,
    success_keywords: tuple[str, ...] = (),
    require_vqa_verification: bool = True,
) -> Callable[..., float]:
    """Combined scorer: 1.0 iff the program did N verified pick-place cycles
    AND a final-state VQA call confirmed the goal was met.

    This is the scorer to use when "the program took shortcuts but ended in
    a state that satisfies the goal" is **not** acceptable for the task —
    e.g. ``stack_three_blocks`` where the LLM might decide to skip
    repositioning the red block because it's already on the table. The
    cycle-count check catches that shortcut; the final-state VQA check
    catches programs that did the right number of cycles but ended in the
    wrong arrangement.

    Both component scorers are stricter than necessary in isolation:
      - Cycle-counter alone passes the shortcut.
      - Final-state-VQA alone passes any program whose final VQA happens
        to return "yes" on the canned synthetic scene answer, even if no
        manipulation took place.
    Their conjunction tightens the bar: structurally correct (3 verified
    cycles) AND goal-state correct (final VQA affirmative).
    """
    cycle_scorer = pick_place_cycles(
        min_cycles=min_cycles,
        require_vqa_verification=require_vqa_verification,
    )
    vqa_scorer = final_state_vqa(success_keywords=success_keywords)

    def score(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float:
        cycle_ok = cycle_scorer(scene_before, scene_after, call_log) >= 0.5
        vqa_ok = vqa_scorer(scene_before, scene_after, call_log) >= 0.5
        return 1.0 if (cycle_ok and vqa_ok) else 0.0

    verified_suffix = "_verified" if require_vqa_verification else ""
    score.__name__ = f"task_complete_min{min_cycles}{verified_suffix}"
    return score


# ---------------------------------------------------------------------------
# LLM-judge scorer (free-form instructions, no predefined keywords)
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge for robot manipulation tasks. You receive "
    "a free-form natural-language instruction the robot was asked to "
    "execute, plus a structured summary of what happened during execution: "
    "the sequence of robot actions and the answers to the visual-question-"
    "answering (VQA) queries the program issued at key moments. Decide "
    "whether the robot completed the instruction.\n\n"
    "Weight the VQA observations as the primary signal — they are the "
    "robot's perceptual evidence about the world state. Pay particular "
    "attention to the most recent VQA call (typically the program's "
    "final-state goal check). If no VQA calls were made, the robot did "
    "not actually verify task completion — return completed=false with a "
    "rationale that flags missing verification. Be strict but reasonable: "
    "a noisy VQA answer that nevertheless reads as affirmative against the "
    "instruction should count as success.\n\n"
    "Respond with strict JSON only, no prose, no markdown fences:\n"
    '{\"completed\": true|false, \"rationale\": \"<one sentence>\"}'
)


def _summarise_actions(call_log: list[dict]) -> str:
    """Tiny categorical summary of the call log for the judge prompt.

    Encodes the structural shape (closes / releases / motions / VQA count)
    so the judge can distinguish "the robot did nothing and asked a
    flattering VQA" from "the robot executed a real manipulation
    sequence". Deliberately terse — the VQA observations carry the real
    signal; this just prevents the judge from being fooled by a
    perceptual claim that has no physical action behind it.
    """
    closes = releases = motions = vqa_count = 0
    for c in call_log:
        name = c.get("name", "")
        if name == "set_gripper_width":
            w = _gripper_width(c)
            if w is None:
                continue
            if w < 0.05:
                closes += 1
            else:
                releases += 1
        elif name.startswith("move_ee_") or name == "execute_waypoint_trajectory":
            motions += 1
        elif name == "get_vqa_response":
            vqa_count += 1
    return (
        f"- gripper closes: {closes}\n"
        f"- gripper releases: {releases}\n"
        f"- end-effector / trajectory motions: {motions}\n"
        f"- get_vqa_response calls: {vqa_count}"
    )


def _format_vqa_observations(call_log: list[dict]) -> str:
    """Render VQA Q/A pairs from the call log in temporal order."""
    lines: list[str] = []
    for c in call_log:
        if c.get("name") != "get_vqa_response":
            continue
        args = c.get("args") or []
        kwargs = c.get("kwargs") or {}
        prompt = args[0] if args else kwargs.get("prompt", "")
        answer = _extract_vqa_answer(c.get("result")) or "<no answer>"
        lines.append(f"Q: {prompt}\nA: {answer}")
    if not lines:
        return "<no VQA calls were made during execution>"
    return "\n\n".join(lines)


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the judge's JSON reply. Robust to a stray ```json fence."""
    import json
    import re as _re
    cleaned = text.strip()
    fence = _re.match(r"```(?:json)?\s*\n(.*?)\n```", cleaned, _re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back: look for the first {...} block. Better than crashing —
        # the judge degraded into prose, but we can still rescue the
        # decision if it embedded the JSON.
        m = _re.search(r"\{[^{}]*\}", cleaned)
        if not m:
            return {"completed": False, "rationale": f"judge returned unparseable response: {text[:120]!r}"}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"completed": False, "rationale": f"judge returned unparseable response: {text[:120]!r}"}
    return {
        "completed": bool(obj.get("completed", False)),
        "rationale": str(obj.get("rationale", "")) or "<judge gave no rationale>",
    }


# Per-trace cache so the LLM judge fires once per execution, shared
# between :func:`score_trace` (reads the score) and
# :func:`explain_failure` (reads the rationale). Keyed by ``id(call_log)``
# — the call_log list object is built fresh per :func:`trace_to_call_log`
# invocation in the CLI, so collisions across trials would require Python
# to recycle the same list-object address while the prior list is still
# referenced. The cache is bounded to the last 8 entries to cap memory.
_JUDGE_CACHE: dict[int, dict[str, Any]] = {}
_JUDGE_CACHE_ORDER: list[int] = []


def _vqa_judge_evaluate(
    instruction: str,
    vqa_guidelines: str,
    call_log: list[dict],
    model: str | None = None,
) -> dict[str, Any]:
    """Run the LLM judge once for a given call_log + instruction.

    Returns ``{"completed": bool, "rationale": str}``. Result is cached
    keyed by ``id(call_log)`` so :func:`explain_failure` can read the
    rationale without firing a second LLM call.
    """
    key = id(call_log)
    cached = _JUDGE_CACHE.get(key)
    if cached is not None:
        return cached

    user_msg = (
        f"Task instruction: {instruction!r}\n\n"
        f"Robot action summary:\n{_summarise_actions(call_log)}\n\n"
        f"VQA observations during execution (chronological):\n"
        f"{_format_vqa_observations(call_log)}\n\n"
        f"VQA phrasing conventions in use (these guide how the robot's "
        f"VQA queries are worded — the conventions don't decide success, "
        f"they just contextualize the Q/A pairs above):\n{vqa_guidelines}\n\n"
        f"Now decide whether the robot completed the instruction. "
        f"Respond with strict JSON only."
    )

    try:
        from architect.llm.client import call_claude
        if model is None:
            # ANTHROPIC_MODEL lives in repo-root config.py (not the architect
            # package), so import it lazily via runpy / sys.path rather
            # than a hard import — the scorer module is otherwise
            # stdlib-only and self-contained.
            from config import ANTHROPIC_MODEL as _default_model
            judge_model = _default_model
        else:
            judge_model = model
        response = call_claude(
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=judge_model,
            max_tokens=256,
            temperature=0.0,
        )
        result = _parse_judge_response(response)
    except Exception as exc:
        # LLM unreachable / API error — fail closed so closed-loop doesn't
        # silently promote bad runs. The rationale surfaces the issue to
        # the operator via explain_failure.
        result = {
            "completed": False,
            "rationale": (
                f"vqa_judge LLM call failed ({type(exc).__name__}: {exc}); "
                f"cannot assess completion. Falling back to FAIL."
            ),
        }

    _JUDGE_CACHE[key] = result
    _JUDGE_CACHE_ORDER.append(key)
    if len(_JUDGE_CACHE_ORDER) > 8:
        old = _JUDGE_CACHE_ORDER.pop(0)
        _JUDGE_CACHE.pop(old, None)
    return result


def vqa_judge(
    instruction: str = "",
    vqa_guidelines: str = "",
    model: str | None = None,
) -> Callable[..., float]:
    """Build an LLM-judge scorer for free-form instructions.

    Unlike :func:`task_complete`, this scorer requires no predefined
    success keywords. Given the task instruction (e.g. "pick up the red
    cup") plus the VQA observations captured in the call_log during
    execution, an LLM judge decides whether the task completed.

    ``vqa_guidelines`` is the contents of ``skills/<robot>/vqa.md``
    (rendered inline into the judge prompt) so the judge understands the
    yes/no phrasing convention the robot's program uses for its VQA
    queries. Without it the judge would have to guess whether ambiguous
    answers like ``"holding"`` count as affirmative.

    Returns 1.0 iff the judge says ``completed=true``, else 0.0.
    """

    def score(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float:
        result = _vqa_judge_evaluate(instruction, vqa_guidelines, call_log, model)
        return 1.0 if result["completed"] else 0.0

    # Short, safe suffix for trace artifacts.
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", instruction[:30]).strip("_").lower() or "any"
    score.__name__ = f"vqa_judge_{suffix}"
    return score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Maps a tasks.json ``success_scorer.name`` to its builder. Adding a new
# scorer = one new entry here + a few lines above.
_REGISTRY: dict[str, Callable[..., Callable[..., float]]] = {
    "pick_place_cycles": pick_place_cycles,
    "final_state_vqa": final_state_vqa,
    "task_complete": task_complete,
    "vqa_judge": vqa_judge,
}


def build_success_scorer(spec: dict[str, Any] | None) -> Callable[..., float] | None:
    """Resolve a tasks.json ``success_scorer`` spec to a probe scorer callable.

    ``spec`` shape::

        {"name": "pick_place_cycles", "params": {"min_cycles": 2}}

    ``None`` (or an unknown name) returns ``None`` — the eval harness then
    falls back to the old exec-only OOD check. This is a soft fallback so
    tasks without explicit scorers continue to run; only new differentiation
    requires the per-task spec.
    """
    if not spec:
        return None
    name = spec.get("name")
    if name not in _REGISTRY:
        return None
    params = dict(spec.get("params") or {})
    return _REGISTRY[name](**params)


# ---------------------------------------------------------------------------
# Real-robot post-execution scoring
# ---------------------------------------------------------------------------


def trace_to_call_log(trace: Any) -> list[dict]:
    """Convert an :class:`architect.corrections.execution_trace.ExecutionTrace`
    into the ``call_log`` shape the scorer functions consume.

    Each ``ExecutionTraceEntry`` becomes
    ``{"name", "args", "kwargs", "result"}`` — ``result`` is the call's
    actual return value, captured by the trace for perception calls
    (see ``_RESULT_CAPTURE_FUNCS`` in
    :mod:`architect.corrections.execution_trace`). Control calls have
    ``result=None`` in the trace, which the cycle-counter scorer
    doesn't need anyway — it keys off ``args[0]`` for
    ``set_gripper_width`` and just notes presence for ``move_ee_*``.

    Typed as ``Any`` rather than the trace dataclass directly so this
    module doesn't grow an import dep on
    :mod:`architect.corrections.execution_trace` (the trace already imports
    from here transitively in the agentic loop).
    """
    out: list[dict] = []
    for e in getattr(trace, "entries", []) or []:
        out.append({
            "name":   getattr(e, "name", ""),
            "args":   list(getattr(e, "args", []) or []),
            "kwargs": dict(getattr(e, "kwargs", {}) or {}),
            "result": getattr(e, "result", None),
        })
    return out


def score_trace(
    trace: Any,
    spec: dict[str, Any] | None,
    *,
    scene_before: dict | None = None,
    scene_after: dict | None = None,
) -> float | None:
    """Score a real-robot (or dry-run) execution trace against a task spec.

    ``trace`` is an :class:`architect.corrections.execution_trace.ExecutionTrace`
    captured by the agentic loop's namespace wrapper. ``spec`` is the
    same dict shape that ``tasks.json``'s ``success_scorer`` field uses
    — typically ``{"name": "task_complete", "params": {"min_cycles": ...,
    "success_keywords": [...]}}``.

    Returns the scorer's float in ``[0, 1]`` (typically binary 1.0 / 0.0
    for ``task_complete``), or ``None`` when ``spec`` is missing /
    unknown so the caller can fall back to a softer check.

    ``scene_before`` / ``scene_after`` are passed through to the scorer
    function. On a real-robot trace these aren't synthetic Scene
    fixtures — pass ``None`` (the function substitutes empty dicts) or
    your own per-execution snapshot if you have one. The bundled
    scorers (``pick_place_cycles``, ``task_complete``,
    ``final_state_vqa``) only read the call_log; the scene args are
    accepted for signature compatibility with future scorers that
    might inspect them.

    This is the bridge between the live agentic loop (which produces
    ExecutionTraces) and the eval harness's scorer machinery (which
    consumes call_logs). With this, a real-robot trial run via
    ``architect_cli.py`` can be post-execution scored with the same
    ``task_complete`` predicate the dry-run sweep uses.
    """
    scorer = build_success_scorer(spec)
    if scorer is None:
        return None
    call_log = trace_to_call_log(trace)
    return float(scorer(scene_before or {}, scene_after or {}, call_log))


# ---------------------------------------------------------------------------
# Failure diagnostic (for closed-loop auto-correction)
# ---------------------------------------------------------------------------


def _find_matching_vqa(
    call_log: list[dict],
    success_keywords: tuple[str, ...],
) -> tuple[dict | None, str | None]:
    """Return the LAST get_vqa_response call whose prompt contains any of
    the keywords, plus its answer string. ``(None, None)`` if no match.
    """
    keywords_lower = tuple(k.lower() for k in success_keywords)
    last_match: dict | None = None
    for c in call_log:
        if c.get("name") != "get_vqa_response":
            continue
        args = c.get("args") or []
        kwargs = c.get("kwargs") or {}
        prompt = args[0] if args else kwargs.get("prompt", "")
        if not isinstance(prompt, str):
            continue
        prompt_lower = prompt.lower()
        if not keywords_lower or any(kw in prompt_lower for kw in keywords_lower):
            last_match = c
    if last_match is None:
        return (None, None)
    return (last_match, _extract_vqa_answer(last_match.get("result")))


def explain_failure(trace: Any, spec: dict[str, Any] | None) -> str | None:
    """Produce a structured, actionable diagnostic for a failed trial.

    Given the same ``(trace, spec)`` that :func:`score_trace` would have
    returned 0.0 for, walk the call log + spec and produce a markdown-
    flavoured explanation of *which* component of the scorer rejected
    the trial and *what* the agent should change to address it.

    Returns ``None`` when:
      * ``spec`` is missing / unknown
      * the trial actually passed (score ≥ 0.5) — caller should check
        the score first and only invoke explain_failure on FAIL
      * the spec is for a scorer this function doesn't know how to
        diagnose (a generic fallback message is returned in that case)

    Designed for the closed-loop auto-correction path: the returned
    text becomes the body of an auto-issued correction prompt, so it's
    written in the imperative voice the agent will see as instructions
    ("Add a final VQA call...", "Refine to actually complete the
    grasp..."). Includes concrete numbers (cycles seen vs required,
    keywords searched for) so the agent has specifics to act on.
    """
    if not spec or trace is None:
        return None

    name = spec.get("name")
    params = spec.get("params") or {}
    call_log = trace_to_call_log(trace)

    if name == "task_complete":
        min_cycles = int(params.get("min_cycles", 1))
        require_vqa = bool(params.get("require_vqa_verification", True))
        success_keywords = tuple(params.get("success_keywords", []))

        diag = _walk_cycles(call_log, require_vqa_verification=require_vqa)
        matched_vqa, vqa_answer = _find_matching_vqa(call_log, success_keywords)
        vqa_affirmative = (
            _vqa_affirmative(matched_vqa.get("result")) if matched_vqa else False
        )

        cycle_ok = diag["verified_cycles"] >= min_cycles
        vqa_ok = matched_vqa is not None and vqa_affirmative

        if cycle_ok and vqa_ok:
            return None  # not actually a failure

        parts: list[str] = []
        if not cycle_ok:
            verified = diag["verified_cycles"]
            structural = diag["close_release_pairs"]
            msg = (
                f"**Cycle check failed.** The program completed "
                f"{verified} fully-verified pick-place cycle(s); the task "
                f"requires at least {min_cycles}."
            )
            if structural > verified:
                missing_vqa = diag["cycles_missing_vqa"]
                missing_motion = diag["cycles_missing_motion"]
                msg += (
                    f" Found {structural} structural close→release pair(s), "
                    f"but {missing_vqa} lacked an affirmative VQA between "
                    f"close and release"
                )
                if missing_motion:
                    msg += f" and {missing_motion} lacked transport motion"
                msg += "."
            if diag["any_close"] == 0:
                msg += " No `set_gripper_width(w < 0.05)` calls were made — the program never closed the gripper."
            msg += (
                " A 'verified cycle' is: `set_gripper_width(close, w<0.05)` "
                "→ at least one `move_ee_*` call → `get_vqa_response` with "
                "an affirmative answer (contains 'yes', not 'no'/'empty'/"
                "'fail') → another `move_ee_*` → "
                "`set_gripper_width(release, w>=0.05)`. Use the "
                "`for attempt in range(2)` retry template so each "
                "iteration's close has its own VQA check."
            )
            parts.append(msg)

        if not vqa_ok:
            if matched_vqa is None:
                kw_str = ", ".join(repr(k) for k in success_keywords) or "(no keywords)"
                msg = (
                    f"**Final-state VQA check failed.** No `get_vqa_response` "
                    f"call's prompt contained any of the task-success "
                    f"keywords [{kw_str}]. Add a final VQA call at the end "
                    f"of the program that asks a yes/no question about whether "
                    f"the goal was met (e.g. \"Is the red block on marker 2? "
                    f"Yes or No?\") — the scorer matches the prompt text "
                    f"against those keywords."
                )
            else:
                msg = (
                    f"**Final-state VQA check failed.** A VQA call matched "
                    f"the task-success keywords but its answer was not "
                    f"affirmative: response was {vqa_answer!r}. The program "
                    f"appears to have *completed its sequence of operations* "
                    f"but did not actually accomplish the task. Refine so "
                    f"the VQA returns yes (the actual physical / scene state "
                    f"matches what the task instruction asks for)."
                )
            parts.append(msg)

        header = (
            f"The program failed the `task_complete` scorer "
            f"(score=0.00). Specific gaps to address:"
        )
        return header + "\n\n" + "\n\n".join(parts)

    if name == "vqa_judge":
        instruction = str(params.get("instruction", ""))
        vqa_guidelines = str(params.get("vqa_guidelines", ""))
        model = params.get("model")
        result = _vqa_judge_evaluate(instruction, vqa_guidelines, call_log, model)
        if result["completed"]:
            return None  # not actually a failure
        rationale = result.get("rationale") or "<no rationale provided>"

        # Sanity flag: zero VQA calls means the program never verified
        # completion. The judge's rationale typically already calls this
        # out, but tacking on an explicit imperative helps the agent
        # generate a more structured fix.
        has_vqa = any(c.get("name") == "get_vqa_response" for c in call_log)
        suffix = ""
        if not has_vqa:
            suffix = (
                "\n\nThe program made zero `get_vqa_response` calls — "
                "add a final-state VQA check phrased as a yes/no question "
                "about the task goal (see `skills/<robot>/vqa.md` for "
                "phrasing conventions) so the judge has perceptual "
                "evidence to evaluate."
            )

        return (
            f"The program failed the `vqa_judge` scorer for instruction "
            f"{instruction!r}.\n\n"
            f"**Judge rationale:** {rationale}{suffix}\n\n"
            f"Refine the program so the robot's actions and final-state "
            f"VQA confirm completion of the instruction. The execution "
            f"trace is in the next section."
        )

    # Generic fallback — we don't know how to introspect other scorer types
    # yet, so just nudge the agent to re-examine the trial.
    return (
        f"The `{name}` scorer rejected the program. The execution trace "
        f"is in the next section; use it to identify what failed and "
        f"refine the program."
    )
