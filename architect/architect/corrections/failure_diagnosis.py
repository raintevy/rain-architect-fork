"""Structured LLM-driven failure diagnosis.

Fires when a trial returns FAIL (closed-loop or human-driven). Reads the
execution trace + the current program + the vqa_judge / scorer's
rationale, and produces a structured JSON diagnosis with a fixed
schema so callers can:

  * Render it in a panel above the human-correction prompt (so the
    operator has an informed read on what went wrong before they
    type a fix)
  * Use it as the auto-correction body in closed-loop mode (replacing
    the older :func:`architect.eval.task_scorers.explain_failure` text)
  * Persist the structured fields to the failure store
    (:mod:`architect.library.failure_store`) for retrospective analysis and
    cross-session lookup at the next failure

This is a richer counterpart to two existing pieces:

  * :func:`architect.eval.task_scorers.explain_failure` — keyword-based
    structural diagnostic from the trace (no LLM). Still useful as a
    fallback when the LLM is unavailable.
  * :func:`architect.corrections.fault_localization.summarize_fault` —
    three-line prose summary used by the agentic correction loop
    (always-on, prepended above the trace). The diagnosis here is
    *richer* (structured fields, category taxonomy) and meant for
    a different surface: human-facing panel + persistent store.

Best-effort by design: on LLM error / malformed JSON, returns a
diagnosis with ``failure_category="diagnosis_unavailable"`` so the
caller still has *something* structured to record / render. We
deliberately don't raise — a flaky diagnosis must never crash the
correction loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


# Canonical failure-category taxonomy. The LLM is asked to pick from
# this list (or use "other"); having a fixed vocabulary lets us
# aggregate failure modes across sessions for the paper. Add categories
# here as we discover new modes; the prompt asks the LLM to use "other"
# + a free-text signature when nothing fits.
FAILURE_CATEGORIES: tuple[str, ...] = (
    "grasp_failed_no_contact",       # gripper closed but no object held
    "grasp_failed_wrong_object",     # picked something other than target
    "vqa_false_positive",            # VQA said success but state was wrong
    "vqa_false_negative",            # VQA said failure but state was right
    "missing_verification",          # program never verified the goal state
    "motion_infeasible",             # CuRobo / planner rejected the motion
    "object_not_detected",           # detect_objects returned empty / wrong
    "placement_misaligned",          # released at wrong pose / off target
    "premature_release",             # opened gripper before reaching goal
    "incomplete_sequence",           # program exited before finishing
    "exec_error",                    # Python exception during execution
    "other",                         # free-form, agent supplies signature
    "diagnosis_unavailable",         # LLM call failed; reserved by parser
)


@dataclass
class FailureDiagnosis:
    """Structured diagnostic produced by :func:`diagnose_failure`.

    ``failure_category`` is one of :data:`FAILURE_CATEGORIES`. The
    ``failure_signature`` is a terse phrase (4-8 words) used as the
    primary lookup key for the failure store — keep it stable across
    semantically-equivalent failures so similarity retrieval works.
    """

    failure_category: str
    failure_signature: str
    root_cause: str
    suggested_correction_directions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Human-readable rendering for the diagnosis panel + correction prompt."""
        lines = [
            f"**Category:** `{self.failure_category}`",
            f"**Signature:** {self.failure_signature}",
            f"**Root cause:** {self.root_cause}",
        ]
        if self.suggested_correction_directions:
            lines.append("")
            lines.append("**Suggested correction directions:**")
            for d in self.suggested_correction_directions:
                lines.append(f"- {d}")
        if self.evidence:
            lines.append("")
            lines.append("**Supporting evidence from the trace:**")
            for e in self.evidence:
                lines.append(f"- {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SYSTEM_PROMPT = (
    "You are a robot-program failure analyst. Given a task instruction, "
    "the program that was executed, the execution trace (call sequence, "
    "per-step VQA Q/A pairs and state snapshots), and the judge's "
    "rationale for the FAIL verdict, produce a structured diagnosis that "
    "a human operator (or a follow-up coding agent) can act on.\n\n"
    "Rules:\n"
    "1. Pick `failure_category` from the provided taxonomy. Use `other` only "
    "when no category fits, and supply a free-form `failure_signature` that "
    "captures the failure mode.\n"
    "2. `failure_signature` is 4-8 words, lowercase-with-underscores or short "
    "phrase form. It is the primary lookup key for cross-session retrieval, "
    "so similar failures should produce similar signatures (e.g. "
    '"gripper closed on empty space", "vqa confirmed grasp but cup not held").\n'
    "3. `root_cause` is 1-2 sentences naming the specific gap between "
    "what the program did and what the instruction required.\n"
    "4. `suggested_correction_directions` is 1-3 imperative-voice strings "
    "the next iteration could try. Each is a *direction*, not a full "
    "code change — keep it action-oriented (e.g. \"add a VQA check "
    "between gripper close and lift to catch empty-grasp failures\").\n"
    "5. `evidence` is 1-3 strings citing specific trace observations "
    "(call indices, VQA Q/A pairs, state values) that support the "
    "diagnosis. Be concrete: \"call 7 (set_gripper_width(0.0)) was "
    "followed by VQA 'is the gripper holding the cup?' → 'no'\".\n\n"
    "Respond with strict JSON only, no prose, no markdown fences. Schema:\n"
    '{"failure_category": "<category from taxonomy>", '
    '"failure_signature": "<4-8 word phrase>", '
    '"root_cause": "<1-2 sentences>", '
    '"suggested_correction_directions": ["<imperative>", ...], '
    '"evidence": ["<observation citing trace>", ...]}'
)


def _format_trace_for_diagnosis(trace: Any) -> str:
    """Compact trace rendering for the diagnosis prompt.

    We don't reuse the agent's full ``format_for_prompt`` rendering
    because (a) it's optimised for the agentic loop's context, not for
    a one-shot diagnosis, and (b) we want to keep this LLM call cheap.
    Pulls the call sequence with VQA Q/A pairs inline.
    """
    if trace is None:
        return "<no trace captured>"
    entries = getattr(trace, "entries", []) or []
    if not entries:
        return "<trace had zero entries>"
    lines: list[str] = []
    for i, e in enumerate(entries):
        name = getattr(e, "name", "?")
        args = list(getattr(e, "args", []) or [])
        kwargs = dict(getattr(e, "kwargs", {}) or {})
        # Render args compactly — full args repr can blow up
        arg_repr = ", ".join(_compact_repr(a) for a in args[:3])
        if len(args) > 3:
            arg_repr += f", … ({len(args)-3} more)"
        kw_repr = ", ".join(f"{k}={_compact_repr(v)}" for k, v in list(kwargs.items())[:3])
        full_args = ", ".join(filter(None, [arg_repr, kw_repr]))
        line = f"  {i}: {name}({full_args})"
        # Inline the VQA answer right under the call so the LLM can read
        # Q/A pairs without scrolling.
        if name == "get_vqa_response":
            ans = _extract_answer(getattr(e, "result", None))
            line += f"\n     → answer: {ans!r}"
        lines.append(line)
    return "\n".join(lines)


def _compact_repr(v: Any) -> str:
    """One-line repr, capped at ~60 chars."""
    s = repr(v)
    if len(s) > 60:
        s = s[:57] + "..."
    return s


def _extract_answer(result: Any) -> str:
    """Pull the VQA answer string from common shapes — Franka dict, bare string."""
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"]
        if isinstance(result.get("answer"), str):
            return result["answer"]
    if isinstance(result, str):
        return result
    return "<no answer>"


def _parse_response(text: str) -> dict[str, Any] | None:
    """Robust JSON parser. Strips fences, falls back to first {...} block.

    Mirrors :func:`architect.eval.task_scorers._parse_judge_response`; kept
    separate so changes to one don't perturb the other.
    """
    if not text:
        return None
    cleaned = text.strip()
    fence = re.match(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def diagnose_failure(
    trace: Any,
    program: str,
    instruction: str,
    judge_rationale: str = "",
    *,
    model: str | None = None,
) -> FailureDiagnosis:
    """Run the diagnostic LLM call and return a FailureDiagnosis.

    Always returns a populated FailureDiagnosis — on LLM error or
    malformed JSON, the diagnosis carries
    ``failure_category="diagnosis_unavailable"`` and an explanatory
    ``root_cause``. Callers can branch on that category if they want
    to fall back to the keyword-based :func:`explain_failure` text.

    Per-trace caching is *not* done here (unlike vqa_judge) because
    the diagnosis is typically fired exactly once per FAIL by the CLI
    — adding a cache layer would only matter if downstream code
    started calling this multiple times for the same trace, which
    isn't a current pattern.
    """
    trace_text = _format_trace_for_diagnosis(trace)
    user_msg = (
        f"## Task instruction\n{instruction!r}\n\n"
        f"## Judge rationale for FAIL\n{judge_rationale or '<no rationale>'}\n\n"
        f"## Program that was executed\n```python\n{program}\n```\n\n"
        f"## Execution trace (call sequence with VQA Q/A inlined)\n"
        f"{trace_text}\n\n"
        f"## Failure-category taxonomy (pick exactly one)\n"
        f"{', '.join(c for c in FAILURE_CATEGORIES if c != 'diagnosis_unavailable')}\n\n"
        f"Now diagnose. Respond with strict JSON only."
    )

    try:
        from architect.llm.client import call_claude
        if model is None:
            from config import ANTHROPIC_MODEL
            model = ANTHROPIC_MODEL
        response = call_claude(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            max_tokens=600,
            temperature=0.0,
        )
    except Exception as exc:
        return FailureDiagnosis(
            failure_category="diagnosis_unavailable",
            failure_signature="diagnosis llm unavailable",
            root_cause=(
                f"Failure-diagnosis LLM call raised "
                f"{type(exc).__name__}: {exc}. Falling back without a "
                f"structured diagnosis — see the raw trace below for "
                f"context."
            ),
            suggested_correction_directions=[],
            evidence=[],
        )

    parsed = _parse_response(response)
    if parsed is None:
        return FailureDiagnosis(
            failure_category="diagnosis_unavailable",
            failure_signature="diagnosis parse failed",
            root_cause=(
                "Failure-diagnosis LLM returned a response that could not "
                "be parsed as JSON. Raw response (clipped): "
                f"{(response or '')[:200]!r}."
            ),
            suggested_correction_directions=[],
            evidence=[],
        )

    category = str(parsed.get("failure_category", "other"))
    if category not in FAILURE_CATEGORIES:
        # LLM made up a category — coerce to "other" and put the
        # invented category into the signature so we don't lose
        # the information.
        category = "other"
    return FailureDiagnosis(
        failure_category=category,
        failure_signature=str(parsed.get("failure_signature", ""))[:200],
        root_cause=str(parsed.get("root_cause", "")),
        suggested_correction_directions=[
            str(d) for d in (parsed.get("suggested_correction_directions") or [])
        ][:5],
        evidence=[
            str(e) for e in (parsed.get("evidence") or [])
        ][:5],
    )
