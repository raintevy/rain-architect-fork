"""Pre-correction fault-localisation summary.

A small LLM call that consumes the last execution trace + the user's
correction text and produces a compact three-line summary the next
agentic correction step can read at the top of its prompt:

    Failure:              <one line>
    Likely root cause:    <one line>
    Correction direction: <one line>

The summary is a *hint*, not a directive. The agentic loop is still
free to investigate via tools and the slate is still free to explore
alternatives; the summary just orients the next step toward the most
likely failure mode so the LLM doesn't have to re-derive it from the
trace each time.

Best-effort by design: on no trace, on LLM failure, or on a malformed
response we return ``None`` and the correction message is built
without the summary section. The agentic loop has always worked
without this summary and must continue to.

Cost: one extra LLM round-trip per correction, capped at ~500 output
tokens. The summary is foreground (synchronous before the agentic
loop) because the loop needs to see it on its first turn — async would
force the loop to either wait at the start or skip the summary.
"""

from __future__ import annotations

from typing import Any, Callable


def summarize_fault(
    correction: str,
    current_program: str | None,
    trace: Any,
    *,
    llm_call: Callable[..., str] | None = None,
    model: str | None = None,
) -> str | None:
    """Return a short fault-localisation summary, or ``None``.

    Returns ``None`` when:
      * ``trace is None`` — first correction in a session, nothing to
        localise against
      * the trace has zero entries AND ``exec_ok`` is True — the program
        didn't run anything observable (e.g. empty submit), no signal
        to summarise
      * the LLM call fails / times out — the summary is best-effort
      * the LLM returns an empty / whitespace-only response

    The returned string is the raw three-line summary from the LLM,
    stripped of leading/trailing whitespace. The caller renders it into
    the correction message under a ``## Fault Localisation Summary``
    section above the raw trace.
    """
    if trace is None:
        return None
    # Nothing to localise on a successful no-op (e.g. submit_program called
    # with a trivially empty body). A failed empty trace IS interesting
    # though — the error message itself is signal — so we only short-circuit
    # the ok+empty case.
    if not getattr(trace, "entries", []) and getattr(trace, "exec_ok", False):
        return None

    from architect.llm.prompts import build_fault_summary_prompt

    llm = llm_call if llm_call is not None else _default_llm_call
    if model is None:
        from config import ANTHROPIC_MODEL  # late import — avoids circular dep
        model = ANTHROPIC_MODEL

    messages = build_fault_summary_prompt(
        correction=correction,
        current_program=current_program or "",
        trace=trace,
    )
    try:
        response = llm(messages, model=model)
    except Exception:
        # A flaky summary must not crash the correction. Bubble up nothing
        # so the loop falls through to the trace-only correction message.
        return None
    summary = (response or "").strip()
    if not summary:
        return None
    return summary


def _default_llm_call(messages: list[dict], model: str) -> str:
    """Default LLM caller for the summary — short cap, deterministic.

    ``max_tokens=500`` covers the three-line summary comfortably: ~100
    words of content per the system prompt = ~150 tokens, with margin
    for the LLM running long or for an evidence-rich trace producing a
    slightly more detailed line. Lower caps (300) risked mid-sentence
    truncation; the summary loses most of its value when "Correction
    direction:" gets cut at the conjunction.
    ``temperature=0.0`` because the summary is a *summary* of trace
    evidence, not a creative output — we want determinism so the same
    trace produces the same summary.
    """
    from architect.llm.client import call_claude
    return call_claude(messages, model=model, max_tokens=500, temperature=0.0)
