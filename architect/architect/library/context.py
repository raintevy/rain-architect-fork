"""Context assembly + budgeting for the agent's LLM calls (P6).

All LLM calls in the agent loop are routed through :func:`build_context`,
so the policy for *what makes it into Claude's context window each turn*
lives in one place rather than spread across the agent. The current
P6-lite policy applies three transformations to the raw session history:

  1. **Tool-output clipping** — ``tool_result`` blocks whose content
     exceeds ``MAX_TOOL_OUTPUT`` characters are truncated with a
     "[... truncated N bytes]" marker. Full payloads are not yet
     addressable elsewhere; if a later phase wants
     :class:`architect.context.ContextChannels.max_tool_output`-aware paging,
     it lands here.

  2. **Stale-read dedupe** — identical ``query_robot_state`` /
     ``get_scene_description`` tool calls (same name + same args) in
     the session keep only the *latest* result. Older identical reads
     are surfaced to Claude as ``[stale; superseded by later call]``
     so the model knows the slot exists but doesn't burn context on
     the redundant payload. Within a single ``_run_loop`` the robot
     doesn't move, so duplicate reads are *guaranteed* redundant.

  3. **Library retrieval** — when a current-turn correction text is
     supplied, the top-k graduated skills from the SkillStore (by
     embedding similarity) are appended to the system prompt under a
     ``## Relevant Skills`` heading. This is the §4.1 wake-phase hook;
     it lets Claude see relevant past helpers as available API surface
     on the very first turn of a new correction.

Deliberately out of scope for P6-lite per §8.1:
rolling summarization of older turns, episodic retrieval into the
prompt, working-memory scratchpad. Those land if §7 step 8's
stress test shows they're needed.

The transformations are *non-destructive*: ``build_context`` returns
freshly-constructed dicts; ``session._message_history`` is never
mutated, so undo/rewind continue to work against the canonical history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# Tool names whose calls represent *reads* of the world (perception,
# proprioception). Duplicate reads with the same args inside one
# ``_run_loop`` are redundant. Tools that *write* (write_primitive,
# run_ros2_command, submit_program) are not subject to dedupe.
_READ_TOOLS: frozenset[str] = frozenset({
    "query_robot_state",
    "get_scene_description",
    "list_primitives",
    "read_primitive",
})


# ---------------------------------------------------------------------------
# Channels (tunables)
# ---------------------------------------------------------------------------


@dataclass
class ContextChannels:
    """Tunable budgets and toggles for :func:`build_context`.

    Defaults match Raschka's mini-coding-agent reference: clip at 4k chars,
    dedupe enabled, retrieval bounded at top-3.
    """

    max_tool_output: int = 4000
    dedupe_enabled: bool = True
    clipping_enabled: bool = True
    retrieval_enabled: bool = True
    retrieval_k: int = 3
    retrieval_status: str | tuple[str, ...] = "graduated"
    # Tools subject to dedupe. Override (e.g. add session-specific reads)
    # without touching the module constant.
    read_tools: frozenset[str] = field(default_factory=lambda: _READ_TOOLS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context(
    *,
    system_prompt: str,
    message_history: list[dict],
    correction_text: str | None = None,
    store: object | None = None,
    channels: ContextChannels | None = None,
) -> list[dict]:
    """Assemble messages for an LLM call from session state.

    Parameters
    ----------
    system_prompt:
        The static agentic system prompt (with API spec + skill markdown).
        Library retrieval, when enabled, appends a ``## Relevant Skills``
        block to this string.
    message_history:
        The session's raw history list — same object the agent appends
        assistant / tool_result entries to. Not mutated.
    correction_text:
        The current user correction (or instruction) string. Used as the
        query for library retrieval. ``None`` skips retrieval.
    store:
        A :class:`architect.skill_store.SkillStore` instance. ``None`` skips
        retrieval. Duck-typed: any object with ``retrieve(query, k,
        status)`` returning ``list[dict]`` works.
    channels:
        Override the default :class:`ContextChannels`.

    Returns
    -------
    list[dict]
        Fresh ``[{"role": "system", ...}, *history]`` ready for the
        Anthropic Messages API. Older identical reads are stale-marked,
        oversized tool outputs are clipped, and the system prompt
        carries any retrieved-skill addendum.
    """
    ch = channels or ContextChannels()
    history = list(message_history)

    if ch.dedupe_enabled:
        history = _dedupe_stale_reads(history, read_tools=ch.read_tools)
    if ch.clipping_enabled:
        history = _clip_tool_outputs(history, max_bytes=ch.max_tool_output)

    system_content = system_prompt
    if ch.retrieval_enabled and store is not None and correction_text:
        retrieved = retrieve_relevant_skills(
            store, correction_text, k=ch.retrieval_k, status=ch.retrieval_status,
        )
        if retrieved:
            system_content = system_prompt + "\n\n" + format_retrieved_skills(retrieved)

    return [{"role": "system", "content": system_content}, *history]


# ---------------------------------------------------------------------------
# Tool-output clipping
# ---------------------------------------------------------------------------


def _clip_tool_outputs(history: list[dict], *, max_bytes: int) -> list[dict]:
    """Truncate ``tool_result`` block content to ``max_bytes`` characters.

    Only the ``user``-role messages carry ``tool_result`` blocks (as dicts).
    Assistant content is left untouched; clipping a model response would
    break the prompt-cache and risk losing argument values the model
    relies on.
    """
    out: list[dict] = []
    for msg in history:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_blocks: list[Any] = []
        clipped = False
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
                and len(block["content"]) > max_bytes
            ):
                trim = len(block["content"]) - max_bytes
                new_blocks.append({
                    **block,
                    "content": (
                        block["content"][:max_bytes]
                        + f"\n\n[... truncated {trim} bytes]"
                    ),
                })
                clipped = True
            else:
                new_blocks.append(block)
        out.append({**msg, "content": new_blocks} if clipped else msg)
    return out


# ---------------------------------------------------------------------------
# Stale-read dedupe
# ---------------------------------------------------------------------------


def _dedupe_stale_reads(
    history: list[dict],
    *,
    read_tools: Iterable[str],
) -> list[dict]:
    """Mark older identical reads as stale; preserve the latest result.

    For each ``(tool_name, json.dumps(input, sort_keys=True))`` key seen
    across assistant turns, every occurrence except the most recent has
    its corresponding ``tool_result`` content replaced with a one-line
    "[stale; superseded by later call]" marker. The ``tool_use`` blocks
    themselves are kept verbatim — they're cheap and Claude needs to see
    the call shape to remember it made the request.
    """
    reads = set(read_tools)
    occurrences: dict[tuple[str, str], list[str]] = {}

    # First pass: collect tool_use_id per (name, input-json) key
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or ():
            name = _block_attr(block, "name")
            if name is None or name not in reads:
                continue
            if _block_attr(block, "type") != "tool_use":
                continue
            tu_id = _block_attr(block, "id")
            if tu_id is None:
                continue
            args = _block_attr(block, "input") or {}
            try:
                key = (name, json.dumps(args, sort_keys=True))
            except (TypeError, ValueError):
                key = (name, repr(args))
            occurrences.setdefault(key, []).append(tu_id)

    # IDs that should be marked stale: all but the last occurrence of each key
    stale_ids: set[str] = set()
    for ids in occurrences.values():
        if len(ids) > 1:
            stale_ids.update(ids[:-1])

    if not stale_ids:
        return history

    # Second pass: rewrite tool_result blocks for stale IDs
    out: list[dict] = []
    for msg in history:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            out.append(msg)
            continue
        changed = False
        new_blocks = []
        for block in msg["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in stale_ids
            ):
                new_blocks.append({
                    **block,
                    "content": "[stale; superseded by later call]",
                })
                changed = True
            else:
                new_blocks.append(block)
        out.append({**msg, "content": new_blocks} if changed else msg)
    return out


# ---------------------------------------------------------------------------
# Library retrieval
# ---------------------------------------------------------------------------


def retrieve_relevant_skills(
    store: object,
    query: str,
    *,
    k: int = 3,
    status: str | tuple[str, ...] = "graduated",
) -> list[dict]:
    """Best-effort top-k retrieval. Empty list on any error / empty store.

    Public counterpart to the private dispatcher used by :func:`build_context`.
    Exposed so the legacy single-shot correction path (the FuncReuse
    baseline in ``run_trial._apply_correction_singleshot``) can append
    the same ``## Relevant Skills`` block to its system prompt that the
    agentic loop gets through :func:`build_context`. Without this,
    library retrieval is silently disabled outside the agentic loop and
    the FuncReuse vs ICL contrast collapses.
    """
    try:
        rows = store.retrieve(query, k=k, status=status)  # type: ignore[attr-defined]
    except Exception:
        return []
    return rows or []


# Back-compat alias for the in-module callers; tests sometimes import the
# private name directly.
_retrieve_skills = retrieve_relevant_skills


def format_retrieved_skills(rows: list[dict]) -> str:
    """Render retrieved skills as a system-prompt addendum.

    The shape is intentionally lightweight — one bullet per skill with the
    docstring and similarity score. Heavier formats (full source preview,
    invariants, success history) are tempting but add tokens that compete
    with the API spec already in the system prompt; defer until P7 metrics
    show retrieval-quality gains from a richer presentation.
    """
    lines = ["## Relevant Skills (retrieved from cross-session library)"]
    lines.append(
        "[dim]Past helpers similar to the current correction. Call them by name "
        "the same way you would call a built-in API function.[/dim]"
    )
    for r in rows:
        sim = r.get("_similarity", 0.0)
        status = r.get("graduation_status", "?")
        doc = (r.get("docstring") or "").strip()
        lines.append(f"- `{r['name']}` — {doc}  [sim={sim:.2f}, status={status}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block helpers (work over both SDK objects and plain dicts)
# ---------------------------------------------------------------------------


def _block_attr(block: Any, name: str) -> Any:
    """Attribute access that works for both Anthropic SDK block objects and dicts.

    The SDK returns ``TextBlock`` / ``ToolUseBlock`` instances (pydantic
    models) in ``response.content``; we also write plain dicts for
    ``tool_result`` blocks. ``getattr`` handles the former; ``.get`` the
    latter. This shim hides the asymmetry from the rest of the module.
    """
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


# ---------------------------------------------------------------------------
# Context measurement (profiling)
# ---------------------------------------------------------------------------


# Empirical bytes-per-token ratio for English+code with Claude's tokenizer.
# Within ~5% of anthropic.count_tokens on typical ARCHITECT prompts. Used when the
# caller doesn't supply a real tokenizer; profile_session can re-derive
# exact tokens from the captured ``response.usage`` numbers.
_BYTES_PER_TOKEN_EST: float = 3.5


# Default Claude model context window. The exact value matters less than
# the *fraction* of it we're consuming over the session — the agent loop
# fails at the limit, but we want to see the curve approach the cap and
# trigger the planned rolling-summarization upgrade well before then.
DEFAULT_MODEL_CONTEXT_WINDOW: int = 200_000


@dataclass
class ContextMetrics:
    """Per-call snapshot of context fill, with per-channel attribution.

    All ``*_bytes`` fields are byte counts of ``json.dumps`` of the
    relevant slice — fast, deterministic, no API call. ``tokens_est`` is
    the bytes-per-token heuristic. When the caller has the matching
    ``response.usage`` numbers from a real LLM call, the profile writer
    fills in ``tokens_actual`` alongside this estimate so we can audit
    the heuristic on real data.
    """

    total_bytes: int = 0
    tokens_est: int = 0

    # Per-channel breakdown — a *partition* of the prompt. The four fields
    # are disjoint and sum to ~total_bytes (modulo a small json-overhead
    # discrepancy from the message envelope, which the consumer treats as
    # ratios). ``system_bytes`` is the *base* system prompt only — the
    # retrieved-skills section, when present, is split out into
    # ``retrieval_addendum_bytes`` and not double-counted here. Pre-fix
    # the per-channel fractions summed to >1.0 (the smoke trace landed at
    # 101.9%) and downstream pie charts were inflated.
    system_bytes: int = 0
    retrieval_addendum_bytes: int = 0
    history_bytes: int = 0
    current_turn_bytes: int = 0

    # Build-time savings vs. the raw history. Set by
    # :func:`build_context_with_metrics`; remain 0 when ``measure_context``
    # is called standalone (no comparison baseline).
    clipped_count: int = 0
    clipped_bytes_saved: int = 0
    stale_count: int = 0
    stale_bytes_saved: int = 0

    # Counts of read calls in the assembled history, by tool name. Used as
    # the denominator in the lost-context-rate proxy:
    #   lost_context_rate = stale_count / total_read_calls
    # in a stress test (§7 step 8).
    read_calls_by_tool: dict[str, int] = field(default_factory=dict)

    # Cap-fraction: total_bytes / (window_bytes_estimate). 1.0 means we're
    # at the configured context limit. Reserved for visualisation /
    # alerting; the agent itself doesn't gate on this.
    window_fill_fraction: float = 0.0


def estimate_tokens(text_or_bytes: str | int) -> int:
    """Cheap token-count estimate from a string or byte count.

    Heuristic only — within ~5 % of Claude's tokenizer on English+code
    in our sampling. For exact counts at runtime, use the
    ``response.usage`` returned by the LLM call. For paper figures
    requiring ground-truth attribution per channel, plug in
    :meth:`anthropic.Anthropic.messages.count_tokens` — the function
    signature is compatible.
    """
    n_bytes = (
        text_or_bytes if isinstance(text_or_bytes, int)
        else len(text_or_bytes.encode("utf-8")) if isinstance(text_or_bytes, str)
        else 0
    )
    return int(n_bytes / _BYTES_PER_TOKEN_EST)


def _bytes_of(obj: Any) -> int:
    """``len(json.dumps(...))`` with a fallback for SDK block objects.

    Anthropic block objects (``ToolUseBlock`` etc.) aren't JSON-
    serialisable by default; ``getattr(o, '__dict__', repr(o))`` mirrors
    the same fallback the eval harness uses when persisting trial
    records.
    """
    try:
        return len(json.dumps(obj, default=lambda o: getattr(o, "__dict__", repr(o))))
    except Exception:
        return 0


def measure_context(
    messages: list[dict],
    *,
    window_bytes: int = DEFAULT_MODEL_CONTEXT_WINDOW * 4,
) -> ContextMetrics:
    """Compute :class:`ContextMetrics` for an assembled messages list.

    Parameters
    ----------
    messages:
        Output of :func:`build_context` — i.e. system + history.
    window_bytes:
        Estimated context-window size in *bytes* (default = 200 K tokens
        × 4 bytes/token, conservative). The agent loop fails when the
        true token count exceeds the model's limit; this gives a
        comparable fraction for visualisation.

    Returns
    -------
    ContextMetrics
        Per-channel byte counts, token estimate, and window-fill
        fraction. ``clipped_*`` / ``stale_*`` fields stay zero — those
        are populated by :func:`build_context_with_metrics`, which has
        the raw history to compare against.
    """
    metrics = ContextMetrics()
    if not messages:
        return metrics

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    if system_msg is not None:
        sys_content = str(system_msg.get("content") or "")
        full_system_bytes = _bytes_of(system_msg)
        # Split the system message into base + retrieval addendum so the
        # per-channel fields partition cleanly. The marker is what
        # architect.context.format_retrieved_skills prepends; if it's not in
        # the system message, the entire payload is base system.
        marker = "## Relevant Skills"
        if marker in sys_content:
            head, _, tail = sys_content.partition(marker)
            addendum_bytes = len((marker + tail).encode("utf-8"))
            metrics.retrieval_addendum_bytes = addendum_bytes
            # Floor at zero in case the system message's json envelope
            # overhead makes the subtraction go slightly negative.
            metrics.system_bytes = max(0, full_system_bytes - addendum_bytes)
        else:
            metrics.system_bytes = full_system_bytes

    history = messages[1:] if system_msg is not None else messages
    if history:
        # Last user message is the "current turn"; everything before it
        # is older history.
        metrics.current_turn_bytes = _bytes_of(history[-1])
        metrics.history_bytes = sum(_bytes_of(m) for m in history[:-1])

    metrics.read_calls_by_tool = _count_read_calls(messages)

    metrics.total_bytes = sum(_bytes_of(m) for m in messages)
    metrics.tokens_est = estimate_tokens(metrics.total_bytes)
    metrics.window_fill_fraction = (
        metrics.total_bytes / window_bytes if window_bytes > 0 else 0.0
    )
    return metrics


def _count_read_calls(messages: list[dict]) -> dict[str, int]:
    """Per-tool counts of assistant ``tool_use`` blocks for read-class tools."""
    out: dict[str, int] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or ():
            if _block_attr(block, "type") != "tool_use":
                continue
            name = _block_attr(block, "name")
            if name in _READ_TOOLS:
                out[name] = out.get(name, 0) + 1
    return out


def build_context_with_metrics(
    *,
    system_prompt: str,
    message_history: list[dict],
    correction_text: str | None = None,
    store: object | None = None,
    channels: ContextChannels | None = None,
    window_bytes: int = DEFAULT_MODEL_CONTEXT_WINDOW * 4,
) -> tuple[list[dict], ContextMetrics]:
    """Same as :func:`build_context`, plus a :class:`ContextMetrics` snapshot.

    Captures the build-time savings: how many tool_results got clipped
    and how many bytes that saved, how many reads got stale-marked and
    how many bytes that saved. The agent loop uses this when
    profile-context is on; the standalone ``build_context`` skips the
    instrumentation overhead for the unprofiled path.
    """
    ch = channels or ContextChannels()
    raw_bytes = sum(_bytes_of(m) for m in message_history)

    messages = build_context(
        system_prompt=system_prompt,
        message_history=message_history,
        correction_text=correction_text,
        store=store,
        channels=ch,
    )

    metrics = measure_context(messages, window_bytes=window_bytes)

    if ch.clipping_enabled or ch.dedupe_enabled:
        clipped, clipped_saved = _count_clipped(messages, message_history,
                                                 max_bytes=ch.max_tool_output)
        stale, stale_saved = _count_stale(messages, message_history)
        metrics.clipped_count = clipped
        metrics.clipped_bytes_saved = clipped_saved
        metrics.stale_count = stale
        metrics.stale_bytes_saved = stale_saved

    return messages, metrics


def _count_clipped(
    processed: list[dict],
    raw: list[dict],
    *,
    max_bytes: int,
) -> tuple[int, int]:
    """Count tool_results in ``raw`` whose content exceeded ``max_bytes``.

    ``processed`` is unused — we know the clipping marker shape and could
    grep for it, but using the raw history (which the processed list was
    built from) makes the count robust to format drift.
    """
    n = 0
    saved = 0
    for msg in raw:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content") or ():
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            content = block.get("content", "")
            if isinstance(content, str) and len(content) > max_bytes:
                n += 1
                saved += len(content) - max_bytes
    return n, saved


def _count_stale(processed: list[dict], raw: list[dict]) -> tuple[int, int]:
    """Count tool_results that build_context replaced with the stale marker."""
    marker = "[stale; superseded by later call]"
    stale_ids: set[str] = set()
    for msg in processed:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content") or ():
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("content") == marker
            ):
                stale_ids.add(block.get("tool_use_id", ""))
    if not stale_ids:
        return 0, 0
    saved = 0
    for msg in raw:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content") or ():
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in stale_ids
            ):
                # Subtract the marker length so we don't credit ourselves
                # for the bytes still in the prompt.
                saved += max(0, len(str(block.get("content", ""))) - len(marker))
    return len(stale_ids), saved


# ---------------------------------------------------------------------------
# Per-turn JSONL writer (used by AgenticSession when --profile-context is on)
# ---------------------------------------------------------------------------


class ContextProfile:
    """Append-only JSONL log of per-turn context-fill metrics.

    One line per LLM call inside :meth:`AgenticSession._run_loop`. The
    line shape is the union of :class:`ContextMetrics` fields plus
    ground-truth token counts from ``response.usage`` and a wall-clock
    latency measurement. ``scripts/eval/profile_session.py`` reads the
    file and produces the §7 step-8 stress-test artifacts: tokens-per-
    call curve, per-channel attribution, lost-context-rate proxy.

    The writer is intentionally minimal — no buffering, no rotation,
    no compression. A 100-turn session writes <100 kB so disk pressure
    isn't a concern; CI logs can grep the JSONL directly.
    """

    def __init__(self, path: str | Path) -> None:
        from pathlib import Path as _P
        self._path = _P(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._turn_index = 0

    @property
    def path(self):
        return self._path

    def record(
        self,
        metrics: ContextMetrics,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_s: float | None = None,
        stop_reason: str | None = None,
        loop_iteration: int | None = None,
        correction_text: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append one per-turn record.

        ``input_tokens`` / ``output_tokens`` come from the SDK's
        ``response.usage`` and are the ground-truth counts; the
        per-channel breakdown stays an estimate.
        """
        entry: dict[str, Any] = {
            "schema": 1,
            "turn_index": self._turn_index,
            "loop_iteration": loop_iteration,
            "correction_text": correction_text,
            "stop_reason": stop_reason,
            "latency_s": latency_s,
            # Estimates from the assembled messages list:
            "total_bytes": metrics.total_bytes,
            "tokens_est": metrics.tokens_est,
            "window_fill_fraction": metrics.window_fill_fraction,
            "system_bytes": metrics.system_bytes,
            "retrieval_addendum_bytes": metrics.retrieval_addendum_bytes,
            "history_bytes": metrics.history_bytes,
            "current_turn_bytes": metrics.current_turn_bytes,
            # Build-time savings:
            "clipped_count": metrics.clipped_count,
            "clipped_bytes_saved": metrics.clipped_bytes_saved,
            "stale_count": metrics.stale_count,
            "stale_bytes_saved": metrics.stale_bytes_saved,
            "read_calls_by_tool": metrics.read_calls_by_tool,
            # Ground truth (from response.usage):
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if extra is not None:
            entry["extra"] = extra
        self._turn_index += 1
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
