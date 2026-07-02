"""Subtask documentation auto-update after each correction.

After ``AgenticSession.correct`` lands a new program, a background daemon
thread asks Claude to emit (or update) a subtask doc that captures what
the correction taught about a generalised sub-operation. The doc is
persisted to ``skills/<robot>/subtasks/<subtask_name>.md`` and loaded
into future sessions' system prompts via
:func:`architect.prompt_builder.build_agentic_system_prompt`, so the LLM sees
accumulated subtask knowledge as part of its domain context.

Separation of concerns:

  * P3 scorers / invariants — *judge* programs (per-program verdict)
  * Subtask docs (this module) — *teach* the next program (generalised
    pattern), accumulated across corrections in human-readable form.

The split matters because scorers are runtime checks; subtask docs are
prompt-time guidance. Both fire as background emissions after a
correction so the user's correction loop never blocks on the
documentation update.

LLM output contract (one fenced ``json`` block):

    ```json
    {
      "subtask_name": "snake_case_name",
      "action":       "create" | "update" | "skip",
      "content":      "full markdown content of the .md file (omit on skip)"
    }
    ```

Validation is strict — bad names (path traversal, special chars,
empty), bad actions, or empty content for create/update → the emission
silently drops the response. A buggy LLM response must not corrupt the
skills dir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Artifact + validation
# ---------------------------------------------------------------------------


_VALID_ACTIONS: frozenset[str] = frozenset({"create", "update", "skip"})


# snake_case identifier; no path components, no leading/trailing underscore,
# 2–48 chars. Restrictive on purpose — the value is interpolated into a
# filesystem path so any path traversal or shell-meta would be a security
# bug. The LLM is told to produce only snake_case.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")


@dataclass
class SubtaskDoc:
    """One emitted subtask doc — name + action + markdown content."""

    name: str
    action: str   # 'create' | 'update' | 'skip'
    content: str
    provenance: dict[str, Any] = field(default_factory=dict)


def _valid_name(name: str) -> bool:
    """``True`` iff ``name`` is a safe snake_case filename stem."""
    return bool(_NAME_RE.fullmatch(name))


# ---------------------------------------------------------------------------
# Parsing the LLM response
# ---------------------------------------------------------------------------


# Tagged-fence parser. We expect one ````json```` block carrying just
# {subtask_name, action} and (for create/update) one ````markdown```` block
# with the doc content.
#
# Outer fences use FOUR backticks instead of the usual three so that
# embedded three-backtick code blocks inside the markdown body don't
# prematurely terminate the outer fence under the regex's lazy ``.*?``.
# Without this, a doc that includes a ``\`\`\`python ... \`\`\`\`` example
# (which is natural for subtask docs that document a pattern with code)
# truncates at the first inner closing fence the regex sees, dropping
# everything after.
#
# Two blocks rather than one fat JSON because the previous "stuff
# multi-line markdown in a JSON string" format died on literal newlines
# whenever the LLM forgot to ``\n``-escape them — which Claude does
# often when the content is long.
_FENCE_TAGGED_RE = re.compile(
    r"`{4}(json|markdown|md)\s*\n(.*?)`{4}",
    re.DOTALL | re.IGNORECASE,
)
# Untagged fallback — first 4-backtick fenced block without a tag, used
# when the LLM emits one block and forgets the language hint. Same
# 4-backtick rule applies so inner code-block fences don't close the
# outer match early.
_FENCE_ANY_RE = re.compile(r"`{4}\s*\n(.*?)`{4}", re.DOTALL)
# Legacy 3-backtick parsers — kept as a fallback for responses generated
# before the 4-backtick prompt change landed (or for LLMs that ignore
# the explicit instruction). Subject to the truncation issue above when
# the markdown body contains embedded code blocks, but better than
# returning ``None`` on parseable-but-3-backtick responses.
_FENCE_TAGGED_RE_LEGACY = re.compile(
    r"```(json|markdown|md)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_ANY_RE_LEGACY = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _extract_blocks(text: str) -> tuple[str | None, str | None]:
    """Return ``(json_blob, markdown_blob)`` from a tagged-fence response.

    Walks every 4-backtick ``<tag>`` block; first json-tagged → metadata,
    first markdown/md-tagged → content. If only one tagged block is
    present the other slot is ``None``. Untagged fences are ignored at
    this stage; ``parse_response`` falls back to the legacy single-JSON
    format only when no tagged json block matches.

    Falls back to 3-backtick fences (the pre-bugfix format) only for the
    JSON block if no 4-backtick blocks were found, since the JSON
    metadata can't contain embedded code-block fences and therefore
    doesn't suffer the inner-fence-truncates-outer collision. The
    markdown block always uses the 4-backtick parser to avoid silently
    truncating docs that include code examples.
    """
    json_blob: str | None = None
    md_blob: str | None = None
    for tag, body in _FENCE_TAGGED_RE.findall(text):
        t = tag.lower()
        if t == "json" and json_blob is None:
            json_blob = body.strip()
        elif t in ("markdown", "md") and md_blob is None:
            md_blob = body.strip()
    # Legacy 3-backtick fallback for the JSON block only. This handles
    # cached / older LLM responses or LLMs that ignore the 4-backtick
    # instruction. We do NOT fall back for the markdown block because
    # 3-backtick markdown fences are the exact pathology this bug fixed
    # — silently returning a truncated body is worse than returning None
    # and surfacing the parse failure.
    if json_blob is None:
        for tag, body in _FENCE_TAGGED_RE_LEGACY.findall(text):
            if tag.lower() == "json":
                json_blob = body.strip()
                break
    return json_blob, md_blob


def parse_response(text: str) -> SubtaskDoc | None:
    """Extract + validate a :class:`SubtaskDoc` from the LLM response.

    Two-block format (preferred):
      * One ```json``` block — ``{"subtask_name": "...", "action": "..."}``
      * For create/update: one ```markdown``` block — the file body.

    Legacy single-block fallback: a ```json``` block carrying
    ``content`` inline. Accepted but discouraged — multi-line markdown
    inside a JSON string is fragile.

    Returns ``None`` when the response is malformed (no JSON block,
    missing fields, invalid name, invalid action, empty content on
    create/update). The daemon thread then logs a dim warning + a
    snippet of the response so the failure mode is debuggable.
    """
    json_blob, md_blob = _extract_blocks(text)

    # Fallback: untagged 4-backtick fence (LLM emitted the right delimiter
    # but forgot the language hint). Then 3-backtick untagged as a deeper
    # fallback — same restriction as above, only applied to find the JSON
    # blob, never the markdown body.
    if json_blob is None:
        m = _FENCE_ANY_RE.search(text)
        if m is not None:
            json_blob = m.group(1).strip()
    if json_blob is None:
        m = _FENCE_ANY_RE_LEGACY.search(text)
        if m is not None:
            json_blob = m.group(1).strip()

    if json_blob is None:
        return None
    try:
        meta = json.loads(json_blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None

    name = meta.get("subtask_name")
    action = meta.get("action")
    if not isinstance(name, str) or not _valid_name(name):
        return None
    if action not in _VALID_ACTIONS:
        return None

    if action == "skip":
        return SubtaskDoc(name=name, action=action, content="")

    # For create/update, prefer the markdown block; fall back to a
    # ``content`` field on the JSON for legacy single-block responses.
    content: str
    if md_blob:
        content = md_blob
    elif isinstance(meta.get("content"), str) and meta["content"].strip():
        content = meta["content"]
    else:
        return None

    return SubtaskDoc(name=name, action=action, content=content)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_subtask_doc(
    doc: SubtaskDoc,
    subtasks_dir: Path,
) -> Path | None:
    """Write the doc to ``<subtasks_dir>/<name>.md``.

    Returns the path that was written, or ``None`` for action='skip' and
    for any I/O failure. ``create`` and ``update`` both overwrite the
    file — the LLM has already seen any existing content and decided
    what the file should look like end-to-end. Future sessions read the
    persisted file via :func:`architect.prompt_builder.build_agentic_system_prompt`.
    """
    if doc.action == "skip":
        return None
    try:
        subtasks_dir.mkdir(parents=True, exist_ok=True)
        path = subtasks_dir / f"{doc.name}.md"
        path.write_text(doc.content)
    except OSError:
        return None
    return path


def load_existing_subtask_docs(subtasks_dir: Path) -> list[tuple[str, str]]:
    """Return ``[(name, content), ...]`` for the prompt's "existing docs" section.

    Sorted by name so the LLM sees a stable list across runs. Missing
    dir returns ``[]`` (legitimate first-correction state). Files with
    unreadable content are skipped silently.
    """
    if not subtasks_dir.exists():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(subtasks_dir.glob("*.md")):
        try:
            out.append((p.stem, p.read_text()))
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Similarity-driven create-vs-update directive
# ---------------------------------------------------------------------------


# Default cosine similarity threshold above which an existing doc is
# considered "similar enough that this correction should update it" rather
# than "create a new doc." Conservative starting value — easy to drop later
# if observed sweeps produce duplicate near-identical docs. The retrieval
# filter in architect/relevance.py uses 0.45 for "this is relevant enough to
# surface"; update is a stricter decision because it overwrites content.
_DEFAULT_UPDATE_THRESHOLD: float = 0.70


def select_target_doc(
    correction: str,
    existing_docs: list[tuple[str, str]],
    *,
    threshold: float = _DEFAULT_UPDATE_THRESHOLD,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> tuple[str, str | None, float]:
    """Decide whether this correction should update an existing doc or create new.

    Computes cosine similarity between the correction text and each existing
    doc's full content; if the max similarity is at or above ``threshold``,
    returns ``("update", <matched_name>, max_sim)``. Otherwise returns
    ``("create", None, max_sim)``.

    Falls back to ``("create", None, 0.0)`` when no existing docs exist or
    when the embedding backend is unavailable — conservative bias toward
    never silently overwriting an existing doc when similarity is
    unmeasurable.

    ``embed_fn`` is injected for testing; defaults to
    :func:`architect.embeddings.get_embeddings`. The default path benefits from
    the bounded LRU cache so repeated doc embeddings within a session are
    cheap.
    """
    if not existing_docs:
        return ("create", None, 0.0)

    if embed_fn is None:
        try:
            from architect.llm.embeddings import get_embeddings
            embed_fn = get_embeddings
        except Exception:
            return ("create", None, 0.0)

    names = [name for name, _ in existing_docs]
    contents = [content for _, content in existing_docs]
    try:
        # Single batch: [correction, doc1, doc2, ...] so we get them all in
        # one Azure call and the cache covers any reuse across corrections.
        vectors = embed_fn([correction, *contents])
    except Exception:
        return ("create", None, 0.0)

    from architect.llm.embeddings import cosine
    correction_vec = vectors[0]
    doc_vecs = vectors[1:]
    sims = [cosine(correction_vec, dv) for dv in doc_vecs]
    if not sims:
        return ("create", None, 0.0)
    max_sim = max(sims)
    best_idx = sims.index(max_sim)
    if max_sim >= threshold:
        return ("update", names[best_idx], max_sim)
    return ("create", None, max_sim)


# ---------------------------------------------------------------------------
# Emission (LLM call → SubtaskDoc → file)
# ---------------------------------------------------------------------------


def _default_llm_call(messages: list[dict], model: str) -> str:
    from architect.llm.client import call_claude
    return call_claude(messages, model=model, max_tokens=4096, temperature=0.2)


def emit_subtask_doc(
    correction: str,
    *,
    current_program: str,
    prior_program: str | None,
    subtasks_dir: Path,
    llm_call: Callable[..., str] | None = None,
    model: str | None = None,
    use_similarity_directive: bool = False,
    similarity_threshold: float = _DEFAULT_UPDATE_THRESHOLD,
) -> tuple[SubtaskDoc | None, str]:
    """One LLM call → ``(doc_or_None, raw_response)``.

    The caller (:meth:`AgenticSession._spawn_subtask_doc_emission`) runs
    this in a daemon thread and persists the result via
    :func:`persist_subtask_doc`. The raw LLM response is returned
    alongside the parsed doc so the daemon can log a snippet on parse
    failures — without that, a malformed response looked identical
    on the console to a successful skip, and was effectively
    undebuggable.

    When ``use_similarity_directive`` is False (the default), the LLM
    decides ``create`` / ``update`` / ``skip`` on its own from the
    existing-docs section of the prompt. This is the behavior used by
    ICL / FuncReuse / SlateGated.

    When True (PatternReuse), :func:`select_target_doc` is called first
    to compute the cosine similarity between the correction text and
    each existing doc's content. The directive (update-this-doc /
    create-new) is then bolted into the prompt so the LLM writes content
    appropriate to the chosen action rather than re-deciding. The
    parsed response is validated against the directive — if the LLM
    emitted an action / name that contradicts what was asked, the
    function returns ``None`` so the daemon thread logs a parse-failure
    diagnostic rather than silently writing the wrong file.
    """
    from architect.llm.prompts import build_subtask_doc_prompt

    llm = llm_call if llm_call is not None else _default_llm_call
    if model is None:
        from config import ANTHROPIC_MODEL  # late import — avoids circular import
        model = ANTHROPIC_MODEL

    existing = load_existing_subtask_docs(subtasks_dir)

    directive: tuple[str, str | None, float] | None = None
    if use_similarity_directive:
        directive = select_target_doc(
            correction, existing, threshold=similarity_threshold,
        )

    messages = build_subtask_doc_prompt(
        correction=correction,
        current_program=current_program,
        prior_program=prior_program,
        existing_docs=existing,
        directive=directive,
    )
    response = llm(messages, model=model)
    doc = parse_response(response)
    if doc is None:
        return None, response

    # Validate the LLM honoured the directive. Skip is always honoured
    # (the LLM may decide the correction is too trivial); but create/update
    # must match the directive's chosen action + name.
    if directive is not None and doc.action != "skip":
        directive_action, directive_name, _max_sim = directive
        if doc.action != directive_action:
            return None, response
        if directive_action == "update" and doc.name != directive_name:
            return None, response

    doc.provenance = {
        "correction": correction,
        "prior_program_excerpt": (prior_program or "")[:500],
        "current_program_excerpt": current_program[:500],
        "n_existing_docs": len(existing),
        "directive": (
            {
                "action": directive[0],
                "target_name": directive[1],
                "max_similarity": directive[2],
            }
            if directive is not None else None
        ),
    }
    return doc, response
