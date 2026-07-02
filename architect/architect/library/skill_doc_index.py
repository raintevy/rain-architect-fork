"""Embedding-indexed markdown skill docs for hybrid retrieval.

Today's path: ``architect.llm.prompts._load_skill_docs`` statically dumps
every ``.md`` file under ``skills/<robot>/`` + ``skills/<robot>/subtasks/``
into the agent's system prompt. That scales badly as the library grows —
docs irrelevant to the current task still cost their tokens every turn.

This module replaces the dump with an embedding-cosine retrieval that
returns *only* the docs relevant to the instruction, plus a small set
of foundational docs tagged ``always_loaded`` that need to be in-prompt
for every task (VQA conventions, motion safety bounds, etc.).

Two affordances:

  * :meth:`SkillDocIndex.retrieve` — k-NN retrieval at session start
    keyed by the instruction's embedding. The default ``k=3`` was
    chosen by the user; it's enough to cover most pick-place tasks
    without dragging the irrelevant docs along.
  * :meth:`SkillDocIndex.get_always_loaded` — returns the foundational
    docs that bypass retrieval (these are tagged in their frontmatter).

Both reuse :mod:`architect.library.skill_store`'s ``Embedder`` abstraction so
embeddings carry the same 4-byte format tag and fall back to lexical
Jaccard when no API backend is available — same fallback policy as the
SkillStore.

Cache: a sidecar ``skill_doc_index.json`` next to the docs maps each
filename → (mtime, description, embedding-base64). Entries with stale
mtime are recomputed on next use; unchanged entries skip the LLM /
embedding round-trip. This is the same pattern as the in-memory text-
embedding cache in :mod:`architect.llm.embeddings`, just persisted so it
survives across sessions.

Frontmatter convention (a single HTML comment near the top of each
``.md`` file — not part of the rendered content):

    <!-- architect-meta
    always_loaded: true
    description: One-line description for retrieval ranking.
    -->

Both fields are optional. ``always_loaded`` defaults to ``false`` (the
doc enters the retrieval pool); ``description`` defaults to the first
heading + first paragraph after it (heuristic — see :func:`_auto_describe`).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from architect.library.skill_store import Embedder, make_default_embedder


_INDEX_FILENAME = "skill_doc_index.json"
_INDEX_SCHEMA_VERSION = 1


_FRONTMATTER_RE = re.compile(
    r"^<!--\s*architect-meta\s*\n(.*?)\n-->",
    re.DOTALL,
)


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse the optional <!-- architect-meta ... --> block at the top of a doc.

    Tiny key:value YAML-ish parser — enough for our two known keys
    (``always_loaded`` boolean, ``description`` string). No nested
    structures supported; this is intentional. Unknown keys are
    silently ignored so docs can carry future metadata without
    breaking older readers.
    """
    m = _FRONTMATTER_RE.search(content)
    if not m:
        return {}
    out: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if key == "always_loaded":
            out["always_loaded"] = value.lower() in ("true", "yes", "1")
        elif key == "description":
            out["description"] = value
    return out


def _auto_describe(content: str) -> str:
    """First heading + first paragraph after it, capped at 280 chars.

    Used when the doc has no explicit ``description:`` in frontmatter.
    Strips the frontmatter block first so the heuristic doesn't latch
    onto its bookkeeping content.
    """
    # Strip frontmatter from the search window
    cleaned = _FRONTMATTER_RE.sub("", content, count=1).strip()
    lines = cleaned.splitlines()
    heading: str | None = None
    para_lines: list[str] = []
    for line in lines:
        ls = line.strip()
        if heading is None:
            if ls.startswith("#"):
                heading = ls.lstrip("#").strip()
            continue
        # After heading, collect lines until we hit a blank line *after*
        # we've already started collecting (so blank lines just after
        # the heading don't terminate immediately).
        if ls:
            para_lines.append(ls)
        elif para_lines:
            break
    parts: list[str] = []
    if heading:
        parts.append(heading)
    if para_lines:
        parts.append(" ".join(para_lines))
    desc = " — ".join(parts)
    if len(desc) > 280:
        desc = desc[:277] + "..."
    return desc or "(no description)"


@dataclass
class IndexedDoc:
    """One row in the skill-doc index."""
    name: str                       # filename, e.g. "grasping.md" or "subtasks/grasp_from_above.md"
    content: str                    # full markdown body (rendered into the prompt)
    description: str                # used for retrieval embedding
    always_loaded: bool             # bypasses retrieval if True
    mtime: float                    # cached for staleness detection
    embedding: bytes | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "mtime": self.mtime,
            "description": self.description,
            "always_loaded": self.always_loaded,
            "embedding_b64": (
                base64.b64encode(self.embedding).decode("ascii")
                if self.embedding is not None else None
            ),
        }

    @classmethod
    def from_json(cls, name: str, content: str, payload: dict[str, Any]) -> "IndexedDoc":
        emb = payload.get("embedding_b64")
        return cls(
            name=name,
            content=content,
            description=str(payload.get("description", "")) or _auto_describe(content),
            always_loaded=bool(payload.get("always_loaded", False)),
            mtime=float(payload.get("mtime", 0.0)),
            embedding=base64.b64decode(emb) if emb else None,
        )


@dataclass
class SkillDocHit:
    """One result from :meth:`SkillDocIndex.retrieve`."""
    name: str
    content: str
    description: str
    similarity: float


class SkillDocIndex:
    """Embedding-indexed markdown skill docs for a single ``skills_dir``.

    Indexes ``*.md`` directly under ``skills_dir`` AND ``subtasks/*.md``
    (the auto-emitted PatternReuse subtask docs) so all markdown the
    static dump used to load is reachable via retrieval.

    The index is built lazily on first method call and cached in
    memory; the persistent sidecar (``skill_doc_index.json``) bridges
    sessions so unchanged docs skip the embedding round-trip.
    """

    def __init__(
        self,
        skills_dir: Path,
        embedder: Embedder | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._embedder: Embedder = embedder or make_default_embedder()
        self._docs: dict[str, IndexedDoc] = {}
        self._loaded = False

    def _index_path(self) -> Path:
        return self._skills_dir / _INDEX_FILENAME

    def _load_sidecar(self) -> dict[str, Any]:
        p = self._index_path()
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if data.get("schema_version") != _INDEX_SCHEMA_VERSION:
            return {}
        return data.get("entries", {}) or {}

    def _save_sidecar(self) -> None:
        try:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": _INDEX_SCHEMA_VERSION,
                "entries": {name: doc.to_json() for name, doc in self._docs.items()},
            }
            self._index_path().write_text(json.dumps(payload, indent=2))
        except OSError:
            # Best-effort: the index works fine without a sidecar; we
            # just pay the embedding cost again next session.
            pass

    def _enumerate_md_files(self) -> list[Path]:
        out: list[Path] = []
        if not self._skills_dir.exists():
            return out
        # Top-level *.md
        out.extend(sorted(self._skills_dir.glob("*.md")))
        # subtasks/*.md
        sub = self._skills_dir / "subtasks"
        if sub.exists():
            out.extend(sorted(sub.glob("*.md")))
        return out

    def _doc_name(self, path: Path) -> str:
        """Return the canonical name we use in the index ('subtasks/x.md' or 'x.md')."""
        try:
            rel = path.relative_to(self._skills_dir)
        except ValueError:
            return path.name
        return str(rel)

    def _build(self) -> None:
        """Build / refresh the in-memory index. Uses sidecar entries
        when their mtime matches; re-embeds anything stale or new.
        """
        sidecar = self._load_sidecar()
        files = self._enumerate_md_files()
        dirty = False
        seen: set[str] = set()
        for path in files:
            name = self._doc_name(path)
            seen.add(name)
            try:
                content = path.read_text()
                cur_mtime = path.stat().st_mtime
            except OSError:
                continue

            cached = sidecar.get(name)
            if cached and float(cached.get("mtime", 0.0)) == cur_mtime:
                doc = IndexedDoc.from_json(name, content, cached)
                # Verify the cached embedding is in the current
                # embedder's format; if not, re-embed below.
                if (
                    doc.embedding is not None
                    and not self._embedder.matches_format(doc.embedding)
                ):
                    doc.embedding = None
            else:
                fm = _parse_frontmatter(content)
                description = fm.get("description") or _auto_describe(content)
                doc = IndexedDoc(
                    name=name,
                    content=content,
                    description=description,
                    always_loaded=bool(fm.get("always_loaded", False)),
                    mtime=cur_mtime,
                    embedding=None,
                )
                dirty = True

            if doc.embedding is None:
                try:
                    emb = self._embedder.embed(doc.description)
                    if isinstance(emb, (bytes, bytearray)):
                        doc.embedding = bytes(emb)
                        dirty = True
                except Exception:
                    # Best-effort: a missing embedding means the doc
                    # won't rank in retrieval but still loads if it's
                    # always_loaded.
                    pass

            self._docs[name] = doc

        # Drop entries for files that no longer exist
        for stale in list(self._docs.keys()):
            if stale not in seen:
                del self._docs[stale]
                dirty = True

        if dirty:
            self._save_sidecar()
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_built(self) -> None:
        if not self._loaded:
            self._build()

    def get_always_loaded(self) -> list[tuple[str, str]]:
        """All docs tagged ``always_loaded: true``, as (name, content) pairs.

        Sorted alphabetically by name for stable prompt assembly across
        sessions.
        """
        self.ensure_built()
        out = [
            (d.name, d.content)
            for d in self._docs.values()
            if d.always_loaded
        ]
        out.sort(key=lambda p: p[0])
        return out

    def retrieve(self, query: str, k: int = 3) -> list[SkillDocHit]:
        """Top-k retrievable docs by cosine on the query embedding.

        ``always_loaded`` docs are excluded from this retrieval pool
        (they're already returned by :meth:`get_always_loaded`, no
        need to double-count). Empty query, empty pool, or embedder
        failure → empty list.
        """
        self.ensure_built()
        if not query.strip() or k <= 0:
            return []
        try:
            q_emb = self._embedder.embed(query)
        except Exception:
            return []
        scored: list[tuple[float, IndexedDoc]] = []
        for doc in self._docs.values():
            if doc.always_loaded:
                continue
            if doc.embedding is None:
                continue
            score = self._embedder.similarity(q_emb, doc.embedding)
            scored.append((score, doc))
        scored.sort(key=lambda t: (-t[0], t[1].name))
        out: list[SkillDocHit] = []
        for score, doc in scored[:k]:
            out.append(SkillDocHit(
                name=doc.name,
                content=doc.content,
                description=doc.description,
                similarity=float(score),
            ))
        return out

    def all_names(self) -> list[str]:
        """All indexed doc names — for ``query_skills`` tool discovery."""
        self.ensure_built()
        return sorted(self._docs.keys())

    def get(self, name: str) -> IndexedDoc | None:
        """Look up a single indexed doc by canonical name."""
        self.ensure_built()
        return self._docs.get(name)
