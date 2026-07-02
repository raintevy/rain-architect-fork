"""SQLite-backed persistent store for session primitives (a.k.a. skills).

Schema (see CREATE TABLE below):
    id, name, code, docstring, description, embedding (BLOB, nullable),
    provenance (JSON), success_count, failure_count,
    graduation_status ('candidate' | 'graduated' | 'archived'),
    probe_suite_id (nullable, populated by P5 graduation gate),
    created_at, updated_at.

This module is the disk backing layer; ``architect.primitive_registry.PrimitiveRegistry``
is the thin in-memory wrapper that the agent loop talks to.

Embedding retrieval is pluggable. The store defaults to :class:`OpenAIEmbedder`
when the OpenAI / Azure embedding pipeline is reachable (same backend as
``architect.library.relevance``'s scorer filter — keeping both retrieval paths on the same
semantic space), and falls back to :class:`TextEmbedder` (difflib over UTF-8)
when no credential is available. Stored embeddings carry a 4-byte format tag
so the store can detect mismatches at retrieval time and lazy-re-embed rows
that were written by a different embedder.
"""

from __future__ import annotations

import array
import difflib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Protocol


# ---------------------------------------------------------------------------
# Embedder protocol + format tags
# ---------------------------------------------------------------------------


# Each embedder prepends one of these to its bytes output so the store can
# tell at retrieval time whether a stored embedding was written by the
# currently-active embedder. Mismatched rows are re-embedded lazily.
_TAG_TEXT = b"TXT0"
_TAG_OPENAI_V1 = b"V001"


class Embedder(Protocol):
    """Pluggable text embedder. Returns opaque ``bytes`` stored as a BLOB."""

    def embed(self, text: str) -> bytes: ...

    def similarity(self, a: bytes, b: bytes) -> float: ...

    def matches_format(self, emb: bytes) -> bool:
        """Whether ``emb`` was written by *this* embedder.

        Used by :meth:`SkillStore.retrieve` to decide whether a stored row's
        embedding is comparable to a freshly computed query embedding, or
        whether the row was persisted under a different embedder and should
        be re-embedded transparently.
        """
        ...


class TextEmbedder:
    """Difflib-over-UTF-8 fallback. No external dependencies, no API calls.

    Used when the OpenAI / Azure embedding pipeline isn't reachable (no API
    key, network down, deployment missing). Adequate for libraries of a
    handful of skills with descriptive docstrings; weak on synonym handling.
    """

    def embed(self, text: str) -> bytes:
        return _TAG_TEXT + text.encode("utf-8")

    def similarity(self, a: bytes, b: bytes) -> float:
        if not (self.matches_format(a) and self.matches_format(b)):
            return 0.0
        sa = a[len(_TAG_TEXT):].decode("utf-8", errors="replace")
        sb = b[len(_TAG_TEXT):].decode("utf-8", errors="replace")
        return difflib.SequenceMatcher(None, sa, sb).ratio()

    def matches_format(self, emb: bytes) -> bool:
        return emb.startswith(_TAG_TEXT)


class OpenAIEmbedder:
    """Cosine over OpenAI / Azure OpenAI embeddings (``architect.llm.embeddings``).

    Shares the embedding pipeline used by the slate-time relevance filter
    (``architect.library.relevance``), so both retrieval surfaces — library lookup and
    scorer relevance — live in the same semantic space. Vectors are packed
    as little-endian float32 after a 4-byte format tag.

    Construct via :func:`make_default_embedder` rather than instantiating
    directly so the fallback to :class:`TextEmbedder` (when no API key is
    set) is handled in one place.
    """

    def embed(self, text: str) -> bytes:
        from architect.llm.embeddings import get_embedding
        vec = get_embedding(text)
        return _TAG_OPENAI_V1 + array.array("f", vec).tobytes()

    def similarity(self, a: bytes, b: bytes) -> float:
        if not (self.matches_format(a) and self.matches_format(b)):
            return 0.0
        from architect.llm.embeddings import cosine
        va = array.array("f")
        va.frombytes(a[len(_TAG_OPENAI_V1):])
        vb = array.array("f")
        vb.frombytes(b[len(_TAG_OPENAI_V1):])
        return cosine(list(va), list(vb))

    def matches_format(self, emb: bytes) -> bool:
        return emb.startswith(_TAG_OPENAI_V1)


def make_default_embedder() -> Embedder:
    """Return :class:`OpenAIEmbedder` if reachable, else :class:`TextEmbedder`.

    Reachability is checked by attempting client initialisation (no API call
    is made here — the first embed will hit the network). If the OpenAI /
    Azure path can't be constructed, the lexical fallback is selected so the
    store remains usable in environments without an API key.
    """
    try:
        from architect.llm.embeddings import is_available
        if is_available():
            return OpenAIEmbedder()
    except Exception:
        pass
    return TextEmbedder()


# ---------------------------------------------------------------------------
# Schema + canonical row shape
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT    NOT NULL UNIQUE,
    code               TEXT    NOT NULL,
    docstring          TEXT    NOT NULL,
    description        TEXT    NOT NULL,
    embedding          BLOB,
    provenance         TEXT,
    success_count      INTEGER NOT NULL DEFAULT 0,
    failure_count      INTEGER NOT NULL DEFAULT 0,
    graduation_status  TEXT    NOT NULL DEFAULT 'candidate',
    probe_suite_id     INTEGER,
    created_at         REAL    NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at         REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(graduation_status);

-- Scorers + invariants emitted by the LLM from a correction (P3).
-- Both kinds share this table; the ``kind`` column distinguishes them.
-- A scorer returns a float in [0, 1]; an invariant returns bool.
-- The runtime contract for both:
--     fn(scene_before: dict, scene_after: dict, call_log: list[dict]) -> float|bool
CREATE TABLE IF NOT EXISTS scorers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL,             -- 'scorer' | 'invariant'
    code         TEXT    NOT NULL,             -- def <name>(...): ...
    preamble     TEXT    NOT NULL DEFAULT '',  -- imports + helper defs prepended at load
    description  TEXT    NOT NULL,             -- the correction text or paraphrase
    provenance   TEXT,                         -- JSON; correction id, session, task
    pass_count   INTEGER NOT NULL DEFAULT 0,   -- # of probes/runs that passed this
    fail_count   INTEGER NOT NULL DEFAULT 0,
    created_at   REAL    NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at   REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_scorers_kind ON scorers(kind);

-- Probe suites attached to a skill at graduation time (P5).
-- One row per graduation event. ``skills.probe_suite_id`` points here so
-- the regression check (re-run the same suite later) is deterministic.
CREATE TABLE IF NOT EXISTS probe_suites (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    seed                 INTEGER NOT NULL,    -- RNG seed for the perturbation suite
    k                    INTEGER NOT NULL,    -- number of probes
    robot_name           TEXT    NOT NULL,
    pass_rate            REAL    NOT NULL,    -- fraction of probes whose exec + scorers + invariants passed
    n_passed             INTEGER NOT NULL,
    n_total              INTEGER NOT NULL,
    invariant_pass_rate  REAL,                -- nullable when no invariants ran
    all_invariants_hold  INTEGER,             -- 1 / 0 / NULL
    report_json          TEXT,                -- full ProbeReport snapshot for audit
    created_at           REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);
"""


_VALID_STATUSES = ("candidate", "graduated", "archived")
_VALID_SCORER_KINDS = ("scorer", "invariant")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("provenance"):
        try:
            d["provenance"] = json.loads(d["provenance"])
        except (TypeError, ValueError):
            pass
    return d


# ---------------------------------------------------------------------------
# SkillStore
# ---------------------------------------------------------------------------


class SkillStore:
    """Persistent store of session primitives.

    Use ``db_path=":memory:"`` for an ephemeral store (tests, dry-run sessions
    that should not persist). Pass a filesystem path for a per-robot durable
    library, conventionally ``skills/<robot>/library.db``.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedder: Embedder | None = None,
    ) -> None:
        self._db_path = str(db_path)
        # Default embedder auto-picks OpenAI/Azure embeddings when reachable,
        # else falls back to lexical. Explicit ``embedder=`` overrides (used
        # by tests that pin a deterministic embedder).
        self._embedder: Embedder = embedder or make_default_embedder()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        code: str,
        docstring: str,
        *,
        description: str | None = None,
        provenance: dict | None = None,
        status: str | None = None,
    ) -> None:
        """Insert or replace a skill row.

        ``description`` defaults to ``docstring`` if not supplied; it is the
        text used for embedding-based retrieval. ``provenance`` is a free-form
        JSON object describing how this skill came to exist (correction text,
        session id, etc.) and is stored verbatim.

        ``status`` is applied verbatim on INSERT (defaulting to ``'candidate'``
        for new rows) and on UPDATE only if explicitly supplied. Re-registering
        an existing skill without specifying ``status`` therefore preserves
        any prior graduation state set via :meth:`admit` / :meth:`archive`.
        """
        if status is not None and status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        desc = description or docstring
        emb = self._embedder.embed(desc) if desc else None
        prov = json.dumps(provenance) if provenance is not None else None

        with closing(self._conn.cursor()) as cur:
            # Preserve the existing id / counts on overwrite so retrieval
            # ordering and outcome history survive re-registration.
            cur.execute("SELECT id FROM skills WHERE name = ?", (name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    INSERT INTO skills
                        (name, code, docstring, description, embedding,
                         provenance, graduation_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, code, docstring, desc, emb, prov, status or "candidate"),
                )
            elif status is None:
                cur.execute(
                    """
                    UPDATE skills
                       SET code = ?, docstring = ?, description = ?,
                           embedding = ?, provenance = ?,
                           updated_at = strftime('%s', 'now')
                     WHERE name = ?
                    """,
                    (code, docstring, desc, emb, prov, name),
                )
            else:
                cur.execute(
                    """
                    UPDATE skills
                       SET code = ?, docstring = ?, description = ?,
                           embedding = ?, provenance = ?,
                           graduation_status = ?,
                           updated_at = strftime('%s', 'now')
                     WHERE name = ?
                    """,
                    (code, docstring, desc, emb, prov, status, name),
                )
        self._conn.commit()

    def record_outcome(self, name: str, success: bool) -> None:
        """Increment the success or failure counter for a skill."""
        col = "success_count" if success else "failure_count"
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"UPDATE skills "
                f"SET {col} = {col} + 1, updated_at = strftime('%s', 'now') "
                f"WHERE name = ?",
                (name,),
            )
        self._conn.commit()

    def admit(self, name: str, probe_suite_id: int | None = None) -> None:
        """Promote a candidate skill to ``graduated`` status (used in P5)."""
        self._set_status(name, "graduated", probe_suite_id=probe_suite_id)

    def archive(self, name: str) -> None:
        """Soft-delete a skill (won't be injected, kept for audit)."""
        self._set_status(name, "archived")

    def forget(self, name: str) -> None:
        """Hard-delete a skill from the store."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM skills WHERE name = ?", (name,))
        self._conn.commit()

    def _set_status(
        self,
        name: str,
        status: str,
        probe_suite_id: int | None = None,
    ) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        with closing(self._conn.cursor()) as cur:
            if probe_suite_id is None:
                cur.execute(
                    "UPDATE skills "
                    "SET graduation_status = ?, updated_at = strftime('%s', 'now') "
                    "WHERE name = ?",
                    (status, name),
                )
            else:
                cur.execute(
                    "UPDATE skills "
                    "SET graduation_status = ?, probe_suite_id = ?, "
                    "    updated_at = strftime('%s', 'now') "
                    "WHERE name = ?",
                    (status, probe_suite_id, name),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> dict[str, Any] | None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM skills WHERE name = ?", (name,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def get_source(self, name: str) -> str | None:
        entry = self.get(name)
        return entry["code"] if entry else None

    def list_names(self, status: str | Iterable[str] | None = None) -> list[str]:
        return [row["name"] for row in self._select(status)]

    def get_all(self, status: str | Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [_row_to_dict(r) for r in self._select(status)]

    def to_spec_string(self, status: str | Iterable[str] | None = None) -> str:
        rows = self._select(status)
        if not rows:
            return ""
        lines = ["## Session Primitives\n"]
        for r in rows:
            lines.append(f"- `{r['name']}(...)` — {r['docstring']}")
        return "\n".join(lines)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        status: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k skills by description similarity to ``query``.

        Uses the configured :class:`Embedder`. Skills with no embedding —
        or whose stored embedding was written by a different embedder —
        are re-embedded transparently from their ``description`` column
        and the new bytes persisted, so libraries built under an older
        embedder migrate on first use without a startup pass.

        Results are sorted by descending similarity; ties broken by
        registration id (older first, matching injection order).
        """
        if not query:
            return []
        q_emb = self._embedder.embed(query)
        rows = self._select(status)
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            emb = r["embedding"]
            if emb is None or not self._embedder.matches_format(emb):
                # Row was persisted under a different embedder (or never
                # embedded). Re-embed from description and update in place;
                # if the re-embed itself fails (e.g. transient API error)
                # treat as similarity 0 so retrieval still functions.
                emb = self._reembed_row(r)
            score = self._embedder.similarity(q_emb, emb) if emb else 0.0
            scored.append((score, r))
        scored.sort(key=lambda t: (-t[0], t[1]["id"]))
        out: list[dict[str, Any]] = []
        for score, row in scored[: max(0, k)]:
            d = _row_to_dict(row)
            d["_similarity"] = score
            out.append(d)
        return out

    def _reembed_row(self, row: sqlite3.Row) -> bytes | None:
        """Compute a fresh embedding for ``row``'s description; persist + return.

        Returns ``None`` on failure (empty description, embedder raised) so
        callers can treat the row as un-rankable without aborting the whole
        retrieval.
        """
        desc = row["description"]
        if not desc:
            return None
        try:
            new_emb = self._embedder.embed(desc)
        except Exception:
            return None
        try:
            with closing(self._conn.cursor()) as cur:
                cur.execute(
                    "UPDATE skills SET embedding = ?, "
                    "updated_at = strftime('%s', 'now') WHERE id = ?",
                    (new_emb, row["id"]),
                )
            self._conn.commit()
        except Exception:
            # Persisting the migration is best-effort; if it fails (locked
            # DB, schema drift) the row still scores against the in-memory
            # value for this retrieval.
            pass
        return new_emb

    def _select(
        self,
        status: str | Iterable[str] | None,
    ) -> list[sqlite3.Row]:
        if status is None:
            sql = "SELECT * FROM skills ORDER BY id ASC"
            params: tuple = ()
        elif isinstance(status, str):
            sql = "SELECT * FROM skills WHERE graduation_status = ? ORDER BY id ASC"
            params = (status,)
        else:
            statuses = tuple(status)
            placeholders = ",".join("?" * len(statuses))
            sql = (
                f"SELECT * FROM skills "
                f"WHERE graduation_status IN ({placeholders}) "
                f"ORDER BY id ASC"
            )
            params = statuses
        with closing(self._conn.cursor()) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Scorers + invariants (P3)
    # ------------------------------------------------------------------

    def register_scorer(
        self,
        name: str,
        code: str,
        description: str,
        *,
        kind: str = "scorer",
        preamble: str = "",
        provenance: dict | None = None,
    ) -> None:
        """Insert or replace a scorer / invariant.

        ``code`` is the full ``def <name>(...): ...`` body. ``preamble`` is
        any imports + helper defs that must be exec'd alongside ``code``
        before the named callable can be invoked. The store does not load
        or validate either field — call ``architect.scorer.load_artifact`` before
        invoking. Re-registering by name preserves id and counters
        (analogous to skill re-registration)."""
        if kind not in _VALID_SCORER_KINDS:
            raise ValueError(f"Invalid scorer kind: {kind!r}")
        prov = json.dumps(provenance) if provenance is not None else None
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT id FROM scorers WHERE name = ?", (name,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    INSERT INTO scorers
                        (name, kind, code, preamble, description, provenance)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, kind, code, preamble, description, prov),
                )
            else:
                cur.execute(
                    """
                    UPDATE scorers
                       SET kind = ?, code = ?, preamble = ?,
                           description = ?, provenance = ?,
                           updated_at = strftime('%s', 'now')
                     WHERE name = ?
                    """,
                    (kind, code, preamble, description, prov, name),
                )
        self._conn.commit()

    def get_scorer(self, name: str) -> dict[str, Any] | None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM scorers WHERE name = ?", (name,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list_scorers(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Return all scorer rows (or filtered to one kind), oldest-first."""
        if kind is None:
            sql = "SELECT * FROM scorers ORDER BY id ASC"
            params: tuple = ()
        else:
            if kind not in _VALID_SCORER_KINDS:
                raise ValueError(f"Invalid scorer kind: {kind!r}")
            sql = "SELECT * FROM scorers WHERE kind = ? ORDER BY id ASC"
            params = (kind,)
        with closing(self._conn.cursor()) as cur:
            cur.execute(sql, params)
            return [_row_to_dict(r) for r in cur.fetchall()]

    def record_scorer_outcome(self, name: str, passed: bool) -> None:
        col = "pass_count" if passed else "fail_count"
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"UPDATE scorers "
                f"SET {col} = {col} + 1, updated_at = strftime('%s', 'now') "
                f"WHERE name = ?",
                (name,),
            )
        self._conn.commit()

    def forget_scorer(self, name: str) -> None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM scorers WHERE name = ?", (name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Probe suites (P5)
    # ------------------------------------------------------------------

    def register_probe_suite(
        self,
        *,
        seed: int,
        k: int,
        robot_name: str,
        pass_rate: float,
        n_passed: int,
        n_total: int,
        invariant_pass_rate: float | None = None,
        all_invariants_hold: bool | None = None,
        report_json: str | None = None,
    ) -> int:
        """Persist a probe suite snapshot; return its row id.

        The id becomes the ``probe_suite_id`` foreign key on the skill row
        when :meth:`admit` is called. Each graduation event creates a new
        probe_suite row even for the same skill — older rows are kept so
        the history of graduation attempts is auditable.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO probe_suites
                    (seed, k, robot_name, pass_rate, n_passed, n_total,
                     invariant_pass_rate, all_invariants_hold, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seed, k, robot_name, pass_rate, n_passed, n_total,
                    invariant_pass_rate,
                    None if all_invariants_hold is None else int(all_invariants_hold),
                    report_json,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def get_probe_suite(self, suite_id: int) -> dict[str, Any] | None:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT * FROM probe_suites WHERE id = ?", (suite_id,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def set_probe_suite_id(self, name: str, suite_id: int) -> None:
        """Attach a probe_suite_id to a skill without changing its graduation status.

        Used by the P5 graduation gate when the skill *failed* the gate —
        we still want the failed attempt visible in ``/library`` so the
        user can inspect why. Successful graduations should use
        :meth:`admit` which writes the suite id atomically with the
        status change.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE skills "
                "SET probe_suite_id = ?, updated_at = strftime('%s', 'now') "
                "WHERE name = ?",
                (suite_id, name),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "SkillStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
