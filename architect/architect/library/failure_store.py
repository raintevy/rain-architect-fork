"""Persistent store of structured failure diagnoses + their corrections.

Sits alongside :mod:`architect.library.skill_store` but in its own database
file (``skills/<robot>/failures.db``) for two reasons:

  * Failure records are a distinct artifact class from skills — they
    have a different schema, different lifecycle (fixed/still_failed/
    unknown outcomes), and a different retrieval pattern (lookup by
    failure-signature similarity, not by skill description).
  * Keeping them separate makes the failure log easy to share /
    aggregate for the paper's failure-mode analysis without dragging
    along the skill library state.

Each row represents one FAILED trial: the structured diagnosis
(:class:`architect.corrections.failure_diagnosis.FailureDiagnosis`), the
program at the time of failure, the correction text issued, the
post-correction program, and the eventual outcome
(``fixed`` / ``still_failed`` / ``unknown``).

Lookup uses the same embedding pipeline as ``SkillStore`` — cosine
similarity on embeddings of ``f"{instruction}\\n{failure_signature}\\n
{root_cause}"`` so two failures of similar shape on a similar task
score high. Falls back to lexical Jaccard when no embedding backend
is available, matching the SkillStore fallback policy.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Reuse the embedder abstraction from skill_store so we don't double-
# implement the API-backend selection + 4-byte format tag handling.
from architect.library.skill_store import (
    Embedder,
    OpenAIEmbedder,
    TextEmbedder,
    make_default_embedder,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    robot                  TEXT    NOT NULL,
    instruction            TEXT    NOT NULL,
    failure_category       TEXT    NOT NULL,
    failure_signature      TEXT    NOT NULL,
    root_cause             TEXT    NOT NULL,
    suggested_directions   TEXT,                -- JSON array
    evidence               TEXT,                -- JSON array
    judge_rationale        TEXT,
    program_before         TEXT,                -- the program that failed
    program_after          TEXT,                -- the refined program (set when correction lands)
    correction_text        TEXT,                -- the correction text issued (set when correction lands)
    outcome                TEXT    NOT NULL DEFAULT 'unknown',   -- 'fixed' | 'still_failed' | 'unknown'
    embedding              BLOB,
    created_at             REAL    NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at             REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_failures_outcome ON failures(outcome);
CREATE INDEX IF NOT EXISTS idx_failures_category ON failures(failure_category);
"""

_VALID_OUTCOMES = ("unknown", "fixed", "still_failed")


def _build_signature_text(instruction: str, signature: str, root_cause: str) -> str:
    """Text used as the embedding key for lookup similarity.

    Concatenates the task instruction with the failure signature and root
    cause so retrieval is biased toward "same task, same failure shape" —
    not just "same failure shape on any task" (which would over-retrieve
    generic patterns) and not just "same instruction" (which ignores the
    failure mode).
    """
    return f"{instruction}\n{signature}\n{root_cause}"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for k in ("suggested_directions", "evidence"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                pass
    return d


@dataclass
class FailureLookupHit:
    """One result from :meth:`FailureStore.lookup_similar`.

    ``similarity`` is the embedder's cosine in [0, 1]; tied rows are
    ordered by recency (descending ``created_at``).
    """
    id: int
    instruction: str
    failure_category: str
    failure_signature: str
    root_cause: str
    correction_text: str | None
    outcome: str
    similarity: float

    def to_markdown(self, *, max_correction_chars: int = 240) -> str:
        ct = (self.correction_text or "<no correction recorded yet>").strip()
        if len(ct) > max_correction_chars:
            ct = ct[: max_correction_chars - 3] + "..."
        return (
            f"- **sig:** {self.failure_signature} "
            f"_(category: `{self.failure_category}`, outcome: `{self.outcome}`, sim: {self.similarity:.2f})_  \n"
            f"  task: {self.instruction!r}  \n"
            f"  applied correction: {ct}"
        )


class FailureStore:
    """Persistent failure log + similarity lookup.

    ``db_path=":memory:"`` for ephemeral stores (tests, dry-run sessions
    that should not persist). Pass a filesystem path for a per-robot
    durable log, conventionally ``skills/<robot>/failures.db``.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedder: Embedder | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._embedder: Embedder = embedder or make_default_embedder()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def record_failure(
        self,
        *,
        robot: str,
        instruction: str,
        diagnosis,  # FailureDiagnosis — duck-typed to avoid the import dep
        judge_rationale: str = "",
        program_before: str = "",
    ) -> int:
        """Insert a failure row and return its id.

        ``diagnosis`` is duck-typed: any object with the attributes
        ``failure_category``, ``failure_signature``, ``root_cause``,
        ``suggested_correction_directions``, ``evidence``.
        """
        text = _build_signature_text(
            instruction, diagnosis.failure_signature, diagnosis.root_cause
        )
        try:
            emb_vec = self._embedder.embed(text)
            emb_blob = emb_vec if isinstance(emb_vec, (bytes, bytearray)) else None
        except Exception:
            emb_blob = None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO failures (
                    robot, instruction, failure_category, failure_signature,
                    root_cause, suggested_directions, evidence, judge_rationale,
                    program_before, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    robot,
                    instruction,
                    diagnosis.failure_category,
                    diagnosis.failure_signature,
                    diagnosis.root_cause,
                    json.dumps(list(diagnosis.suggested_correction_directions or [])),
                    json.dumps(list(diagnosis.evidence or [])),
                    judge_rationale,
                    program_before,
                    emb_blob,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_with_correction(
        self,
        failure_id: int,
        *,
        correction_text: str,
        program_after: str,
    ) -> None:
        """Attach the correction text + refined program to a recorded failure.

        Called after the correction is issued and the next program lands.
        Outcome stays ``unknown`` until :meth:`mark_outcome` is called
        based on the next execution's PASS/FAIL.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                UPDATE failures
                   SET correction_text = ?,
                       program_after   = ?,
                       updated_at      = strftime('%s', 'now')
                 WHERE id = ?
                """,
                (correction_text, program_after, failure_id),
            )
            self._conn.commit()

    def mark_outcome(self, failure_id: int, outcome: str) -> None:
        """Set the outcome for a failure row to ``fixed`` / ``still_failed``."""
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {_VALID_OUTCOMES!r}, got {outcome!r}"
            )
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                UPDATE failures
                   SET outcome    = ?,
                       updated_at = strftime('%s', 'now')
                 WHERE id = ?
                """,
                (outcome, failure_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get(self, failure_id: int) -> dict[str, Any] | None:
        with closing(self._conn.cursor()) as cur:
            row = cur.execute(
                "SELECT * FROM failures WHERE id = ?", (failure_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def count(self) -> int:
        with closing(self._conn.cursor()) as cur:
            return int(cur.execute("SELECT COUNT(*) FROM failures").fetchone()[0])

    def all(
        self,
        *,
        outcome: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return all rows, optionally filtered by outcome — for paper aggregation."""
        sql = "SELECT * FROM failures"
        params: list[Any] = []
        if outcome is not None:
            outcomes = (outcome,) if isinstance(outcome, str) else tuple(outcome)
            placeholders = ",".join("?" for _ in outcomes)
            sql += f" WHERE outcome IN ({placeholders})"
            params.extend(outcomes)
        sql += " ORDER BY created_at DESC"
        with closing(self._conn.cursor()) as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def lookup_similar(
        self,
        instruction: str,
        failure_signature: str,
        root_cause: str = "",
        *,
        k: int = 3,
        only_fixed: bool = True,
        exclude_id: int | None = None,
    ) -> list[FailureLookupHit]:
        """Top-k past failures by embedding cosine, optionally restricted to
        outcome='fixed' so we only surface successful patches.

        ``exclude_id`` lets callers skip a specific row (typically the
        just-recorded failure itself, which would otherwise be its own
        best match).

        Returns an empty list when the store is empty or no row's
        embedding is comparable to the query — never raises.
        """
        if k <= 0:
            return []
        query_text = _build_signature_text(
            instruction, failure_signature, root_cause
        )
        try:
            q_emb = self._embedder.embed(query_text)
        except Exception:
            return []

        sql = "SELECT * FROM failures"
        params: list[Any] = []
        clauses: list[str] = []
        if only_fixed:
            clauses.append("outcome = ?")
            params.append("fixed")
        if exclude_id is not None:
            clauses.append("id != ?")
            params.append(exclude_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        with closing(self._conn.cursor()) as cur:
            rows = cur.execute(sql, params).fetchall()

        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            emb = r["embedding"]
            if emb is None or not self._embedder.matches_format(emb):
                # Re-embed and persist back so subsequent lookups don't
                # take the slow path. Mirrors SkillStore._reembed_row.
                emb = self._reembed_row(r)
            score = self._embedder.similarity(q_emb, emb) if emb else 0.0
            scored.append((score, r))
        scored.sort(key=lambda t: (-t[0], -float(t[1]["created_at"])))

        out: list[FailureLookupHit] = []
        for score, row in scored[:k]:
            out.append(
                FailureLookupHit(
                    id=int(row["id"]),
                    instruction=row["instruction"],
                    failure_category=row["failure_category"],
                    failure_signature=row["failure_signature"],
                    root_cause=row["root_cause"],
                    correction_text=row["correction_text"],
                    outcome=row["outcome"],
                    similarity=float(score),
                )
            )
        return out

    def _reembed_row(self, row: sqlite3.Row) -> bytes | None:
        text = _build_signature_text(
            row["instruction"], row["failure_signature"], row["root_cause"]
        )
        if not text.strip():
            return None
        try:
            emb = self._embedder.embed(text)
        except Exception:
            return None
        if not isinstance(emb, (bytes, bytearray)):
            return None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE failures SET embedding = ?, updated_at = strftime('%s','now') WHERE id = ?",
                (emb, int(row["id"])),
            )
            self._conn.commit()
        return emb

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
