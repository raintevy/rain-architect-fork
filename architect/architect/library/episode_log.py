"""Append-only episode log for slate decisions (lightweight §4.7).

One line of JSON per slate the user (or the auto-apply rule) resolved. The
log is *not* the full episodic memory described in §4.7 of the design doc —
P4 only persists the minimum needed to compute the *useful-disagreement
rate* (the fraction of slates where the user's pick was not the rank-1
candidate) and to feed multi-axis supervision to later phases. Full
embedding-indexed retrieval lands later if the paper needs it.

File layout: ``skills/<robot>/episodes.jsonl``, one JSON object per line.
Open-append only — no rewriting, no compaction. A field ``schema=1`` lets
us migrate later.

One entry shape::

    {
      "schema": 1,
      "ts": 1715000000.123,
      "correction": "approach from the side, not above",
      "n_candidates": 3,
      "picked_index": 0,           # null if cancelled / abandoned
      "auto_applied": false,       # true if dominant + user wasn't asked
      "candidates": [              # sorted by score descending
        {
          "axis_label":   "approach from +y, 5cm offset, cache detection",
          "fingerprint":  ["detect_objects", "move_ee_to_pose", ...],
          "score":        0.92,
          "score_breakdown": {"pass_rate": 0.875, "scorer_mean": 1.0, "audit_penalty": 0.0},
          "audit_severity": {"high": 0, "medium": 0, "low": 0}
        }, ...
      ],
      "rejected_axes": ["...", "..."]   # the un-picked candidates' labels
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from architect.probes.audit import severity_summary
from architect.probes.sampling import Candidate, Slate


_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------


class EpisodeLog:
    """Append-only writer over a per-robot ``episodes.jsonl``.

    Constructed with the file path; creates the parent dir if needed. All
    writes are line-buffered ``json.dumps`` + newline so consumers (eval
    scripts, dashboards) can stream-read.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record_slate(
        self,
        slate: Slate,
        *,
        picked_index: int | None,
        auto_applied: bool = False,
        rejected_axes: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one slate-resolution record.

        ``picked_index`` is the index *in slate.candidates* the user / auto-rule
        chose (or ``None`` when cancelled). ``rejected_axes`` defaults to the
        axis labels of every other candidate; pass an explicit list to record
        a different supervision shape (e.g. multi-pick).
        """
        if rejected_axes is None and picked_index is not None:
            rejected_axes = [
                c.axis_label for i, c in enumerate(slate.candidates) if i != picked_index
            ]
        elif rejected_axes is None:
            rejected_axes = []

        entry: dict[str, Any] = {
            "schema": _SCHEMA_VERSION,
            "ts": time.time(),
            "correction": slate.correction,
            "n_candidates": len(slate.candidates),
            "picked_index": picked_index,
            "auto_applied": bool(auto_applied),
            "candidates": [_candidate_summary(c) for c in slate.candidates],
            "rejected_axes": rejected_axes,
        }
        if extra is not None:
            entry["extra"] = extra
        self._append(entry)

    # ------------------------------------------------------------------
    # Read helpers (mostly for tests + eval aggregation)
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def useful_disagreement_rate(self) -> float | None:
        """Across all surfaced (non-auto-applied) slates, the fraction where
        the picked candidate was *not* the rank-1 (index 0). Returns ``None``
        if no surfaced slates exist yet — distinguishing "no signal" from
        "perfect agreement"."""
        surfaced = [
            r for r in self.read_all()
            if not r.get("auto_applied") and r.get("picked_index") is not None
        ]
        if not surfaced:
            return None
        disagree = sum(1 for r in surfaced if r["picked_index"] != 0)
        return disagree / len(surfaced)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, entry: dict[str, Any]) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def _candidate_summary(c: Candidate) -> dict[str, Any]:
    sev = severity_summary(c.audit_flags)
    # Copy the breakdown so we don't share mutable dicts with the live
    # Candidate; nested per_scorer / per_invariant dicts are already plain
    # JSON-serialisable types from sampling._aggregate_*.
    breakdown = dict(c.score_breakdown)
    for key in ("per_scorer", "per_invariant_violation_rate"):
        if key in breakdown and breakdown[key] is not None:
            breakdown[key] = dict(breakdown[key])
    return {
        "axis_label": c.axis_label,
        "fingerprint": list(c.fingerprint),
        "score": c.score,
        "score_breakdown": breakdown,
        "audit_severity": sev,
    }
