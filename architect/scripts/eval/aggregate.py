#!/usr/bin/env python3
"""Read trial JSONs and produce the §8.4 metric tables.

Headline output: held-out OOD success rate per (task, condition), with
95% bootstrap CIs over seeds. Secondary metrics: mean corrections-applied,
useful-disagreement rate (from per-trial episode logs), library
compactness (graduated vs. total).

A paired-Wilcoxon column compares each baseline to ``SlateGated`` within
each task, using the per-seed OOD pass-rate vector as the paired sample.
Holm correction across tasks is applied to the displayed *p* values when
more than one task is reported. Below the per-task table, an "overall"
row summarises the macro-mean across tasks.

Usage:
    python -m scripts.eval.aggregate --trial-dir /tmp/eval/
    python -m scripts.eval.aggregate --trial-dir /tmp/eval/ --csv /tmp/eval/summary.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys

# Display order for condition columns. Keep in sync with the conditions
# block of scripts/eval/tasks.json. New conditions append to the right
# of the existing baselines.
_CONDITION_ORDER: tuple[str, ...] = (
    "ICL", "FuncReuse", "SlateGated", "PatternReuse",
)
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI on the sample mean.

    Use over per-seed pass-rates so the CI reflects seed-variance.
    Returns ``(lo, hi)``. Degenerate input (≤1 value) returns
    ``(mean, mean)``.
    """
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        v = values[0]
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int(alpha / 2 * n_resamples)
    hi_i = int((1 - alpha / 2) * n_resamples) - 1
    return means[lo_i], means[hi_i]


# ---------------------------------------------------------------------------
# Paired Wilcoxon (signed-rank) — minimal stdlib-only implementation
# ---------------------------------------------------------------------------


def _wilcoxon_signed_rank_p(a: list[float], b: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank ``p``-value on paired (a, b).

    Uses normal approximation with continuity correction; appropriate
    when ``n >= 8`` and there aren't too many ties. Returns ``None``
    when ``n < 5`` (the test is uninformative on very small samples).

    Avoids ``scipy`` so the eval harness has no heavy dependencies.
    """
    import math

    if len(a) != len(b) or len(a) < 5:
        return None
    diffs = [ai - bi for ai, bi in zip(a, b) if ai != bi]
    n = len(diffs)
    if n < 5:
        return None
    abs_diffs = sorted(((abs(d), 1 if d > 0 else -1, i) for i, d in enumerate(diffs)),
                       key=lambda t: t[0])
    # Average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[abs_diffs[k][2]] = avg_rank
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return 1.0
    z = (w - mu + 0.5) / sigma   # continuity correction
    # Two-sided p
    return 2 * (1 - _phi(abs(z)))


def _phi(x: float) -> float:
    """Standard-normal CDF via the error function."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _holm_correct(pvals: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down correction across the dict's keys.

    Inputs with ``None`` p-values pass through. Returns a fresh dict.
    """
    items = [(k, p) for k, p in pvals.items() if p is not None]
    items.sort(key=lambda t: t[1])
    m = len(items)
    out: dict[str, float | None] = dict(pvals)
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - rank))
        # enforce monotonicity
        if rank > 0:
            prev_k = items[rank - 1][0]
            prev_adj = out[prev_k]
            if isinstance(prev_adj, float):
                adj = max(adj, prev_adj)
        out[k] = adj
    return out


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------


def load_trials(trial_dir: Path) -> list[dict]:
    """Glob ``trial_*.json`` (top-level only) and parse them."""
    paths = sorted(trial_dir.glob("trial_*.json"))
    out: list[dict] = []
    for p in paths:
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    n = len(values)
    m = mean(values)
    s = stdev(values) if n > 1 else 0.0
    lo, hi = _bootstrap_ci(values)
    return {"n": n, "mean": m, "std": s, "ci_lo": lo, "ci_hi": hi}


def aggregate(trials: list[dict]) -> dict:
    """Group trials by (task, condition); compute per-cell summaries.

    Returns a nested dict::

        {
          "tasks": {
            "pick_block_on_marker": {
              "ICL":        { ood: {...}, corrections: {...}, library: {...} },
              "FuncReuse":  {...},
              "SlateGated": {...},
              "p_vs_SlateGated": { "ICL": 0.03, "FuncReuse": 0.12 }
                                  # Wilcoxon, Holm-corrected
            },
            ...
          },
          "overall": { "ood_by_condition": {...}, "useful_disagreement_rate": ... }
        }
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in trials:
        by_cell[(t["task_id"], t["condition"])].append(t)

    tasks: dict[str, dict] = {}
    for (task_id, cond), records in by_cell.items():
        cell = tasks.setdefault(task_id, {})
        # Useful-disagreement: per-cell aggregate over the trials in this cell
        # that actually carried a non-None UD rate (i.e. surfaced slates were
        # logged with a user pick). Only SlateGated produces these in current
        # sweeps.
        ud_values = [
            r["useful_disagreement_rate"] for r in records
            if r.get("useful_disagreement_rate") is not None
        ]
        cell[cond] = {
            "ood":         _summarise([r["ood_pass_rate"] for r in records]),
            "corrections": _summarise([r["corrections_applied"] for r in records]),
            "lib_total":   _summarise([r["library"]["total"] for r in records]),
            "lib_grad":    _summarise([r["library"]["graduated"] for r in records]),
            "useful_disagreement": _summarise(ud_values),
            "seeds":       sorted(r["seed"] for r in records),
            "raw_ood":     [r["ood_pass_rate"] for r in records],
            "raw_seeds":   [r["seed"] for r in records],
        }

    # Paired Wilcoxon vs. SlateGated within each task.
    for task_id, cell in tasks.items():
        if "SlateGated" not in cell:
            continue
        sg_by_seed = dict(zip(cell["SlateGated"]["raw_seeds"],
                              cell["SlateGated"]["raw_ood"]))
        raw_p: dict[str, float | None] = {}
        for other_cond in [c for c in cell
                           if c != "SlateGated"
                           and "raw_seeds" in (cell.get(c) or {})]:
            other_seeds = cell[other_cond]["raw_seeds"]
            paired_a, paired_b = [], []
            for s in other_seeds:
                if s in sg_by_seed:
                    paired_a.append(sg_by_seed[s])
                    paired_b.append(cell[other_cond]["raw_ood"][other_seeds.index(s)])
            raw_p[other_cond] = _wilcoxon_signed_rank_p(paired_a, paired_b)
        cell["p_vs_SlateGated"] = _holm_correct(raw_p)

    # Useful-disagreement: aggregated across SlateGated trials.
    ud_values = [
        t.get("useful_disagreement_rate") for t in trials
        if t.get("condition") == "SlateGated" and t.get("useful_disagreement_rate") is not None
    ]
    overall = {
        "useful_disagreement_rate": (sum(ud_values) / len(ud_values)) if ud_values else None,
        "ud_n_trials": len(ud_values),
    }

    # Macro-mean OOD per condition (mean over per-task means)
    conditions_seen: set[str] = set()
    for cell in tasks.values():
        for c in cell:
            if c in ("p_vs_SlateGated",):
                continue
            conditions_seen.add(c)
    ood_by_condition = {}
    for c in conditions_seen:
        per_task = [cell[c]["ood"]["mean"] for cell in tasks.values() if c in cell]
        ood_by_condition[c] = _summarise(per_task) if per_task else {
            "n": 0, "mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
        }
    overall["ood_by_condition"] = ood_by_condition

    # Per-sequence position curves. Only populated when trials carry a
    # ``sequence_id`` + ``position_in_sequence`` — i.e. the sweep was run
    # in sequence mode. Groups by (sequence_id, position, condition) so
    # we can plot OOD as a function of position-in-sequence per condition.
    # If FuncReuse/SlateGated win mostly at later positions, that's the
    # C3 library-compounding evidence the sweep was designed to expose.
    sequences: dict[str, dict] = {}
    sequenced = [
        t for t in trials
        if t.get("sequence_id") and t.get("position_in_sequence") is not None
    ]
    if sequenced:
        by_seq_pos_cond: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
        for t in sequenced:
            key = (t["sequence_id"], int(t["position_in_sequence"]), t["condition"])
            by_seq_pos_cond[key].append(t)
        for (sid, pos, cond), recs in by_seq_pos_cond.items():
            seq_entry = sequences.setdefault(sid, {})
            positions = seq_entry.setdefault("positions", {})
            pos_entry = positions.setdefault(pos, {})
            pos_entry["task_id"] = recs[0]["task_id"]
            pos_entry.setdefault("by_condition", {})[cond] = {
                "ood":      _summarise([r["ood_pass_rate"] for r in recs]),
                "lib_grad": _summarise([r["library"]["graduated"] for r in recs]),
                "lib_total":_summarise([r["library"]["total"] for r in recs]),
                "n":        len(recs),
            }
    return {"tasks": tasks, "overall": overall, "sequences": sequences}


# ---------------------------------------------------------------------------
# Pretty printing (rich-optional)
# ---------------------------------------------------------------------------


def _format_value(s: dict) -> str:
    return f"{s['mean']:.0%} ± {s['std']:.0%}  [n={s['n']}]"


def _format_ci(s: dict) -> str:
    return f"[{s['ci_lo']:.0%}, {s['ci_hi']:.0%}]"


def render_text(report: dict) -> str:
    """Plain-text table for environments without rich (CI / paper appendices)."""
    lines = []
    lines.append("=" * 88)
    lines.append(f"{'task':<32s}  {'condition':<10s}  {'OOD (mean±std)':<22s}  "
                 f"{'95% CI':<18s}  {'p vs SlateGated':<16s}")
    lines.append("-" * 88)
    for task_id, cell in sorted(report["tasks"].items()):
        for cond in _CONDITION_ORDER:
            if cond not in cell or "ood" not in cell[cond]:
                continue
            s = cell[cond]["ood"]
            p = cell.get("p_vs_SlateGated", {}).get(cond)
            p_str = f"{p:.3f}" if isinstance(p, float) else "—"
            lines.append(f"{task_id:<32s}  {cond:<10s}  {_format_value(s):<22s}  "
                         f"{_format_ci(s):<18s}  {p_str:<10s}")
        lines.append("")
    lines.append("-" * 88)
    o = report["overall"]
    lines.append("OVERALL — macro-mean OOD pass-rate (per condition, across tasks):")
    for cond, s in sorted(o["ood_by_condition"].items()):
        lines.append(f"  {cond:<10s}  {_format_value(s):<22s}  {_format_ci(s):<18s}")
    ud = o.get("useful_disagreement_rate")
    if ud is not None:
        lines.append(f"\nUseful-disagreement rate (SlateGated, surfaced slates):  "
                     f"{ud:.0%}  [n={o['ud_n_trials']}]")
    lines.append("=" * 88)

    # Per-sequence position curves. Only rendered when the sweep ran in
    # sequence mode (tasks.json defines `sequences` and --per-task wasn't
    # set). This is the headline view for the C3 "library compounds
    # across sessions" claim — read down each column to see how a
    # condition's OOD evolves over the sequence.
    sequences = report.get("sequences") or {}
    if sequences:
        lines.append("")
        lines.append("Per-sequence position curves (OOD mean ± std):")
        for sid in sorted(sequences):
            seq = sequences[sid]
            positions = seq.get("positions", {})
            if not positions:
                continue
            lines.append("")
            lines.append(f"  sequence: {sid}")
            header = f"    {'pos':>3}  {'task':<28s}  " + "  ".join(
                f"{c:<22s}" for c in _CONDITION_ORDER
            )
            lines.append(header)
            for pos in sorted(positions):
                pe = positions[pos]
                row = f"    {pos:>3}  {pe.get('task_id', ''):<28s}  "
                cells = []
                for cond in _CONDITION_ORDER:
                    by_c = pe.get("by_condition", {}).get(cond)
                    cells.append(_format_value(by_c["ood"]) if by_c else "—")
                row += "  ".join(f"{c:<22s}" for c in cells)
                lines.append(row)
        lines.append("=" * 88)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_csv(report: dict, path: Path) -> None:
    """One row per (task, condition); columns include OOD pass-rate, CIs,
    library compactness, paired-Wilcoxon p-value vs SlateGated, and (for
    the SlateGated rows) the per-cell useful-disagreement rate the
    user-study arm of P8 will populate. When the sweep ran in sequence
    mode, a second CSV (``<path>.sequence_positions.csv``) is written
    alongside with one row per (sequence, position, condition).

    The headline-graph CSV for the paper.
    """
    import csv
    fieldnames = (
        "task_id", "condition",
        "ood_mean", "ood_std", "ood_ci_lo", "ood_ci_hi", "n_seeds",
        "mean_corrections", "lib_graduated", "lib_total",
        "useful_disagreement_rate", "ud_n_trials",
        "p_vs_SlateGated",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for task_id, cell in sorted(report["tasks"].items()):
            for cond in _CONDITION_ORDER:
                if cond not in cell or "ood" not in cell[cond]:
                    continue
                ood = cell[cond]["ood"]
                p = cell.get("p_vs_SlateGated", {}).get(cond)
                ud = cell[cond].get("useful_disagreement")
                ud_mean = ud["mean"] if (ud and ud["n"] > 0) else None
                ud_n = ud["n"] if ud else 0
                w.writerow({
                    "task_id":          task_id,
                    "condition":        cond,
                    "ood_mean":         f"{ood['mean']:.4f}",
                    "ood_std":          f"{ood['std']:.4f}",
                    "ood_ci_lo":        f"{ood['ci_lo']:.4f}",
                    "ood_ci_hi":        f"{ood['ci_hi']:.4f}",
                    "n_seeds":          ood["n"],
                    "mean_corrections": f"{cell[cond]['corrections']['mean']:.2f}",
                    "lib_graduated":    f"{cell[cond]['lib_grad']['mean']:.2f}",
                    "lib_total":        f"{cell[cond]['lib_total']['mean']:.2f}",
                    "useful_disagreement_rate":
                                        f"{ud_mean:.4f}" if ud_mean is not None else "",
                    "ud_n_trials":      ud_n,
                    "p_vs_SlateGated":  f"{p:.4f}" if isinstance(p, float) else "",
                })

    # Sequence-positions companion CSV. Only written when the sweep ran in
    # sequence mode; one row per (sequence_id, position, condition). Lets
    # external plotters draw the OOD-vs-position curve per condition.
    sequences = report.get("sequences") or {}
    if not sequences:
        return
    seq_path = path.with_suffix(".sequence_positions.csv")
    seq_fields = (
        "sequence_id", "position", "task_id", "condition",
        "ood_mean", "ood_std", "ood_ci_lo", "ood_ci_hi",
        "lib_graduated", "lib_total", "n_seeds",
    )
    with seq_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seq_fields)
        w.writeheader()
        for sid in sorted(sequences):
            positions = sequences[sid].get("positions", {})
            for pos in sorted(positions):
                pe = positions[pos]
                for cond in _CONDITION_ORDER:
                    by_c = pe.get("by_condition", {}).get(cond)
                    if not by_c:
                        continue
                    ood = by_c["ood"]
                    w.writerow({
                        "sequence_id":   sid,
                        "position":      pos,
                        "task_id":       pe.get("task_id", ""),
                        "condition":     cond,
                        "ood_mean":      f"{ood['mean']:.4f}",
                        "ood_std":       f"{ood['std']:.4f}",
                        "ood_ci_lo":     f"{ood['ci_lo']:.4f}",
                        "ood_ci_hi":     f"{ood['ci_hi']:.4f}",
                        "lib_graduated": f"{by_c['lib_grad']['mean']:.2f}",
                        "lib_total":     f"{by_c['lib_total']['mean']:.2f}",
                        "n_seeds":       by_c["n"],
                    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-dir", type=Path, required=True,
                        help="Directory containing trial_*.json files.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write a summary CSV here.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full aggregated report as JSON here.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the table on stdout; useful when only writing CSV.")
    args = parser.parse_args()

    trials = load_trials(args.trial_dir)
    if not trials:
        print(f"No trial_*.json files found in {args.trial_dir}", file=sys.stderr)
        return 1

    report = aggregate(trials)
    if not args.quiet:
        print(render_text(report))
    if args.csv is not None:
        write_csv(report, args.csv)
        print(f"\nWrote CSV → {args.csv}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"Wrote JSON → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
