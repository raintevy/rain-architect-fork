#!/usr/bin/env python3
"""Read a context_profile JSONL and produce the §7 step-8 stress-test artifacts.

The CLI's ``--profile-context`` flag writes one JSON line per LLM call
inside ``_run_loop``. This script aggregates those lines into:

  * **Tokens-per-call curve** — per-turn input/output tokens (exact from
    response.usage when available; falls back to bytes-based estimate).
  * **Per-channel attribution** — share of bytes consumed by system /
    retrieval addendum / history / current turn, averaged across turns.
  * **Build-time savings curve** — cumulative bytes saved by clipping +
    stale-read dedupe, plotted against raw history growth. This is the
    sub-linear growth claim from P6.
  * **Lost-context-rate proxy** — fraction of read calls that got
    stale-marked. Reported per tool.
  * **Window-fill projection** — fraction of the configured model
    context window consumed, peak + curve. The rolling-summarisation
    trigger fires (in a future phase) at ~0.7 of the cap.

Usage:
    python -m scripts.eval.profile_session \\
        --jsonl skills/franka/context_profile_20260514_142022.jsonl

    # Bulk: aggregate every profile in a directory
    python -m scripts.eval.profile_session --dir skills/franka/

    # Export the per-turn rows as CSV for plotting
    python -m scripts.eval.profile_session --jsonl <path> --csv out.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_dir(directory: Path) -> dict[Path, list[dict]]:
    return {p: load_jsonl(p) for p in sorted(directory.glob("context_profile_*.jsonl"))}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _tokens_for_row(row: dict) -> int:
    """Prefer ground-truth ``input_tokens`` from response.usage; fall back
    to the bytes-based estimate when the SDK didn't return usage."""
    actual = row.get("input_tokens")
    if isinstance(actual, int):
        return actual
    est = row.get("tokens_est")
    return int(est) if isinstance(est, (int, float)) else 0


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Produce the §7 step-8 numbers from one profile."""
    if not rows:
        return {"n_turns": 0}

    tokens_per_call = [_tokens_for_row(r) for r in rows]
    bytes_per_call = [int(r.get("total_bytes") or 0) for r in rows]
    latencies = [float(r.get("latency_s") or 0.0) for r in rows]
    output_tokens = [
        int(r["output_tokens"]) for r in rows
        if isinstance(r.get("output_tokens"), int)
    ]

    # Per-channel (mean fraction of total_bytes).
    channels: dict[str, list[float]] = {
        "system":     [],
        "retrieval":  [],
        "history":    [],
        "current":    [],
    }
    for r in rows:
        total = max(1, int(r.get("total_bytes") or 0))
        channels["system"].append(int(r.get("system_bytes") or 0) / total)
        channels["retrieval"].append(int(r.get("retrieval_addendum_bytes") or 0) / total)
        channels["history"].append(int(r.get("history_bytes") or 0) / total)
        channels["current"].append(int(r.get("current_turn_bytes") or 0) / total)

    # Build-time savings cumulatives — the P6 sub-linear-growth claim.
    cum_clipped: list[int] = []
    cum_stale: list[int] = []
    running_clipped = running_stale = 0
    for r in rows:
        running_clipped += int(r.get("clipped_bytes_saved") or 0)
        running_stale += int(r.get("stale_bytes_saved") or 0)
        cum_clipped.append(running_clipped)
        cum_stale.append(running_stale)

    # Lost-context-rate proxy, per tool.
    stale_total = sum(int(r.get("stale_count") or 0) for r in rows)
    read_total: dict[str, int] = defaultdict(int)
    for r in rows:
        for name, n in (r.get("read_calls_by_tool") or {}).items():
            read_total[name] += int(n)
    read_total_sum = sum(read_total.values())
    lost_context_rate = (
        stale_total / read_total_sum if read_total_sum > 0 else 0.0
    )

    # Window fill: peak + final.
    fills = [float(r.get("window_fill_fraction") or 0.0) for r in rows]

    # Heuristic-vs-actual: compare tokens_est to input_tokens where both
    # are available. The ratio tells us how to trust the estimate when
    # the SDK didn't return usage (e.g. SDK-stub trials).
    matched = [
        (int(r["input_tokens"]), int(r.get("tokens_est") or 0))
        for r in rows
        if isinstance(r.get("input_tokens"), int)
    ]
    if matched:
        ratios = [est / max(1, actual) for actual, est in matched if actual > 0]
        heuristic_quality = {
            "n_matched":    len(matched),
            "ratio_mean":   _safe_mean(ratios),
            "ratio_min":    min(ratios) if ratios else 0.0,
            "ratio_max":    max(ratios) if ratios else 0.0,
        }
    else:
        heuristic_quality = None

    return {
        "n_turns":                    len(rows),
        "tokens_per_call":            tokens_per_call,
        "bytes_per_call":             bytes_per_call,
        "latencies_s":                latencies,
        "output_tokens":              output_tokens,
        "channels_mean_fraction":     {k: _safe_mean(v) for k, v in channels.items()},
        "cum_clipped_bytes":          cum_clipped,
        "cum_stale_bytes":            cum_stale,
        "stale_count_total":          stale_total,
        "read_count_total_by_tool":   dict(read_total),
        "lost_context_rate":          lost_context_rate,
        "window_fill_peak":           max(fills) if fills else 0.0,
        "window_fill_final":          fills[-1] if fills else 0.0,
        "tokens_mean":                _safe_mean([float(t) for t in tokens_per_call]),
        "tokens_peak":                max(tokens_per_call) if tokens_per_call else 0,
        "latency_mean_s":             _safe_mean(latencies),
        "heuristic_quality":          heuristic_quality,
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def render_text(summary: dict, *, label: str | None = None) -> str:
    lines: list[str] = []
    bar = "─" * 72
    if label:
        lines.append(f"\n{label}")
    lines.append(bar)
    n = summary.get("n_turns", 0)
    if n == 0:
        lines.append("(empty profile)")
        lines.append(bar)
        return "\n".join(lines)

    lines.append(f"turns:                  {n}")
    lines.append(f"tokens / call  mean:    {summary['tokens_mean']:,.0f}")
    lines.append(f"               peak:    {summary['tokens_peak']:,d}")
    lines.append(f"latency / call mean:    {summary['latency_mean_s']:.2f}s")
    lines.append(f"window fill    peak:    {summary['window_fill_peak']:.1%}")
    lines.append(f"               final:   {summary['window_fill_final']:.1%}")
    lines.append("")
    lines.append("per-channel byte share (mean across turns):")
    for ch, frac in sorted(summary["channels_mean_fraction"].items(),
                            key=lambda kv: -kv[1]):
        bar_width = int(frac * 40)
        lines.append(f"  {ch:<10s}  {frac:5.1%}  {'█' * bar_width}")
    lines.append("")
    lines.append(f"build-time savings cumulative:")
    lines.append(
        f"  clipped:    {summary['cum_clipped_bytes'][-1]:>12,d} bytes "
        f"(across {len(summary['cum_clipped_bytes'])} turns)"
    )
    lines.append(
        f"  stale:      {summary['cum_stale_bytes'][-1]:>12,d} bytes  "
        f"(across {summary['stale_count_total']:>3d} stale-marked reads)"
    )
    lines.append("")
    lines.append(
        f"lost-context-rate proxy: "
        f"{summary['lost_context_rate']:.1%} "
        f"(stale_count {summary['stale_count_total']} / "
        f"read_total {sum(summary['read_count_total_by_tool'].values())})"
    )
    if summary["read_count_total_by_tool"]:
        for tool, cnt in sorted(summary["read_count_total_by_tool"].items()):
            lines.append(f"  {tool:<24s} {cnt}")

    hq = summary.get("heuristic_quality")
    if hq is not None:
        lines.append("")
        lines.append(
            f"tokens_est / input_tokens   "
            f"mean {hq['ratio_mean']:.2f}  "
            f"range [{hq['ratio_min']:.2f}, {hq['ratio_max']:.2f}]  "
            f"(n={hq['n_matched']})"
        )

    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV (one row per turn)
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        return
    fieldnames = (
        "turn_index", "loop_iteration", "correction_text", "stop_reason",
        "latency_s", "total_bytes", "tokens_est", "input_tokens", "output_tokens",
        "window_fill_fraction",
        "system_bytes", "retrieval_addendum_bytes", "history_bytes",
        "current_turn_bytes",
        "clipped_count", "clipped_bytes_saved", "stale_count", "stale_bytes_saved",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path,
                     help="One profile JSONL (single-session view).")
    src.add_argument("--dir", type=Path,
                     help="Directory of profile JSONLs (one summary per file).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write per-turn CSV (only valid with --jsonl).")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the aggregated summary as JSON.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the text table on stdout.")
    args = parser.parse_args()

    aggregate_payload: dict[str, Any] = {}

    if args.jsonl is not None:
        rows = load_jsonl(args.jsonl)
        summary = summarise(rows)
        aggregate_payload[str(args.jsonl)] = summary
        if not args.quiet:
            print(render_text(summary, label=f"\nProfile: {args.jsonl}"))
        if args.csv is not None:
            write_csv(rows, args.csv)
            print(f"Wrote per-turn CSV → {args.csv}")
    else:
        files = load_dir(args.dir)
        if not files:
            print(f"No context_profile_*.jsonl files in {args.dir}", file=sys.stderr)
            return 1
        for path, rows in files.items():
            summary = summarise(rows)
            aggregate_payload[str(path)] = summary
            if not args.quiet:
                print(render_text(summary, label=f"\nProfile: {path.name}"))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(aggregate_payload, indent=2, default=str))
        print(f"Wrote JSON → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
