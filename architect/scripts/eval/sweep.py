#!/usr/bin/env python3
"""Sweep run_trial across (tasks × conditions × seeds) and emit a summary.

This is the §8 evaluation entry point. The default sweep matches the
paper's protocol — all tasks in ``tasks.json``, all three conditions
(ICL / FuncReuse / SlateGated / PatternReuse), and 10 seeds — but each axis can be narrowed via flag
for ablations or for the P8 real-robot run where each cell costs ~5 min
of robot time.

The sweep writes one ``trial_<task>__<cond>__seed<n>.json`` per cell
into ``--output-dir``, then immediately invokes
:mod:`scripts.eval.aggregate` so the operator sees the headline table
on stdout when the sweep finishes.

Sequential by default for determinism; ``--workers N`` runs trials in a
process pool when the LLM backend is ``stub`` (the ``claude`` backend
shouldn't be parallelised here — let the user manage their own rate
limits at the call site).

Usage:
    # Full P7 dry-run sweep
    python -m scripts.eval.sweep --output-dir /tmp/eval/p7

    # Single condition narrowed to one task
    python -m scripts.eval.sweep --conditions SlateGated \\
        --tasks pick_block_on_marker --seeds 1,2,3 \\
        --output-dir /tmp/eval/ablation

    # Real-LLM cell (one trial, useful for spot-checking before P8)
    python -m scripts.eval.sweep --llm claude --conditions SlateGated \\
        --tasks pick_block_on_marker --seeds 1 --output-dir /tmp/eval/real
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Same dotenv bootstrap as run_trial.py — sweep is an alternative entry point
# and needs the same credential surface.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.eval import run_trial as run_trial_mod
from scripts.eval.aggregate import aggregate, load_trials, render_text, write_csv


# ---------------------------------------------------------------------------
# Sweep matrix
# ---------------------------------------------------------------------------


def _all_task_ids(tasks: dict) -> list[str]:
    return [t["task_id"] for t in tasks["tasks"]]


def _all_conditions(tasks: dict) -> list[str]:
    return list(tasks["conditions"].keys())


def _has_sequences(tasks: dict) -> bool:
    return bool(tasks.get("sequences"))


def _all_sequence_ids(tasks: dict) -> list[str]:
    return [s["sequence_id"] for s in tasks.get("sequences", [])]


def _get_sequence(tasks: dict, sequence_id: str) -> dict:
    for s in tasks.get("sequences", []):
        if s["sequence_id"] == sequence_id:
            return s
    raise KeyError(
        f"Unknown sequence: {sequence_id!r}. Available: "
        f"{_all_sequence_ids(tasks)}"
    )


def _parse_csv_list(s: str) -> list[str]:
    return [item.strip() for item in s.split(",") if item.strip()]


def _parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


# ---------------------------------------------------------------------------
# One trial via the run_trial module (in-process)
# ---------------------------------------------------------------------------


def _run_one(args: dict) -> dict:
    """Worker entry for per-task mode — single trial in, single record out."""
    return run_trial_mod.run_trial(**args)


def _run_sequence_cell(args: dict) -> list[dict]:
    """Worker entry for sequence mode — run every position in one cell.

    A "cell" here is a single ``(sequence_id, condition, seed)`` tuple. The
    positions run sequentially in the same Python process so the workspace
    state (library.db rows, episodes.jsonl entries, scorers) accumulates
    across positions exactly the way it would across CLI sessions on the
    same robot. Returns one trial record per position; the caller prints
    one line per record.
    """
    sequence_id: str = args["sequence_id"]
    task_ids: list[str] = args["task_ids"]
    base_args = {k: v for k, v in args.items()
                 if k not in ("sequence_id", "task_ids")}
    records: list[dict] = []
    for position, task_id in enumerate(task_ids):
        trial_args = dict(base_args)
        trial_args["task_id"] = task_id
        trial_args["sequence_id"] = sequence_id
        trial_args["position_in_sequence"] = position
        records.append(run_trial_mod.run_trial(**trial_args))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", type=Path,
                        default=Path(__file__).parent / "tasks.json",
                        help="Path to tasks.json (default: scripts/eval/tasks.json)")
    parser.add_argument("--tasks", type=_parse_csv_list, default=None,
                        help="Comma-separated task_ids (per-task mode only; ignored "
                             "when tasks.json defines sequences).")
    parser.add_argument("--sequences", type=_parse_csv_list, default=None,
                        help="Comma-separated sequence_ids to narrow to "
                             "(sequence mode only; default: all sequences in tasks.json).")
    parser.add_argument("--per-task", action="store_true",
                        help="Force legacy per-task iteration even when tasks.json "
                             "defines sequences. Use for ablations that want to "
                             "isolate each task from cross-position library carryover.")
    parser.add_argument("--conditions", type=_parse_csv_list, default=None,
                        help="Comma-separated conditions "
                             "(ICL, FuncReuse, SlateGated, PatternReuse; "
                             "default: all).")
    parser.add_argument("--seeds", type=_parse_seeds, default=list(range(1, 11)),
                        help="Comma-separated seed list, ranges OK (default: 1-10).")
    parser.add_argument("--llm", default="stub", choices=("stub", "claude"))
    parser.add_argument("--robot", default="franka", choices=("franka",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool size for parallel trials. Must be 1 "
                             "when --llm=claude (don't parallelise paid API).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write the aggregate CSV here (defaults to "
                             "<output-dir>/summary.csv).")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip running aggregate.py after the sweep.")
    args = parser.parse_args()

    if args.workers > 1 and args.llm == "claude":
        parser.error("--workers > 1 is forbidden with --llm=claude")

    tasks = json.loads(args.tasks_file.read_text())
    conditions = args.conditions or _all_conditions(tasks)
    seeds = args.seeds

    # Decide between sequence mode (default when tasks.json has sequences) and
    # per-task mode (legacy fallback). Sequence mode runs positions within a
    # (sequence, condition, seed) cell sequentially in the same process so the
    # workspace state — library.db, episodes.jsonl, scorers — accumulates
    # across positions, exactly the way it would across CLI sessions on the
    # same robot. That accumulation is what makes the C3 "library-compounds-
    # across-sessions" claim measurable.
    sequence_mode = _has_sequences(tasks) and not args.per_task
    if sequence_mode:
        seq_ids = args.sequences or _all_sequence_ids(tasks)
        cells = [(sid, cond, seed)
                 for sid in seq_ids for cond in conditions for seed in seeds]
        trial_args = [
            {
                "sequence_id": sid,
                "task_ids":    _get_sequence(tasks, sid)["task_ids"],
                "condition":   cond,
                "seed":        seed,
                "tasks_path":  args.tasks_file,
                "output_dir":  args.output_dir,
                "llm":         args.llm,
                "robot_name":  args.robot,
            }
            for sid, cond, seed in cells
        ]
        n_positions = sum(
            len(_get_sequence(tasks, sid)["task_ids"]) for sid in seq_ids
        )
        n_trials = n_positions * len(conditions) * len(seeds)
        print(f"Sweep: {len(seq_ids)} sequences × {len(conditions)} conditions × "
              f"{len(seeds)} seeds = {len(cells)} cells "
              f"({n_trials} trials across positions, llm={args.llm}, "
              f"workers={args.workers})")
        runner = _run_sequence_cell
    else:
        task_ids = args.tasks or _all_task_ids(tasks)
        cells = [(task, cond, seed)
                 for task in task_ids for cond in conditions for seed in seeds]
        trial_args = [
            {
                "task_id":    task,
                "condition":  cond,
                "seed":       seed,
                "tasks_path": args.tasks_file,
                "output_dir": args.output_dir,
                "llm":        args.llm,
                "robot_name": args.robot,
            }
            for task, cond, seed in cells
        ]
        print(f"Sweep: {len(task_ids)} tasks × {len(conditions)} conditions × "
              f"{len(seeds)} seeds = {len(cells)} trials  "
              f"(llm={args.llm}, workers={args.workers})")
        runner = _run_one

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_failed = 0
    if args.workers <= 1:
        for ta in trial_args:
            cell_label = (
                f"{ta.get('sequence_id', '')}/{ta.get('task_ids', [''])[0]}…"
                if sequence_mode else ta["task_id"]
            )
            try:
                result = runner(ta)
            except Exception as exc:
                print(f"  FAIL {cell_label:<32s} {ta['condition']:<6s} "
                      f"seed={ta['seed']}  {type(exc).__name__}: {exc}")
                n_failed += 1
                continue
            # In sequence mode runner returns list[dict]; in per-task it's a
            # single dict. Normalise so the print + gc paths handle both.
            records = result if isinstance(result, list) else [result]
            for rec in records:
                _print_trial_line(rec)
            # Force a collection pass between trials. Trial-local objects
            # (SkillStore probe reports, slate Candidate lists, large
            # program strings, response objects) are no longer referenced
            # after the record is printed, but cyclic references can
            # delay reclamation. The v3 sweep was OOM-killed around trial
            # 51; explicit gc + the llm_client / embeddings cache fixes
            # together keep memory flat across long sweeps.
            del result, records
            gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(runner, ta): ta for ta in trial_args}
            for fut in as_completed(futs):
                ta = futs[fut]
                cell_label = (
                    f"{ta.get('sequence_id', '')}"
                    if sequence_mode else ta["task_id"]
                )
                try:
                    result = fut.result()
                except Exception as exc:
                    print(f"  FAIL {cell_label:<32s} {ta['condition']:<6s} "
                          f"seed={ta['seed']}  {type(exc).__name__}: {exc}")
                    n_failed += 1
                    continue
                records = result if isinstance(result, list) else [result]
                for rec in records:
                    _print_trial_line(rec)

    dt = time.time() - t0
    unit = "cells" if sequence_mode else "trials"
    print(f"\nFinished {len(cells)} {unit} in {dt:.1f}s "
          f"({n_failed} failed).")

    if args.no_aggregate:
        return 0

    trials = load_trials(args.output_dir)
    if not trials:
        print("No trial JSONs found — skipping aggregate.")
        return 1
    report = aggregate(trials)
    print("\n" + render_text(report))
    csv_path = args.csv if args.csv else (args.output_dir / "summary.csv")
    write_csv(report, csv_path)
    print(f"\nWrote CSV → {csv_path}")
    return 0


def _print_trial_line(rec: dict) -> None:
    ud = rec["useful_disagreement_rate"]
    ud_str = f"{ud:.0%}" if ud is not None else "n/a"
    # Prefix the line with the sequence position when present so a cell's
    # positions are visually grouped in the sweep output.
    pos = rec.get("position_in_sequence")
    sid = rec.get("sequence_id") or ""
    if pos is not None and sid:
        label = f"{sid}/p{pos}/{rec['task_id']}"
    else:
        label = rec["task_id"]
    print(
        f"  {label:<44s} {rec['condition']:<6s} seed={rec['seed']:<3d} "
        f"OOD={rec['ood_pass_rate']:.0%}  "
        f"lib={rec['library']['graduated']}/{rec['library']['total']}  "
        f"UD={ud_str}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
