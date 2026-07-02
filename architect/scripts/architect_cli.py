#!/usr/bin/env python3
"""ARCHITECT interactive CLI — agentic program synthesis, execution, and correction.

Claude runs in a tool-use loop: it introspects the robot API, queries robot/scene
state, registers new primitives (with approval), and submits the final program.

Usage:
    python scripts/architect_cli.py --instruction "Pick up the red cup"
    python scripts/architect_cli.py --program output/program.py   # skip generation
    python scripts/architect_cli.py --instruction "..." --dry-run  # no live robot needed
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ANTHROPIC_MODEL
from architect.agent.session import AgenticSession
from architect.ui.diff_ui import prompt_pick
from architect.library.episode_log import EpisodeLog
from architect.agent.primitive_registry import PrimitiveRegistry
from architect.robots import get_robot_config
from architect.library.skill_store import SkillStore
from architect.agent.synthesis import execute_with_synthesis
from architect.midexec import wrap_vqa_for_midexec, unwrap_vqa_for_midexec

console = Console()

_HELP = (
    "[dim]Type a description of a change to apply it as a correction, "
    "or use a command:[/dim]  "
    "[cyan]undo[/cyan]  "
    "[cyan]edit[/cyan]  "
    "[cyan]show[/cyan]  "
    "[cyan]reset[/cyan]  "
    "[cyan]/library[/cyan]  "
    "[cyan]/graduate <name>[/cyan]  "
    "[cyan]/forget <name>[/cyan]  "
    "[cyan]done[/cyan]"
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_program(program: str, title: str = "Program") -> None:
    syntax = Syntax(program, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"[bold]{title}[/bold]", border_style="blue"))


def show_diff(old: str, new: str) -> None:
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="before",
        tofile="after",
        lineterm="",
    ))
    if not diff:
        console.print("[dim]  (no changes)[/dim]")
        return

    text = Text()
    for line in diff:
        if line.startswith(("---", "+++")):
            text.append(line + "\n", style="bold")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        else:
            text.append(line + "\n", style="dim")

    console.print(Panel(text, title="[bold]Changes[/bold]", border_style="yellow"))


def _add_curobo_validation(namespace: dict, dry_run: bool) -> None:
    """Wrap move_ee_to_pose with pre-execution CuRobo feasibility validation.

    In dry-run mode validation is skipped (always returns feasible=True).

    In live mode this is currently a stub.
    TODO: implement validate_motion_plan() by adding a validate_only flag to
    the CuRobo WebSocket motion-planning service. The service currently plans
    and executes in one step; a validate_only=True JSON request field should
    plan but skip the articulation controller loop.
    Expected JSON request: {"target_pose": [x,y,z,qw,qx,qy,qz], "validate_only": true}
    Expected response on infeasible: {"status": "error", "reason": "..."}
    """

    def validate_motion_plan(target_pose: dict) -> dict:
        if dry_run:
            return {"feasible": True, "message": "dry-run: validation skipped"}
        # TODO: send validate_only request to CuRobo WebSocket service
        return {"feasible": True, "message": "live: CuRobo validation not yet implemented"}

    namespace["validate_motion_plan"] = validate_motion_plan

    _orig_move_ee = namespace["move_ee_to_pose"]

    def move_ee_with_validation(target_pose: dict):
        result = validate_motion_plan(target_pose)
        if not result["feasible"]:
            raise RuntimeError(
                f"CuRobo pre-execution check failed: {result.get('message', '')}"
            )
        if not dry_run and "not yet implemented" not in result.get("message", ""):
            console.print(f"  [green]✓ CuRobo: motion feasible[/green]")
        return _orig_move_ee(target_pose)

    namespace["move_ee_to_pose"] = move_ee_with_validation


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _make_approve_primitive(auto_approve: bool):
    """Return an approve_primitive callable. ``auto_approve=True`` short-
    circuits the interactive prompt for benchmark-mode sweeps (P7/P8).

    The agentic graduation gate (P5) still runs on the auto-approved
    primitive, so a buggy benchmark-mode generation gets caught by the
    probe pass-rate threshold rather than slipping silently into the
    library as graduated.
    """
    if auto_approve:
        def _auto(func_name: str, func_code: str) -> tuple[bool, str]:
            return True, func_code
        return _auto
    return _approve_primitive


def _approve_primitive(func_name: str, func_code: str) -> tuple[bool, str]:
    """Show a synthesized/proposed primitive and ask for approval.

    Returns (approved, final_code).
    """
    current = func_code
    while True:
        try:
            choice = input("\nRegister this primitive? [y/n/e(dit)] › ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False, current

        if choice in ("y", "yes", ""):
            return True, current
        elif choice in ("n", "no"):
            return False, current
        elif choice in ("e", "edit"):
            edited = _open_in_editor(current)
            if edited.strip() != current.strip():
                show_diff(current, edited)
                current = edited
                syntax = Syntax(current, "python", theme="monokai", line_numbers=True)
                console.print(Panel(
                    syntax,
                    title=f"[bold yellow]Edited primitive:[/bold yellow] [cyan]{func_name}()[/cyan]",
                    border_style="yellow",
                ))
            else:
                console.print("[dim]  (no changes)[/dim]")
        else:
            console.print("[dim]  y = register   n = cancel   e = open in $EDITOR[/dim]")


def _run(
    program: str,
    namespace: dict,
    model: str,
    registry: PrimitiveRegistry,
    *,
    api_spec: str,
    robot_label: str,
    session=None,
    vlm_tracing: bool = True,
    instruction: str | None = None,
    robot_name: str = "franka",
):
    """Execute a program; catch undefined names and synthesize them on demand.

    Returns ``(ok, trace, interrupt)`` — the exec-success bool, the populated
    :class:`architect.execution_trace.ExecutionTrace` recorded as the program
    ran, and an optional :class:`~architect.midexec.MidExecInterrupt` (``None``
    when execution completed normally). When ``session`` is supplied, the
    trace is also stored on it via
    :meth:`AgenticSession.set_last_execution_trace` so the next
    correction's prompt includes ``## Last Execution Trace``.

    ``api_spec`` and ``robot_label`` are forwarded to the synthesis prompt so
    helpers generated for missing names are constrained to the current
    robot's primitives.
    """
    registry.inject_into_namespace(namespace)

    def _approve_synthesis(func_name: str, func_code: str) -> tuple[bool, str]:
        console.print()
        syntax = Syntax(func_code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(
            syntax,
            title=f"[bold yellow]Synthesized helper:[/bold yellow] [cyan]{func_name}()[/cyan]",
            border_style="yellow",
        ))
        return _approve_primitive(func_name, func_code)

    ok, trace, interrupt = execute_with_synthesis(
        program, namespace, model, console, _approve_synthesis,
        api_spec=api_spec, robot_label=robot_label,
        vlm_enabled=vlm_tracing,
    )
    if session is not None:
        session.set_last_execution_trace(trace)
        # Reset the score every run; only update when scoring actually
        # produces a value. Without the reset, a previous PASS would
        # linger if the current execution crashed or wasn't scoreable.
        session.set_last_task_score(None)
        # Post-execution task scoring (optional). When the session was
        # constructed with a task_spec (via --task), run the trace
        # through the same task_complete predicate the eval harness
        # uses and print a PASS / FAIL line. This is a *measurement*
        # of whether the program met the task, not a feedback signal —
        # the correction loop still belongs to the user.
        spec = session.task_spec
        if spec is not None and trace is not None and ok:
            from architect.eval.task_scorers import score_trace
            try:
                score = score_trace(trace, spec)
            except Exception as exc:
                console.print(
                    f"[dim]\\[task-scorer] error: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
            else:
                if score is not None:
                    session.set_last_task_score(score)
                    label = spec.get("name", "task scorer")
                    is_pass = score >= 0.5
                    if is_pass:
                        console.print(
                            f"\n[bold green]✓ {label}: PASS[/bold green] "
                            f"[dim](score {score:.2f})[/dim]"
                        )
                    else:
                        console.print(
                            f"\n[bold red]✗ {label}: FAIL[/bold red] "
                            f"[dim](score {score:.2f})[/dim]"
                        )

                    # Failure store outcome bookkeeping. We mark a prior
                    # failure's outcome based on the *current* execution's
                    # verdict: PASS after FAIL → 'fixed', FAIL after FAIL
                    # → 'still_failed'. The prior failure_id was stashed
                    # on the session at the last FAIL; clear it once
                    # marked so we don't double-mark.
                    prior_fid = session.last_failure_id
                    if prior_fid is not None and session.failure_store is not None:
                        try:
                            session.failure_store.mark_outcome(
                                prior_fid, "fixed" if is_pass else "still_failed"
                            )
                        except Exception as exc:
                            console.print(
                                f"[dim]\\[failure-store] mark_outcome failed: "
                                f"{type(exc).__name__}: {exc}[/dim]"
                            )
                        session.set_last_failure_id(None)

                    # On PASS, clear any stale diagnosis. Without this,
                    # a stale FailureDiagnosis from the previous FAIL
                    # could leak into the next session-level read.
                    if is_pass:
                        session.set_last_diagnosis(None)

                    # On FAIL, fire the diagnosis call + record + lookup.
                    if not is_pass and session.failure_diagnosis_enabled:
                        from architect.corrections.failure_diagnosis import diagnose_failure
                        judge_rationale = ""
                        # vqa_judge stashes its rationale on the per-trace
                        # cache; lift it out if available so the diagnosis
                        # call has the judge's read on what failed.
                        if spec.get("name") == "vqa_judge":
                            try:
                                from architect.eval.task_scorers import (
                                    _vqa_judge_evaluate, trace_to_call_log,
                                )
                                judge_rationale = _vqa_judge_evaluate(
                                    spec["params"].get("instruction", ""),
                                    spec["params"].get("vqa_guidelines", ""),
                                    trace_to_call_log(trace),
                                ).get("rationale", "")
                            except Exception:
                                judge_rationale = ""

                        diagnosis = diagnose_failure(
                            trace=trace,
                            program=program,
                            instruction=instruction or "",
                            judge_rationale=judge_rationale,
                        )
                        session.set_last_diagnosis(diagnosis)

                        console.print()
                        console.print(Panel(
                            diagnosis.to_markdown(),
                            title="[bold red]Failure diagnosis[/bold red]",
                            border_style="red",
                        ))

                        # Lookup similar past failures BEFORE recording the
                        # new one, so the new failure can't be its own
                        # neighbour. Render top-3 if any.
                        if session.failure_store is not None:
                            try:
                                hits = session.failure_store.lookup_similar(
                                    instruction or "",
                                    diagnosis.failure_signature,
                                    diagnosis.root_cause,
                                    k=3,
                                    only_fixed=True,
                                )
                            except Exception:
                                hits = []
                            if hits:
                                lines = ["**Similar past failures that were fixed:**", ""]
                                for h in hits:
                                    lines.append(h.to_markdown())
                                console.print(Panel(
                                    "\n".join(lines),
                                    title="[bold yellow]Cross-session failure recall[/bold yellow]",
                                    border_style="yellow",
                                ))

                            try:
                                fid = session.failure_store.record_failure(
                                    robot=robot_name,
                                    instruction=instruction or "",
                                    diagnosis=diagnosis,
                                    judge_rationale=judge_rationale,
                                    program_before=program,
                                )
                                session.set_last_failure_id(fid)
                            except Exception as exc:
                                console.print(
                                    f"[dim]\\[failure-store] record_failure failed: "
                                    f"{type(exc).__name__}: {exc}[/dim]"
                                )
    return ok, trace, interrupt


# ---------------------------------------------------------------------------
# Editor helper
# ---------------------------------------------------------------------------

def _open_in_editor(program: str) -> str:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, prefix="architect_") as f:
        f.write(program)
        tmp_path = f.name
    subprocess.run([editor, tmp_path])
    edited = Path(tmp_path).read_text()
    Path(tmp_path).unlink(missing_ok=True)
    return edited


# ---------------------------------------------------------------------------
# Approval prompt
# ---------------------------------------------------------------------------

def approval_prompt(
    program: str,
    namespace: dict,
    model: str,
    registry: PrimitiveRegistry,
    *,
    api_spec: str,
    robot_label: str,
    session=None,
    vlm_tracing: bool = True,
    instruction: str | None = None,
    robot_name: str = "franka",
) -> tuple:
    """Show program, ask y/n/e, execute if approved.

    Returns (was_executed, final_program, interrupt_or_none). When ``session`` is supplied,
    the execution trace is stashed on it via :meth:`AgenticSession.
    set_last_execution_trace` so the next correction's prompt receives
    a ``## Last Execution Trace`` section.

    ``vlm_tracing`` is forwarded to :func:`_run`; set False to disable
    the per-call Layer B VQA observations during execution.
    """
    current = program
    while True:
        try:
            choice = input("\nExecute? [y/n/e(dit)] › ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False, current, None

        if choice in ("y", "yes", ""):
            console.print("\n[bold]Executing...[/bold]")
            success, _trace, _interrupt = _run(
                current, namespace, model, registry,
                api_spec=api_spec, robot_label=robot_label,
                session=session,
                vlm_tracing=vlm_tracing,
                instruction=instruction,
                robot_name=robot_name,
            )
            if success:
                console.print("[green]✓ Execution complete.[/green]")
            return success, current, _interrupt

        elif choice in ("n", "no"):
            return False, current, None

        elif choice in ("e", "edit"):
            edited = _open_in_editor(current)
            if edited.strip() != current.strip():
                show_diff(current, edited)
                current = edited
                show_program(current, title="Edited Program")
            else:
                console.print("[dim]  (no changes)[/dim]")

        else:
            console.print("[dim]  y = execute   n = skip   e = open in $EDITOR[/dim]")


def _correct_with_slate(
    session,
    correction: str,
    current_ee_pose,
    *,
    n: int,
    k_probes: int,
    robot_name: str,
    episode_log: EpisodeLog,
    console,
):
    """One slate-mode correction turn: sample, rank, surface, log.

    Returns the picked candidate's program (str) or ``None`` if the user
    cancelled or the LLM produced nothing usable. The episode log is
    appended whether the slate was auto-applied or surfaced; cancelled
    slates are also recorded with ``picked_index=None`` so the
    useful-disagreement-rate computation has a denominator that reflects
    *all* surfaced slates.
    """
    console.print(
        f"[bold]Sampling slate (n={n}, probes={k_probes})...[/bold]"
    )
    slate = session.correct_with_slate(
        correction,
        n=n, current_ee_pose=current_ee_pose,
        robot_name=robot_name, k_probes=k_probes,
    )

    if not slate.candidates:
        console.print(
            "[yellow]Slate empty (LLM produced no parseable candidates). "
            "Falling back to single-shot correction.[/yellow]"
        )
        return session.correct(correction, current_ee_pose=current_ee_pose)

    if slate.dominant:
        top = slate.candidates[0]
        console.print(
            f"[green]Auto-applied:[/green] [cyan]{top.axis_label}[/cyan] "
            f"(score {top.score:.2f})"
        )
        episode_log.record_slate(slate, picked_index=0, auto_applied=True)
        session.apply_slate_pick(top, correction=correction)
        return top.code

    pick = prompt_pick(
        slate, console, current_program=session.current_program,
    )
    if pick.index is None or pick.picked is None:
        console.print("[dim]Slate cancelled.[/dim]")
        episode_log.record_slate(slate, picked_index=None)
        return None

    episode_log.record_slate(
        slate, picked_index=pick.index,
        rejected_axes=pick.rejected_axes,
    )
    session.apply_slate_pick(pick.picked, correction=correction)
    return pick.picked.code


# ---------------------------------------------------------------------------
# /library — /graduate <name> — /forget <name>  (P5 / C2)
# ---------------------------------------------------------------------------


def _print_library(store, console) -> None:
    """Render the per-robot SkillStore as a status table.

    Columns: name, status, ✓ / ✗, last probe pass-rate (if any). Empty
    libraries print a hint pointing at the slate / write_primitive flows
    that populate the library so users new to the codebase know how to
    grow it.
    """
    from rich.table import Table

    skills = store.get_all()
    if not skills:
        console.print(
            "[dim]Library is empty. Approve primitives via the agentic "
            "tool-use loop (or --slate mode) to populate it.[/dim]"
        )
        return

    table = Table(title="Skill library", title_style="bold", border_style="cyan")
    table.add_column("name", style="cyan")
    table.add_column("status")
    table.add_column("✓", justify="right")
    table.add_column("✗", justify="right")
    table.add_column("last probe", justify="right")
    table.add_column("docstring", style="dim")

    status_style = {
        "graduated": "green",
        "candidate": "yellow",
        "archived":  "dim",
    }
    for s in skills:
        st = s["graduation_status"]
        sty = status_style.get(st, "white")
        last_probe = "—"
        if s.get("probe_suite_id"):
            suite = store.get_probe_suite(s["probe_suite_id"])
            if suite:
                last_probe = f"{suite['pass_rate']:.0%} ({suite['n_passed']}/{suite['n_total']})"
        table.add_row(
            s["name"], f"[{sty}]{st}[/{sty}]",
            str(s["success_count"]), str(s["failure_count"]),
            last_probe,
            (s["docstring"] or "")[:60],
        )
    console.print(table)


def _graduate(name: str, store, console) -> None:
    """Force-promote a candidate skill to graduated, bypassing the gate.

    Used when the user disagrees with the gate's verdict — e.g. when the
    probe's synthesized stub call doesn't exercise the primitive's intent.
    The skill keeps its existing ``probe_suite_id`` (if any) so audit
    history is preserved.
    """
    row = store.get(name)
    if row is None:
        console.print(f"[red]No skill named {name!r} in the library.[/red]")
        return
    if row["graduation_status"] == "graduated":
        console.print(f"[dim]{name} is already graduated.[/dim]")
        return
    store.admit(name)
    console.print(f"[green]✓ {name} force-graduated (was {row['graduation_status']}).[/green]")


def _forget(name: str, store, namespace: dict, console) -> None:
    """Hard-delete a skill from the library and remove it from the namespace.

    The namespace removal is best-effort — if the user already overwrote
    the name with something else, we leave that alone rather than risk
    breaking the current session.
    """
    row = store.get(name)
    if row is None:
        console.print(f"[red]No skill named {name!r} in the library.[/red]")
        return
    store.forget(name)
    if name in namespace:
        try:
            del namespace[name]
        except Exception:
            pass
    console.print(f"[yellow]🗑  forgot {name} (was {row['graduation_status']}).[/yellow]")


# ---------------------------------------------------------------------------
# Condition flag bundles
# ---------------------------------------------------------------------------


# Maps the four eval conditions (see scripts/eval/tasks.json's `conditions`
# block) to the CLI-level toggles each implies. Used by --condition to wire
# the right SkillStore / slate / AgenticSession kwargs for a real-robot
# ablation run.
#
# - library_persistent: True → path-backed SkillStore on skills/<robot>/
#   library.db (helpers survive across sessions). False → ":memory:"
#   SkillStore (helpers discarded on session end).
# - use_slate: forwards to args.slate behavior — slate-based corrections
#   if True, single-shot if False.
# - agentic_kwargs: keyword args spread into AgenticSession() to control
#   the auxiliary emission paths (scorers, subtask docs, fault summary,
#   in-program retry pattern).
#
# IMPORTANT alignment gap: the CLI uses the agentic tool-use loop for
# corrections, while the dry-run harness uses single-shot prompts. The
# CLI's SlateGated does NOT invoke the graduation gate — that's an
# unfilled gap (task #61). For real-robot ablations, SlateGated here
# means "slate + library + retrieval, but no graduation-of-helpers."
_CONDITION_FLAGS = {
    "ICL": {
        "library_persistent": False,
        "use_slate": False,
        "agentic_kwargs": {
            "emit_scorers": False,
            "emit_subtask_docs": False,
            "emit_fault_summary": False,
            "inprogram_retry": False,
        },
    },
    "FuncReuse": {
        "library_persistent": True,
        "use_slate": False,
        "agentic_kwargs": {
            "emit_scorers": False,
            "emit_subtask_docs": False,
            "emit_fault_summary": False,
            "inprogram_retry": False,
        },
    },
    "SlateGated": {
        "library_persistent": True,
        "use_slate": True,
        "agentic_kwargs": {
            "emit_scorers": True,
            "emit_subtask_docs": True,
            "emit_fault_summary": True,
            "inprogram_retry": False,
        },
    },
    "PatternReuse": {
        "library_persistent": False,
        "use_slate": False,
        "agentic_kwargs": {
            "emit_scorers": False,
            "emit_subtask_docs": True,
            "emit_fault_summary": True,
            "inprogram_retry": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instruction", "-n", default=None,
                        help="Natural-language task instruction.")
    parser.add_argument("--program", "-p", default=None,
                        help="Path to a pre-generated .py file (skips generation).")
    parser.add_argument("--save", "-s", default=None,
                        help="Save the final program to this path on exit.")
    parser.add_argument("--model", "-m", default=ANTHROPIC_MODEL,
                        help=f"Model to use (default: {ANTHROPIC_MODEL}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub robot API calls — no live robot / ROS required.")
    parser.add_argument("--robot", "-r", default="franka",
                        choices=["franka"],
                        help="Robot platform to use (default: franka).")
    parser.add_argument("--slate", action="store_true",
                        help=(
                            "Active counterfactual sampling (P4 / C1): each "
                            "free-text correction is answered with N candidate "
                            "programs ranked by probe + scorer signal, with the "
                            "user's pick logged as multi-axis supervision."
                        ))
    parser.add_argument("--slate-n", type=int, default=3,
                        help="Number of candidates per slate (default: 3).")
    parser.add_argument("--slate-probes", type=int, default=8,
                        help="Probes per candidate (default: 8).")
    parser.add_argument("--auto-approve-primitives", action="store_true",
                        help=(
                            "Benchmark mode: auto-approve every write_primitive "
                            "and run_ros2_command without prompting. The P5 "
                            "graduation gate still runs — bad primitives stay "
                            "candidate. Use only for eval sweeps (P7/P8) where "
                            "user-input latency would dominate the metric."
                        ))
    parser.add_argument("--profile-context", action="store_true",
                        help=(
                            "Log per-turn context-fill metrics to "
                            "skills/<robot>/context_profile_<ts>.jsonl. Captures "
                            "tokens-per-call (exact, from response.usage), "
                            "per-channel byte attribution, clipped/stale "
                            "savings, and wall-clock latency. Read with "
                            "scripts/eval/profile_session.py."
                        ))
    parser.add_argument("--no-vlm-tracing", dest="vlm_tracing",
                        action="store_false", default=True,
                        help=(
                            "Disable the trace wrapper's per-call Layer B "
                            "VLM observations — the contextual VQA queries "
                            "fired after every traced control/perception "
                            "primitive (\"did the gripper grasp?\", \"where "
                            "is the gripper now?\", etc.). Useful when VQA "
                            "is unreliable or expensive: state snapshots "
                            "and call-args still land in the trace, just "
                            "without the per-call VLM round-trip. Does NOT "
                            "affect VQA calls the generated program itself "
                            "makes — those are part of the program logic, "
                            "controlled by --condition."
                        ))
    parser.add_argument("--no-fault-summary", dest="fault_summary",
                        action="store_false", default=None,
                        help=(
                            "Disable the foreground LLM call that "
                            "renders a three-line fault-localisation "
                            "summary above the trace in correction "
                            "prompts. Independent of --condition: lets "
                            "you ablate the fault-summary channel "
                            "while keeping subtask-docs / in-program "
                            "retry / etc. from PatternReuse or "
                            "SlateGated. Default (no flag) inherits "
                            "the --condition setting, or True when "
                            "no --condition is given."
                        ))
    parser.add_argument("--no-trace-injection", dest="trace_injection",
                        action="store_false", default=None,
                        help=(
                            "Disable the raw `## Last Execution Trace` "
                            "section that gets injected into correction "
                            "prompts. With this flag, the agent sees "
                            "only the user's correction text + the "
                            "current program — not the call-by-call "
                            "record of what happened last run. Trace "
                            "is still captured (so vqa_judge / "
                            "explain_failure still score on it) — only "
                            "the prompt injection is suppressed. Use "
                            "for clean ablations measuring correction "
                            "quality without prior-run context leaking "
                            "in. Combine with --no-fault-summary and "
                            "--no-vlm-tracing for a fully execution-"
                            "blind correction baseline."
                        ))
    parser.add_argument("--closed-loop", action="store_true", default=False,
                        help=(
                            "Closed-loop feedback: when an execution "
                            "returns FAIL, automatically issue a "
                            "correction (instead of prompting the user) "
                            "using a structured diagnostic of what "
                            "failed. The refined program is auto-"
                            "approved and re-executed. Loop exits on "
                            "PASS or after --max-auto-corrections "
                            "rounds, at which point the user is "
                            "prompted as today. Source of the PASS/FAIL "
                            "signal: --task uses its tasks.json "
                            "success_scorer (keyword-based), --"
                            "instruction-only mode auto-derives a "
                            "vqa_judge scorer (LLM interprets the "
                            "program's VQA answers against the free-"
                            "form instruction; no keywords required)."
                        ))
    parser.add_argument("--max-auto-corrections", type=int, default=2,
                        help=(
                            "Cap on closed-loop auto-correction rounds. "
                            "After this many automatic refinements without "
                            "reaching PASS, the CLI falls back to manual "
                            "user-correction prompts. Default 2."
                        ))
    parser.add_argument("--hil", action="store_true", default=False,
                        help=(
                            "Human-in-the-loop mid-execution intervention. "
                            "When a VQA verification retry exhausts (two "
                            "consecutive 'no' answers), execution pauses and "
                            "the user is prompted for a language correction. "
                            "A continuation program is generated from the "
                            "current robot state. Implies PatternReuse-like "
                            "settings (in-program retry, subtask docs, fault "
                            "summary)."
                        ))
    parser.add_argument("--vlm", action="store_true", default=False,
                        help=(
                            "VLM-based mid-execution intervention. Same "
                            "interruption mechanism as --hil, but the "
                            "correction is auto-generated from the partial "
                            "execution trace instead of prompting the user. "
                            "Single-variable comparison with --hil: identical "
                            "infrastructure, only the correction source "
                            "differs (auto vs human)."
                        ))
    parser.add_argument("--failure-store", action="store_true", default=False,
                        help=(
                            "Persist structured failure diagnoses to "
                            "skills/<robot>/failures.db. On each FAIL, "
                            "the failure (category, signature, root "
                            "cause, evidence) is recorded; on the "
                            "next FAIL, similar past failures are "
                            "retrieved by embedding similarity and "
                            "shown to the operator (and/or used as "
                            "additional context in closed-loop). "
                            "Outcome of each correction is tracked: a "
                            "PASS marks the prior failure as 'fixed', "
                            "another FAIL marks it 'still_failed'. "
                            "Off by default; opt in for cross-session "
                            "learning + the paper's failure-mode log. "
                            "Implies --failure-diagnosis (the diagnosis "
                            "is what gets stored)."
                        ))
    parser.add_argument("--retrieval", default="static",
                        choices=("static", "hybrid"),
                        help=(
                            "How the system prompt's Domain Knowledge "
                            "section is assembled. 'static' (default): "
                            "every skill markdown doc under "
                            "skills/<robot>/ is included — current "
                            "behaviour. 'hybrid': always-loaded docs "
                            "(tagged in their frontmatter — typically "
                            "vqa.md, motion.md, verification.md, "
                            "ros1.md) plus the top-3 task-specific "
                            "docs retrieved by embedding cosine "
                            "against the instruction. Also exposes a "
                            "query_skills tool the agent can call "
                            "mid-loop to fetch additional docs. "
                            "Reduces in-prompt context size for "
                            "irrelevant docs but adds an embedding "
                            "round-trip at session start (cached by "
                            "mtime in skills/<robot>/skill_doc_index.json)."
                        ))
    parser.add_argument("--retrieval-k", type=int, default=3,
                        help=(
                            "k for --retrieval hybrid (default 3). "
                            "How many task-specific docs to retrieve "
                            "in addition to the always-loaded ones."
                        ))
    parser.add_argument("--scratchpad", action="store_true", default=False,
                        help=(
                            "Enable the agent-driven session scratchpad — "
                            "an explicit working-memory layer the agent "
                            "curates via the update_scratchpad tool "
                            "(current focus, working hypotheses, "
                            "invalidated hypotheses, key observations). "
                            "Rendered at the top of correction prompts "
                            "above the fault summary and trace. Off by "
                            "default; opt in for long multi-correction "
                            "sessions where hypothesis tracking across "
                            "turns matters more than per-turn token "
                            "cost. The scratchpad is in-session only — "
                            "not persisted across sessions."
                        ))
    parser.add_argument("--no-failure-diagnosis", dest="failure_diagnosis",
                        action="store_false", default=True,
                        help=(
                            "Disable the foreground LLM call that "
                            "produces a structured failure diagnosis "
                            "on FAIL. Without this flag, every FAIL "
                            "fires the diagnosis call (~1 extra LLM "
                            "round-trip) and renders the result above "
                            "the ARCHITECT › prompt for manual corrections / "
                            "uses it as auto-correction body for "
                            "closed-loop. With --no-failure-diagnosis, "
                            "the CLI falls back to the older keyword-"
                            "based explain_failure text. The "
                            "--failure-store flag forces diagnosis on "
                            "since there's nothing meaningful to "
                            "persist without it."
                        ))
    parser.add_argument("--task", "-t", default=None,
                        help=(
                            "Task id from scripts/eval/tasks.json. When set, "
                            "the task's instruction auto-fills --instruction "
                            "(unless --instruction is also given) AND the task's "
                            "success_scorer spec is attached to the AgenticSession "
                            "so each execution gets a PASS / FAIL line printed "
                            "after it completes. On PASS the correction loop "
                            "auto-exits. Available ids: pick_block_on_marker, "
                            "stack_two_blocks, stack_three_blocks, "
                            "open_cabinet_and_place_bottle."
                        ))
    parser.add_argument("--condition", "-c", default=None,
                        choices=("ICL", "FuncReuse", "SlateGated", "PatternReuse"),
                        help=(
                            "Eval-style condition setting. When set, the CLI "
                            "wires the SkillStore / slate / AgenticSession "
                            "toggles consistently with scripts/eval/tasks.json's "
                            "conditions block: "
                            "ICL (no library / no slate), "
                            "FuncReuse (Python helper library + retrieval), "
                            "SlateGated (FuncReuse + slate of N candidates), "
                            "PatternReuse (markdown subtask doc library + "
                            "in-program VQA-verified-retry as a hard requirement). "
                            "Default (no flag) preserves today's interactive-CLI "
                            "behaviour (library on, all emissions on, slate "
                            "gated by --slate). Pass an explicit condition for "
                            "strict real-robot ablation runs."
                        ))
    args = parser.parse_args()

    # Resolve --condition into the underlying toggles, or fall back to
    # today's defaults when --condition is not set.
    if args.condition is not None:
        cond_flags = _CONDITION_FLAGS[args.condition]
        if args.slate and not cond_flags["use_slate"]:
            parser.error(
                f"--slate is incompatible with --condition {args.condition} — "
                f"slate is only used by SlateGated."
            )
        library_persistent = cond_flags["library_persistent"]
        use_slate = cond_flags["use_slate"]
        agentic_kwargs = dict(cond_flags["agentic_kwargs"])
    else:
        library_persistent = True
        use_slate = args.slate
        agentic_kwargs = {}  # AgenticSession defaults
    # vlm_tracing is independent of --condition: it's a debug toggle, not a
    # condition mechanism. The user can disable trace-wrapper VLM
    # observations under any condition.
    vlm_tracing = args.vlm_tracing

    # Per-channel correction-feedback ablation flags. Independent of
    # --condition: lets you measure the LLM's correction quality with
    # specific feedback channels surgically removed (e.g. PatternReuse
    # minus fault summary, or any condition with raw trace injection
    # off). Default None on each flag means "no user override" — the
    # --condition's setting (or AgenticSession's True default) wins.
    # Only an explicit --no-fault-summary / --no-trace-injection
    # overrides to False.
    if args.fault_summary is False:
        agentic_kwargs["emit_fault_summary"] = False
    if args.trace_injection is False:
        agentic_kwargs["emit_execution_trace"] = False

    # --hil / --vlm imply PatternReuse-like defaults when no explicit
    # --condition was set.  These are additive — if --condition was given
    # its settings win, and only the mid-exec wrapper is activated.
    midexec_mode = args.hil or args.vlm
    if midexec_mode and args.condition is None:
        agentic_kwargs.setdefault("emit_subtask_docs", True)
        agentic_kwargs.setdefault("emit_fault_summary", True)
        agentic_kwargs.setdefault("inprogram_retry", True)

    # Dry-run isolation. Regardless of --condition, a dry-run session must
    # never write to the persistent stores that live-robot sessions read
    # from — otherwise dry-run-derived skills / subtask docs / episode log
    # entries pollute the retrieval surfaces that live PatternReuse /
    # FuncReuse / SlateGated runs use. The stub-VQA always-yes return
    # makes dry-run "successes" qualitatively different from real
    # successes; their emissions would bias future generations in the
    # wrong direction. So: in dry-run, force the skill store to
    # ``:memory:``, silence subtask-doc and scorer emission, and route
    # episode-log entries to a sandbox file the eval harness can ignore.
    if args.dry_run:
        library_persistent = False
        agentic_kwargs["emit_subtask_docs"] = False
        agentic_kwargs["emit_scorers"] = False

    # Resolve --task into instruction + task_spec.
    task_spec: dict | None = None
    if args.task is not None:
        from pathlib import Path as _Path
        import json as _json
        tasks_path = _Path(__file__).resolve().parent / "eval" / "tasks.json"
        try:
            tasks = _json.loads(tasks_path.read_text())
        except FileNotFoundError:
            parser.error(f"--task supplied but tasks.json not found at {tasks_path}")
        match = next((t for t in tasks.get("tasks", []) if t["task_id"] == args.task), None)
        if match is None:
            available = ", ".join(t["task_id"] for t in tasks.get("tasks", []))
            parser.error(f"Unknown --task {args.task!r}. Available: {available}")
        # Auto-fill instruction if user didn't pass --instruction explicitly.
        if args.instruction is None:
            args.instruction = match.get("instruction")
        task_spec = match.get("success_scorer")  # may be None on legacy tasks

    if args.instruction is None and args.program is None:
        parser.error("Provide at least one of --instruction, --program, or --task.")

    console.print()
    console.rule("[bold blue]ARCHITECT — Synthesizing Shared Autonomy[/bold blue]")
    console.print()

    # ------------------------------------------------------------------
    # Build robot execution namespace
    # ------------------------------------------------------------------
    robot_config = get_robot_config(args.robot)
    robot_client = None
    if args.dry_run:
        namespace = robot_config.make_dry_run_namespace(console)
    else:
        namespace, robot_client = robot_config.make_live_namespace()

    # Free-form fallback: when --closed-loop is on but --task wasn't
    # supplied, synthesise a vqa_judge task_spec from the free-form
    # --instruction. Without this, closed-loop is a silent no-op for
    # free-form runs (the trigger at session.task_spec is not None
    # never fires). The judge reads skills/<robot>/vqa.md for VQA
    # phrasing conventions so it can interpret the program's VQA
    # answers correctly. Plain interactive runs (no --closed-loop, no
    # --task) keep their existing no-scoring behavior — no surprise
    # LLM judge calls.
    if task_spec is None and args.closed_loop and args.instruction is not None:
        vqa_md_path = robot_config.skills_dir / "vqa.md"
        try:
            vqa_guidelines = vqa_md_path.read_text()
        except FileNotFoundError:
            vqa_guidelines = ""
            console.print(
                f"[dim]\\[vqa_judge] no vqa.md at {vqa_md_path}; judge "
                f"runs without phrasing guidelines.[/dim]"
            )
        task_spec = {
            "name": "vqa_judge",
            "params": {
                "instruction": args.instruction,
                "vqa_guidelines": vqa_guidelines,
            },
        }
        console.print(
            "[dim]Free-form closed-loop: vqa_judge scorer auto-derived "
            "from --instruction.[/dim]"
        )

    _add_curobo_validation(namespace, dry_run=args.dry_run)

    # Per-robot persistent skill library.  Lives next to the curated markdown
    # skills for the same robot (skills/<robot>/library.db).  Skills written
    # via write_primitive (and approved by the user) survive session restart
    # and are auto-injected into the execution namespace next session.
    #
    # When --condition selects ICL or PatternReuse, the SkillStore is
    # backed by ":memory:" instead — helpers don't persist, retrieval
    # surfaces nothing, and the ablation matches the harness's
    # library: false flag for those conditions.
    #
    # ``--dry-run`` also forces ``:memory:`` (see the dry-run isolation
    # block above) so stub-tested primitives can't leak into live
    # sessions.
    library_db = robot_config.skills_dir / "library.db"
    if library_persistent:
        skill_store = SkillStore(db_path=library_db)
    else:
        skill_store = SkillStore(db_path=":memory:")
    registry = PrimitiveRegistry(store=skill_store)
    if registry.list_names():
        console.print(
            f"[dim]Library:[/dim] {len(registry.list_names())} skill(s) loaded "
            f"from {library_db}"
        )
    # Per-robot append-only slate decisions log (P4 / §4.7-lite). Used by the
    # eval harness in P7 to compute the useful-disagreement rate. In
    # dry-run, route to a sandbox file so dry-run-derived slate decisions
    # don't pollute the live log the eval harness consumes.
    episodes_filename = "episodes.dry_run.jsonl" if args.dry_run else "episodes.jsonl"
    episode_log = EpisodeLog(robot_config.skills_dir / episodes_filename)

    # Failure store + diagnosis. --failure-store gates the persistent
    # log; without it we use a :memory: store so the diagnosis path
    # still works (lookups return empty since nothing was ever
    # persisted). --no-failure-diagnosis disables the LLM diagnosis
    # call entirely; the CLI falls back to the older keyword-based
    # explain_failure text. --failure-store implies diagnosis on
    # (there's nothing useful to store without it). In dry-run, the
    # store is also :memory: regardless of flag — same isolation
    # rationale as library.db.
    failure_diagnosis_enabled = args.failure_diagnosis or args.failure_store
    failure_store = None
    if failure_diagnosis_enabled:
        from architect.library.failure_store import FailureStore
        if args.failure_store and not args.dry_run:
            failures_db = robot_config.skills_dir / "failures.db"
            failure_store = FailureStore(db_path=failures_db)
            existing = failure_store.count()
            if existing:
                console.print(
                    f"[dim]Failure store:[/dim] {existing} prior failure(s) "
                    f"loaded from {failures_db}"
                )
        else:
            # Diagnosis on, but no persistence — :memory: store keeps the
            # lookup interface available (returns empty) without touching
            # disk. Cheap and uniform.
            failure_store = FailureStore(db_path=":memory:")

    # Optional per-turn context-fill profile. Filename is timestamped so
    # repeated --profile-context runs accumulate side-by-side traces in
    # skills/<robot>/ rather than clobbering each other.
    profile_path: Path | None = None
    if args.profile_context:
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        profile_path = robot_config.skills_dir / f"context_profile_{ts}.jsonl"
        console.print(
            f"[dim]Context profile →[/dim] {profile_path}"
        )

    try:
        # ------------------------------------------------------------------
        # Step 1: Obtain the initial program
        # ------------------------------------------------------------------
        if args.program is not None:
            program_code = Path(args.program).read_text()
            console.print(f"[dim]Loaded:[/dim] {args.program}")
            show_program(program_code, title="Loaded Program")
            session = AgenticSession(
                model=args.model,
                registry=registry,
                namespace=namespace,
                console=console,
                approve_primitive_fn=_make_approve_primitive(args.auto_approve_primitives),
                dry_run=args.dry_run,
                api_spec=robot_config.api_spec,
                robot_label=robot_config.display_name,
                robot_name=args.robot,
                skills_dir=robot_config.skills_dir,
                task_spec=task_spec,
                context_profile_path=profile_path,
                failure_store=failure_store,
                failure_diagnosis_enabled=failure_diagnosis_enabled,
                scratchpad_enabled=args.scratchpad,
                retrieval_mode=args.retrieval,
                retrieval_k=args.retrieval_k,
                retrieval_instruction=args.instruction,
                **agentic_kwargs,
            )
            session.apply_edit(program_code)
        else:
            console.print(f"[dim]Mode:[/dim]  language_only  (agentic)")
            console.print(f"[dim]Model:[/dim] {args.model}")
            if args.dry_run:
                console.print("[dim]Dry-run:[/dim] robot API calls will be stubbed")
            console.print()

            session = AgenticSession(
                model=args.model,
                registry=registry,
                namespace=namespace,
                console=console,
                approve_primitive_fn=_make_approve_primitive(args.auto_approve_primitives),
                dry_run=args.dry_run,
                api_spec=robot_config.api_spec,
                robot_label=robot_config.display_name,
                robot_name=args.robot,
                skills_dir=robot_config.skills_dir,
                task_spec=task_spec,
                context_profile_path=profile_path,
                failure_store=failure_store,
                failure_diagnosis_enabled=failure_diagnosis_enabled,
                scratchpad_enabled=args.scratchpad,
                retrieval_mode=args.retrieval,
                retrieval_k=args.retrieval_k,
                retrieval_instruction=args.instruction,
                **agentic_kwargs,
            )

            console.print("[bold]Generating program (agentic)...[/bold]")
            program_code = session.generate(instruction=args.instruction)

            if program_code is None:
                console.print("[red]Generation failed.[/red]")
                return

            show_program(program_code, title="Generated Program")

        # ------------------------------------------------------------------
        # Step 2: Approval + initial execution
        # ------------------------------------------------------------------
        if midexec_mode:
            wrap_vqa_for_midexec(namespace)

        _, initial_program, last_interrupt = approval_prompt(
            program_code, namespace, args.model, registry,
            api_spec=robot_config.api_spec,
            robot_label=robot_config.display_name,
            session=session,
            vlm_tracing=vlm_tracing,
            instruction=args.instruction,
            robot_name=args.robot,
        )
        if initial_program.strip() != session.current_program.strip():
            session.apply_edit(initial_program)

        # ------------------------------------------------------------------
        # Step 3: Interactive correction loop
        # ------------------------------------------------------------------
        console.print()
        console.print(_HELP)

        # Closed-loop auto-correction round counter. Local to this CLI
        # session; resets if the user takes manual control by typing a
        # correction (since manual input means the user has decided to
        # drive the loop themselves, and any earlier failed auto-rounds
        # are no longer relevant).
        auto_correction_count = 0

        while True:
            # Auto-exit when the most recent execution PASSED the task
            # scorer. The user only sees the correction prompt when the
            # task failed (or no task spec was set, so we can't tell —
            # behave the same as before in that case). This makes the
            # correction loop event-driven: a failure (indicated by the
            # in-program VQA + scorer verdict) surfaces a prompt;
            # success ends the session cleanly. Doesn't fire when
            # task_spec is None (e.g. ad-hoc CLI runs without --task) —
            # those keep today's "always prompt until done" behaviour.
            if (
                session.task_spec is not None
                and session.last_task_score is not None
                and session.last_task_score >= 0.5
            ):
                console.print(
                    "\n[bold green]✓ Task complete — session done.[/bold green]"
                )
                break

            # Mid-execution intervention (--hil / --vlm). When a
            # MidExecInterrupt was raised during execution, the program
            # paused at a VQA verification failure. Get a correction
            # (from user or auto-diagnosed), generate a continuation
            # program, and execute it.
            if midexec_mode and last_interrupt is not None:
                console.print()
                console.print(
                    f"[bold yellow]VQA verification failed after retries:[/bold yellow]"
                )
                console.print(
                    f"  [dim]Question:[/dim] {last_interrupt.vqa_question}"
                )
                console.print(
                    f"  [dim]Answer:[/dim]   {last_interrupt.vqa_answer}"
                )
                console.print()

                if args.hil:
                    try:
                        correction = input("HIL correction › ").strip()
                    except (EOFError, KeyboardInterrupt):
                        console.print()
                        break
                    if not correction:
                        continue
                    if correction.lower() in ("done", "exit", "quit"):
                        break
                    if correction.lower() == "skip":
                        last_interrupt = None
                        continue
                else:
                    # --vlm: auto-diagnose from the partial trace.
                    from architect.eval.task_scorers import explain_failure
                    correction = explain_failure(
                        session._last_execution_trace,
                        session.task_spec,
                    )
                    if correction is None:
                        correction = (
                            "The verification check failed. Adjust the "
                            "approach and retry the failed step."
                        )
                    console.print(
                        f"[bold yellow]VLM auto-correction:[/bold yellow]"
                    )
                    for ln in correction.splitlines():
                        console.print(f"[dim]│ {ln}[/dim]")
                    console.print()

                current_ee_pose = None
                try:
                    current_ee_pose = namespace["get_current_ee_pose"]()
                except Exception:
                    pass

                continuation = session.continue_from_failure(
                    correction, last_interrupt, current_ee_pose,
                )
                if continuation is None:
                    console.print(
                        "[red]Continuation generation failed.[/red]"
                    )
                    last_interrupt = None
                    continue

                show_program(continuation, title="Continuation Program")

                # Reset wrapper state for the continuation execution.
                unwrap_vqa_for_midexec(namespace)
                wrap_vqa_for_midexec(namespace)

                namespace["reset_robot"]()
                console.print("[bold]Executing continuation...[/bold]")
                ok, _trace, last_interrupt = _run(
                    continuation, namespace, args.model, registry,
                    api_spec=robot_config.api_spec,
                    robot_label=robot_config.display_name,
                    session=session,
                    vlm_tracing=vlm_tracing,
                )
                if ok:
                    console.print("[green]✓ Execution complete.[/green]")
                continue

            # Closed-loop auto-correction. When --closed-loop is on and
            # the most recent execution FAILED the task scorer (and we
            # haven't yet exhausted the auto-correction budget), build a
            # structured diagnostic of WHY the trial failed and issue
            # the correction automatically. The refined program is
            # auto-approved and re-executed; the next loop iteration
            # re-checks the score. Falls through to the manual prompt
            # below once auto_correction_count reaches
            # --max-auto-corrections.
            #
            # Skipped when there's no task_spec (can't compute pass/fail)
            # or when the most recent execution exec-failed (trace is
            # incomplete; let the user decide).
            if (
                args.closed_loop
                and session.task_spec is not None
                and session.last_task_score is not None
                and session.last_task_score < 0.5
                and auto_correction_count < args.max_auto_corrections
                and session.current_program is not None
            ):
                # Prefer the structured FailureDiagnosis when available
                # (--failure-diagnosis on, which is the default). It's
                # richer than explain_failure's keyword-based diagnostic
                # — includes category, signature, root cause, suggested
                # directions, and trace evidence — so the refining LLM
                # has a more actionable starting point. Falls back to
                # the keyword diagnostic when diagnosis is disabled or
                # the LLM call failed.
                feedback: str | None = None
                if session.last_diagnosis is not None and session.last_diagnosis.failure_category != "diagnosis_unavailable":
                    feedback = session.last_diagnosis.to_markdown()
                else:
                    from architect.eval.task_scorers import explain_failure
                    feedback = explain_failure(
                        session._last_execution_trace, session.task_spec,
                    )
                    if feedback is None:
                        feedback = (
                            "The program failed the task scorer. Refine to "
                            "satisfy the task's success criteria — see the "
                            "execution trace for what the program actually did."
                        )

                console.print()
                console.print(
                    f"[bold yellow]Auto-correction "
                    f"{auto_correction_count + 1}/{args.max_auto_corrections}[/bold yellow]"
                    f"  [dim](closed-loop feedback)[/dim]"
                )
                # Indent the diagnostic so it visually clusters with the
                # auto-correction header. Shorten very long diagnostics for
                # the on-console display (the full text still goes to the LLM).
                preview = feedback if len(feedback) <= 600 else feedback[:597] + "..."
                for ln in preview.splitlines():
                    console.print(f"[dim]│ {ln}[/dim]")
                console.print()

                refined = session.correct(feedback)
                if refined is None:
                    console.print(
                        "[red]Auto-correction failed (LLM call returned None). "
                        "Falling through to manual prompt.[/red]"
                    )
                    auto_correction_count = args.max_auto_corrections
                    continue

                show_diff(session.current_program, refined)
                show_program(refined, title="Auto-Refined Program")
                # Auto-approve + execute, no y/n prompt. Closed-loop
                # implies the user has opted into autonomous correction
                # so we don't gate each iteration on user input.
                namespace["reset_robot"]()
                console.print("[bold]Auto-executing refined program...[/bold]")
                ok, _trace, _intr = _run(
                    refined, namespace, args.model, registry,
                    api_spec=robot_config.api_spec,
                    robot_label=robot_config.display_name,
                    session=session,
                    vlm_tracing=vlm_tracing,
                    instruction=args.instruction,
                    robot_name=args.robot,
                )
                if ok:
                    console.print("[green]✓ Execution complete.[/green]")
                else:
                    console.print(
                        "[yellow]Auto-refined program raised during exec; "
                        "letting the loop iterate.[/yellow]"
                    )
                auto_correction_count += 1
                continue  # back to top; re-check score

            try:
                raw = input("\nSSA › ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not raw:
                continue

            cmd = raw.lower()

            # User took manual control — reset the auto-correction budget
            # so subsequent failures can re-enter the closed-loop path
            # rather than skipping straight to manual prompts. Runs before
            # the command dispatch so subsequent failures after a manual
            # input get a fresh auto-correction budget.
            auto_correction_count = 0

            if cmd in ("done", "exit", "quit"):
                break

            elif cmd == "show":
                show_program(session.current_program)

            elif cmd == "undo":
                if session.can_undo:
                    restored = session.undo()
                    console.print("[yellow]↩  Undone.[/yellow]")
                    show_program(restored or "", title="Restored Program")
                    if restored:
                        namespace["reset_robot"]()
                        console.print("[bold]Re-executing...[/bold]")
                        ok, _trace, last_interrupt = _run(
                            restored, namespace, args.model, registry,
                            api_spec=robot_config.api_spec,
                            robot_label=robot_config.display_name,
                            session=session,
                            vlm_tracing=vlm_tracing,
                            instruction=args.instruction,
                            robot_name=args.robot,
                        )
                        if ok:
                            console.print("[green]✓ Execution complete.[/green]")
                else:
                    console.print("[dim]  Nothing to undo.[/dim]")

            elif cmd == "edit":
                edited = _open_in_editor(session.current_program)
                if edited.strip() != session.current_program.strip():
                    show_diff(session.current_program, edited)
                    session.apply_edit(edited)
                    show_program(edited, title="Edited Program")
                    namespace["reset_robot"]()
                    console.print("[bold]Re-executing...[/bold]")
                    ok, _trace, last_interrupt = _run(
                        edited, namespace, args.model, registry,
                        api_spec=robot_config.api_spec,
                        robot_label=robot_config.display_name,
                        session=session,
                        vlm_tracing=vlm_tracing,
                        instruction=args.instruction,
                        robot_name=args.robot,
                    )
                    if ok:
                        console.print("[green]✓ Execution complete.[/green]")
                else:
                    console.print("[dim]  (no changes)[/dim]")

            elif cmd == "reset":
                console.print("[bold]Resetting and re-executing...[/bold]")
                namespace["reset_robot"]()
                ok, _trace, last_interrupt = _run(
                    session.current_program, namespace, args.model, registry,
                    api_spec=robot_config.api_spec,
                    robot_label=robot_config.display_name,
                    session=session,
                    vlm_tracing=vlm_tracing,
                    instruction=args.instruction,
                    robot_name=args.robot,
                )
                if ok:
                    console.print("[green]✓ Execution complete.[/green]")

            elif cmd == "/library" or cmd == "library":
                _print_library(skill_store, console)

            elif cmd.startswith("/graduate ") or cmd.startswith("graduate "):
                target = raw.split(None, 1)[1].strip()
                _graduate(target, skill_store, console)

            elif cmd.startswith("/forget ") or cmd.startswith("forget "):
                target = raw.split(None, 1)[1].strip()
                _forget(target, skill_store, namespace, console)

            else:
                current_ee_pose = None
                try:
                    current_ee_pose = namespace["get_current_ee_pose"]()
                except Exception:
                    pass

                old_program = session.current_program

                if use_slate:
                    # Active counterfactual sampling (P4 / C1). Active under
                    # --condition SlateGated and under the legacy --slate
                    # flag (when no --condition is set).
                    refined = _correct_with_slate(
                        session, raw, current_ee_pose,
                        n=args.slate_n, k_probes=args.slate_probes,
                        robot_name=args.robot, episode_log=episode_log,
                        console=console,
                    )
                else:
                    console.print("[bold]Refining (agentic)...[/bold]")
                    refined = session.correct(raw, current_ee_pose=current_ee_pose)

                if refined is None:
                    console.print("[red]Correction failed.[/red]")
                    continue

                show_diff(old_program, refined)
                show_program(refined, title="Revised Program")

                namespace["reset_robot"]()
                _, approved, last_interrupt = approval_prompt(
                    refined, namespace, args.model, registry,
                    api_spec=robot_config.api_spec,
                    robot_label=robot_config.display_name,
                    session=session,
                    vlm_tracing=vlm_tracing,
                    instruction=args.instruction,
                    robot_name=args.robot,
                )
                if approved.strip() != refined.strip():
                    session.apply_edit(approved)

        # ------------------------------------------------------------------
        # Step 4: Save final program if requested
        # ------------------------------------------------------------------
        if args.save and session.current_program:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(session.current_program)
            console.print(f"\n[dim]Saved to {save_path}[/dim]")

        # Wait for any in-flight emission threads (P3 scorers + subtask docs)
        # so their writes complete before we close the SkillStore / process.
        # The threads are daemons, so a hard Ctrl-C still exits promptly; this
        # just ensures the clean-exit path persists artifacts the user paid for.
        if "session" in dir() and hasattr(session, "join_pending_scorers"):
            session.join_pending_scorers(timeout=30.0)
        if "session" in dir() and hasattr(session, "join_pending_subtask_docs"):
            session.join_pending_subtask_docs(timeout=30.0)

        console.print()
        console.rule("[dim]Session ended[/dim]")

    finally:
        if robot_client is not None:
            robot_config.shutdown(robot_client)
        registry.close()


if __name__ == "__main__":
    main()
