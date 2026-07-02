"""CaP-style hierarchical function synthesis.

When exec() raises NameError for an undefined function, we:
  1. Extract the undefined name from the exception
  2. Find its call site in the parent program via AST to infer arguments
  3. Ask Claude to synthesize the function body using only the primitive API
  4. Show the synthesized function to the user for approval (with optional edit)
  5. Inject it into the namespace and retry execution

This recurses up to MAX_SYNTHESIS_DEPTH times so synthesized helpers can
themselves call other not-yet-defined helpers.
"""

from __future__ import annotations

import ast
import re
from typing import Callable

from architect.llm.client import call_claude
from architect.llm.prompts import build_synthesis_prompt

MAX_SYNTHESIS_DEPTH = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _undefined_name(exc: NameError) -> str | None:
    """Extract the undefined identifier from a NameError message."""
    m = re.search(r"name '(\w+)' is not defined", str(exc))
    return m.group(1) if m else None


def _call_example(func_name: str, program: str) -> str:
    """Return the first call site of func_name found in program, as a string.

    Used to give Claude a concrete example of how the function is called so it
    can infer the expected arguments.
    """
    try:
        tree = ast.parse(program)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name
            ):
                args = [ast.unparse(a) for a in node.args]
                kwargs = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
                return f"{func_name}({', '.join(args + kwargs)})"
    except Exception:
        pass
    return f"{func_name}(...)"


def _strip_fences(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def synthesize_function(
    func_name: str,
    parent_program: str,
    model: str,
    *,
    api_spec: str,
    robot_label: str,
) -> str:
    """Ask Claude to synthesize a function body for an undefined name.

    Args:
        func_name: The undefined function name to implement.
        parent_program: The full parent program (for context on intent and args).
        model: Claude model name.
        api_spec: The active robot's primitive-API string. Required so
            synthesis stays grounded in the runtime robot's primitives —
            without this, the LLM was previously primed with a Stretch
            spec and produced ``hello_helpers``-importing code on Franka.
        robot_label: Human-readable robot name for the synthesis prompt role.

    Returns:
        A string containing only the function definition (def ...: ...).
    """
    example = _call_example(func_name, parent_program)
    messages = build_synthesis_prompt(
        func_name, example, parent_program,
        api_spec=api_spec, robot_label=robot_label,
    )
    raw = call_claude(messages, model=model, max_tokens=2048)
    return _strip_fences(raw)


# ---------------------------------------------------------------------------
# Execution with recursive synthesis
# ---------------------------------------------------------------------------

def execute_with_synthesis(
    program: str,
    namespace: dict,
    model: str,
    console,
    approve_fn: Callable[[str, str], tuple[bool, str]],
    *,
    api_spec: str,
    robot_label: str,
    depth: int = 0,
    trace: "ExecutionTrace | None" = None,
    vlm_enabled: bool = True,
) -> "tuple[bool, ExecutionTrace]":
    """Execute program, synthesizing undefined helper functions on demand.

    When a NameError is raised for an undefined function, Claude synthesizes
    its body from the primitive API, the user approves it, and execution
    retries. This recurses until all calls resolve or the depth limit is hit.

    Args:
        program: Python source to execute.
        namespace: Robot API namespace dict (mutated in place with new helpers).
        model: Claude model name for synthesis.
        console: Rich Console for display.
        approve_fn: Called as approve_fn(func_name, func_code) -> (approved, final_code).
                    Should display the function, ask for approval, and return
                    whether to proceed and the (possibly edited) final code.
        depth: Current recursion depth.
        trace: Carry across recursive calls so a synthesized helper's calls
               land in the same timeline as the parent program's. Created on
               the first (outermost) call; passed through unchanged on
               recursion. The namespace is wrapped exactly once.
        vlm_enabled: When True (default), the trace wrapper fires a
               contextual VQA question after every traced control /
               perception call ("did the gripper grasp?", "where is the
               gripper now?", etc.) — these are the Layer B observations.
               Pass False to disable: useful when the VQA backend is
               unreliable or expensive and you want clean per-call
               state snapshots without the per-call VLM round-trip.
               Forwarded to :func:`attach_trace`.

    Returns:
        ``(ok, trace, interrupt)`` — exec success bool, the populated
        :class:`architect.execution_trace.ExecutionTrace`, and an optional
        :class:`~architect.midexec.MidExecInterrupt` (``None`` when execution
        completed or errored normally). The trace is always returned
        (even on failure) so callers can persist it for the next
        correction's prompt.
    """
    from architect.midexec import MidExecInterrupt
    from architect.corrections.execution_trace import (
        attach_trace, seal_trace_error, seal_trace_interrupted, seal_trace_ok,
    )

    # Attach the trace once, on the outermost call. Recursive calls reuse
    # the same trace so synthesized-helper calls show up in-line with the
    # parent's call sequence rather than starting a fresh timeline.
    if trace is None:
        trace = attach_trace(namespace, program, vlm_enabled=vlm_enabled)

    if depth >= MAX_SYNTHESIS_DEPTH:
        console.print(
            f"[red]Synthesis depth limit ({MAX_SYNTHESIS_DEPTH}) reached — aborting.[/red]"
        )
        seal_trace_error(
            trace, namespace,
            RuntimeError(f"synthesis depth limit ({MAX_SYNTHESIS_DEPTH})"),
        )
        return False, trace, None

    try:
        exec(program, namespace)  # noqa: S102
        seal_trace_ok(trace, namespace)
        return True, trace, None

    except NameError as exc:
        func_name = _undefined_name(exc)
        if func_name is None:
            console.print(f"[red bold]NameError:[/red bold] {exc}")
            seal_trace_error(trace, namespace, exc)
            return False, trace, None

        console.print(
            f"\n[yellow]Undefined function:[/yellow] [cyan bold]{func_name}[/cyan bold] "
            f"— synthesizing from primitive API..."
        )

        with console.status(f"[bold]Synthesizing {func_name}()...[/bold]", spinner="dots"):
            func_code = synthesize_function(
                func_name, program, model=model,
                api_spec=api_spec, robot_label=robot_label,
            )

        approved, final_code = approve_fn(func_name, func_code)
        if not approved:
            console.print("[dim]Synthesis cancelled.[/dim]")
            seal_trace_error(
                trace, namespace,
                RuntimeError(f"user cancelled synthesis of {func_name}"),
            )
            return False, trace, None

        # Inject the synthesized function (may itself trigger further synthesis)
        try:
            exec(final_code, namespace)  # noqa: S102
        except Exception as inject_exc:
            console.print(f"[red]Error defining {func_name}:[/red] {inject_exc}")
            seal_trace_error(trace, namespace, inject_exc)
            return False, trace, None

        # Retry the full program now that func_name is defined — keep the
        # same trace so the recursive call's entries append to the existing
        # timeline.
        return execute_with_synthesis(
            program, namespace, model, console, approve_fn,
            api_spec=api_spec, robot_label=robot_label, depth=depth + 1,
            trace=trace,
        )

    except MidExecInterrupt as exc:
        console.print(
            f"\n[bold yellow]Mid-execution interrupt:[/bold yellow] "
            f"{exc.vqa_question} → {exc.vqa_answer}"
        )
        seal_trace_interrupted(trace, namespace, exc)
        return False, trace, exc

    except Exception as exc:
        console.print(f"\n[red bold]Execution error:[/red bold] {exc}")
        seal_trace_error(trace, namespace, exc)
        return False, trace, None
