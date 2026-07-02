"""Anthropic tool-use schemas and executor for the agentic ARCHITECT loop.

Tools available to Claude during agentic synthesis:
  list_primitives      — introspect the full robot API + session primitives
  read_primitive       — get source of a session primitive
  query_robot_state    — read current EE pose / joints / gripper
  get_scene_description — detect markers + VQA scene description
  write_primitive      — register a new reusable helper (needs user approval)
  run_ros2_command     — run a ros2 CLI command (needs user approval)
  submit_program       — deliver the final program (handled by the agent loop)
"""

from __future__ import annotations

import json
import subprocess

from rich.panel import Panel
from rich.syntax import Syntax

from architect.agent.primitive_registry import PrimitiveRegistry


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_primitives",
        "description": (
            "List all available robot API functions — both the built-in primitives "
            "and any helper functions registered during this session. "
            "Call this first to understand what's available before writing a program."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_primitive",
        "description": "Get the full source code of a session-registered primitive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function name to look up."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "query_robot_state",
        "description": (
            "Read the current robot state: end-effector pose, joint positions, "
            "and gripper width."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_scene_description",
        "description": (
            "Get a description of the current scene: all detected ArUco markers "
            "(with poses in the base_link frame) and a visual description of "
            "objects visible to the robot."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_primitive",
        "description": (
            "Register a new reusable helper function as a session primitive. "
            "The function will be injected into the execution namespace and "
            "listed in future list_primitives() calls. "
            "Requires user approval before registration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Python function name (valid identifier).",
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python function definition starting with 'def'. "
                        "Must use only functions from list_primitives() output or "
                        "previously registered primitives. No imports."
                    ),
                },
                "docstring": {
                    "type": "string",
                    "description": "One-line description of what the function does.",
                },
            },
            "required": ["name", "code", "docstring"],
        },
    },
    {
        "name": "run_ros2_command",
        "description": (
            "Run a ROS2 CLI command for environment introspection or service invocation "
            "(e.g. 'ros2 topic list', 'ros2 service call ...'). "
            "Requires user approval before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command string starting with 'ros2'.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "query_skills",
        "description": (
            "Search the markdown skill-doc library by semantic similarity. "
            "Returns the top-k docs whose descriptions best match your "
            "query, with their full markdown contents. Use this when the "
            "Domain Knowledge section in the system prompt doesn't already "
            "cover what you need — typically when the task touches an "
            "unfamiliar primitive (keypoint detection, placement pose, "
            "etc.) or when a correction needs you to look up a specific "
            "pattern. The session prompt already pre-loads the foundational "
            "docs (VQA, motion bounds, verification, ROS) plus the top "
            "matches against the original instruction; call query_skills "
            "to fetch *more* docs mid-loop. Only available when "
            "--retrieval hybrid is on; in static mode the tool returns "
            "a no-op notice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query to match against doc descriptions.",
                },
                "k": {
                    "type": "integer",
                    "description": "Max number of docs to return. Default 3, max 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_scratchpad",
        "description": (
            "Update the session scratchpad — your working memory carried "
            "across correction turns. Use this to record beliefs that "
            "should survive past the current LLM turn: the current focus, "
            "working hypotheses about why prior attempts failed, "
            "observations confirmed by the trace, and theories already "
            "invalidated so you don't re-try them. Provide only the "
            "fields you want to update; omit the rest. Each list field "
            "is deduplicated case-insensitively and capped at 8 entries "
            "(oldest evicted first). Invalidated hypotheses are matched "
            "softly (substring) against working_hypotheses and moved "
            "rather than duplicated. Call this AT MOST once per turn "
            "and only when there's something new worth remembering; "
            "calling it with no fields is wasteful."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_goal": {
                    "type": "string",
                    "description": (
                        "Optional. Sets the current focus — what you're "
                        "actively trying to solve right now, possibly "
                        "narrower than the original task (e.g. 'fix the "
                        "grasp before tackling placement')."
                    ),
                },
                "add_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Trace-confirmed facts worth remembering "
                        "across corrections (e.g. 'AnyGrasp returns low "
                        "confidence on this cup', 'VQA reports red object "
                        "visible but mislabelled')."
                    ),
                },
                "add_working_hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Current beliefs about cause or fix that "
                        "haven't been confirmed yet (e.g. 'object is "
                        "lighter than expected — gripper closes too far')."
                    ),
                },
                "invalidate_hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Hypotheses you've now disproved. Soft "
                        "substring match against working_hypotheses, "
                        "moved to invalidated. Lets you avoid retrying "
                        "the same theory across corrections."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_program",
        "description": (
            "Submit the completed Python program for user approval and execution. "
            "Call this when you are ready to deliver the final program. "
            "The program must use only functions from list_primitives() output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete Python program to submit.",
                },
            },
            "required": ["code"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Dispatches Claude tool calls to their implementations.

    ``submit_program`` is intentionally NOT handled here — the agent loop
    intercepts it directly and uses it as the loop-exit signal.
    """

    def __init__(
        self,
        registry: PrimitiveRegistry,
        namespace: dict,
        console,
        approve_primitive_fn,
        api_spec: str = "",
        dry_run: bool = False,
        robot_name: str = "franka",
        graduation_enabled: bool = True,
        graduation_threshold: float | None = None,
        graduation_k: int | None = None,
        scratchpad=None,
        correction_count_getter=None,
        skill_doc_index=None,
    ) -> None:
        self._registry = registry
        self._namespace = namespace
        self._console = console
        self._approve_primitive_fn = approve_primitive_fn
        self._api_spec = api_spec
        self._dry_run = dry_run
        self._robot_name = robot_name
        self._graduation_enabled = graduation_enabled
        self._graduation_threshold = graduation_threshold
        self._graduation_k = graduation_k
        # SessionScratchpad ref (or None when the scratchpad feature is
        # off). The update_scratchpad tool handler mutates it in place;
        # session.py renders it into correction prompts. correction_count_
        # getter is a zero-arg callable returning the current correction
        # round number so the scratchpad can stamp updates with it
        # (used for the harness's "agent hasn't updated in N rounds"
        # detection). Both None means update_scratchpad is a no-op that
        # tells the agent the feature is off.
        self._scratchpad = scratchpad
        self._correction_count_getter = correction_count_getter
        # Optional :class:`architect.library.skill_doc_index.SkillDocIndex` —
        # set when --retrieval hybrid is on. Powers the query_skills
        # tool. None in static mode; the tool handler returns a noop
        # notice rather than erroring so the agent's turn stays clean.
        self._skill_doc_index = skill_doc_index

    # Tools that print their own rich UI — skip the generic log line for these
    _SELF_REPORTING = {"write_primitive", "run_ros2_command"}

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return a JSON-serialisable result string."""
        if tool_name not in self._SELF_REPORTING:
            detail = ""
            if tool_name == "read_primitive" and "name" in tool_input:
                detail = f"([cyan]{tool_input['name']}[/cyan])"
            elif tool_name == "run_ros2_command" and "command" in tool_input:
                detail = f"([cyan]{tool_input['command']}[/cyan])"
            self._console.print(
                f"  [dim]→[/dim] [bold]{tool_name}[/bold]{detail}"
            )
        try:
            if tool_name == "list_primitives":
                return self._list_primitives()
            elif tool_name == "read_primitive":
                return self._read_primitive(tool_input.get("name", ""))
            elif tool_name == "query_robot_state":
                return self._query_robot_state()
            elif tool_name == "get_scene_description":
                return self._get_scene_description()
            elif tool_name == "write_primitive":
                return self._write_primitive(
                    tool_input.get("name", ""),
                    tool_input.get("code", ""),
                    tool_input.get("docstring", ""),
                )
            elif tool_name == "run_ros2_command":
                return self._run_ros2_command(tool_input.get("command", ""))
            elif tool_name == "update_scratchpad":
                return self._update_scratchpad(tool_input)
            elif tool_name == "query_skills":
                return self._query_skills(tool_input)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------

    def _list_primitives(self) -> str:
        spec = self._api_spec
        session_spec = self._registry.to_spec_string()
        if session_spec:
            spec = spec + "\n\n" + session_spec
        return spec

    def _read_primitive(self, name: str) -> str:
        src = self._registry.get_source(name)
        if src is None:
            return json.dumps({"error": f"No session primitive named '{name}'. Use list_primitives() to see available functions."})
        return src

    def _query_robot_state(self) -> str:
        try:
            ee = self._namespace["get_current_ee_pose"]()
            joints = self._namespace["get_current_joints"]()
            gripper = self._namespace["get_current_gripper_width"]()
            return json.dumps({
                "ee_pose": ee,
                "joints": joints,
                "gripper_width": gripper,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _get_scene_description(self) -> str:
        try:
            markers = self._namespace["detect_markers"]()
            description = self._namespace["get_vqa_response"](
                "Describe the objects visible to the robot and their approximate locations."
            )
            return json.dumps({"markers": markers, "scene": description}, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _write_primitive(self, name: str, code: str, docstring: str) -> str:
        if not name or not code:
            return json.dumps({"error": "name and code are required."})

        self._console.print()
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self._console.print(Panel(
            syntax,
            title=f"[bold yellow]New primitive:[/bold yellow] [cyan]{name}()[/cyan]",
            border_style="yellow",
        ))
        self._console.print(f"[dim]  {docstring}[/dim]")

        approved, final_code = self._approve_primitive_fn(name, code)
        if not approved:
            return json.dumps({"status": "rejected", "message": "User rejected the primitive."})

        self._registry.register(name, final_code, docstring)
        self._registry.inject_into_namespace(self._namespace)

        # P5 graduation gate. Runs the primitive against a counterfactual
        # probe suite; admits to the cross-session library only on pass.
        # Failure leaves the skill at 'candidate' — usable in this session,
        # surfaced in ``/library``, force-promotable via ``/graduate <name>``.
        result_info: dict[str, object] = {"status": "registered", "name": name}
        store = getattr(self._registry, "store", None)
        if self._graduation_enabled and store is not None:
            try:
                from architect.probes.graduation import (
                    DEFAULT_K, DEFAULT_THRESHOLD,
                    persist_and_admit, try_graduate,
                )
                result = try_graduate(
                    name=name,
                    code=final_code,
                    robot_name=self._robot_name,
                    threshold=self._graduation_threshold or DEFAULT_THRESHOLD,
                    k=self._graduation_k or DEFAULT_K,
                )
                suite_id = persist_and_admit(store, result, console=self._console)
                # The "✓ graduated" UX claim must reflect what actually
                # landed in the DB, not just the gate's verdict. If
                # persistence failed (returned None), persist_and_admit
                # already printed a dim warning and the skill row is still
                # at 'candidate'.
                persisted = result.admitted and suite_id is not None
                if persisted:
                    self._console.print(
                        f"  [green]✓ graduated:[/green] {name} "
                        f"[dim]({result.reason})[/dim]"
                    )
                else:
                    reason = result.reason if not result.admitted else (
                        "persistence failed (see warning above)"
                    )
                    self._console.print(
                        f"  [yellow]⚠ stays candidate:[/yellow] {name} "
                        f"[dim]({reason})[/dim]"
                    )
                result_info.update({
                    "graduated": persisted,
                    "pass_rate": result.pass_rate,
                    "reason": result.reason,
                })
            except Exception as exc:
                # A buggy gate must not crash the whole tool-use loop —
                # surface the error and continue with the primitive in
                # candidate status.
                self._console.print(
                    f"  [dim]\\[graduation] gate raised "
                    f"{type(exc).__name__}: {exc} — skill stays candidate[/dim]"
                )

        return json.dumps(result_info)

    def _run_ros2_command(self, command: str) -> str:
        if not command.strip().startswith("ros2"):
            return json.dumps({"error": "Only ros2 commands are permitted."})

        if self._dry_run:
            self._console.print(
                f"  [dim]\\[dry-run][/dim] [cyan]ros2 command skipped:[/cyan] {command}"
            )
            return json.dumps({
                "status": "dry-run",
                "message": "ROS2 command skipped in dry-run mode.",
                "stdout": "",
            })

        self._console.print()
        self._console.print(Panel(
            f"[bold]{command}[/bold]",
            title="[bold yellow]ROS2 command[/bold yellow]",
            border_style="yellow",
        ))

        try:
            choice = input("Run this command? [y/n] › ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._console.print()
            return json.dumps({"status": "rejected", "message": "User cancelled."})

        if choice not in ("y", "yes"):
            return json.dumps({"status": "rejected", "message": "User rejected the command."})

        try:
            result = subprocess.run(
                command,
                shell=True,  # noqa: S602
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.dumps({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Command timed out after 30 seconds."})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _update_scratchpad(self, tool_input: dict) -> str:
        """Apply an agent-emitted scratchpad update.

        When the scratchpad feature is off (no scratchpad ref on the
        executor), returns a benign 'feature disabled' notice. We
        deliberately don't error: an agent that calls the tool when
        it's off should just see that nothing happened and move on,
        not have its turn poisoned by an error.
        """
        if self._scratchpad is None:
            return json.dumps({
                "status": "noop",
                "message": (
                    "Scratchpad feature is disabled for this session. "
                    "The update was not applied."
                ),
            })

        # correction_count_getter is the harness's hook for stamping
        # updates with the current round. Treat absent / failing
        # getter as round 0 — best-effort.
        cc = 0
        if self._correction_count_getter is not None:
            try:
                cc = int(self._correction_count_getter() or 0)
            except Exception:
                cc = 0

        status = self._scratchpad.apply_update(
            current_goal=tool_input.get("current_goal"),
            add_observations=tool_input.get("add_observations") or [],
            add_working_hypotheses=tool_input.get("add_working_hypotheses") or [],
            invalidate_hypotheses=tool_input.get("invalidate_hypotheses") or [],
            correction_count=cc,
        )
        # Light console echo so the operator can see what the agent
        # committed to working memory — useful when reading the trace
        # to understand why the next correction took a particular
        # direction.
        try:
            self._console.print(
                f"  [dim]\\[scratchpad] applied: {status}[/dim]"
            )
        except Exception:
            pass
        return json.dumps({"status": "ok", **status})

    def _query_skills(self, tool_input: dict) -> str:
        """Search the skill-doc index by similarity and return matching docs.

        Returns a JSON list of {name, description, similarity, content}
        objects sorted by descending similarity. When the index isn't
        available (static-retrieval mode), returns a noop notice so the
        agent's turn isn't poisoned by an error.
        """
        if self._skill_doc_index is None:
            return json.dumps({
                "status": "noop",
                "message": (
                    "query_skills is only available when --retrieval "
                    "hybrid is on. In static mode the full Domain "
                    "Knowledge section is already in the system prompt."
                ),
            })
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"})
        try:
            k = int(tool_input.get("k", 3))
        except (TypeError, ValueError):
            k = 3
        k = max(1, min(5, k))
        try:
            hits = self._skill_doc_index.retrieve(query, k=k)
        except Exception as exc:
            return json.dumps({"error": f"retrieval failed: {type(exc).__name__}: {exc}"})
        out = [
            {
                "name": h.name,
                "description": h.description,
                "similarity": round(h.similarity, 4),
                "content": h.content,
            }
            for h in hits
        ]
        return json.dumps({"status": "ok", "hits": out})
