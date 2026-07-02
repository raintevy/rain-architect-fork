"""Multi-turn correction session for the ARCHITECT interactive CLI.

CorrectionSession maintains the full Claude conversation history across
corrections so the model has memory of all prior changes when applying
new ones. Manual edits are tracked in program history but not in the
message history — the current program is always included explicitly in
each correction turn so the LLM sees manual edits too.
"""

from __future__ import annotations

from architect.llm.client import call_claude
from architect.llm.prompts import correction_system_prompt, format_ee_pose


def _strip_fences(program: str) -> str:
    stripped = program.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return program


class CorrectionSession:
    """Manages iterative program refinement with undo support.

    History tracks both Claude corrections and manual edits so undo
    works uniformly across both. Only Claude turns are stored in the
    message history used for LLM memory.
    """

    def __init__(
        self,
        initial_program: str,
        model: str,
        *,
        api_spec: str | None = None,
        robot_label: str | None = None,
    ) -> None:
        if api_spec is None or robot_label is None:
            # Callers that don't pass a robot config fall back to the default
            # Franka configuration.
            from architect.robots import get_robot_config
            cfg = get_robot_config("franka")
            api_spec = api_spec or cfg.api_spec
            robot_label = robot_label or cfg.display_name
        self._system_prompt = correction_system_prompt(api_spec, robot_label)
        self._model = model
        self._program_history: list[str] = [initial_program]
        # "initial" | "claude" | "edit"
        self._source: list[str] = ["initial"]
        # alternating user / assistant dicts for multi-turn memory
        self._message_history: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_program(self) -> str:
        return self._program_history[-1]

    @property
    def can_undo(self) -> bool:
        return len(self._program_history) > 1

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def apply_correction(self, correction: str, current_ee_pose: dict | None = None) -> str:
        """Refine the current program via Claude using the given correction.

        The full current program is always included in the user message so that
        manual edits made outside the LLM are visible to Claude. The accumulated
        message history gives Claude memory of all prior corrections.

        Returns the refined program string.
        """
        parts = [
            f"## Current Program\n\n```python\n{self.current_program}\n```",
        ]
        if current_ee_pose is not None:
            parts.append(f"## Current EE Pose\n\n{format_ee_pose(current_ee_pose)}")
        parts.append(f"## Correction\n\n{correction}")

        user_msg = {"role": "user", "content": "\n\n".join(parts)}
        self._message_history.append(user_msg)

        all_messages = (
            [{"role": "system", "content": self._system_prompt}]
            + self._message_history
        )
        raw = call_claude(all_messages, model=self._model)
        refined = _strip_fences(raw)

        self._message_history.append({"role": "assistant", "content": refined})
        self._program_history.append(refined)
        self._source.append("claude")
        return refined

    def apply_edit(self, edited_program: str) -> None:
        """Record a manual edit.

        Does not update message_history — the next apply_correction call will
        include the edited program explicitly so the LLM sees it.
        """
        self._program_history.append(edited_program)
        self._source.append("edit")

    def undo(self) -> str | None:
        """Undo the most recent correction or edit.

        Returns the restored program, or None if there is nothing to undo.
        """
        if not self.can_undo:
            return None
        source = self._source.pop()
        self._program_history.pop()
        if source == "claude" and len(self._message_history) >= 2:
            self._message_history.pop()  # assistant turn
            self._message_history.pop()  # user turn
        return self.current_program
