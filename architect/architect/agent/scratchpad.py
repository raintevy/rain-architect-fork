"""Session-scoped working memory ('scratchpad') for the agentic loop.

Follows the two-layer pattern Raschka describes for AI coding agents
(magazine.sebastianraschka.com/p/components-of-a-coding-agent):

  * **Full transcript** — every user request, tool output, and LLM
    response. Already present as ``AgenticSession._message_history``.
  * **Working memory (this module)** — a smaller, explicitly-maintained
    summary layer focused on "task continuity": the current task,
    what the agent currently believes is happening, what it has
    invalidated, and key observations carried across corrections.

The scratchpad is *agent-driven*: the LLM emits ``update_scratchpad``
tool calls to mutate the fields, picking what's worth remembering for
the next correction. This matches Raschka's "distilled state" framing
where the LLM is best positioned to decide what's load-bearing — as
opposed to a harness-side auto-summariser that has to guess.

Persistence is in-session only — the scratchpad is created fresh at
session start (when the feature flag is on) and dropped at ``done``.
Cross-session persistence would conflate "this task's working
memory" with "stuff to remember about the robot in general"; the
latter belongs in the skill library, not here.

Schema (kept deliberately small — Raschka warns against bloated
working memory):

  current_goal              str        — what the agent thinks it's solving
  working_hypotheses        list[str]  — current beliefs about cause/fix
  invalidated_hypotheses    list[str]  — disproved theories, with reason
  key_observations          list[str]  — cross-correction facts worth remembering
  correction_count_at_update int       — last correction count when update fired
                                         (used by the harness's forced-refresh fallback)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


# Cap on each list field so a runaway agent can't bloat the prompt. If
# the agent appends past the cap, the oldest entries are evicted in
# FIFO order — the assumption is recent observations are more relevant
# than ancient ones for an in-session scratchpad.
_MAX_LIST_LENGTH = 8


@dataclass
class SessionScratchpad:
    """The agent's working memory for one session.

    Always carries the original ``instruction`` so the agent doesn't
    lose sight of what it was asked to do (Raschka's "current task" in
    the working-memory definition). ``current_goal`` can drift from
    that (e.g. "currently focused on fixing the grasp before tackling
    placement") so we keep them as separate fields.
    """

    instruction: str
    current_goal: str = ""
    working_hypotheses: list[str] = field(default_factory=list)
    invalidated_hypotheses: list[str] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    # Bookkeeping for the harness's forced-refresh fallback: the
    # correction count at the time of the last successful update.
    # Compared against the current correction count to decide whether
    # to force a refresh. Not surfaced in to_markdown — internal.
    correction_count_at_update: int = 0

    # ------------------------------------------------------------------
    # Mutation (called from the update_scratchpad tool handler)
    # ------------------------------------------------------------------

    def apply_update(
        self,
        *,
        current_goal: str | None = None,
        add_observations: list[str] | None = None,
        add_working_hypotheses: list[str] | None = None,
        invalidate_hypotheses: list[str] | None = None,
        correction_count: int | None = None,
    ) -> dict[str, int]:
        """Apply the agent-emitted update. Returns a small status dict the
        tool handler echoes back to the model so it can verify what landed.

        Field semantics:
          * ``current_goal`` (str): replaces current_goal (or no-op if None)
          * ``add_observations`` / ``add_working_hypotheses``: appended;
            duplicates are deduplicated against the existing list (case-
            insensitive exact-string match) so repeated turns don't bloat
            the scratchpad with the same string.
          * ``invalidate_hypotheses``: matched against working_hypotheses
            (case-insensitive substring match — Raschka-style soft match
            so the agent doesn't need to remember exact prior wording),
            moved to invalidated_hypotheses.
        """
        status = {
            "current_goal_set": 0,
            "observations_added": 0,
            "hypotheses_added": 0,
            "hypotheses_invalidated": 0,
            "list_evictions": 0,
        }

        if current_goal is not None and current_goal.strip():
            self.current_goal = current_goal.strip()
            status["current_goal_set"] = 1

        if add_observations:
            existing_lower = {o.lower() for o in self.key_observations}
            for obs in add_observations:
                if not obs or not obs.strip():
                    continue
                if obs.lower() in existing_lower:
                    continue
                self.key_observations.append(obs.strip())
                existing_lower.add(obs.lower())
                status["observations_added"] += 1
            status["list_evictions"] += _enforce_cap(self.key_observations)

        if add_working_hypotheses:
            existing_lower = {h.lower() for h in self.working_hypotheses}
            for h in add_working_hypotheses:
                if not h or not h.strip():
                    continue
                if h.lower() in existing_lower:
                    continue
                self.working_hypotheses.append(h.strip())
                existing_lower.add(h.lower())
                status["hypotheses_added"] += 1
            status["list_evictions"] += _enforce_cap(self.working_hypotheses)

        if invalidate_hypotheses:
            for needle in invalidate_hypotheses:
                if not needle or not needle.strip():
                    continue
                needle_lower = needle.lower().strip()
                # Soft match: any working hypothesis whose lowercased form
                # contains the needle, or is contained by it. Catches small
                # rewording across turns.
                matched_idx: list[int] = []
                for i, h in enumerate(self.working_hypotheses):
                    hl = h.lower()
                    if needle_lower in hl or hl in needle_lower:
                        matched_idx.append(i)
                if not matched_idx:
                    # Nothing matched in working_hypotheses; still record
                    # it as an invalidated belief so the agent's intent
                    # isn't lost.
                    if needle.strip().lower() not in {x.lower() for x in self.invalidated_hypotheses}:
                        self.invalidated_hypotheses.append(needle.strip())
                        status["hypotheses_invalidated"] += 1
                else:
                    # Remove in reverse so indices stay valid
                    for i in reversed(matched_idx):
                        h = self.working_hypotheses.pop(i)
                        if h.lower() not in {x.lower() for x in self.invalidated_hypotheses}:
                            self.invalidated_hypotheses.append(h)
                            status["hypotheses_invalidated"] += 1
            status["list_evictions"] += _enforce_cap(self.invalidated_hypotheses)

        if correction_count is not None:
            self.correction_count_at_update = correction_count

        return status

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Compact markdown rendering for prompt injection.

        Empty fields are omitted from the rendering so the section
        doesn't bloat the prompt before the agent has populated it.
        Returns an empty string when the scratchpad is wholly empty
        (apart from instruction); the caller should skip injection in
        that case.
        """
        lines: list[str] = []
        lines.append(f"**Original task:** {self.instruction}")
        if self.current_goal:
            lines.append(f"**Current focus:** {self.current_goal}")
        if self.working_hypotheses:
            lines.append("")
            lines.append("**Working hypotheses (not yet confirmed):**")
            for h in self.working_hypotheses:
                lines.append(f"- {h}")
        if self.invalidated_hypotheses:
            lines.append("")
            lines.append("**Invalidated hypotheses (tried and disproved — do not revisit):**")
            for h in self.invalidated_hypotheses:
                lines.append(f"- {h}")
        if self.key_observations:
            lines.append("")
            lines.append("**Key observations carried across corrections:**")
            for o in self.key_observations:
                lines.append(f"- {o}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """True iff nothing beyond the original instruction has been recorded."""
        return not any([
            self.current_goal,
            self.working_hypotheses,
            self.invalidated_hypotheses,
            self.key_observations,
        ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _enforce_cap(lst: list[str]) -> int:
    """Trim list to :data:`_MAX_LIST_LENGTH` (FIFO). Returns # evicted."""
    if len(lst) <= _MAX_LIST_LENGTH:
        return 0
    n_evicted = len(lst) - _MAX_LIST_LENGTH
    del lst[:n_evicted]
    return n_evicted
