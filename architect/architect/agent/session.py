"""Agentic synthesis loop for ARCHITECT.

Claude runs in a tool-use loop where it can:
  - introspect the robot API
  - query live robot / scene state
  - register new reusable primitives (with user approval)
  - run ROS2 CLI commands (with user approval)
  - submit the final program

The loop terminates when Claude calls submit_program() or exceeds
MAX_LOOP_ITERATIONS.  The session maintains full message history across
generate + correction turns so Claude remembers prior context.
"""

from __future__ import annotations

import re
import secrets
import threading
from pathlib import Path

import time

from architect.agent.tools import TOOL_SCHEMAS, ToolExecutor
from architect.library.context import (
    ContextChannels,
    ContextProfile,
    build_context,
    build_context_with_metrics,
)
from architect.llm.client import call_claude_with_tools
from architect.agent.primitive_registry import PrimitiveRegistry
from architect.llm.prompts import (
    build_agentic_correction_message,
    build_agentic_generation_message,
    build_agentic_system_prompt,
    build_continuation_message,
)
from architect.corrections.llm_scorer import emit_scorer_and_invariants

MAX_LOOP_ITERATIONS = 30


def _extract_code_block(text: str) -> str | None:
    """Pull the first fenced code block out of a text response, if any."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


class AgenticSession:
    """Manages the full generate → correct lifecycle with tool-use memory.

    Message history spans the whole session so Claude has context of all
    prior tool calls and corrections when applying new ones.
    """

    def __init__(
        self,
        model: str,
        registry: PrimitiveRegistry,
        namespace: dict,
        console,
        approve_primitive_fn,
        dry_run: bool = False,
        api_spec: str = "",
        robot_label: str = "",
        robot_name: str = "franka",
        skills_dir: Path | None = None,
        emit_scorers: bool = True,
        emit_subtask_docs: bool = True,
        emit_fault_summary: bool = True,
        emit_execution_trace: bool = True,
        inprogram_retry: bool = False,
        task_spec: dict | None = None,
        context_channels: ContextChannels | None = None,
        context_profile_path: Path | str | None = None,
        failure_store=None,                       # FailureStore | None
        failure_diagnosis_enabled: bool = False,
        scratchpad_enabled: bool = False,
        retrieval_mode: str = "static",            # "static" | "hybrid"
        retrieval_k: int = 3,
        retrieval_instruction: str | None = None,
    ) -> None:
        self._model = model
        self._registry = registry
        self._namespace = namespace
        self._console = console
        self._approve_primitive_fn = approve_primitive_fn
        self._dry_run = dry_run
        self._api_spec = api_spec
        self._robot_label = robot_label
        self._robot_name = robot_name
        # Stored so downstream emission paths (subtask docs in particular)
        # can branch on the condition: PatternReuse swaps the subtask-doc
        # emitter's create-vs-update logic from LLM-judgment to
        # embedding-similarity. Default False preserves ICL / FuncReuse /
        # SlateGated behavior unchanged.
        self._inprogram_retry = inprogram_retry
        # Hybrid retrieval: when retrieval_mode='hybrid', the system
        # prompt's Domain Knowledge section is assembled from
        # always-loaded docs + top-k retrieved-by-similarity docs
        # against retrieval_instruction (typically the task instruction).
        # Static mode preserves the current static-dump behaviour.
        self._retrieval_mode = retrieval_mode
        self._retrieval_k = retrieval_k
        # Build a doc index ahead of time when in hybrid mode so the
        # query_skills tool can call into it without re-indexing.
        self._skill_doc_index = None
        if retrieval_mode == "hybrid" and skills_dir is not None:
            try:
                from architect.library.skill_doc_index import SkillDocIndex
                self._skill_doc_index = SkillDocIndex(skills_dir)
                self._skill_doc_index.ensure_built()
            except Exception:
                # Best-effort: a failed index build means the agent
                # gets the static dump + no query_skills tool, but the
                # session still runs.
                self._skill_doc_index = None
        self._system_prompt = build_agentic_system_prompt(
            robot_label=robot_label, skills_dir=skills_dir,
            inprogram_retry=inprogram_retry,
            retrieval_instruction=(
                retrieval_instruction if retrieval_mode == "hybrid" else None
            ),
            retrieval_k=retrieval_k,
        )
        self._emit_scorers = emit_scorers
        self._emit_subtask_docs = emit_subtask_docs
        # Foreground LLM call before the agentic loop that produces a
        # three-line fault-localisation summary from the prior
        # execution trace + the user's correction text. Disabled in
        # tests / dry-runs by passing emit_fault_summary=False; the
        # agentic loop still works without the summary (it just sees
        # the raw trace instead).
        self._emit_fault_summary = emit_fault_summary
        # When False, the raw execution-trace section is omitted from the
        # correction prompt — the agent sees only the user's correction
        # text + the current program, not the call-by-call record of
        # what happened last run. The trace is still *captured* (so
        # vqa_judge / explain_failure can score on it) — only the prompt
        # injection is suppressed. Default True preserves trace-as-
        # feedback behaviour; set False for clean ablations where you
        # want to measure the LLM's correction quality without prior-
        # run context leaking in.
        self._emit_execution_trace = emit_execution_trace
        # tasks.json-style success_scorer spec, attached for post-execution
        # scoring. ``None`` means no scorer is configured — the CLI will
        # silently skip the pass/fail line after each execution. When set,
        # the spec is consumed by :func:`architect.eval.task_scorers.score_trace`
        # via the CLI's ``_run`` after each program execution. The session
        # itself doesn't act on the spec; it just carries it for the
        # caller to use post-exec.
        self._task_spec: dict | None = task_spec
        # The score (float in [0, 1] or None) returned by score_trace on the
        # most recent execution. Single-slot like _last_execution_trace; the
        # CLI's correction loop reads this after every _run() call to decide
        # whether to auto-exit (PASS) or keep prompting (FAIL / no spec).
        # The session itself does not act on this — it's caller-managed,
        # mirroring how _last_execution_trace is set externally.
        self._last_task_score: float | None = None
        # Cache the per-robot skills dir so background subtask-doc emission
        # can resolve `subtasks/` under it without re-passing the path.
        # ``None`` is valid — in-memory sessions / tests don't write docs.
        self._skills_dir: Path | None = skills_dir
        # Random 8-hex session id, prepended to every scorer/invariant name
        # this session emits. Without this, the per-session c{idx}_ counter
        # restarts from 1 each CLI run, and register_scorer's INSERT-OR-UPDATE
        # silently overwrites earlier sessions' artifacts when correction
        # indices line up. The session id is intentionally not persisted —
        # nothing downstream needs to query *by* session, the prefix is just
        # a uniqueness guard on scorers.name.
        self._session_id: str = secrets.token_hex(4)
        # P6: budgets / toggles for build_context. ``None`` uses defaults
        # (4k tool-output clip, dedupe on, top-3 retrieval on graduated skills).
        self._context_channels = context_channels or ContextChannels()
        # Optional per-turn context-fill profile (one JSONL line per LLM call).
        # Used by scripts/eval/profile_session.py to produce the §7 step-8
        # stress-test artifacts: tokens-per-call curve, per-channel
        # attribution, lost-context-rate proxy. Disabled by default — only
        # the --profile-context CLI flag and explicit constructor argument
        # turn it on.
        self._context_profile: ContextProfile | None = (
            ContextProfile(context_profile_path) if context_profile_path else None
        )
        # Full multi-turn conversation history (tool use + results included)
        self._message_history: list[dict] = []
        # Program produced at each generate / correct step
        self._program_history: list[str] = []
        # "claude" | "edit" — controls undo behavior
        self._source: list[str] = []
        # Index into _message_history before each claude-turn (for undo)
        self._history_checkpoints: list[int] = []
        # Background scorer-emission threads, kept around so we can join() on
        # shutdown without deadlocking the CLI.
        self._scorer_threads: list[threading.Thread] = []
        # Background subtask-doc-emission threads. Same pattern as
        # _scorer_threads — daemon thread per correction, joined on
        # shutdown so an in-flight write isn't killed mid-file.
        self._subtask_doc_threads: list[threading.Thread] = []
        # Single-slot store of the most-recent ExecutionTrace.
        # Populated by architect_cli._run via set_last_execution_trace after
        # every live execution; rendered into the next correction's
        # prompt as ``## Last Execution Trace`` so Claude can reason
        # about what the program actually did rather than guess from the
        # source + a single post-execution pose.
        self._last_execution_trace = None
        # Stashed by generate() so continue_from_failure() can include the
        # original task instruction in continuation prompts.
        self._instruction: str | None = None
        # Failure-diagnosis + store state (P? — agent_memory_features).
        # Both default to off; the CLI flips them on per --failure-store /
        # the diagnosis-enabled toggle. ``failure_store`` is duck-typed
        # to avoid a hard dep on architect.library.failure_store at construction
        # time (tests can pass any object with record_failure /
        # mark_outcome / lookup_similar / update_with_correction).
        self._failure_store = failure_store
        self._failure_diagnosis_enabled = failure_diagnosis_enabled
        # ID of the most recently recorded failure (None until a FAIL is
        # diagnosed + recorded). Used to mark outcome ('fixed' /
        # 'still_failed') on the *next* run, and to attach the
        # correction text once correct() returns. Reset to None on every
        # PASS so an old failure can't be marked twice.
        self._last_failure_id: int | None = None
        # Most recent FailureDiagnosis dataclass, mirroring last_failure_id
        # but holding the structured diagnosis itself so the closed-loop
        # path can use it as correction text without re-querying the
        # store. None when no diagnosis has been run for the current
        # trial.
        self._last_diagnosis = None
        # Scratchpad: agent-driven working memory carried across turns.
        # Constructed lazily on first need (in :meth:`set_instruction` or
        # at the first generate() / correct() call) so a disabled flag
        # incurs zero cost. ``_correction_count`` is bumped on every
        # successful correction; the scratchpad's apply_update stamps
        # this on the row so the harness can detect "agent hasn't
        # updated working memory in N rounds" if we later add a
        # forced-refresh fallback.
        self._scratchpad_enabled = scratchpad_enabled
        self._scratchpad = None
        self._correction_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_program(self) -> str | None:
        return self._program_history[-1] if self._program_history else None

    @property
    def can_undo(self) -> bool:
        return bool(self._program_history)

    @property
    def task_spec(self) -> dict | None:
        """The task's ``success_scorer`` spec, or ``None`` if not configured.

        Read-only; set once at construction. The CLI's ``_run`` uses this
        to decide whether to call :func:`architect.eval.task_scorers.score_trace`
        on the last execution trace and print a PASS / FAIL verdict.
        """
        return self._task_spec

    @property
    def last_task_score(self) -> float | None:
        """Score returned by ``score_trace`` on the most recent execution.

        ``None`` when no task_spec is set, when the most recent execution
        was an exec failure, or when the spec was unrecognised. ``>= 0.5``
        means the task's ``task_complete`` predicate accepted the trial.
        Reset by :meth:`set_last_task_score` after each ``_run`` in the CLI.
        """
        return self._last_task_score

    def set_last_task_score(self, score: float | None) -> None:
        """Stash the score for the most recent execution. CLI-managed."""
        self._last_task_score = score

    @property
    def failure_store(self):
        """Optional FailureStore for persisting + retrieving FAIL diagnoses."""
        return self._failure_store

    @property
    def failure_diagnosis_enabled(self) -> bool:
        return self._failure_diagnosis_enabled

    @property
    def last_failure_id(self) -> int | None:
        return self._last_failure_id

    def set_last_failure_id(self, failure_id: int | None) -> None:
        self._last_failure_id = failure_id

    @property
    def last_diagnosis(self):
        """Most recent FailureDiagnosis, or None.

        Set by the CLI's ``_run`` after a FAIL when failure_diagnosis is
        enabled; cleared on PASS. The closed-loop correction path reads
        this to use the structured diagnosis as feedback (richer than
        the keyword-based ``explain_failure`` fallback).
        """
        return self._last_diagnosis

    def set_last_diagnosis(self, diagnosis) -> None:
        self._last_diagnosis = diagnosis

    @property
    def scratchpad(self):
        """The session's working-memory scratchpad, or None when disabled."""
        return self._scratchpad

    @property
    def scratchpad_enabled(self) -> bool:
        return self._scratchpad_enabled

    def _ensure_scratchpad(self, instruction: str) -> None:
        """Lazy-construct the scratchpad on first generate/correct.

        Idempotent — subsequent calls with a non-empty instruction
        update the stored instruction (rare, but useful when generate
        is called multiple times with different instructions in the
        same session). No-op when the feature is off.
        """
        if not self._scratchpad_enabled:
            return
        if self._scratchpad is None:
            from architect.agent.scratchpad import SessionScratchpad
            self._scratchpad = SessionScratchpad(instruction=instruction or "")
        elif instruction and not self._scratchpad.instruction:
            self._scratchpad.instruction = instruction

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, instruction: str | None = None) -> str | None:
        """Run the agentic loop for initial program generation."""
        self._instruction = instruction
        self._ensure_scratchpad(instruction or "")
        msg = build_agentic_generation_message(instruction=instruction)
        return self._run_loop(msg, query_text=instruction)

    def set_last_execution_trace(self, trace) -> None:
        """Stash the most recent live-execution trace for the next correction.

        Called by architect_cli._run after the program runs. The trace becomes
        the ``## Last Execution Trace`` section in the next correction
        message — giving Claude a structured view of what the program
        actually did, not just the source code + a single
        post-execution pose. Single-slot: previous trace is dropped.
        """
        self._last_execution_trace = trace

    def correct(self, correction: str, current_ee_pose: dict | None = None) -> str | None:
        """Run the agentic loop to refine the current program.

        On success, schedule a *background* LLM call to operationalize the
        correction as a scorer + invariant artifacts (P3). The thread is
        daemonized so it never blocks the user's correction loop; failures
        are logged to the console but never raised. Disable with
        ``AgenticSession(emit_scorers=False)`` in tests / dry-runs.

        Trace injection: when a prior live execution stashed a trace
        via :meth:`set_last_execution_trace`, the trace is rendered
        into the correction message under ``## Last Execution Trace``
        so Claude can reason about the call sequence + per-step state,
        not just the source + a single post-execution pose.

        Fault-localisation summary: when ``emit_fault_summary`` is
        enabled and a trace is available, a foreground LLM call
        produces a three-line summary that gets rendered into the
        correction message under ``## Fault Localisation Summary``
        *above* the trace. The agentic loop reads the summary first
        and consults the raw trace only when it needs evidence beyond
        the hint. Best-effort: a failed summary call drops to ``None``
        and the message is built without the section, never blocking
        the correction.
        """
        self._correction_count += 1
        self._ensure_scratchpad("")
        fault_summary = self._compute_fault_summary(correction)

        msg = build_agentic_correction_message(
            correction, self.current_program, current_ee_pose,
            execution_trace=(
                self._last_execution_trace if self._emit_execution_trace else None
            ),
            fault_summary=fault_summary,
            scratchpad=(
                self._scratchpad.to_markdown()
                if self._scratchpad is not None and not self._scratchpad.is_empty()
                else None
            ),
        )
        # Snapshot the prior program *before* _run_loop appended the new one,
        # so the subtask-doc emission can show the LLM what the correction
        # changed.
        prior_program = (
            self._program_history[-2] if len(self._program_history) >= 2 else None
        )
        program = self._run_loop(msg, query_text=correction)
        if program is not None and self._emit_scorers:
            self._spawn_scorer_emission(correction, program)
        if program is not None and self._emit_subtask_docs:
            self._spawn_subtask_doc_emission(correction, program, prior_program)
        # Failure-store correction attachment. When a failure was just
        # recorded (set_last_failure_id called by the CLI's _run after a
        # FAIL), pin the correction text + refined program to that row.
        # Outcome will be set on the *next* run by _run when it sees the
        # new PASS / FAIL verdict. Best-effort: a failed store update
        # must not break the correction.
        if (
            program is not None
            and self._failure_store is not None
            and self._last_failure_id is not None
        ):
            try:
                self._failure_store.update_with_correction(
                    self._last_failure_id,
                    correction_text=correction,
                    program_after=program,
                )
            except Exception:
                pass
        return program

    def continue_from_failure(
        self,
        correction: str,
        interrupt: "MidExecInterrupt",
        current_ee_pose: dict | None = None,
    ) -> str | None:
        """Generate a continuation program after a mid-execution interrupt.

        Used by ``--hil`` and ``--vlm`` modes. Builds a continuation
        prompt that includes the original instruction, what succeeded so
        far (from the partial trace), what failed (the VQA question +
        answer), and the correction. Runs the agentic tool-use loop and
        returns the continuation program.

        Spawns subtask doc emission (same as :meth:`correct`) so the
        correction is captured as a persistent skill.
        """
        from architect.midexec import MidExecInterrupt  # noqa: F811

        self._correction_count += 1
        self._ensure_scratchpad("")
        fault_summary = self._compute_fault_summary(correction)

        msg = build_continuation_message(
            instruction=self._instruction or "(no instruction)",
            correction=correction,
            vqa_question=interrupt.vqa_question,
            vqa_answer=interrupt.vqa_answer,
            current_ee_pose=current_ee_pose,
            execution_trace=(
                self._last_execution_trace if self._emit_execution_trace else None
            ),
            fault_summary=fault_summary,
            scratchpad=(
                self._scratchpad.to_markdown()
                if self._scratchpad is not None and not self._scratchpad.is_empty()
                else None
            ),
        )
        prior_program = (
            self._program_history[-2] if len(self._program_history) >= 2 else None
        )
        program = self._run_loop(msg, query_text=correction)
        if program is not None and self._emit_subtask_docs:
            self._spawn_subtask_doc_emission(correction, program, prior_program)
        return program

    def join_pending_scorers(self, timeout: float | None = None) -> None:
        """Wait for outstanding scorer-emission threads to finish.

        Called on session shutdown so artifacts in flight don't get killed
        with the daemon thread mid-write. Pass ``timeout=None`` to block
        indefinitely (typical CLI exit) or a small float to drop pending
        emissions on a forced quit.
        """
        for t in list(self._scorer_threads):
            if t.is_alive():
                t.join(timeout)
        self._scorer_threads = [t for t in self._scorer_threads if t.is_alive()]

    def join_pending_subtask_docs(self, timeout: float | None = None) -> None:
        """Wait for outstanding subtask-doc-emission threads to finish.

        Same shape as :meth:`join_pending_scorers` — called on session
        shutdown so an in-flight markdown write isn't killed mid-file.
        """
        for t in list(self._subtask_doc_threads):
            if t.is_alive():
                t.join(timeout)
        self._subtask_doc_threads = [
            t for t in self._subtask_doc_threads if t.is_alive()
        ]

    def _compute_fault_summary(self, correction: str) -> str | None:
        """Foreground LLM call → three-line fault-localisation summary, or ``None``.

        Shared by :meth:`correct` and :meth:`correct_with_slate` so both
        paths get the same summary on the same correction (and a single
        bug fix to the surrounding control flow lands in one place).
        Returns ``None`` when ``emit_fault_summary`` is disabled, when
        there's no prior execution trace to localise against, or when
        the underlying LLM call fails — none of which should block the
        correction. Successful summaries print a one-line dim
        ``[fault-summary] localised`` confirmation; the full body is
        rendered into the LLM message, not reprinted to the console.
        """
        if not self._emit_fault_summary or self._last_execution_trace is None:
            return None
        from architect.corrections.fault_localization import summarize_fault
        try:
            summary = summarize_fault(
                correction,
                self.current_program,
                self._last_execution_trace,
                model=self._model,
            )
        except Exception as exc:
            # Best-effort — log + fall through so a flaky summary call
            # doesn't block the correction. summarize_fault already
            # swallows LLM-level errors; this catch is for unexpected
            # import/build failures.
            self._console.print(
                f"[dim]\\[fault-summary] skipped: "
                f"{type(exc).__name__}: {exc}[/dim]"
            )
            return None
        if summary:
            self._console.print("[dim]\\[fault-summary] localised[/dim]")
        return summary

    def _spawn_scorer_emission(self, correction: str, program: str) -> None:
        """Fire-and-forget LLM call → ScorerArtifacts → SkillStore."""
        store = getattr(self._registry, "store", None)
        if store is None:
            return  # in-memory registry without a persistent store; nothing to do

        # Anchor artifact names to (session_id, correction_index). Session id
        # prevents cross-session collisions on scorers.name (different CLI
        # runs both numbering from c1_); correction index keeps within-session
        # ordering legible and avoids same-session UNIQUE violations.
        idx = max(0, len(self._program_history) - 1)
        prefix = f"{self._session_id}_c{idx}_"
        model = self._model
        console = self._console

        def _emit() -> None:
            try:
                scorer, invariants = emit_scorer_and_invariants(
                    correction,
                    current_program=program,
                    llm_call=None,  # use real call_claude
                    model=model,
                    name_prefix=prefix,
                )
            except Exception as exc:
                console.print(
                    f"[dim]\\[scorer] emission failed: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
                return

            wrote = 0
            try:
                if scorer is not None:
                    store.register_scorer(
                        scorer.name, scorer.code, scorer.description,
                        kind="scorer",
                        preamble=scorer.preamble,
                        provenance=scorer.provenance,
                    )
                    wrote += 1
                for inv in invariants:
                    store.register_scorer(
                        inv.name, inv.code, inv.description,
                        kind="invariant",
                        preamble=inv.preamble,
                        provenance=inv.provenance,
                    )
                    wrote += 1
            except Exception as exc:
                console.print(
                    f"[dim]\\[scorer] store write failed: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
                return

            if wrote:
                kinds = (
                    f"1 scorer + {len(invariants)} invariant"
                    + ("s" if len(invariants) != 1 else "")
                ) if scorer is not None else (
                    f"{len(invariants)} invariant"
                    + ("s" if len(invariants) != 1 else "")
                )
                console.print(f"[dim]\\[scorer] {kinds} saved[/dim]")

        t = threading.Thread(target=_emit, name=f"scorer-emit-{idx}", daemon=True)
        self._scorer_threads.append(t)
        t.start()

    def _spawn_subtask_doc_emission(
        self,
        correction: str,
        program: str,
        prior_program: str | None,
    ) -> None:
        """Fire-and-forget LLM call → SubtaskDoc → ``skills/<robot>/subtasks/<name>.md``.

        Same daemon-thread pattern as :meth:`_spawn_scorer_emission`. The
        thread runs an LLM call that classifies the correction as
        create/update/skip on a per-subtask basis, then writes the
        markdown file when appropriate. Failures are logged dim and
        never raised — a broken emission must not crash the user's
        correction loop. Disabled by ``AgenticSession(emit_subtask_docs=
        False)`` in tests / dry-runs that don't want filesystem side
        effects.

        Skipped silently when ``skills_dir`` wasn't supplied at construction
        — there's no per-robot dir to write into.
        """
        if self._skills_dir is None:
            return
        subtasks_dir = Path(self._skills_dir) / "subtasks"
        idx = max(0, len(self._program_history) - 1)
        model = self._model
        console = self._console
        # PatternReuse: swap the LLM-judgment create-vs-update path for
        # the embedding-similarity directive. Other conditions keep the
        # LLM-judgment path (use_similarity_directive=False).
        use_sim = self._inprogram_retry

        def _emit() -> None:
            try:
                from architect.corrections.subtask_docs import emit_subtask_doc, persist_subtask_doc
                doc, raw_response = emit_subtask_doc(
                    correction,
                    current_program=program,
                    prior_program=prior_program,
                    subtasks_dir=subtasks_dir,
                    model=model,
                    use_similarity_directive=use_sim,
                )
            except Exception as exc:
                console.print(
                    f"[dim]\\[subtask] emission failed: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
                return

            if doc is None:
                # Surface a snippet of what the LLM actually emitted so
                # the failure mode is debuggable — previously the user
                # saw only "parse / validation failed" with no signal.
                snippet = " ".join(raw_response.split())[:200]
                console.print(
                    f"[dim]\\[subtask] no doc emitted (parse / validation "
                    f"failed). Response snippet: {snippet!r}[/dim]"
                )
                return
            if doc.action == "skip":
                console.print(
                    f"[dim]\\[subtask] skipped — correction is too narrow "
                    f"for a doc[/dim]"
                )
                return

            try:
                path = persist_subtask_doc(doc, subtasks_dir)
            except Exception as exc:
                console.print(
                    f"[dim]\\[subtask] persist failed: "
                    f"{type(exc).__name__}: {exc}[/dim]"
                )
                return

            if path is not None:
                console.print(
                    f"[dim]\\[subtask] {doc.action}: {path.name}[/dim]"
                )

        t = threading.Thread(
            target=_emit, name=f"subtask-doc-emit-{idx}", daemon=True,
        )
        self._subtask_doc_threads.append(t)
        t.start()

    def apply_edit(self, edited_program: str) -> None:
        """Record a manual edit (does not touch message history)."""
        self._program_history.append(edited_program)
        self._source.append("edit")

    def correct_with_slate(
        self,
        correction: str,
        *,
        n: int = 3,
        current_ee_pose: dict | None = None,
        base_scene: object | None = None,
        robot_name: str = "franka",
        k_probes: int = 8,
        seed: int | None = None,
    ):
        """Active counterfactual sampling path (C1).

        Calls the LLM once for ``n`` behaviorally-different candidates,
        runs the probe + scorer pipeline on each, and returns a
        :class:`architect.sampling.Slate`. The caller (typically
        :mod:`scripts.architect_cli`) uses ``slate.dominant`` to decide whether
        to auto-apply or surface the slate to the user via
        :func:`architect.diff_ui.prompt_pick`.

        Stored scorers + invariants are pulled from the SkillStore (when
        the registry is path-backed) and fed to the ranker so the C1
        likelihood signal compounds with C3's preference memory.
        """
        from architect.library.relevance import filter_artifacts
        from architect.probes.sampling import build_slate
        from architect.corrections.llm_scorer import ScorerArtifact

        store = getattr(self._registry, "store", None)
        scorer_artifacts: tuple[ScorerArtifact, ...] = ()
        invariant_artifacts: tuple[ScorerArtifact, ...] = ()
        if store is not None:
            try:
                scorer_artifacts = tuple(
                    ScorerArtifact(
                        name=r["name"], kind=r["kind"], code=r["code"],
                        description=r["description"],
                        preamble=r.get("preamble", "") or "",
                        provenance=r["provenance"] or {},
                    )
                    for r in store.list_scorers(kind="scorer")
                )
                invariant_artifacts = tuple(
                    ScorerArtifact(
                        name=r["name"], kind=r["kind"], code=r["code"],
                        description=r["description"],
                        preamble=r.get("preamble", "") or "",
                        provenance=r["provenance"] or {},
                    )
                    for r in store.list_scorers(kind="invariant")
                )
            except Exception:
                # Missing or unreadable scorers shouldn't kill the slate
                # path; fall back to unscored ranking.
                scorer_artifacts = ()
                invariant_artifacts = ()

        # Filter out stored artifacts whose description is irrelevant to the
        # active correction. Without this, every prior scorer feeds into
        # rank_candidates and rates candidates on criteria unrelated to the
        # current correction — see architect.library.relevance for context. Backend
        # defaults to "auto": prefer Azure embeddings, fall back to the
        # Jaccard lexical filter when the embedding deployment is
        # unreachable.
        kept_scorers, skipped_scorers = filter_artifacts(
            correction, scorer_artifacts, console=self._console,
        )
        kept_invariants, skipped_invariants = filter_artifacts(
            correction, invariant_artifacts, console=self._console,
        )
        scorer_artifacts = tuple(kept_scorers)
        invariant_artifacts = tuple(kept_invariants)
        self._report_filter(
            "scorer", skipped_scorers, kept_count=len(scorer_artifacts),
        )
        self._report_filter(
            "invariant", skipped_invariants, kept_count=len(invariant_artifacts),
        )

        # Same pre-correction summary the agentic correct() path uses.
        # The slate generator has no tool loop to investigate the trace
        # on its own, so the summary is especially load-bearing here:
        # it's the only condensed read on what went wrong that each
        # candidate's design choice can branch off.
        fault_summary = self._compute_fault_summary(correction)

        return build_slate(
            correction,
            current_program=self.current_program or "",
            n=n,
            current_ee_pose=current_ee_pose,
            execution_trace=(
                self._last_execution_trace if self._emit_execution_trace else None
            ),
            fault_summary=fault_summary,
            base_scene=base_scene,
            robot_name=robot_name,
            api_spec=self._api_spec,
            robot_label=self._robot_label,
            scorers=scorer_artifacts,
            invariants=invariant_artifacts,
            k_probes=k_probes,
            seed=seed,
            model=self._model,
        )

    def _report_filter(
        self,
        kind: str,
        skipped: list,
        *,
        kept_count: int,
    ) -> None:
        """Show the user which stored artifacts were filtered out of this slate."""
        total = kept_count + len(skipped)
        if total == 0:
            return
        if not skipped:
            return
        # Don't spam the console with every skipped artifact when many fire;
        # show the top few by overlap score so the user can sanity-check.
        top = sorted(skipped, key=lambda pair: -pair[1])[:4]
        preview = ", ".join(f"{a.name} ({s:.2f})" for a, s in top)
        more = f" +{len(skipped) - len(top)} more" if len(skipped) > len(top) else ""
        self._console.print(
            f"[dim]\\[slate] {kind} filter: {total}→{kept_count} "
            f"(skipped: {preview}{more})[/dim]"
        )

    def apply_slate_pick(self, candidate, *, correction: str | None = None) -> None:
        """Record a candidate from a slate as the new current program.

        Mirrors :meth:`apply_edit` but tags the source as ``"slate"`` so
        undo, history, and future episode-log queries can distinguish
        slate-driven changes from manual edits / agentic corrections.
        When ``correction`` is supplied, fires background scorer +
        subtask-doc emissions on the picked candidate's program — same
        mechanism as :meth:`correct`. Snapshots the prior program
        *before* appending so the subtask-doc emission sees the actual
        diff the correction induced.
        """
        prior_program = (
            self._program_history[-1] if self._program_history else None
        )
        self._program_history.append(candidate.code)
        self._source.append("slate")
        if correction and self._emit_scorers:
            self._spawn_scorer_emission(correction, candidate.code)
        if correction and self._emit_subtask_docs:
            self._spawn_subtask_doc_emission(
                correction, candidate.code, prior_program,
            )

    def undo(self) -> str | None:
        """Revert the most recent correction or edit.

        For Claude turns, the corresponding message history is also rolled back
        so the next correction doesn't see the undone turn.
        """
        if not self._program_history:
            return None
        source = self._source.pop()
        self._program_history.pop()
        if source == "claude" and self._history_checkpoints:
            checkpoint = self._history_checkpoints.pop()
            self._message_history = self._message_history[:checkpoint]
        return self.current_program

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run_loop(self, user_message: dict, *, query_text: str | None = None) -> str | None:
        """Tool-use loop: append user_message, iterate until submit_program.

        ``query_text`` is the user's natural-language correction (or instruction)
        that drives library retrieval in :func:`architect.context.build_context`.
        ``None`` disables retrieval for this loop — appropriate for internal
        synthetic loops where retrieval would add noise rather than signal.
        """
        checkpoint = len(self._message_history)
        self._message_history.append(user_message)

        executor = ToolExecutor(
            registry=self._registry,
            namespace=self._namespace,
            console=self._console,
            approve_primitive_fn=self._approve_primitive_fn,
            api_spec=self._api_spec,
            dry_run=self._dry_run,
            robot_name=self._robot_name,
            scratchpad=self._scratchpad,
            correction_count_getter=lambda: self._correction_count,
            skill_doc_index=self._skill_doc_index,
        )

        store = getattr(self._registry, "store", None)

        for iteration in range(MAX_LOOP_ITERATIONS):
            # P6: route all message assembly through build_context so
            # clipping + dedupe + retrieval are applied uniformly to every
            # turn. ``_message_history`` is *not* mutated — build_context
            # returns a fresh list with stale reads marked and oversized
            # tool outputs clipped. When profiling is on we also capture
            # per-turn metrics (per-channel byte counts, clipped/stale
            # savings) for the §7 step-8 stress-test artifacts.
            if self._context_profile is not None:
                all_messages, metrics = build_context_with_metrics(
                    system_prompt=self._system_prompt,
                    message_history=self._message_history,
                    correction_text=query_text,
                    store=store,
                    channels=self._context_channels,
                )
            else:
                all_messages = build_context(
                    system_prompt=self._system_prompt,
                    message_history=self._message_history,
                    correction_text=query_text,
                    store=store,
                    channels=self._context_channels,
                )
                metrics = None
            t_call_start = time.perf_counter()
            with self._console.status(
                f"[bold]Thinking[/bold] [dim](step {iteration + 1})[/dim]",
                spinner="dots",
            ):
                # Filter out tools whose underlying feature is off, so
                # the agent doesn't see no-op tools in its vocabulary.
                # Tools list is small enough that this filter is cheap
                # per turn.
                disabled: set[str] = set()
                if not self._scratchpad_enabled:
                    disabled.add("update_scratchpad")
                if self._skill_doc_index is None:
                    disabled.add("query_skills")
                tools_for_turn = (
                    TOOL_SCHEMAS if not disabled
                    else [t for t in TOOL_SCHEMAS if t["name"] not in disabled]
                )
                response = call_claude_with_tools(
                    all_messages, tools_for_turn, model=self._model
                )
            latency_s = time.perf_counter() - t_call_start

            if self._context_profile is not None and metrics is not None:
                usage = getattr(response, "usage", None)
                self._context_profile.record(
                    metrics,
                    input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                    output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                    latency_s=latency_s,
                    stop_reason=getattr(response, "stop_reason", None),
                    loop_iteration=iteration,
                    correction_text=query_text,
                )
            self._message_history.append(
                {"role": "assistant", "content": response.content}
            )

            if response.stop_reason == "end_turn":
                # Claude finished without calling submit_program — try extracting
                # a code block from the text as a fallback.
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                program = _extract_code_block(text) or (text.strip() or None)
                if program:
                    self._history_checkpoints.append(checkpoint)
                    self._program_history.append(program)
                    self._source.append("claude")
                    return program
                self._console.print(
                    "[yellow]Agent finished without submitting a program.[/yellow]"
                )
                self._message_history = self._message_history[:checkpoint]
                return None

            # --- Process tool calls ---
            tool_results = []
            submitted_program: str | None = None

            for block in response.content:
                if not hasattr(block, "type") or block.type != "tool_use":
                    continue

                if block.name == "submit_program":
                    submitted_program = block.input.get("code", "").strip()
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Program received.",
                    })
                else:
                    result = executor.execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                self._message_history.append(
                    {"role": "user", "content": tool_results}
                )

            if submitted_program is not None:
                self._history_checkpoints.append(checkpoint)
                self._program_history.append(submitted_program)
                self._source.append("claude")
                return submitted_program

        self._console.print(
            f"[red]Agent loop exceeded {MAX_LOOP_ITERATIONS} iterations — aborting.[/red]"
        )
        self._message_history = self._message_history[:checkpoint]
        return None
