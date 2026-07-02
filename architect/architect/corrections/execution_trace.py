"""Live-execution trace capture.

Wraps the execution namespace so that every control or perception call the
program makes during a live `_run` is recorded as a structured timeline:

    [
      {call_index, name, args, kwargs, state_after, timestamp,
       vlm_question, vlm_observation},
      ...
    ]

Plus the program's exit status (exec_ok / error / error_type) and a final
state snapshot. The next correction's prompt embeds this trace under a
``## Last Execution Trace`` section so Claude knows *what the program
actually did* — not just the source code + a single post-execution EE
pose. Without this, corrections are driven purely by the user's
free-text feedback and Claude has to guess at the failure mode.

The trace is recorded **per execution** and stored single-slot on the
:class:`architect.agent.AgenticSession`. Older traces are dropped — corrections
are about the *most recent* run. If a future feature wants a multi-run
timeline (e.g. for fault localisation across retries), this module's
single-slot policy is the one place that changes.

State snapshots are captured cheaply from the namespace's existing
proprioception primitives (``get_current_ee_pose`` /
``get_current_joints`` / ``get_current_gripper_width``), which are
bound by the per-robot config in live mode and stubbed in dry-run.
Each traced call can also fire one contextual VLM question after it
returns — questions are keyed on the call type (gripper-close → "did
you grasp?", move-ee → "where is the gripper now?", etc.) and the
answer lands in the trace entry as ``vlm_observation``. The post-call
VLM hook uses the *unwrapped* ``get_vqa_response`` captured at attach
time so it doesn't recurse through the trace wrapper. Disable with
``attach_trace(..., vlm_enabled=False)``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# Functions whose calls represent meaningful "logical blocks" — control
# primitives that move the robot OR perception primitives that determine
# what the program does next. Both are useful as boundaries between
# trace entries because they're where the program's intent becomes
# observable. The set deliberately matches probe_runner's _LOGGED_FUNCS
# plus the perception calls (which probe_runner excludes because scenes
# capture perception there; in live execution we want them).
_TRACED_FUNCS: frozenset[str] = frozenset({
    # Control
    "move_ee_to_pose",
    "move_ee_to_rel_pose",
    "set_gripper_width",
    "set_arm_joints",
    "set_camera_pose",
    "move_base",
    "move_base_to_rel",
    "reset_robot",
    # Franka extensions (ported from franka_entire_program)
    "move_ee_guarded",
    "rotate_wrist",
    "execute_waypoint_trajectory",
    # Perception — gives Claude visibility into what the program *saw*
    # before each action
    "detect_objects",
    "detect_markers",
    "get_vqa_response",
    "get_placement_pose",
    "get_keypoints",
    "get_keypoints_trajectory",
})


# Proprioception functions whose return values constitute the "state
# snapshot" appended after each traced call. Failures (ROS service
# missing, network blip) are caught and the slot is left as ``None`` so
# a flaky proprioception call never crashes the program under test.
_STATE_FUNCS: frozenset[str] = frozenset({
    "get_current_ee_pose",
    "get_current_gripper_width",
    "get_current_joints",
})


@dataclass
class ExecutionTraceEntry:
    """One entry in the timeline — produced after each traced API call.

    ``state_after`` is the proprioception snapshot taken *immediately
    after* the call returned. For perception calls (``detect_objects``,
    ``detect_markers``, ``get_vqa_response``) it's still captured but
    won't have moved since the prior entry; for control calls it
    reflects the post-motion state.

    ``result`` is the call's actual return value. Only stored for the
    names in :data:`_RESULT_CAPTURE_FUNCS` (``get_vqa_response``,
    ``detect_objects``, etc.) — these are the perception payloads that
    real-robot post-execution scoring needs to inspect. Control calls
    (``move_ee_to_pose``, ``set_gripper_width``) typically return bare
    success bools and their *effect* is what matters, captured via
    ``state_after``. Leaving control results uncaptured keeps the trace
    cheap to serialise into the next correction's prompt.

    ``vlm_question`` / ``vlm_observation`` hold a contextual VLM query
    asked *after* the call and its answer. Populated by the namespace
    wrapper when ``attach_trace(vlm_enabled=True)`` and the namespace
    exposes a ``get_vqa_response`` primitive. In live runs these are
    real camera-grounded observations; in dry-run they're the canned
    fixture answer (still useful so the trace shape is identical
    across modes). Either field is ``None`` when the call type didn't
    trigger a VLM query (e.g. ``get_vqa_response`` itself, to avoid
    recursion) or when the VLM call failed.
    """

    call_index: int
    name: str
    args: list[Any]
    kwargs: dict[str, Any]
    state_after: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    result: Any = None
    vlm_question: str | None = None
    vlm_observation: str | None = None


# Names whose return values are captured into the trace entry's ``result``
# field. Perception calls only — their responses carry information the
# program acted on, which post-execution scorers (e.g.
# :func:`architect.eval.task_scorers.task_complete`) need to inspect.
_RESULT_CAPTURE_FUNCS: frozenset[str] = frozenset({
    "get_vqa_response",
    "detect_objects",
    "detect_markers",
    "get_placement_pose",
    "get_keypoints",
    "get_keypoints_trajectory",
})


@dataclass
class ExecutionTrace:
    """The full timeline + outcome for one program execution.

    Lifecycle: constructed empty before exec, populated by the namespace
    wrapper as the program runs, sealed by the caller after exec returns
    or raises. Persisted single-slot on :class:`AgenticSession` so the
    next correction's prompt has it; older traces are dropped.

    ``_originals`` snapshots the unwrapped namespace entries at attach
    time so :func:`seal_trace_ok` / :func:`seal_trace_error` can restore
    them after exec finishes. Without restoration, a wrapped namespace
    persists across runs: subsequent ``namespace["reset_robot"]()``
    calls between corrections would fire a state snapshot + VLM query,
    and a second ``attach_trace`` would wrap the wrapper, doubling the
    instrumentation per call. The leading underscore + ``repr=False``
    keep the slot out of the rendered trace + dataclass repr.
    """

    program: str
    entries: list[ExecutionTraceEntry] = field(default_factory=list)
    exec_ok: bool = False
    error: str | None = None
    error_type: str | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    _originals: dict[str, Callable[..., Any]] = field(
        default_factory=dict, repr=False
    )

    def as_dict(self) -> dict[str, Any]:
        # Strip _originals — they're live callable refs, not serialisable
        # diagnostic data. Consumers that need to inspect the trace
        # (profile_session, eval harness) only care about entries.
        d = asdict(self)
        d.pop("_originals", None)
        return d

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


# ---------------------------------------------------------------------------
# Post-call VLM observation (contextual question keyed on the call type)
# ---------------------------------------------------------------------------


def _coerce_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _vlm_prompt_for_call(
    name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> str | None:
    """Return a short contextual VLM question for a traced call, or ``None``.

    The question is asked *after* the call returned, so the camera sees
    the post-call scene. Prompts are deliberately one-sentence and ask
    for actionable info — the LLM consuming the trace later needs
    facts ("the cube is still on the table" / "gripper is empty"), not
    free-form description.

    Returns ``None`` for:
      * ``get_vqa_response`` itself — would recurse infinitely
      * ``get_current_*`` proprioception calls — already captured by the
        state snapshot, the VLM would add nothing
      * Anything unrecognised — adding a fallback "describe the scene"
        is tempting but pollutes the trace with redundant generic answers
    """
    if name == "set_gripper_width":
        w = _coerce_float(args[0] if args else kwargs.get("width"))
        if w is not None and w < 0.05:
            return (
                "Did the gripper successfully grasp an object? "
                "If yes, name it in one phrase. If no, explain briefly "
                "what's preventing the grasp."
            )
        if w is not None and w >= 0.05:
            return (
                "Did the gripper release whatever it was holding? "
                "Where did the released object end up?"
            )
        return "Describe the current gripper state."
    if name in ("move_ee_to_pose", "move_ee_to_rel_pose"):
        return (
            "Where is the gripper now in relation to the target object? "
            "Answer in one short sentence."
        )
    if name == "detect_objects":
        target = args[0] if args else kwargs.get("text_prompt")
        if isinstance(target, str) and target:
            return (
                f"Confirm whether {target!r} is clearly visible in the "
                "workspace, and list any other prominent objects you see."
            )
        return "List the visible objects in the workspace."
    if name == "detect_markers":
        return "List the visible ArUco markers and their approximate positions."
    if name == "reset_robot":
        return (
            "Describe the workspace state and where the gripper is "
            "resting in one short sentence."
        )
    if name == "set_arm_joints":
        return "Where is the arm now, and is anything obstructing it?"
    if name in ("move_base", "move_base_to_rel"):
        return "Where is the robot base relative to the workspace now?"
    # get_vqa_response (recursion risk), get_current_* (redundant with
    # state snapshot), set_camera_pose (no scene change), and anything
    # else fall through — no VLM query for these.
    return None


def _extract_vqa_answer(resp: Any) -> str | None:
    """Pull the answer string out of a VQA response across schemas.

    Franka's live + dry-run shape: ``{"data": {"answer": "<string>"}}``.
    Stretch's live shape returns a bare string. Generic fallback handles
    ``{"answer": "..."}`` too. Returns ``None`` when nothing answer-like
    is recoverable; the caller leaves ``vlm_observation`` unset and the
    failure shows up as an empty cell in the trace render.
    """
    if isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"]
        ans = resp.get("answer")
        if isinstance(ans, str):
            return ans
    if isinstance(resp, str):
        return resp
    return None


# ---------------------------------------------------------------------------
# Namespace wrapping
# ---------------------------------------------------------------------------


def _snapshot_state(namespace: dict) -> dict[str, Any]:
    """Call every proprioception primitive currently bound in ``namespace``.

    Each function that raises (e.g. live robot service unavailable) is
    caught and its slot left as ``None`` — a flaky proprioception call
    can't be allowed to crash a live execution.
    """
    snap: dict[str, Any] = {}
    for name in _STATE_FUNCS:
        fn = namespace.get(name)
        if fn is None:
            continue
        try:
            snap[name] = fn()
        except Exception:
            snap[name] = None
    return snap


def _make_traced(
    name: str,
    original: Callable[..., Any],
    trace: ExecutionTrace,
    namespace: dict,
    vlm_query_fn: Callable[[str], Any] | None,
) -> Callable[..., Any]:
    """Wrap ``original`` so calls land in the trace with a post-call state snapshot.

    The wrapper preserves the original's return value untouched — the
    program under test must see exactly what it would see without
    tracing. Side effects (the trace append, the state snapshot, and
    when enabled the post-call VLM query) happen out of band.

    ``vlm_query_fn`` is the *unwrapped* ``get_vqa_response`` from the
    namespace at the moment ``attach_trace`` ran. We hold the original
    reference so calling it post-trace doesn't go back through the
    wrapper (which would recurse: the VLM call is traced → triggers
    another VLM call → traced → …). When ``None``, the VLM step is
    skipped.
    """
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        idx = len(trace.entries)
        result = original(*args, **kwargs)
        entry = ExecutionTraceEntry(
            call_index=idx,
            name=name,
            args=list(args),
            kwargs=dict(kwargs),
            state_after=_snapshot_state(namespace),
            timestamp=time.time(),
            # Capture the return value for perception calls so
            # post-execution scoring can read what the program saw.
            # Control-call results (typically True/False) are skipped to
            # keep the trace cheap to serialise.
            result=result if name in _RESULT_CAPTURE_FUNCS else None,
        )
        # Contextual VLM question after the call. Skipped for calls
        # that shouldn't trigger one (get_vqa_response itself,
        # proprioception) and silently when the VLM call raises — a
        # flaky camera shouldn't crash the program under test.
        if vlm_query_fn is not None:
            q = _vlm_prompt_for_call(name, list(args), dict(kwargs))
            if q is not None:
                entry.vlm_question = q
                try:
                    raw = vlm_query_fn(q)
                except Exception:
                    raw = None
                entry.vlm_observation = _extract_vqa_answer(raw)
        trace.entries.append(entry)
        return result
    # Sentinel — guards against double-wrapping if attach_trace is ever
    # called on a namespace that already carries wrappers (e.g. a bug
    # in the restoration path, or a caller forgetting to seal). See
    # attach_trace for the check.
    wrapped.__architect_traced__ = True  # type: ignore[attr-defined]
    return wrapped


def attach_trace(
    namespace: dict,
    program: str,
    *,
    vlm_enabled: bool = True,
) -> ExecutionTrace:
    """Wrap every traced primitive in ``namespace``; return the populated trace.

    Call before ``exec(program, namespace)``. The returned :class:`ExecutionTrace`
    starts empty and is mutated as the program runs. The caller seals it
    via :func:`seal_trace_ok` or :func:`seal_trace_error` once the exec
    completes.

    Only names already in ``namespace`` get wrapped — primitives that
    aren't bound (e.g. a Stretch-only API on a Franka run) are skipped
    rather than synthesized as stubs. The synthesis machinery
    (:mod:`architect.agent.synthesis`) handles undefined names separately.

    When ``vlm_enabled=True`` and ``get_vqa_response`` is bound in the
    namespace, each traced call also fires a contextual VLM question
    after the call returns. The *original* ``get_vqa_response`` is
    captured here, before wrapping, so the VLM hook doesn't recurse
    through the wrapper. Dry-run runs get the canned fixture answer
    (still useful for trace-shape parity with live mode); live runs
    get a real camera-grounded observation.
    """
    trace = ExecutionTrace(program=program, started_at=time.time())
    # Capture the unwrapped get_vqa_response before any wrapping happens,
    # so post-call VLM queries don't go through the wrapper. None when
    # the namespace doesn't expose VQA (test stubs) or when the user has
    # disabled VLM-during-execution.
    vlm_fn: Callable[[str], Any] | None = None
    if vlm_enabled:
        candidate = namespace.get("get_vqa_response")
        if callable(candidate):
            vlm_fn = candidate
    for fn_name in _TRACED_FUNCS:
        if fn_name not in namespace:
            continue
        original = namespace[fn_name]
        # Defensive: if the slot is already a tracer-wrapped callable
        # (e.g. a prior seal didn't restore — a bug, but still), don't
        # wrap again. Doing so would double-fire snapshots + VLM calls
        # per primitive invocation and would also leak the *real*
        # original on restore.
        if getattr(original, "__architect_traced__", False):
            continue
        trace._originals[fn_name] = original
        namespace[fn_name] = _make_traced(fn_name, original, trace, namespace, vlm_fn)
    return trace


def _restore_namespace(trace: ExecutionTrace, namespace: dict) -> None:
    """Reinstall the unwrapped originals captured at attach time.

    Without this, the wrappers persist past the end of exec: subsequent
    ``namespace["reset_robot"]()`` calls in the CLI between corrections
    would fire a state snapshot + VLM query (visible in the user's
    interactive output as ``[dry-run] get_current_*`` and ``[dry-run]
    get_vqa_response(...)`` lines emitted *before* the next ``Execute?``
    prompt), and a re-attach on the next run would wrap the wrappers.
    Idempotent — restoring an already-restored namespace is a no-op,
    so doubled seal calls don't misbehave.
    """
    for name, original in trace._originals.items():
        if name in namespace:
            namespace[name] = original
    trace._originals.clear()


def seal_trace_ok(trace: ExecutionTrace, namespace: dict) -> ExecutionTrace:
    """Mark the trace as successfully completed; capture the final state."""
    trace.exec_ok = True
    trace.finished_at = time.time()
    trace.final_state = _snapshot_state(namespace)
    _restore_namespace(trace, namespace)
    return trace


def seal_trace_error(
    trace: ExecutionTrace,
    namespace: dict,
    exc: BaseException,
) -> ExecutionTrace:
    """Mark the trace as failed; record the exception type + message."""
    trace.exec_ok = False
    trace.error = str(exc)
    trace.error_type = type(exc).__name__
    trace.finished_at = time.time()
    trace.final_state = _snapshot_state(namespace)
    _restore_namespace(trace, namespace)
    return trace


def seal_trace_interrupted(
    trace: ExecutionTrace,
    namespace: dict,
    exc: BaseException,
) -> ExecutionTrace:
    """Mark the trace as interrupted mid-execution (e.g. MidExecInterrupt).

    Like ``seal_trace_error`` but signals a deliberate pause rather than
    a crash. The partial trace is valid for building continuation prompts.
    """
    trace.exec_ok = False
    trace.error = str(exc)
    trace.error_type = "interrupted"
    trace.finished_at = time.time()
    trace.final_state = _snapshot_state(namespace)
    _restore_namespace(trace, namespace)
    return trace


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def format_for_prompt(trace: ExecutionTrace, *, max_entries: int = 80) -> str:
    """Render the trace as a compact text section for the next correction's prompt.

    Output shape::

        ## Last Execution Trace

        status:   ok | error: <type>: <message>
        duration: 4.3s
        calls:    24

         #  call                              state_after
         0  reset_robot()                     ee=(0.40, 0.00, 0.40)  gw=0.085
         1  detect_objects('red cube')        ee=(0.40, 0.00, 0.40)  gw=0.085
         2  set_gripper_width(0.085)          ee=(0.40, 0.00, 0.40)  gw=0.085
         3  move_ee_to_pose({...})            ee=(0.45, 0.00, 0.40)  gw=0.085
         ...

        final state: ee=(0.45, 0.00, 0.55)  gw=0.000

    The condensed state column shows ``ee=(x, y, z)`` and ``gw=<width>``
    only — full joint configs would dominate the budget without adding
    much signal to a correction prompt. If the trace has more than
    ``max_entries`` entries (a chatty program with lots of relative
    moves), the middle is collapsed with ``[... N more]``.
    """
    lines = ["## Last Execution Trace", ""]
    if trace.exec_ok:
        lines.append("status:   ok")
    else:
        lines.append(f"status:   error: {trace.error_type}: {trace.error}")
    if trace.duration_s is not None:
        lines.append(f"duration: {trace.duration_s:.2f}s")
    lines.append(f"calls:    {len(trace.entries)}")
    lines.append("")
    lines.append(f"{'#':>3}  {'call':<48s}  state_after")

    rendered = [_render_entry(e) for e in _maybe_truncate(trace.entries, max_entries)]
    lines.extend(rendered)

    final = _short_state(trace.final_state)
    if final:
        lines.append("")
        lines.append(f"final state: {final}")
    return "\n".join(lines)


def _maybe_truncate(
    entries: list[ExecutionTraceEntry],
    max_entries: int,
) -> list[ExecutionTraceEntry | str]:
    """Return entries unchanged if short, else head + ellipsis + tail."""
    if len(entries) <= max_entries:
        return list(entries)
    head = max_entries // 2
    tail = max_entries - head
    return [
        *entries[:head],
        f"  ...  [{len(entries) - head - tail} middle entries omitted]",
        *entries[-tail:],
    ]


def _render_entry(entry: ExecutionTraceEntry | str) -> str:
    if isinstance(entry, str):
        return entry
    call_str = _short_call(entry)
    state_str = _short_state(entry.state_after)
    line = f"{entry.call_index:>3}  {call_str:<48s}  {state_str}"
    # Append the VLM observation as an indented sub-line when one was
    # captured. Truncated to keep the trace readable; full answers
    # live in the dataclass for downstream consumers.
    if entry.vlm_observation:
        obs = " ".join(entry.vlm_observation.split())
        if len(obs) > 100:
            obs = obs[:97] + "..."
        line += f"\n     vlm: {obs}"
    return line


def _short_call(entry: ExecutionTraceEntry) -> str:
    """Compact one-line repr of the call, truncated at ~48 chars."""
    arg_strs = [_short_value(a) for a in entry.args]
    arg_strs += [f"{k}={_short_value(v)}" for k, v in entry.kwargs.items()]
    inner = ", ".join(arg_strs)
    s = f"{entry.name}({inner})"
    if len(s) > 48:
        s = s[:45] + "...)"
    return s


def _short_value(v: Any) -> str:
    """Truncate a single arg's repr — dict poses get position-only summary."""
    if isinstance(v, dict):
        pos = v.get("position")
        if isinstance(pos, dict):
            x, y, z = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
            return f"{{pos:({x:.2f},{y:.2f},{z:.2f}),...}}"
        return "{...}"
    if isinstance(v, (list, tuple)) and len(v) > 3:
        return f"[{len(v)} items]"
    if isinstance(v, str) and len(v) > 24:
        return repr(v[:21] + "...")
    return repr(v)


def _short_state(state: dict[str, Any]) -> str:
    """One-line ``ee=(x,y,z)  gw=<w>`` summary of the per-call state snapshot."""
    parts: list[str] = []
    ee = state.get("get_current_ee_pose")
    if isinstance(ee, dict):
        pos = ee.get("position", {})
        if isinstance(pos, dict):
            parts.append(
                f"ee=({pos.get('x', 0.0):.2f},{pos.get('y', 0.0):.2f},"
                f"{pos.get('z', 0.0):.2f})"
            )
    gw = state.get("get_current_gripper_width")
    if isinstance(gw, dict) and "gripper_width" in gw:
        parts.append(f"gw={gw['gripper_width']:.3f}")
    elif isinstance(gw, (int, float)):
        parts.append(f"gw={gw:.3f}")
    return "  ".join(parts)
