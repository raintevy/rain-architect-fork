"""Deterministic stub LLM responses keyed by task + prompt shape.

Eval trials run without network access by replacing ``call_claude`` and
``call_claude_with_tools`` with these stubs. Each stub is keyed by a
``task_id`` and dispatches on the *shape* of the prompt — generation
prompt, correction prompt, slate prompt, scorer-emission prompt — so a
single stub covers an entire trial.

These responses are **not** representative of LLM quality. They exist so
the P7 sweep can run end-to-end in CI / dry-run and produce metric tables
without burning API credits. The paper's P8 sweep replaces them with
real ``call_claude``.
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Prompt classification — what is the LLM being asked to do?
# ---------------------------------------------------------------------------


def _classify(messages: list[dict]) -> str:
    """Return one of ``generation`` / ``correction`` / ``slate`` /
    ``scorer`` / ``other``. Looks at the system prompt + last user message
    for canonical ARCHITECT prompt markers."""
    if not messages:
        return "other"
    sys_text = ""
    if messages[0].get("role") == "system":
        sys_text = str(messages[0].get("content", ""))
    last = str(messages[-1].get("content", ""))

    if "alternative refined programs" in sys_text.lower() or "Candidate A:" in sys_text:
        return "slate"
    if "operationalizing a single user correction" in sys_text.lower():
        return "scorer"
    if "## Correction" in last:
        return "correction"
    # Recognise both the legacy markers ("## Instruction") and the canonical
    # build_prompt output ("## User Instruction" / "## Demonstrated
    # Trajectory"); the eval harness uses build_prompt for initial generation
    # so Claude gets the API spec in the system message.
    if any(
        marker in last
        for marker in ("## User Instruction", "## Demonstrated Trajectory",
                       "## Instruction", "## Demonstration")
    ):
        return "generation"
    return "other"


# ---------------------------------------------------------------------------
# Canned programs — used by the stubs to answer generation / correction
# ---------------------------------------------------------------------------


# Canned programs share a single helper def — ``grasp_at_pose`` — so the
# eval harness's graduation gate has something to actually evaluate. With a
# flat program (no defs), conditions that toggle ``graduation: true`` are
# silently no-ops and the per-cell ``lib_graduated`` column reads 0.
_HELPER_GRASP_AT_POSE = '''def grasp_at_pose(pose):
    """Move the end-effector to ``pose`` and close the gripper."""
    move_ee_to_pose(pose)
    set_gripper_width(0.0)
'''


_BASELINE_PICKUP = _HELPER_GRASP_AT_POSE + '''
reset_robot()
detection = detect_objects("target object")
g = detection["best_grasp"]
target = {"position": {"x": g["translation_wrt_base"][0],
                      "y": g["translation_wrt_base"][1],
                      "z": g["translation_wrt_base"][2]},
          "orientation": g["quaternion_wrt_base"]}
grasp_at_pose(target)
'''


_TOP_APPROACH = _HELPER_GRASP_AT_POSE + '''
reset_robot()
detection = detect_objects("target object")
g = detection["best_grasp"]
pre = {"position": {"x": g["translation_wrt_base"][0],
                    "y": g["translation_wrt_base"][1],
                    "z": g["translation_wrt_base"][2] + 0.10},
       "orientation": g["quaternion_wrt_base"]}
move_ee_to_pose(pre)
move_ee_to_rel_pose({"x": 0.0, "y": 0.0, "z": -0.10})
set_gripper_width(0.0)
'''


_SIDE_APPROACH = _HELPER_GRASP_AT_POSE + '''
reset_robot()
detection = detect_objects("target object")
g = detection["best_grasp"]
side = {"position": {"x": g["translation_wrt_base"][0],
                     "y": g["translation_wrt_base"][1] + 0.05,
                     "z": g["translation_wrt_base"][2]},
        "orientation": g["quaternion_wrt_base"]}
move_ee_to_pose(side)
move_ee_to_rel_pose({"x": 0.0, "y": -0.05, "z": 0.0})
set_gripper_width(0.0)
'''


def _wrap_lift(program: str, dz: float = 0.15) -> str:
    return program.rstrip() + f"\nmove_ee_to_rel_pose({{\"x\": 0.0, \"y\": 0.0, \"z\": {dz}}})\n"


# ---------------------------------------------------------------------------
# Per-task stubs
# ---------------------------------------------------------------------------


def _stub_correction(task_id: str, correction_text: str, current_program: str) -> str:
    """Single-shot correction: edit the current program to satisfy the correction.

    Returns a complete Python program (no markdown fences).
    """
    txt = correction_text.lower()
    if "approach the block from above" in txt or "from above, not from the side" in txt:
        return _TOP_APPROACH
    if "approach from the side" in txt:
        return _SIDE_APPROACH
    if "lift 15 cm" in txt or "lift" in txt and "15" in txt:
        return _wrap_lift(current_program, dz=0.15)
    if "open the gripper wider" in txt:
        return current_program.replace("set_gripper_width(0.0)", "set_gripper_width(0.085)", 1)
    if "lower more carefully" in txt:
        return current_program.rstrip() + "\nmove_ee_to_rel_pose({\"x\": 0.0, \"y\": 0.0, \"z\": -0.01})\n"
    if "release the block only after" in txt:
        return current_program.rstrip() + "\nset_gripper_width(0.085)\n"
    # default: tiny no-op edit
    return current_program


def _stub_generation(task_id: str) -> str:
    return _BASELINE_PICKUP


_HARDCODED_FALLBACK_POSE = '''hardcoded_target = {"position": {"x": 0.45, "y": 0.0, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
move_ee_to_pose(hardcoded_target)
move_ee_to_rel_pose({"x": 0.0, "y": 0.0, "z": -0.10})
set_gripper_width(0.0)
'''


def _stub_slate(task_id: str, correction_text: str, current_program: str) -> str:
    """Two behaviorally-differentiable candidates wrapped in the slate format.

    Candidate A is audit-clean: routes the pose through detect_objects so the
    audit module finds no high-severity flags. Candidate C uses a *hardcoded
    fallback pose* — exactly the pattern architect.probes.audit's ``hardcoded_pose`` rule
    flags as ``severity=high``. After rank_candidates applies the 0.05 audit
    penalty per high flag, A scores 1.0 and C scores 0.95 — small but real
    differentiation. Without this, the stub's slate path emitted two
    candidates that scored identically and the ranker had no signal at all.

    A third candidate (the comment-only variant in the legacy stub) is
    deliberately removed: ast.unparse normalises away its comment and the
    slate dedupe collapses it back into A, so it added no signal anyway.
    """
    a = _stub_correction(task_id, correction_text, current_program)
    # C: same correction intent but with a hardcoded fallback pose that
    # triggers the audit module's hardcoded_pose=HIGH flag.
    c = _HELPER_GRASP_AT_POSE + "\nreset_robot()\n" + _HARDCODED_FALLBACK_POSE
    return (
        f"# Candidate A: detect-driven (audit-clean)\n```python\n{a}```\n\n"
        f"# Candidate C: hardcoded fallback pose (audit-flagged)\n```python\n{c}```\n"
    )


_CANNED_SCORER = '''```python
def score(scene_before, scene_after, call_log):
    """Reward programs that issue at least one move_ee_to_pose call."""
    if not call_log:
        return 0.0
    moves = [e for e in call_log if e.get("name") == "move_ee_to_pose"]
    return 1.0 if moves else 0.0
```

```python
def invariant_0(scene_before, scene_after, call_log):
    """Gripper should be opened (>0) at least once during the trajectory."""
    for entry in call_log:
        if entry.get("name") == "set_gripper_width":
            args = entry.get("args") or []
            kw = entry.get("kwargs") or {}
            w = args[0] if args else kw.get("width", 0.0)
            try:
                if float(w) > 0:
                    return True
            except Exception:
                pass
    return False
```
'''


# ---------------------------------------------------------------------------
# Public: stub registry
# ---------------------------------------------------------------------------


def make_call_claude_stub(task_id: str, *, current_program: dict[str, str]) -> Callable:
    """Return a ``call_claude``-shaped stub closed over ``task_id`` + a
    mutable ``current_program`` dict the trial runner updates after each
    correction. ``current_program[k]`` is the latest program text;
    the stub reads it when emitting a correction response so successive
    corrections compound rather than each starting from scratch."""

    def stub(messages: list[dict], model: str, **kwargs: Any) -> str:
        kind = _classify(messages)
        if kind == "generation":
            return _stub_generation(task_id)
        if kind == "scorer":
            return _CANNED_SCORER
        if kind == "slate":
            last = str(messages[-1].get("content", ""))
            correction = _extract_correction(last)
            return _stub_slate(task_id, correction, current_program.get("program", ""))
        if kind == "correction":
            last = str(messages[-1].get("content", ""))
            correction = _extract_correction(last)
            return _stub_correction(task_id, correction, current_program.get("program", ""))
        return ""

    return stub


def make_call_claude_with_tools_stub(task_id: str, *, current_program: dict[str, str]) -> Callable:
    """Tool-use stub: returns a fake response object that always calls
    ``submit_program`` with the canned correction output, so the agentic
    loop terminates on the first turn. This mirrors what a perfect
    tool-using LLM would do for trivial corrections."""

    class _FakeBlock:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _FakeResp:
        def __init__(self, content, stop_reason="tool_use"):
            self.content = content
            self.stop_reason = stop_reason

    def stub(messages: list[dict], tools: list[dict], model: str, **kwargs: Any):
        kind = _classify(messages)
        if kind == "generation":
            program = _stub_generation(task_id)
        elif kind == "correction":
            last = str(messages[-1].get("content", ""))
            correction = _extract_correction(last)
            program = _stub_correction(task_id, correction, current_program.get("program", ""))
        else:
            program = current_program.get("program", "")

        submit_block = _FakeBlock(
            type="tool_use",
            id=f"toolu_{task_id}",
            name="submit_program",
            input={"code": program},
        )
        return _FakeResp(content=[submit_block])

    return stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_correction(user_content: str) -> str:
    """Pull the correction text out of a build_correction_prompt user message."""
    marker = "## Correction"
    if marker in user_content:
        tail = user_content.split(marker, 1)[1]
        for line in tail.splitlines():
            line = line.strip()
            if line and not line.startswith("##"):
                return line
    return ""
