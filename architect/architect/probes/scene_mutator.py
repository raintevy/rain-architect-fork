"""Counterfactual scene representation + perturbation suite for probe runs.

A :class:`Scene` is the snapshot of perception/state values that the probe
runner injects into the dry-run namespace before executing a candidate
program. Mutations alter what perception calls return — translating an
object's grasp pose, rotating it, jittering markers, permuting detection
order — so we can ask the program to handle a *different* scene than the
one the user corrected on.

Probes never re-execute physics. They check a logic-level question:
*"if perception had returned these slightly different values, would the
program still route them through correctly?"* That signal is what feeds
both the C1 active-sampling ranker (per-candidate generalization rate)
and the C2 graduation gate (skill admitted only if pass-rate ≥ τ).

Mutations:
  * ``object_translation``  — jitter the default grasp's translation_wrt_base
  * ``object_rotation``     — small yaw rotation on the grasp quaternion
  * ``marker_jitter``       — translate every detected marker's pose
  * ``detection_order``     — permute the marker / object detection order
  * ``vqa_negative_grasp``  — first N ``get_vqa_response`` calls return a
                              negative answer simulating a failed grasp, then
                              subsequent calls return affirmative — exercises
                              the retry pattern documented in
                              ``skills/<robot>/verification.md``. Programs
                              that verify-once-without-retry score 0 on this
                              axis; programs with a one-retry budget recover.

Distractor add/drop and attribute swap are deferred — they need a richer
scene model with multiple labelled objects.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """Per-call return values for the dry-run perception / state stubs.

    Attributes match the structure of the live robot client's return values
    (see ``architect/robots/franka_client.py``) so a probe
    namespace can be built without per-robot branching beyond the defaults.

    ``vqa_answer`` is the constant response used when the scene has no
    multi-call sequence configured. ``vqa_answer_sequence`` is the
    adversarial knob: when set to a non-empty list, ``bind_scene`` installs
    a stateful ``get_vqa_response`` closure that walks the sequence on
    successive calls (and stays on the last element once exhausted). The
    ``vqa_negative_grasp`` mutation axis uses this to make the first call
    return "no" and the retry return "yes", so programs that have a retry
    budget recover and programs that don't fail.
    """

    markers: list[dict[str, Any]] = field(default_factory=list)
    grasp: dict[str, Any] = field(default_factory=dict)
    vqa_answer: dict[str, Any] = field(default_factory=lambda: {"data": {"answer": "yes"}})
    vqa_answer_sequence: list[dict[str, Any]] | None = None
    ee_pose: dict[str, Any] = field(default_factory=dict)
    joints: dict[str, Any] = field(default_factory=dict)
    gripper_width: dict[str, Any] = field(default_factory=lambda: {"gripper_width": 0.085})
    odom: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Scene":
        return Scene(
            markers=deepcopy(self.markers),
            grasp=deepcopy(self.grasp),
            vqa_answer=deepcopy(self.vqa_answer),
            vqa_answer_sequence=deepcopy(self.vqa_answer_sequence),
            ee_pose=deepcopy(self.ee_pose),
            joints=deepcopy(self.joints),
            gripper_width=deepcopy(self.gripper_width),
            odom=deepcopy(self.odom),
        )


def default_scene(robot_name: str = "franka") -> Scene:
    """Return a reasonable starting Scene for the given robot.

    The defaults match the existing dry-run stubs in the robot config
    (see ``architect/robots/franka.py``) so probe runs see the same nominal scene
    that an interactive dry-run session would.
    """
    if robot_name == "franka":
        ee = {
            "position": {"x": 0.4, "y": 0.0, "z": 0.4},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        joints = {
            f"panda_joint{i}": {"position": 0.0, "velocity": 0.0, "effort": 0.0}
            for i in range(1, 8)
        }
        grasp = {
            "best_grasp": {
                "translation_wrt_base": [0.45, 0.0, 0.30],
                "quaternion_wrt_base": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "score": 0.9,
                "width": 0.05,
            }
        }
        # Three AprilTags at distinct positions so probes that mutate marker
        # pose have something to mutate AND tasks like "place on marker 2"
        # actually find their target. Schema matches the documented contract
        # in architect.franka_robot_api.API_SPEC: ``marker_id`` (not ``id``) and a
        # ``grasp_pose`` field distinct from ``pose_wrt_base_link``. Without
        # the multiple-markers fixture, programs that correctly iterate
        # detect_markers() looking for marker_id=2 (or 3) bail with "marker
        # not found" and the eval scorer correctly registers a 0% pass —
        # but the failure is the fixture's, not the program's.
        markers = [
            {
                "marker_id": 1,
                "pose_wrt_base_link": {
                    "position": {"x": 0.45, "y": 0.0, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "grasp_pose": {
                    "position": {"x": 0.45, "y": 0.0, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            {
                "marker_id": 2,
                "pose_wrt_base_link": {
                    "position": {"x": 0.55, "y": 0.15, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "grasp_pose": {
                    "position": {"x": 0.55, "y": 0.15, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            {
                "marker_id": 3,
                "pose_wrt_base_link": {
                    "position": {"x": 0.55, "y": -0.15, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "grasp_pose": {
                    "position": {"x": 0.55, "y": -0.15, "z": 0.30},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ]
        gripper = {"gripper_width": 0.085}
    else:
        raise ValueError(
            f"Unsupported robot: {robot_name!r} (only 'franka' is supported)"
        )

    odom = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    return Scene(
        markers=markers,
        grasp=grasp,
        vqa_answer={"data": {"answer": "yes"}},
        ee_pose=ee,
        joints=joints,
        gripper_width=gripper,
        odom=odom,
    )


# ---------------------------------------------------------------------------
# Bind a Scene into a dry-run namespace
# ---------------------------------------------------------------------------


def bind_scene(namespace: dict, scene: Scene) -> dict:
    """Override the perception / state functions in *namespace* with closures
    that return ``scene``'s values. Returns *namespace* for chaining.

    The non-perception functions (``move_ee_to_pose``, ``set_gripper_width``,
    etc.) are left untouched, so the existing per-robot dry-run stubs continue
    to no-op those calls. Only what the program *reads* from the world is
    Scene-controlled.

    When ``scene.vqa_answer_sequence`` is a non-empty list, a stateful
    closure walks it across calls — each ``get_vqa_response`` call returns
    the next element, clamping at the last once exhausted. That's the
    mechanism the ``vqa_negative_grasp`` mutation axis uses to differentiate
    retry-aware programs from verify-once / open-loop ones.
    """
    namespace["detect_markers"] = lambda: {"markers": deepcopy(scene.markers)}
    namespace["detect_objects"] = lambda text_prompt=None: deepcopy(scene.grasp)
    if scene.vqa_answer_sequence:
        seq = scene.vqa_answer_sequence
        counter = [0]
        def _vqa_stateful(prompt=None):
            idx = min(counter[0], len(seq) - 1)
            counter[0] += 1
            return deepcopy(seq[idx])
        namespace["get_vqa_response"] = _vqa_stateful
    else:
        namespace["get_vqa_response"] = lambda prompt=None: deepcopy(scene.vqa_answer)
    namespace["get_current_ee_pose"] = lambda: deepcopy(scene.ee_pose)
    namespace["get_current_joints"] = lambda: deepcopy(scene.joints)
    namespace["get_current_gripper_width"] = lambda: deepcopy(scene.gripper_width)
    namespace["get_current_odom"] = lambda: deepcopy(scene.odom)
    return namespace


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


# Axis names — kept as constants so the runner can request a specific axis or
# iterate over the set without typo'ing strings.
AXIS_OBJECT_TRANSLATION = "object_translation"
AXIS_OBJECT_ROTATION = "object_rotation"
AXIS_MARKER_JITTER = "marker_jitter"
AXIS_DETECTION_ORDER = "detection_order"
AXIS_VQA_NEGATIVE_GRASP = "vqa_negative_grasp"

ALL_AXES = (
    AXIS_OBJECT_TRANSLATION,
    AXIS_OBJECT_ROTATION,
    AXIS_MARKER_JITTER,
    AXIS_DETECTION_ORDER,
    AXIS_VQA_NEGATIVE_GRASP,
)


# Canned VQA responses for the adversarial axis. Phrased so both the scorer's
# word-token classifier (``architect.eval_scorers._vqa_affirmative``) and any
# program-side string check land on the intended verdict. The negative phrasing
# also names what's wrong ("gripper is empty"), which is the kind of
# diagnostic answer a real VLM would return on a failed grasp — useful both
# for the scorer's classifier and for any program that branches on the
# answer's content rather than just yes/no.
_VQA_NEGATIVE_GRASP_ANSWER = {
    "data": {"answer": "no, the gripper is empty and the object is still on the surface"},
}
_VQA_AFFIRMATIVE_GRASP_ANSWER = {
    "data": {"answer": "yes, the gripper is firmly holding the object"},
}


def _yaw_rotate_quat(q: dict[str, float], yaw: float) -> dict[str, float]:
    """Compose a yaw rotation onto an existing quaternion (q' = q_yaw * q)."""
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    qz = {"x": 0.0, "y": 0.0, "z": sy, "w": cy}
    # Hamilton product q1 * q2
    a, b, c, d = qz["w"], qz["x"], qz["y"], qz["z"]
    e, f, g, h = q["w"], q["x"], q["y"], q["z"]
    return {
        "w": a * e - b * f - c * g - d * h,
        "x": a * f + b * e + c * h - d * g,
        "y": a * g - b * h + c * e + d * f,
        "z": a * h + b * g - c * f + d * e,
    }


def mutate(scene: Scene, axis: str, magnitude: float, rng: random.Random) -> Scene:
    """Return a new Scene perturbed along ``axis`` by ``magnitude``.

    ``magnitude`` is interpreted per-axis:
      * ``object_translation``   — meters (uniform per-coordinate jitter, ±)
      * ``object_rotation``      — radians (signed yaw, ±)
      * ``marker_jitter``        — meters (per-coordinate ±)
      * ``detection_order``      — ignored (permutation is unconditional)
      * ``vqa_negative_grasp``   — integer count of leading "no" answers
                                   before the sequence flips to "yes" (rounded
                                   down, minimum 1). Default magnitude 1 matches
                                   the one-retry budget in verification.md.

    The original ``scene`` is not modified.
    """
    s = scene.copy()

    if axis == AXIS_OBJECT_TRANSLATION:
        bg = s.grasp.get("best_grasp", {})
        t = list(bg.get("translation_wrt_base", [0.0, 0.0, 0.0]))
        for i in range(3):
            t[i] = float(t[i]) + rng.uniform(-magnitude, magnitude)
        bg["translation_wrt_base"] = t

    elif axis == AXIS_OBJECT_ROTATION:
        bg = s.grasp.get("best_grasp", {})
        q = bg.get("quaternion_wrt_base", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        yaw = rng.uniform(-magnitude, magnitude)
        bg["quaternion_wrt_base"] = _yaw_rotate_quat(q, yaw)

    elif axis == AXIS_MARKER_JITTER:
        for m in s.markers:
            pos = m.get("pose_wrt_base_link", {}).get("position")
            if isinstance(pos, dict):
                for k in ("x", "y", "z"):
                    pos[k] = float(pos.get(k, 0.0)) + rng.uniform(-magnitude, magnitude)

    elif axis == AXIS_DETECTION_ORDER:
        rng.shuffle(s.markers)

    elif axis == AXIS_VQA_NEGATIVE_GRASP:
        # N initial "no" answers followed by an affirmative, where N is
        # max(1, int(magnitude)). With the default magnitude=1, this is
        # exactly one failure then success — programs with the
        # verification.md one-retry pattern recover; verify-once or
        # open-loop programs do not. The clamp-to-final behavior in
        # bind_scene means programs that ask more than len(seq) times
        # keep seeing "yes" — they don't accidentally loop forever
        # consuming negatives.
        n_negative = max(1, int(magnitude))
        s.vqa_answer_sequence = (
            [deepcopy(_VQA_NEGATIVE_GRASP_ANSWER) for _ in range(n_negative)]
            + [deepcopy(_VQA_AFFIRMATIVE_GRASP_ANSWER)]
        )

    else:
        raise ValueError(f"Unknown mutation axis: {axis!r}")

    return s


# ---------------------------------------------------------------------------
# Probe suite generation
# ---------------------------------------------------------------------------


# Magnitudes chosen to match the §8.2 perturbation envelope (±10cm translation,
# ±45° rotation). Marker jitter uses the same translation budget. They can be
# overridden per-call when an experiment wants tighter or wider bounds.
DEFAULT_MAGNITUDES: dict[str, float] = {
    AXIS_OBJECT_TRANSLATION: 0.10,
    AXIS_OBJECT_ROTATION: math.radians(45.0),
    AXIS_MARKER_JITTER: 0.10,
    AXIS_DETECTION_ORDER: 0.0,
    AXIS_VQA_NEGATIVE_GRASP: 1.0,
}


def generate_probe_suite(
    base: Scene,
    k: int,
    *,
    seed: int | None = None,
    axes: tuple[str, ...] = ALL_AXES,
    magnitudes: dict[str, float] | None = None,
) -> list[tuple[str, Scene]]:
    """Generate ``k`` perturbed scenes spread across ``axes``.

    Each result is ``(axis, scene)``. Axes are cycled round-robin so the
    suite covers all dimensions evenly even when ``k`` is small.
    Returns the *base scene* as the first element (``axis="identity"``)
    so a probe can also report whether the program runs on the unperturbed
    scene — without that it's hard to diagnose perturbation-specific
    failures.
    """
    if k < 1:
        return []
    rng = random.Random(seed)
    mags = {**DEFAULT_MAGNITUDES, **(magnitudes or {})}
    suite: list[tuple[str, Scene]] = [("identity", base.copy())]
    if k == 1:
        return suite
    for i in range(k - 1):
        axis = axes[i % len(axes)]
        suite.append((axis, mutate(base, axis, mags[axis], rng)))
    return suite
