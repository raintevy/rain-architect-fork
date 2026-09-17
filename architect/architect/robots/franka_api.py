"""Franka Panda + Robotiq 2F-85 API spec (used in the LLM system prompt)."""

API_SPEC = """\
## Perception

detect_markers()
    Detect ArUco markers in the scene using the scene camera or wrist camera.
    Returns a dict with a list of detected markers:
    {
        "markers": [
            {
                "marker_id": int,
                "pose_wrt_base_link": {
                    "position": {"x": float, "y": float, "z": float},
                    "orientation": {"x": float, "y": float, "z": float, "w": float}
                },
                "grasp_pose": {
                    "position": {"x": float, "y": float, "z": float},
                    "orientation": {"x": float, "y": float, "z": float, "w": float}
                }
            }
        ]
    }
    Note: Use grasp_pose (not pose_wrt_base_link) as the EE target for grasping.
    Two cameras available: scene camera (wide view) and wrist camera (close-range).

detect_objects(text_prompt: str, x_offset: float = 0.0,
               y_offset: float = 0.0, z_offset: float = 0.0, camera: str = "scene")
    Detect objects matching text_prompt via Grounded-SAM2 + AnyGrasp.
    Optional ``*_offset`` parameters shift the returned grasp pose in
    end-effector local frame meters — useful when the detected grasp lands a few
    centimeters off from where you actually need to grasp. Defaults are
    0.0 (no offset). See `grasping.md` for offset usage examples.
    camera selects which RGB feed to run detection on ("scene" or "wrist").
    Returns best grasp pose:
    {
        "best_grasp": {
            "translation_wrt_base": [x, y, z],
            "quaternion_wrt_base": {"x": float, "y": float, "z": float, "w": float},
            "score": float,
            "width": float
        }
    }

get_vqa_response(prompt: str, camera: str)
    Answer a visual question about the scene via Qwen2.5-VL.
    prompt: question string (e.g. "Is the cup upright? Yes or No?")
    camera: camera name (e.g. "scene" or "wrist")
    Returns: {"data": {"answer": "<string>"}}
    Note: Returns a dict — always access the answer via result["data"]["answer"].

verify_grasp(object_name: str)
    Verify if the current grasp is stable for the named object (e.g. "red apple", "baseball", etc.).
    object_name: String describing the object to verify grasp for (e.g. "red apple", "baseball", etc.)
    Returns a dict with the grasp verification result with following parameters:
    {
    "data": {
        "grasped": true,
        "reason": "stable grasp with good contact on all sides of the object"
        }
    }
    Note: this will be used to check if the robot has successfully grasped the object before proceeding with the next steps in the program, so the response should include a boolean "grasped" field.
    
get_placement_pose(base_text_prompt: str, target_text_prompt: str, x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.0)
    Recommend a placement pose for the target object on / next to the base
    object. base_text_prompt names the support (e.g. "the red block"),
    target_text_prompt names the object being placed (e.g. "the green block").
    Combines segmentation with spatial reasoning to pick a free surface or
    a stable stacking position. See `placement.md` for orientation guidance.
    Optional ``*_offset`` parameters shift the returned placement pose in
    end-effector local frame meters — useful when the detected placement is a few
    centimeters off from where you actually need to place the object. Defaults are
    0.0 (no offset).
    Returns:
    {
        "placement_pose_base": {
            "position": {"x": float, "y": float, "z": float},
            "orientation": {"x": float, "y": float, "z": float, "w": float}
        }
    }

get_keypoints(text_prompt: str, task: str)
    Return task-conditioned semantic keypoints on the named object
    (ReKep-style). ``task`` is a natural-language description of what the
    keypoints are for — the same object can yield different keypoints for
    different tasks (e.g. a mug's handle endpoints for grasp vs. its rim
    for pour). See `get_keypoints.md` for full schema and usage.
    Returns: {"keypoints": [{"position": {...}, "label": str, ...}, ...]}

get_keypoints_trajectory(text_prompt: str, task: str)
    Return a task-conditioned waypoint trajectory through keypoints. Same
    backend as get_keypoints but the response is an ordered sequence of
    EE waypoints — useful for articulated motions (open a drawer, pour,
    rotate). See `get_keypoints_trajectory.md` for schema + usage.
    Returns: {"waypoints": [{"position": {...}, "orientation": {...}}, ...]}

## Proprioception

get_current_joints()
    Get the current state of all arm joints.
    Returns a dict mapping joint name to {position, velocity, effort}:
    {
        "panda_joint1": {"position": float, "velocity": float, "effort": float},
        ...  (panda_joint1 through panda_joint7)
    }

get_current_gripper_width()
    Get current Robotiq 2F-85 gripper opening width.
    Returns: {"gripper_width": float}
    Range: 0.0 (fully closed) to 0.085 (fully open) meters.

get_current_ee_pose()
    Get the current end-effector pose in the robot base frame.
    Returns:
    {
        "position": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float}
    }

## Control

set_gripper_width(width: float)
    Set the Robotiq 2F-85 gripper width.
    width: 0.0 (fully closed) to 0.085 (fully open) meters.
    Recommended states: open=0.085, pre-grasp=0.06, closed=0.02, tight=0.0
    Returns: True on success.

move_ee_to_pose(target_pose: dict)
    Move the end-effector to an absolute pose via CuRobo motion planning.
    target_pose: {"position": {"x": float, "y": float, "z": float},
                  "orientation": {"x": float, "y": float, "z": float, "w": float}}
    Valid range (panda_link0 frame, approximate):
        x: [0.2, 0.75] m  (forward reach)
        y: [-0.5, 0.5] m  (left/right)
        z: [0.05, 0.8] m  (height)
    Returns: True on success.

move_ee_to_rel_pose(delta: dict)
    Move the end-effector by a relative offset from its current position.
    delta: {"x": float, "y": float, "z": float}  (meters)
        x = forward (+) / backward (−)
        y = left (+) / right (−)
        z = up (+) / down (−)
    Returns: True on success.

move_ee_guarded(axis: str, distance: float, force_threshold: float)
    Move the EE along ``axis`` ("x" / "y" / "z") for up to ``distance``
    meters, stopping early if measured contact force exceeds
    ``force_threshold`` (Newtons). Useful for "press until contact"
    patterns where you don't know the exact contact depth in advance
    (e.g. placing a block on top of another without crashing).
    Returns a dict:
    {
        "actual_distance": float,   # meters traveled before stop
        "contact_detected": bool
    }

rotate_wrist(angle_degrees: float, direction: str)
    Rotate the wrist by ``angle_degrees`` in the named ``direction``
    ("cw" / "ccw"). Used for in-place reorientations and screw / unscrew
    motions that move_ee_to_pose can't express cleanly.
    Returns: True on success.

execute_waypoint_trajectory(waypoints: list[dict])
    Execute a smooth trajectory through the given waypoints. Pair with
    get_keypoints_trajectory for articulated motions (open a door,
    pour from a mug). Each waypoint is a full EE pose dict.
    Returns: {"completed": bool, "n_waypoints_reached": int}

reset_robot()
    Move the robot to its home pose and fully open the gripper.
    Returns: True on success.

insert(obj: str, socket: dict)
    Insert a held ``obj`` into a ``socket`` using a learned insertion policy.
    Handles the contact-rich alignment and seating that move_ee_to_pose and
    move_ee_guarded cannot express. Currently supports the 8 mm round peg only.
    obj: name of the object currently held (e.g. "peg"). Used to select the
        policy and to verify the grasp — it does not steer the motion.
    socket: {"position": {"x": float, "y": float, "z": float},
             "orientation": {"x": float, "y": float, "z": float, "w": float}}
        Socket opening pose in the panda_link0 base frame.
    Preconditions:
        - ``obj`` is already grasped along its central axis with the gripper
          closed (as returned by detect_objects for the peg).
        - EE is above the socket, pointing down, within about 2 cm laterally
          and 4–6 cm vertically. Call move_ee_to_pose first to get there.
        Outside this envelope the policy is out of distribution and will fail.
    Returns a dict:
    {
        "success": bool,          # peg seated to within 1 mm of full depth
        "reason": str,            # "success" | "timeout" | "not_grasped" | "out_of_range"
        "depth": float,           # meters inserted at termination
        "steps": int              # policy steps executed (max 150)
    }
    This is a learned policy and succeeds probabilistically, unlike the
    planner-backed motion functions. Always check ``success``; on failure,
    re-approach with move_ee_to_pose and retry rather than assuming the
    object is seated. See `insertion.md` for retry patterns.

"""