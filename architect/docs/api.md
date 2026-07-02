# Robot API reference

Generated programs use these functions, backed by live ROS services.
Both robots share the same perception and proprioception interface; the
control API differs by platform. The same function names also exist as
stubs in dry-run mode for offline testing.

> **Active target.** This project is currently focused on the **Franka
> Panda**. The Stretch entries below remain in the codebase
> (`architect/robots/stretch*.py`) and are documented for completeness, but
> aren't actively maintained on this branch.

## Perception (both robots)

| Function | Description |
|---|---|
| `detect_markers()` | Detect ArUco markers. Returns `{"markers": [{marker_id, pose_wrt_base_link, grasp_pose}]}` |
| `detect_objects(text_prompt, x_offset=0.0, y_offset=0.0, z_offset=0.0)` | Detect objects via Grounded-SAM2 + AnyGrasp. Returns `{"best_grasp": {"translation_wrt_base": [x,y,z], "quaternion_wrt_base": {x,y,z,w}, "score": float, "width": float}}`. The `*_offset` params shift the returned grasp pose in end-effectors local frame in meters. |
| `get_vqa_response(prompt)` | Visual question answering via Qwen2.5-VL. Franka returns `{"data": {"answer": "<string>"}}`; Stretch returns a bare string. |
| `get_placement_pose(base_text_prompt, target_text_prompt)` *(Franka)* | Recommended placement pose for `target` on top of / next to `base`. Returns `{"placement_pose_base": {...}}`. |
| `get_keypoints(text_prompt, task)` *(Franka)* | Task-conditioned semantic keypoints on the named object. ReKep-style. |
| `get_keypoints_trajectory(text_prompt, task)` *(Franka)* | Task-conditioned waypoint trajectory through keypoints. Pair with `execute_waypoint_trajectory` for articulated motions. |

## Proprioception (both robots)

| Function | Description |
|---|---|
| `get_current_joints()` | Joint states dict. Franka: `panda_joint1`–`panda_joint7`. Stretch: 5 arm joints. |
| `get_current_gripper_width()` | Gripper width dict. Franka: 0.0–0.085 m. Stretch: 0.0–0.152 m. |
| `get_current_ee_pose()` | `{position: {x,y,z}, orientation: {x,y,z,w}}` in base frame |
| `get_current_odom()` *(Stretch)* | Base odometry (position, orientation, velocities) |

## Control — Franka

| Function | Description |
|---|---|
| `set_gripper_width(width)` | Set Robotiq 2F-85 width (0.0–0.085 m). 0.085 = open, 0.0 = fully closed |
| `move_ee_to_pose(target_pose)` | Move EE to absolute pose via CuRobo. Valid range: x [0.2, 0.75], y [−0.5, 0.5], z [0.05, 0.8] m |
| `move_ee_to_rel_pose(delta)` | Move EE by relative offset `{x, y, z}` in meters |
| `move_ee_guarded(axis, distance, force_threshold)` | Move along `axis` until contact exceeds `force_threshold` (N). Returns `{actual_distance, contact_detected}`. |
| `rotate_wrist(angle_degrees, direction)` | Rotate wrist `cw` / `ccw` by the given angle |
| `execute_waypoint_trajectory(waypoints)` | Execute a smooth multi-waypoint EE trajectory (typically from `get_keypoints_trajectory`) |
| `reset_robot()` | Move to home pose and fully open the gripper |

## Control — Stretch *(legacy, not actively maintained)*

| Function | Description |
|---|---|
| `set_arm_joints(target)` | Move arm joints. `target`: subset of `{joint_lift, wrist_extension, joint_wrist_yaw/pitch/roll}` |
| `set_gripper_width(width)` | Set gripper width (0.0–0.152 m). 0.152 = fully open, 0.05 = closed |
| `move_ee_to_pose(target_pose)` | Move EE to absolute pose via CuRobo. `{position: {x,y,z}, orientation: {x,y,z,w}}` |
| `move_ee_to_rel_pose(delta)` | Move EE by relative offset `{x, y, z}` in meters |
| `move_base(x)` | Move base to absolute odom x coordinate |
| `move_base_to_rel(x)` | Move base by relative x displacement |
| `set_camera_pose(target)` | Set head pan/tilt. `{joint_head_pan, joint_head_tilt}` |
| `reset_robot()` | Reset to home pose and fully open gripper |

For the dry-run stub return values and ROS service bindings, see
`architect/robots/franka.py`, `architect/robots/franka_client.py`, and the matching
Stretch files.
