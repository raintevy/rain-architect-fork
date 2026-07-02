# Keypoint Detection for Deformable / Articulated Objects (Franka Panda)

## When to use `get_keypoints`
Use `get_keypoints(text_prompt, task)` when the object to be manipulated is **deformable** (e.g. cloth, towel, plastic bag) or **articulated** (e.g. drawer, lid, door, cabinet, oven handle) — i.e. cases where a single grasp pose from AnyGrasp is not meaningful or reliable.
This primitive uses DIFT (Diffusion Features) + SAM2 to pick a *task-relevant* keypoint on the object and returns a pre-pick pose in the robot base frame, already adjusted for the recommended grasp approach.

**Use this primitive *instead of* `detect_objects` + `get_placement_pose`** for non-rigid or articulated targets. Those two are designed for rigid, graspable objects via AnyGrasp; `get_keypoints` is the correct tool for cloth corners, drawer, lid knobs, etc.
The returned pose can be passed directly to `move_ee_to_pose` — no separate placement query is needed because the task itself encodes both *where* and *how* to interact.
Run this primitive before issuing any motion command; the scene is assumed static during the call.

## Arguments
| Parameter | Type | Description |
|-----------|------|-------------|
| `text_prompt` | str | SAM2 segmentation prompt that isolates the object in the scene (e.g. `"white cloth on table"`, `"top drawer"`, `"pot lid"`) |
| `task` | str | Natural-language manipulation task with a verb on a deformable / articulated object (e.g. `"fold the cloth in half"`, `"close drawer from the cabinet"`, `"lift the pot lid"`) |

## Return format
```python
{
    "status": "success",
    "selected_reason": str,           # LLM rationale for why this keypoint was chosen
    "selected_index": int,            # which candidate keypoint was picked
    "keypoint_prepick_pose": {
        "position":    {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float}
    },
    "grasp_approach": str             # e.g. "top_down", "side", etc.
}
```
The `keypoint_prepick_pose` is already in the robot base frame (`panda_link0`) and oriented for the recommended `grasp_approach`, so it can be passed directly to `move_ee_to_pose`. **Do not** overwrite this orientation with one from a different primitive — unlike `get_placement_pose`, the orientation here is task-critical.

## Typical sequence — fold a cloth
```python
# 1. Get the task-relevant keypoint pre-pick pose for the cloth
result = get_keypoints(
    text_prompt="white cloth on table",
    task="fold the cloth in half"
)
prepick_pose = result["keypoint_prepick_pose"]

# 2. Open the gripper before approach
set_gripper_width(0.085)

# 3. Move to the pre-pick pose
move_ee_to_pose(prepick_pose)

# 4. Use move_ee_guarded to descend along z-axis until contact with the cloth surface
move_ee_guarded(axis="z", distance=-0.07, force_threshold=5.0)

# 5. Close gripper to pinch the cloth corner
set_gripper_width(0.0)

# 6. Lift and execute the fold (relative motions follow naturally from the task)
move_ee_to_rel_pose({"x": 0, "y": 0, "z": 0.10})
```

## Typical sequence — close a drawer
```python
# 1. Get the keypoint on the drawer handle
result = get_keypoints(
    text_prompt="drawer handle",
    task="close the open drawer by pushing its handle",
    num_candidates=50
)
prepick_pose = result["keypoint_prepick_pose"]

# 2. Open the gripper (we're pushing, not grasping)
set_gripper_width(0.085)

# 3. Override orientation to a top-down vertical approach
top_down_orientation = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

# 4.1 Move to a position before the drawer keypoint with an offset towards robot base (-X) and base (-Z) This positions the gripper behind the drawer face so it can push forward 
approach_pose = {
    "position": {
        "x": prepick_pose["position"]["x"] - 0.15,
        "y": prepick_pose["position"]["y"],
        "z": prepick_pose["position"]["z"] + 0.10
    },
    "orientation": top_down_orientation
}
move_ee_to_pose(approach_pose)

# 4.2 Move down to descend towards original keypoint height + some -Z offset
move_ee_to_rel_pose({"x": 0.0, "y": 0.0, "z": 0.15})

# 5. Push the drawer closed in +X direction using guarded move. Stop when contact is made with greater force (drawer fully closed hits the cabinet)
move_ee_guarded(axis="x", distance=0.30, force_threshold=10.0)

# 6. Retract the arm back and up after closing the drawer
move_ee_to_rel_pose({"x": -0.10, "y": 0.0, "z": 0.10})
```

## Notes
- Use this primitive for **deformable** (cloth, towel, bag) and **articulated** (drawer, lid, door, handle) objects. For rigid graspable objects, use `detect_objects` + `get_placement_pose` instead.
- `task` must be a manipulation verb phrase that names the action (`"fold ..."`, `"open ..."`, `"close ..."`, `"lift ..."`, `"pull ..."`). A bare object name will not produce a useful keypoint.
- `text_prompt` should isolate the object cleanly for SAM2; include disambiguating context like `"on table"` or `"top"` when multiple similar objects are in view.
- The returned orientation is task-specific (matches `grasp_approach`) — keep it as-is when calling `move_ee_to_pose`.
- The object must be visible to the scene camera at call time; the call is a one-shot perception query, not a closed-loop tracker.
- Inspect `selected_reason` when debugging — it explains why the LLM picked that keypoint and is useful for verifying the task was interpreted correctly.
- Use `move_ee_guarded` to descend along aproach axis and make contact with the target object for grasping
- For closing drawer tasks, be precise to write `"front face in center of drawer"` in the `task` prompt rather than just `"drawer handle"`. Keep `text_prompt` as short and precise as possible, ex: `"drawer handle"`