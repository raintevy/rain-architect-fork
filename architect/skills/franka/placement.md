# Placement Pose Estimation (Franka Panda)

## When to use `get_placement_pose`
Use `get_placement_pose(base_text_prompt, target_text_prompt)` when you need to determine where to place an object relative to another object in the scene. This is a perception function that uses vision to compute a valid placement pose in the robot base frame.
Note: Use this primitive alongwith `detect_objects` prior to making any control commands using `move_ee_to_pose`. And always replace the orientation of placement pose with grasping orientation.
Run this primitive before `detect_objects` as the scene is static.

## Arguments
| Parameter | Type | Description |
|-----------|------|-------------|
| `base_text_prompt` | str | Text description of the reference object (e.g. `"white plate"`, `"wooden shelf"`) |
| `target_text_prompt` | str | Text description of the object being placed (e.g. `"red cup"`, `"yellow block"`) |

## Return format
```python
{
    "placement_pose_base": {
        "position": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float}
    }
}
```
The returned pose is in the robot base frame (`panda_link0`) and can be passed directly to `move_ee_to_pose`.

## Typical placement sequence
```python
# 1: Detect the banana on the shelf to get grasp pose
banana = detect_objects("banana on wooden shelf")
banana_grasp_pose = {
    "position": {
        "x": banana["best_grasp"]["translation_wrt_base"][0],
        "y": banana["best_grasp"]["translation_wrt_base"][1],
        "z": banana["best_grasp"]["translation_wrt_base"][2]
    },
    "orientation": banana["best_grasp"]["quaternion_wrt_base"]
}

# 2. Get placement pose for banana relative to frying pan
placement = get_placement_pose("frying pan", "banana")
place_pose = {"position": placement["placement_pose_base"]["position"], "orientation": banana_grasp_pose["orientation"]}

# 3. Move above the placement target
move_ee_to_pose(place_pose)

# 4. Release the object
set_gripper_width(0.085)

# 5. Retract upward
move_ee_to_rel_pose({"x": 0, "y": 0, "z": 0.1})
```

## Notes
- Call `get_placement_pose` **before** moving to the place location — it is a perception query, not a motion command
- The base object must be visible to the camera; if not, reposition or use a known pose instead
- The returned pose accounts for the target object's size, so the EE can move directly to it without manual height offset
- For multi-step tasks, get all placement pose before grasping any object as the scene is static and have no dynamic changes during the run. Do not call `get_placement_pose` after grasping/picking/lifting/moving towards the object.
