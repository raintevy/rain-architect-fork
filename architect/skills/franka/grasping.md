# Grasping (Franka Panda + Robotiq 2F-85)

## Preferred: marker-based grasping
Use `detect_objects()` with an ArUco text prompt, or `detect_markers()` when markers are attached to objects. Marker entries contain:
- `marker_id` — integer ID
- `pose_wrt_base_link` — `{position: {x,y,z}, orientation: {x,y,z,w}}`
- `grasp_pose` — `{position: {x,y,z}, orientation: {x,y,z,w}}` — use this as the grasp target

Two cameras are available:
- **Scene camera**: mounted opposite the Franka, wide field of view for workspace overview
- **Wrist camera**: mounted on the Robotiq gripper, close-range view for precise grasping

Grasp sequence:
1. `set_gripper_width(0.085)` — fully open before approaching
2. `move_ee_to_pose(marker.grasp_pose)` — move to pre-grasp pose
3. `move_ee_guarded(axis="z", distance=-0.06, force_threshold=5.0)` — descend 6 cm to ensure contact safely in guarded manner (top-down approach)
4. `set_gripper_width(0.0)` — close tightly around object

**IMPORTANT:**
- The grasp pose returned by `detect_objects` is already at the correct height for grasping. Do NOT add any z-offset to the returned position. Use the translation_wrt_base values as-is. If you need to adjust the grasp height, use the `z_offset` parameter of `detect_objects()` itself (e.g. `detect_objects(text_prompt, z_offset=0.02)`), NOT by manually modifying the returned z coordinate. `detect_objects()` returns the pre-grasp pose, so adding z-offset on returned position will go to incorrect grasping point.
- ONLY For objects such as fruits (ex: "banana", "watermelon", "carrot", etc. and NOT round objects such as "baseball", "tomato", "apple"), always add a negative z_offset of 3cm while getting the grasp pose using `detect_objects(text_prompt, z_offset=-0.03)`, then move using guarded to descend down.
- If the grasp approach is top-down vertical grasp, then apply a negative z_offset of 3cm while getting the grasp pose to ALL type of objects using `detect_objects(text_prompt, z_offset=-0.03)`

## Fallback: detect_objects grasping
Use `detect_objects(text_prompt)` when no marker is available. Returns:
```
{"best_grasp": {"translation_wrt_base": [x,y,z], "quaternion_wrt_base": {x,y,z,w}, "score": float, "width": float}}
```
Convert to pose dict before passing to `move_ee_to_pose`.

## Post-grasp behavior
- After closing gripper, lift vertically (`move_ee_to_rel_pose({"x":0,"y":0,"z":0.1})`) before any lateral movement.
- For placing: move above target (placement position received from `get_placement_pose` primitive), descend down and retract

## Robotiq 2F-85 gripper width reference
| State | Width |
|-------|-------|
| Fully open | 0.085 |
| Pre-grasp (wide clearance) | 0.06 |
| Closed around object | 0.02 |
| Tight grasp | 0.0 |

## Correction Guidelines
- The x, y, z offsets are defined in meters. 
- Following are few corrections examples, if the language corrections states:
    - Sample examples where z_offset can be used:
        - "grasp the cup a bit higher" - `detect_objects(text_prompt, z_offset=0.02)`
        - "grasp the cup with an offset height of 5 cm" - `detect_objects(text_prompt, z_offset=0.05)`
    - Sample examples where y_offset can be used:
        - "move the gripper 3 cm left while grasping" - `detect_objects(text_prompt, y_offset=0.03)`
        - "move the gripper a bit right while grasping" - `detect_objects(text_prompt, y_offset=-0.02)`
    - Sample examples where x_offset can be used:
        - "move the gripper 3 cm front while grasping" - `detect_objects(text_prompt, x_offset=0.03)`
        - "move the gripper little closer to robot base while grasping" - `detect_objects(text_prompt, x_offset=-0.02)`