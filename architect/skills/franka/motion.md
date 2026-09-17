<!-- architect-meta
always_loaded: true
description: Franka motion planning safety bounds — valid EE pose ranges, joint limits, approach conventions.
-->
# Motion Planning (Franka Panda)

## Valid EE pose range (panda_link0 frame, approximate)
| Axis | Range | Notes |
|------|-------|-------|
| x | [0.2, 0.75] m | forward reach |
| y | [-0.5, 0.5] m | left/right |
| z | [0.05, 0.8] m | height above base |

These are approximate bounds for a table-mounted Franka. CuRobo will catch hard infeasibility, but staying within these bounds avoids near-singular configurations.

## `move_ee_to_rel_pose` axis conventions
`delta = {"x": dx, "y": dy, "z": dz}`
- x: forward (+) / backward (−)
- y: left (+) / right (−)
- z: up (+) / down (−)

## CuRobo pre-execution validation
`move_ee_to_pose` is automatically wrapped with a CuRobo feasibility check. If the planned trajectory is collision-free and kinematically reachable, motion proceeds. If not, a `RuntimeError` is raised before any physical motion.

> Note: The validation endpoint is not yet active — the stub always passes. Once enabled, infeasible poses will be caught before execution.

## Common motion patterns

**Approach and grasp:**
```python
move_ee_to_pose(grasp_pose)                           # move to pre-grasp
move_ee_to_rel_pose({"x": 0, "y": 0, "z": -0.05})    # descend onto object
set_gripper_width(0.0)                                 # close
```

**Lift after grasp:**
```python
move_ee_to_rel_pose({"x": 0, "y": 0, "z": 0.15})    # lift before lateral movement
```

**Place and retreat:**
```python
move_ee_to_pose(place_pose)                            # move above target
set_gripper_width(0.085)                               # release
move_ee_to_rel_pose({"x": -0.05, "y": 0, "z": 0.1}) # retract and up
```

**Insert and retract:**
```python
socket = get_placement_pose("the hole", "the peg")["placement_pose_base"]
approach = {
    "position": {
        "x": socket["position"]["x"],
        "y": socket["position"]["y"],
        "z": socket["position"]["z"] + 0.05,
    },
    "orientation": socket["orientation"],
}
move_ee_to_pose(approach)                              # within 2 cm laterally
result = insert("peg", socket)
if not result["success"]:                              # e.g. "timeout", "out_of_range"
    move_ee_to_pose(approach)                          # re-approach and retry once
    result = insert("peg", socket)
set_gripper_width(0.085)                               # release
move_ee_to_rel_pose({"x": -0.05, "y": 0, "z": 0.1})    # retract and up
```

Do **not** hand-roll insertion with `move_ee_guarded` and a release — a guarded
descent stops at first contact, which for a peg means jamming on the chamfer,
not seating. Use `insert()` for any peg / socket / hole task.

## Notes on the Franka Panda
- 7-DOF arm: `panda_joint1` through `panda_joint7`
- No mobile base — `move_base` and `move_base_to_rel` are NOT available
- No articulated camera head — `set_camera_pose` is NOT available
- `set_arm_joints` is NOT available; use `move_ee_to_pose` for arm motion
