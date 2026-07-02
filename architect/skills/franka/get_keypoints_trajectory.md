# Multi-Stage Keypoint Trajectory for Deformable / Articulated Manipulation (Franka Panda)

## When to use `get_keypoints_trajectory`
Use `get_keypoints_trajectory(text_prompt, task)` when the task requires a **multi-phase motion plan** through the scene — not just a single grasp point. The DIFT + ReKep planner returns a sequence of waypoint groups, each containing per-waypoint pose **and** gripper-action labels, so the entire manipulation (approach → grasp → execute path → release → retreat) can be played back group-by-group with `execute_waypoint_trajectory`.

This is the right primitive for **cloth folding** (corner-to-corner path), **drawer close** (handle approach + linear pull), **lid lift / place** (vertical lift along a curved retract), and similar tasks where the *path* between keypoints matters, not just the start pose.

### `get_keypoints` vs `get_keypoints_trajectory`

| Use `get_keypoints` when... | Use `get_keypoints_trajectory` when... |
|---|---|
| You only need a single pre-pick pose | You need a full path with per-waypoint gripper actions |
| The path between approach and grasp is a simple straight descent (use `move_ee_guarded`) | The path is curved or has multiple semantically distinct phases |
| The task is "make contact and grasp" | The task is "fold", "close drawer", "lift and place lid" |
| The LLM will plan the post-grasp motion itself | The LLM should defer path planning to ReKep |

**Both** primitives replace `detect_objects` + `get_placement_pose` for deformable / articulated targets — AnyGrasp is not appropriate here. Pick between them based on whether you need just a contact point or the full motion plan.

## Arguments
| Parameter | Type | Description |
|-----------|------|-------------|
| `text_prompt` | str | SAM2 segmentation prompt that isolates the object in the scene (e.g. `"white cloth on table"`, `"top drawer"`, `"pot lid"`) |
| `task` | str | Natural-language manipulation task with a verb (e.g. `"fold the cloth in half"`, `"close the top drawer"`, `"lift the pot lid and place it beside the pot"`) |

## Return format
```python
{
    "status":  "success",
    "message": str,
    "waypoints": [          # list of waypoint groups, one per contiguous label run
        [                   # group 0 — "pregrasp" (typically a single waypoint)
            {
                "position_base":        [x, y, z],            # base frame, metres
                "quaternion_base_xyzw": [x, y, z, w],         # base frame
                "gripper_action":       "open"|"close"|"keep",
                "label":                str                   # e.g. "pregrasp"
            }
        ],
        [ ... group 1 — "grasp_position" ... ],
        [ ... group 2 — "grasp_close" ... ],
        [ ... group 3 — "manipulation" (usually many waypoints forming the path) ... ],
        [ ... group 4 — "release_open" ... ],
        [ ... group 5 — "release_retreat" ... ]
    ]
}
```
All positions and orientations are in the robot base frame (`panda_link0`). The `gripper_action` field is interpreted by `execute_waypoint_trajectory` — `"open"` opens the gripper at that waypoint, `"close"` closes it, `"keep"` leaves it as-is. Each group corresponds to a contiguous run of waypoints sharing the same `label`; the labels (`"pregrasp"`, `"grasp_position"`, `"grasp_close"`, `"manipulation"`, `"release_open"`, `"release_retreat"`) describe the semantic phase of the motion.

The exact set of groups depends on the task, but the typical ordering for a pick-path-release task is:
`pregrasp → grasp_position → grasp_close → manipulation → release_open → release_retreat`.
The first three and last two groups are normally single-waypoint; the `manipulation` group contains the dense path (often 15–25 waypoints).

## Standard execution pattern
Each group in the returned `"waypoints"` is a list of waypoint dicts in exactly the format `execute_waypoint_trajectory` expects. Loop over the groups and pass each to the executor — no `move_ee_to_pose`, `set_gripper_width`, or `move_ee_guarded` calls are needed.

```python
result = get_keypoints_trajectory(
    text_prompt="white cloth on table",
    task="fold the cloth in half"
)

for group in result["waypoints"]:
    execute_waypoint_trajectory(group)
```
That's the whole execution. The groups run in order and cover the full task end-to-end — approach, grasp close, manipulation path, release, retreat.

## Typical sequence — fold a cloth in half
```python
# 1. Plan the full fold trajectory
result = get_keypoints_trajectory(
    text_prompt="white cloth on table",
    task="fold the cloth in half"
)

# 2. Execute each group in order: pregrasp → grasp_position → grasp_close
#    → manipulation path → release_open → release_retreat
for group in result["waypoints"]:
    execute_waypoint_trajectory(group)
```

## Typical sequence — lift a pot lid
```python
result = get_keypoints_trajectory(
    text_prompt="pot lid",
    task="lift the pot lid and place it beside the pot"
)

for group in result["waypoints"]:
    execute_waypoint_trajectory(group)
```

## Notes
- Waypoints are grouped by contiguous `label`, not by stage. The number of groups depends on the task; iterating all of them runs the full task, while iterating a subset stops early (e.g. iterate only through `grasp_close` to grasp without executing the manipulation path).
- Pass each `group` (a list of waypoint dicts) directly to `execute_waypoint_trajectory` — do **not** flatten all groups into one big list and do **not** call `move_ee_to_pose` / `set_gripper_width` manually. The gripper actions embedded in the waypoints are the source of truth.
- Do not insert your own `set_gripper_width(0.085)` before the loop — the `pregrasp` group's `gripper_action` already handles initial gripper state.
- Waypoint orientations are task-critical. Do not overwrite them with poses from `detect_objects` or `get_keypoints`; the orientations encode the planned approach axis and the curve of the manipulation path.
- The planner is one-shot against the current scene — call it after the scene is set, not while objects are still being moved into position.
- For tasks that only need a single contact point (e.g. "press the button", "tap the surface"), use `get_keypoints` instead — `get_keypoints_trajectory` is over-kill.

## Correction Guidelines
- **If and only if** the language correction states - "move the arm little down before grasping the cloth" or "go down a bit to grasp the cloth", then use the `move_ee_guarded` to move the gripper a little down using guarded move until contact with the cloth surface. Example -
```python
# 1: Get the full fold trajectory using keypoints trajectory planner
result = get_keypoints_trajectory(
    text_prompt="white cloth on table",
    task="fold the cloth in half from right corner to left"
)

# 2: Execute each waypoint group in order, inserting a guarded descent before grasping
for group in result["waypoints"]:
    # Check the label of the first waypoint in this group
    label = group[0].get("label", "")

    # If this is the grasp_close group, first do a guarded move down to ensure contact with the cloth
    if "grasp_close" in label or "close" in label:
        # Move the gripper a little down using guarded move until contact with the cloth surface
        move_ee_guarded(axis="z", distance=-0.07, force_threshold=5.0)
    
    # Execute the waypoint group
    execute_waypoint_trajectory(group)
```