<!-- architect-meta
always_loaded: true
description: Verification-by-default pattern — VQA-verified retry for sub-operations that lack proprioceptive success signals.
-->
# Verification-by-default: VQA-verified retry

For any sub-operation whose success isn't observable from proprioception
alone — **grasps**, **placements**, **door / drawer / lid opening** — the
default pattern is *do the action, ask VQA, retry on failure*. Open-loop
sequences that omit verification are acceptable only for trivial steps
(`reset_robot`, `home pose`, `move_base` to a free area).

## Pattern

```python
def grasp_with_verification(target_pose, object_name, max_attempts=2):
    for attempt in range(max_attempts):
        # DO NOT add any x, y or z offsets to the grasp or pre-grasp pose
        # move to grasp pose
        move_ee_to_pose(target_pose)
        # try picking the target
        set_gripper_width(0.0)
        move_ee_to_rel_pose({"x":0,"y":0,"z":0.05})
        # verify grasp
        resp = verify_grasp(
            f"{object_name}"
        )
        answer = resp["data"]["grasped"]
        if answer:
            return True
        # Release before retrying so the next attempt isn't fighting the
        # previous closed gripper by resetting the robot
        reset_robot()
    return False
```

The pattern generalises to placement (`is the cube resting on the marker
without tipping?`) and to articulation (`is the drawer open enough to
reach inside?`).

## Phrasing the VQA question

- One question per outcome. Yes/no, never open-ended.
- Phrase so **"no" is the failure mode** — the VQA backend is more
  reliable at affirming what it sees than denying what it doesn't.
  - Good: `"Is the gripper holding the cube? Yes or No?"`
  - Avoid: `"Did the grasp fail?"` (double negative on "no")
- Name the object, not just the action. `"holding the red cube"` beats
  `"holding the object"` — concrete referents get more accurate answers.

## Failure handling

- Cap at **1 retry** total (`max_attempts=2`: one initial try + one
  retry). If the retry also fails, give up rather than looping
  indefinitely — a fragile sub-op that fails twice usually needs a
  re-plan, not more retries.
- On ambiguous answers (`"I can't tell"`, `"unclear"`), bias toward
  **continuing** — treat as soft-success — so the single retry isn't
  burned on VQA noise. The next outcome-check downstream will catch a
  true failure.
- Don't VQA-verify after every primitive. Only after the
  *outcome-bearing* one in each sub-operation. VQA latency in live mode
  is real (~0.5–2s per call).

## Return-shape reminder

Franka VQA returns `{"data": {"answer": "<string>"}}`. Always go through
`resp["data"]["answer"]`. Other robots may differ — see their `vqa.md`.
