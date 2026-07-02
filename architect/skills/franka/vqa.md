<!-- architect-meta
always_loaded: true
description: VQA prompt phrasing conventions — yes/no question style, response shape, when to use VQA vs detect_*.
-->
# Visual Question Answering

## When to use `get_vqa_response(prompt)`
- **Grasp verification**: "Is the robot currently holding an object?" — check after closing gripper before lifting
- **Task completion**: "Is the cup on the shelf?" — verify placement before releasing
- **Disambiguation**: "Which of the two cups is red?" — when multiple objects match a description
- **Spatial reasoning**: "Is the marker to the left or right of the bowl?" — when relative position matters and markers alone are insufficient
- **State check before a critical action**: "Is the path in front of the robot clear?"

## When NOT to use VQA
- Object localization — use `detect_markers()` or `detect_objects()` instead; VQA does not return poses
- Joint or EE state — use `get_current_joints()`, `get_current_ee_pose()`, `get_current_gripper_width()`
- Routine steps with predictable outcomes — don't add VQA checks to every motion step

## Return format
`get_vqa_response` returns `{"data": {"answer": "<string>"}}`. Always access via `result["data"]["answer"]`.

## Prompt style
Ask yes/no or short-answer questions. Avoid open-ended prompts.
- Good: `"Is the gripper holding the stuffed animal? Yes or No?"`
- Good: `"Is the cup upright or tipped over?"`
- Avoid: `"Describe the scene."` (use `get_scene_description` tool instead)

## When to use "wrist" versus "scene" camera
- Use the `wrist` camera when verifying a grasp
- Use the `scene` camera to validate the state of the environment such as the end effectors and the object positions, number of objects in the scene, get overall understanding of the scene