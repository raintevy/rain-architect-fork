<!-- architect-meta
always_loaded: true
description: ROS1 / Noetic CLI introspection patterns — when and how to use run_ros2_command for environment debugging.
-->
# ROS1 CLI Introspection (Franka / Noetic)

## When to use `run_ros2_command` (runs as a shell command)
On the Franka system, ROS1 Noetic is used. The tool still accepts any shell command — use `rosservice`, `rostopic`, and `rosnode` commands instead of `ros2` equivalents.

> Note: `run_ros2_command` checks that the command starts with `ros2` by default. On the Franka, pass the full ROS1 command (e.g. `rosservice list`) and expect the executor to run it as-is.

## Useful ROS1 commands
```bash
rosservice list                                    # all active services
rostopic list                                      # all active topics
rosservice type /robot/control/move_ee_to_pose     # inspect a service type
rosservice info /robot/perception/detect_objects   # service details
rosnode list                                       # active nodes
```

## Service naming convention
All Franka robot API services follow the same pattern as the Stretch:
`/robot/{category}/{function_name}`

Categories:
- `/robot/perception/` — detect_objects, get_vqa_response, verify_grasp
- `/robot/proprioception/` — get_current_ee_pose, get_current_joints, get_gripper_width
- `/robot/control/` — move_ee_to_pose, move_ee_to_rel_pose, set_gripper_width, reset_robot, move_ee_guarded

## Caution
- Use only read-only commands (`list`, `type`, `info`, `echo`) unless deliberately adding a new service
- `move_ee_guarded` is available as `/robot/control/move_ee_guarded` but is not yet in the Python API — use `run_ros2_command` to inspect its interface before implementing a primitive that calls it
