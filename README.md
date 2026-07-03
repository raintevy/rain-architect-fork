# ARCHITECT

### A Few Words Go a Long Way: Language Guided Robot Policy Synthesis

[![Project Page](https://img.shields.io/badge/project-page-blue)](https://robo-architect.github.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ROS](https://img.shields.io/badge/ROS-Noetic-22314E)](https://wiki.ros.org/noetic)

ARCHITECT synthesizes executable robot manipulation policies from natural-language
instructions and human corrections, treating policy acquisition as interactive
program synthesis: an LLM coding agent composes modular programs from a library of
perception and control *tools*, and successful programs are saved to a persistent
skill library that carries forward to new tasks.

This repository has two parts that run together:

1. **`architect/` — the agent application (main entry point).** An LLM coding agent that
   turns a natural-language instruction into an executable robot program, runs it
   closed-loop with VQA-gated success checks, and refines it from human
   corrections — accumulating reusable skills across sessions. See
   **[`architect/README.md`](architect/README.md)**.
2. **The robot-side software stack (a ROS 1 Noetic workspace).** Turns a Franka Emika Panda, a
   Robotiq 2F gripper, and RGB-D cameras into the tool layer the agent calls:
   motion, gripper, proprioception, and perception exposed as uniform
   JSON-over-ROS services. Heavy computation (IK/trajectory generation, grasp and
   placement synthesis, keypoints, VQA) is delegated to model servers over
   WebSocket/HTTP.

For the method, experiments, and videos, see the **[project page](https://robo-architect.github.io/)**.

## Features

- **Uniform service API** — every capability is a `/robot/...` ROS service with
  JSON request/response payloads, easy for an LLM agent (or any client) to call.
- **Cartesian-impedance motion** with absolute, relative, and contact-*guarded*
  moves; automatic recovery from reflex/collision stops.
- **Perception tool suite** — open-vocabulary detection & segmentation, grasp
  synthesis, placement-pose prediction, semantic keypoints, and VQA.
- **Multi-camera streaming** for RealSense and ZED (scene + wrist) with calibrated
  intrinsics/extrinsics.

```
.
├── architect/                        # agent application (main entry point) — see architect/README.md
├── src/
│   ├── franka_robot_apis/      # robot tool API: motion/gripper/perception services, camera clients
│   ├── robot_api_interfaces/   # RobotCommand / RobotQuery service + ResultCode message
│   └── third_party/            # vendored dependencies (see Acknowledgements)
└── docs/INSTALL.md             # libfranka is cloned + built separately (see docs/INSTALL.md - step 3)
```

## Prerequisites
(our testing was done on Ubuntu 22.04 linux workstation with RoboStack environment setup for Ubuntu 20.04 as follows)
- Ubuntu 20.04 with ROS Noetic, or a [RoboStack](https://robostack.github.io/) conda environment
- Franka Panda/FE with FCI, Robotiq 2F gripper
- RealSense and/or ZED RGB-D cameras
- Reachable model servers for IK and perception (see [External services](#external-services))
- Python 3.9, `numpy==1.26.4`

## Installation

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full, version-pinned setup
(RoboStack environment, building libfranka 0.8.0, `catkin_make` flags, Python
dependencies, and camera/gripper setup).

## Configuration

Runtime configuration is kept out of the code, in `src/franka_robot_apis/`:

```bash
cd src/franka_robot_apis

# API Keys + camera serials
cp .env.example .env && $EDITOR .env

# Servers / cameras / robot (intrinsics, endpoints, workspace bounds, ...)
cp config/servers.example.yaml config/servers.yaml
cp config/cameras.example.yaml config/cameras.yaml
cp config/robot.example.yaml   config/robot.yaml
```

The most common values are also exposed as launch arguments: `robot_ip`,
`gripper_device`, and `inference_host` (the host serving the perception/IK models).

## Usage

```bash
conda activate env_franka && source devel/setup.bash

roslaunch franka_robot_apis franka_robot_core.launch \
  robot_ip:=172.16.0.2 \
  gripper_device:=/dev/ttyUSB0 \
  inference_host:=<model-server-host>
```

This starts the Cartesian impedance controller, the gripper node, the camera
streaming clients, and all `franka_robot_apis` service nodes.

Then, in a **second terminal**, run the ARCHITECT agent against the live stack:

```bash
conda activate env_franka && source devel/setup.bash
cd architect && ./run_architect.sh "Pick up the red cup"
```

See **[`architect/README.md`](architect/README.md)** for the agent's setup, flags, and usage.

## Service API

All endpoints live under `/robot/...` and use two service types from
`robot_api_interfaces`, with JSON-encoded payloads:

- **`RobotCommand`** (actions) — request `string req` (JSON); response
  `ResultCode result_code` + `string data` (JSON).
- **`RobotQuery`** (reads) — empty request; response `ResultCode result_code` +
  `string data` (JSON).

`ResultCode`: `SUCCESS=0, FAILURE=1, INVALID_INPUT=2, TIMEOUT=3, SERVICE_NOT_RUNNING=4`.

| Namespace | Services |
| --- | --- |
| `/robot/control/*` | `move_ee_to_pose`, `move_ee_to_rel_pose`, `move_ee_guarded`, `reset_robot`, `execute_waypoint_trajectory`, `set_gripper_width`, `rotate_wrist`, `recover_from_reflex` |
| `/robot/proprioception/*` | `get_current_ee_pose`, `get_current_joints` |
| `/robot/perception/*` | `detect_objects`, `get_object_mask`, `get_grasp_config`, `get_keypoints`, `get_placing_pose`, `detect_marker`, `get_vqa_response`, `verify_grasp` |

Example:

```bash
rosservice call /robot/control/move_ee_to_rel_pose \
  "req: '{\"delta_position\": {\"x\": 0.0, \"y\": 0.1, \"z\": 0.3}}'"
```

Each service node's module docstring documents its exact request/response JSON
schema and a `rosservice call` example.

## External services

Several perception/motion services are thin ROS wrappers around model servers
reached over WebSocket/HTTP; point them at your hosts via `inference_host` and
`config/servers.yaml`. Default ports: IK / trajectory `8765`, VLM (Qwen) `8000`,
segmentation `8766`, grasp synthesis `8767`, placement `8768`, keypoints `8769`.
VQA can alternatively use OpenAI / Azure OpenAI via `.env`. These servers are not
part of this repository.

## Acknowledgements

This project builds on several open-source works, vendored under
`src/third_party/`:

- [franka_ros](https://github.com/frankaemika/franka_ros) and [libfranka](https://github.com/frankaemika/libfranka) — Franka arm driver and ROS interface
- [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers) — Cartesian impedance controller
- [robotiq](https://github.com/ros-industrial/robotiq) — Robotiq gripper drivers
- [ros_canopen](https://github.com/ros-industrial/ros_canopen) and [soem](https://github.com/OpenEtherCATsociety/SOEM) — CANopen / EtherCAT support
- [agentlace](https://github.com/youliangtan/agentlace) — distributed data streaming

Each retains its own license under its package directory. The perception and
planning model servers (e.g. cuRobo, Grounded-SAM2, AnyGrasp, AnyPlace, DIFT)
are external components; see the [project page](https://robo-architect.github.io/)
for references.

## License

This project is released under the [MIT License](LICENSE). Dependencies vendored
under `src/third_party/` are licensed separately under their respective licenses.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{architect2026,
  title  = {A Few Words Go a Long Way: Language Guided Robot Policy Synthesis},
  author = {Anonymous Author(s)},
  year   = {2026},
  note   = {Under review. Project page: https://robo-architect.github.io/}
}
```
