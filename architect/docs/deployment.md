# Deployment

A real-robot run spans two machines:

| Machine | Role | Runs |
|---------|------|------|
| **GPU server** | Perception + motion-planning backends | CuRobo motion planning, Grounded-SAM2, AnyGrasp, AnyPlace, DIFT keypoints, and (optionally) a local Qwen2.5-VL VLM |
| **Robot control PC** | ROS1 + agent | ROS1 Noetic, the Franka robot API services (`franka_robot_apis`), the camera streamer, and the ARCHITECT CLI |

The two machines communicate over the LAN: the ROS services on the control PC
forward heavy perception / planning requests to the model servers on the GPU
server via WebSocket/HTTP.

## Model servers (GPU server)

Start each backend in its own environment. Point the robot stack at the GPU
server with the `inference_host` launch arg (see the workspace README). Default
ports:

| Server | Port | Purpose |
|---|---|---|
| CuRobo motion planning | `8765` | IK / trajectory generation (`ws://<gpu_host>:8765`) |
| Grounded-SAM2 | `8766` | open-vocabulary detection + segmentation |
| AnyGrasp | `8767` | grasp-pose synthesis |
| AnyPlace | `8768` | placement-pose prediction |
| DIFT | `8769` | task-conditioned semantic keypoints |
| Qwen2.5-VL *(optional)* | `8000` | local VLM backend for VQA (only if not using a cloud VLM) |

These servers are external components and are **not** part of this repository;
see the project page for references to each.

## Robot control PC

Three processes, in order. Each shell must have the ROS1 workspace sourced and
the `env_franka` conda env active (see the workspace `docs/INSTALL.md`).

1. **Camera streamer** — publishes the wrist + scene cameras over agentlace.
2. **Franka robot API** — `roslaunch franka_robot_apis franka_robot_core.launch`
   (the ROS1 service nodes; pass `inference_host:=<gpu_host>`).
3. **ARCHITECT CLI** — the interactive agent, started in a **separate terminal**:

```bash
# Terminal 2 (after the robot stack is up in Terminal 1)
conda activate env_franka
source <catkin_ws>/devel/setup.bash
python3 scripts/architect_cli.py --instruction "Pick up the red cup"
# or the convenience wrapper:
./run_architect.sh "Pick up the red cup"
```

`run_architect.sh` wraps the workspace sourcing + conda activation around
`scripts/architect_cli.py`. Two environment overrides let you adapt it to your machine:

- `ARCHITECT_CONDA_ENV=<name>` — conda env to activate (default `env_franka`)
- `ARCHITECT_ROS_WS=<path>` — ROS1 catkin workspace root (default: the parent of this
  `architect/` directory)

## Debug helpers

- **CuRobo / AnyGrasp visualisation.** Most model servers have a debug/viewer
  mode; run the server with its debug flag to inspect planned motions or detected
  grasps. Note AnyGrasp's viewer blocks the service until dismissed.
- **Rebuild after editing ROS packages.** Re-run `catkin_make` in the catkin
  workspace and re-source `devel/setup.bash`.
</content>
