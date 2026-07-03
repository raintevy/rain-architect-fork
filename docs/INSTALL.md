# Installation

This is a ROS 1 (Noetic) catkin workspace. The repository **is** the workspace:
packages live under `src/` (first-party at the top level, vendored dependencies
under `src/third_party/`). **libfranka is not shipped in this repo** — you clone
and build it separately (step 3); do **not** install it from conda/robostack.

The instructions use a [RoboStack](https://robostack.github.io/) conda
environment so the whole stack (ROS Noetic + toolchain) is self-contained. Adapt
freely if you run native ROS Noetic on Ubuntu 20.04.

## Verified versions

| Component | Version |
| --- | --- |
| gcc | 13.3.0 |
| Python | 3.9.18 |
| Franka FCI | 4.0.4 |
| libfranka | 0.8.0 |
| franka_ros | 0.8.0 |
| serl_franka_controllers | main |

Pick a workspace location and export it once; the commands below reference it.
Clone this repository into `$CATKIN_WS` (its `src/` becomes the workspace source;
`libfranka` is added in step 3):

```bash
export CATKIN_WS=~/architect_ws
# git clone <repo-url> "$CATKIN_WS"
cd "$CATKIN_WS"
```

## 1. Create the conda (RoboStack) environment

```bash
conda activate                       # base
conda config --env --remove channels https://repo.anaconda.com/pkgs/main
conda config --env --remove channels https://repo.anaconda.com/pkgs/r
conda config --env --remove channels defaults
conda create -n env_franka python=3.9 ros-noetic-desktop -c robostack -c conda-forge
conda activate env_franka
conda config --env --add channels robostack-noetic
conda deactivate && conda activate env_franka
conda install -c conda-forge ros-dev-tools

# Pin the toolchain (RoboStack may ship a newer gcc)
conda install -c conda-forge gcc=13.3.0
gcc --version        # 13.3.0
which python && python --version   # conda env, 3.9.x
which gcc            # conda env

# Build dependencies
conda install -c conda-forge eigen poco zeromq lz4
```

Confirm ROS works: `roscore` should start a master.

## 2. Initialize and resolve the workspace

The top-level `src/CMakeLists.txt` is environment-specific and is **not** checked
in — generate it, then resolve ROS dependencies:

```bash
cd "$CATKIN_WS"
catkin_init_workspace src
rosdep install --from-paths src --ignore-src --rosdistro noetic -y \
  --skip-keys "libfranka franka_gazebo"
```

Unused `robotiq` / `ros_canopen` subpackages are already marked with
`CATKIN_IGNORE` in the tree, so catkin skips them automatically.

> **NOTE:** Do **not** install libfranka via conda / robostack
> (e.g. `ros-noetic-libfranka`). Build it from source as below.

## 3. Clone, patch, and install libfranka (0.8.0)

libfranka is **not** part of this repository. Clone the pinned version into the
workspace root:

```bash
git clone --recursive --branch 0.8.0 \
  https://github.com/frankaemika/libfranka "$CATKIN_WS/libfranka"
```

Apply two small portability `#include`s — required for the gcc-13 / ROS Noetic
toolchain used here:

```bash
sed -i '5a #include <stdexcept>' "$CATKIN_WS/libfranka/src/control_types.cpp"
sed -i '6a #include <string>'    "$CATKIN_WS/libfranka/include/franka/control_tools.h"
```

Build and install (it installs into `$CONDA_PREFIX`, which the catkin build later
finds via `franka_DIR` — the in-tree source is not needed after this):

```bash
cd "$CATKIN_WS/libfranka"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j"$(nproc)"
cmake --install .
```

> The vendored `franka_ros` in this repo already carries its one portability
> include (`<cstdint>` in `franka_hw/resource_helpers.h`), so no patch is needed
> there.

## 4. Build the workspace

```bash
cd "$CATKIN_WS"
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_CXX_FLAGS="-include cstdint -include stdexcept -include string" \
  -Dfranka_DIR=$CONDA_PREFIX/lib/cmake/Franka
```

The `-DCMAKE_CXX_FLAGS="-include ..."` flags are required for the Franka/ROS
Noetic toolchain combination. `serl_franka_controllers` is already configured for
C++14 (for `std::allocator::rebind` compatibility with ROS Noetic) — no manual
edit is needed. To rebuild just that package:

```bash
catkin_make --pkg serl_franka_controllers \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_CXX_FLAGS="-include cstdint -include stdexcept -include string" \
  -Dfranka_DIR=$CONDA_PREFIX/lib/cmake/Franka
```

## 5. Python dependencies (agentlace + services)

`agentlace` carries a `CATKIN_IGNORE` and is installed with pip:

```bash
cd "$CATKIN_WS/src/third_party/agentlace"
pip install -e . --no-deps
pip install "numpy==1.26.4"     # required; newer numpy breaks the stack
pip install "scipy==1.13.1"
pip install pyzmq lz4 opencv-python-headless rosnumpy openai aiohttp open3d

# sanity check
python -c "from agentlace.action import ActionClient, ActionConfig; print('agentlace OK')"
python -c "import numpy; print(numpy.__version__)"   # 1.26.4
```

## 6. Cameras (optional, per hardware)

RealSense:

```bash
pip install pyrealsense2
python -c "import pyrealsense2 as rs; print(len(rs.context().devices), 'device(s)')"
```

ZED — install the ZED SDK, then:

```bash
cd /usr/local/zed && python get_python_api.py
python -c "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])"
```

## 7. Robotiq 2F gripper

```bash
pip install pymodbus==2.5.3
sudo usermod -a -G dialout $USER        # then re-login
# or, per-session: sudo chmod 666 /dev/ttyUSB0
```

Sanity-check the gripper standalone (use your device, e.g. `/dev/ttyUSB0`):

```bash
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
# Open:
rostopic pub /Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output \
  "{rACT: 1, rGTO: 1, rATR: 0, rPR: 0,   rSP: 255, rFR: 150}" --once
# Close:
rostopic pub /Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output \
  "{rACT: 1, rGTO: 1, rATR: 0, rPR: 255, rSP: 255, rFR: 150}" --once
```

## 8. Activate the environment (every shell)

```bash
conda activate env_franka && source "$CATKIN_WS/devel/setup.bash"
```

Next: configure the runtime (camera serials, server host, robot IP, API keys) —
see [Configuration](../README.md#configuration).

## 9. ARCHITECT agent application (second terminal)

The `architect/` agent runs in the **same `env_franka` env** as the robot stack (its
live path uses `rospy`). Install its extra Python dependencies into that env:

```bash
conda activate env_franka
pip install -r architect/requirements.txt      # anthropic, python-dotenv, rich, numpy
cp architect/.env.example architect/.env             # then add your ANTHROPIC_API_KEY etc.
```

That's the only ARCHITECT-specific setup — everything else (ROS, cameras, gripper) is
shared with the robot stack above. See [`architect/README.md`](../architect/README.md) for the
agent's configuration, flags, and usage.

## Troubleshooting

- **numpy got upgraded to 2.x** (a transitive pip dependency can do this):
  ```bash
  pip uninstall numpy -y && pip install "numpy==1.26.4"
  ```
- **`pip show` errors / stray `typing` backport** shadowing the stdlib:
  ```bash
  pip uninstall typing -y
  rm -f "$CONDA_PREFIX/lib/python3.9/site-packages/typing.py"
  ```
- **cryptography import errors in conda base**: `conda update -n base cryptography --no-plugins`
- **`pyrealsense2` fails to load `libudev.so.0`** (Ubuntu 22.x ships `.so.1`):
  ```bash
  sudo ln -s /lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libudev.so.0
  ```
- **Robotiq Python-3 indentation errors**: the vendored
  `robotiq_modbus_rtu/.../comModbusRtu.py` and
  `robotiq_2f_gripper_control/.../baseRobotiq2FGripper.py` were ported to
  Python 3; if you re-pull upstream copies you may need to re-apply those fixes.
