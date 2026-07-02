#!/usr/bin/env bash
# Run ARCHITECT agentic synthesis on the Franka Panda. Wraps the environment setup
# (ROS1 workspace sourcing + conda activation) around scripts/architect_cli.py:
#
#   conda activate env_franka
#   source <catkin_ws>/devel/setup.bash
#   export PATH="$CONDA_PREFIX/bin:$PATH"
#   python3 scripts/architect_cli.py ...
#
# Three invocation patterns:
#
#   # 1. Instruction-driven generation:
#   ./run_architect.sh "Pick up the red cup"
#   ./run_architect.sh "Pick up the red cup" --dry-run
#   ./run_architect.sh "Pick up the red cup" --task pick_block_on_marker --condition PatternReuse
#
#   # 2. Pre-written program debugging (no instruction needed — first arg
#   #    is a flag, the script skips the positional-instruction step):
#   ./run_architect.sh --program /path/to/program.py
#   ./run_architect.sh --program /path/to/program.py --task pick_block_on_marker
#
#   # 3. Task-only (instruction auto-fills from tasks.json):
#   ./run_architect.sh --task pick_block_on_marker --condition PatternReuse

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The Franka ROS1 catkin workspace is the parent of this repo by default
# (this `architect/` package lives inside the catkin workspace). Try setup.bash,
# then setup.zsh (or vice-versa); warn-don't-fail if neither exists so
# --dry-run still works on machines without ROS. Override with ARCHITECT_ROS_WS=...
ROS_WS="${ARCHITECT_ROS_WS:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ENV_NAME="${ARCHITECT_CONDA_ENV:-env_franka}"

# --- Argument validation ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [\"<instruction>\"] [extra architect_cli.py args...]" >&2
    echo "       (instruction is optional when --program or --task is supplied)" >&2
    exit 1
fi

# If the first arg starts with `-` it's a flag, not the instruction.
# Skip the positional-instruction step in that case so callers can run
# program-only or task-only invocations without a placeholder string.
if [[ "$1" == -* ]]; then
    INSTRUCTION=""
else
    INSTRUCTION="$1"
    shift  # remaining args (e.g. --dry-run, --save) are passed through
fi

# --- ROS1 workspace ---
# Try setup.zsh first, then setup.bash as a fallback for bash shells.
# Non-fatal: --dry-run works without ROS.
if [[ -f "$ROS_WS/devel/setup.zsh" ]]; then
    # shellcheck source=/dev/null
    source "$ROS_WS/devel/setup.zsh"
elif [[ -f "$ROS_WS/devel/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "$ROS_WS/devel/setup.bash"
else
    echo "Warning: ROS1 workspace not found at $ROS_WS/devel/setup.{zsh,bash} — skipping." >&2
    echo "         rospy will not be available; use --dry-run for offline testing." >&2
    echo "         Override the workspace path with ARCHITECT_ROS_WS=... if needed." >&2
fi

# --- Conda environment ---
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null)" || true
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
    else
        echo "Error: conda not found and env '$CONDA_ENV_NAME' is not active." >&2
        echo "       Activate it manually and re-run," >&2
        echo "       or override with ARCHITECT_CONDA_ENV=<name>." >&2
        exit 1
    fi
fi

# --- PATH: ensure the conda env's Python wins over ROS-prepended ones ---
# ROS sourcing prepends /opt/ros/.../bin and system Python paths to PATH;
# without re-exporting the conda env's bin dir first, generated programs
# can end up running against the wrong Python and lose access to the
# env's installed packages.
CONDA_BIN="${CONDA_PREFIX:-$HOME/miniconda3/envs/$CONDA_ENV_NAME}/bin"
if [[ -d "$CONDA_BIN" ]]; then
    export PATH="$CONDA_BIN:$PATH"
fi

# --- Run ---
cd "$SCRIPT_DIR"
if [[ -n "$INSTRUCTION" ]]; then
    python3 scripts/architect_cli.py \
        --robot franka \
        --instruction "$INSTRUCTION" \
        "$@"
else
    # No instruction supplied — let architect_cli.py source it from --task
    # or rely on --program. The CLI's own argparse check will error
    # cleanly if none of --instruction / --task / --program / --demo
    # is given.
    python3 scripts/architect_cli.py \
        --robot franka \
        "$@"
fi
