#!/usr/bin/env python3
"""End-to-end CLI: generate a robot program and execute it on the live robot.

Reference only — NOT part of the supported release. This is the historical,
non-interactive precursor to ``scripts/architect_cli.py`` (the interactive agentic
CLI, which supersedes it). It depends on the demonstration-preprocessing
pipeline, which has been removed from this release, so it is retained for
reference and is not runnable as-is. Use ``scripts/architect_cli.py`` instead.

Generation modes (same as generate_program.py):
  demo + instruction  ->  demo_language
  demo only           ->  demo_only
  instruction only    ->  language_only

At least one of --demo, --instruction, or --program must be provided.

Usage:
    # Generate from instruction and run
    python scripts/run_program.py \\
        --instruction "Pick up the cup from the table" \\
        --save output/program_cup.py

    # Generate from demo + instruction and run
    python scripts/run_program.py \\
        --demo output/22_3.json \\
        --instruction "Place the bowl on the rack"

    # Run a pre-generated program directly
    python scripts/run_program.py --program output/program_22_3.py

    # Test generation + correction loop without a live robot
    python scripts/run_program.py \\
        --instruction "Pick up the red cup" \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ANTHROPIC_MODEL
from generate_program import generate, refine

DEMO_CODE = """
# Detect the yellow cup and get its best grasp pose
yellow_cup = detect_objects("yellow cup")
grasp = yellow_cup["best_grasp"]

# Open the gripper wider than the grasp width
set_gripper_width(grasp["width"] + 0.02)

# Move end-effector to the grasp pose
target_pose = {
    "position": {
        "x": grasp["translation_wrt_base"][0],
        "y": grasp["translation_wrt_base"][1],
        "z": grasp["translation_wrt_base"][2]
    },
    "orientation": grasp["quaternion_wrt_base"]
}
move_ee_to_pose(target_pose)

# Close the gripper to grasp the cup
set_gripper_width(0.05)

# Lift the cup by moving the end-effector up
move_ee_to_rel_pose({"x": 0.0, "y": 0.0, "z": 0.10})
"""

# Fake EE pose returned by the dry-run stub
_DRY_RUN_EE_POSE = {
    "position": {"x": 0.0, "y": -0.6, "z": 0.8},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


def _make_dry_run_namespace() -> dict:
    """Return a namespace of stub robot API functions that print calls instead of executing them."""

    def _stub(name, *args, **kwargs):
        arg_strs = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        print(f"  [dry-run] {name}({', '.join(arg_strs)})")

    return {
        "detect_markers": lambda: (
            _stub("detect_markers") or {"markers": []}
        ),
        "detect_objects": lambda text_prompt: (
            _stub("detect_objects", text_prompt) or {
                "best_grasp": {
                    "translation_wrt_base": [0.0, -0.65, 0.85],
                    "quaternion_wrt_base": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "score": 0.9,
                    "width": 0.08,
                }
            }
        ),
        "get_vqa_response": lambda prompt: (
            _stub("get_vqa_response", prompt) or "yes"
        ),
        "verify_grasp": lambda object_name: (
            _stub("verify_grasp", object_name) or {"data": {"grasped": True, "reason": "stable grasp"}}
        ),
        "get_current_joints": lambda: (
            _stub("get_current_joints") or {
                "joint_lift": {"position": 0.6, "velocity": 0.0, "effort": 0.0},
                "wrist_extension": {"position": 0.1, "velocity": 0.0, "effort": 0.0},
                "joint_wrist_yaw": {"position": 0.0, "velocity": 0.0, "effort": 0.0},
                "joint_wrist_pitch": {"position": -0.6, "velocity": 0.0, "effort": 0.0},
                "joint_wrist_roll": {"position": 0.0, "velocity": 0.0, "effort": 0.0},
            }
        ),
        "get_current_gripper_width": lambda: (
            _stub("get_current_gripper_width") or {"gripper_width": 0.152}
        ),
        "get_current_ee_pose": lambda: (
            _stub("get_current_ee_pose") or _DRY_RUN_EE_POSE
        ),
        "get_current_odom": lambda: (
            _stub("get_current_odom") or {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        ),
        "set_arm_joints": lambda target_joints, **kw: (
            _stub("set_arm_joints", target_joints, **kw) or True
        ),
        "set_gripper_width": lambda width: (
            _stub("set_gripper_width", width) or True
        ),
        "move_base": lambda x, **kw: (
            _stub("move_base", x, **kw) or True
        ),
        "move_base_to_rel": lambda x, **kw: (
            _stub("move_base_to_rel", x, **kw) or True
        ),
        "move_ee_to_pose": lambda target_pose: (
            _stub("move_ee_to_pose", target_pose) or True
        ),
        "move_ee_to_rel_pose": lambda delta_position: (
            _stub("move_ee_to_rel_pose", delta_position) or True
        ),
        "set_camera_pose": lambda target_joints, **kw: (
            _stub("set_camera_pose", target_joints, **kw) or True
        ),
        "reset_robot": lambda: (
            _stub("reset_robot") or True
        ),
    }


def _make_live_namespace() -> tuple:
    """Initialise rclpy + RobotClient and return (namespace, client)."""
    import rclpy
    from architect.robots.stretch_client import RobotClient

    rclpy.init()
    client = RobotClient()
    namespace = {
        "detect_markers": client.detect_markers,
        "detect_objects": client.detect_objects,
        "get_vqa_response": client.get_vqa_response,
        "verify_grasp": client.verify_grasp,
        "get_current_joints": client.get_current_joints,
        "get_current_gripper_width": client.get_current_gripper_width,
        "get_current_ee_pose": client.get_current_ee_pose,
        "get_current_odom": client.get_current_odom,
        "set_arm_joints": client.set_arm_joints,
        "set_gripper_width": client.set_gripper_width,
        "move_base": client.move_base,
        "move_base_to_rel": client.move_base_to_rel,
        "move_ee_to_pose": client.move_ee_to_pose,
        "move_ee_to_rel_pose": client.move_ee_to_rel_pose,
        "set_camera_pose": client.set_camera_pose,
        "reset_robot": client.reset_robot,
    }
    return namespace, client


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo", "-d",
        default=None,
        help="Path to preprocessed demo JSON (output of preprocess_demo.py).",
    )
    parser.add_argument(
        "--instruction", "-n",
        default=None,
        help="Natural-language task instruction for the program.",
    )
    parser.add_argument(
        "--program", "-p",
        default=None,
        help="Path to a pre-generated .py file. Skips the generation step.",
    )
    parser.add_argument(
        "--save", "-s",
        default=None,
        help="Save the generated program to this path before running.",
    )
    parser.add_argument(
        "--model", "-m",
        default=ANTHROPIC_MODEL,
        help=f"Anthropic model to use (default: {ANTHROPIC_MODEL}).",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Use hardcoded demo program instead of generating one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub out all robot API calls (no ROS required). Useful for testing generation and the correction loop.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive correction loop after execution.",
    )
    args = parser.parse_args()

    if args.demo is None and args.instruction is None and args.program is None and not args.test:
        parser.error("At least one of --demo, --instruction, or --program is required.")

    # ------------------------------------------------------------------
    # Step 1: Obtain program code
    # ------------------------------------------------------------------
    if args.test:
        program_code = DEMO_CODE
        print("Test mode: using hardcoded demo program.")
    elif args.program is not None:
        program_code = Path(args.program).read_text()
        print(f"Loaded pre-generated program from {args.program}")
    else:
        demo = None
        if args.demo is not None:
            with open(args.demo) as f:
                demo = json.load(f)

        if demo is not None and args.instruction is not None:
            mode = "demo_language"
        elif demo is not None:
            mode = "demo_only"
        else:
            mode = "language_only"

        print(f"Mode: {mode}")
        if demo is not None:
            print(f"Demo: {demo['source_file']}  ({demo['num_segments']} segments, {demo['total_duration_s']:.1f}s)")
        if args.instruction is not None:
            print(f"Instruction: {args.instruction}")
        print(f"Model: {args.model}")
        if args.dry_run:
            print("Dry-run: robot API calls will be stubbed out.")
        print()

        program_code = generate(demo=demo, instruction=args.instruction, model=args.model)

        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(program_code)
            print(f"Saved generated program to {save_path}")

    print()
    print("--- Generated Program ---")
    print(program_code)
    print("--- End ---")
    print()

    # ------------------------------------------------------------------
    # Step 2: Build execution namespace (live robot or dry-run stubs)
    # ------------------------------------------------------------------
    client = None
    if args.dry_run:
        namespace = _make_dry_run_namespace()
    else:
        namespace, client = _make_live_namespace()

    try:
        print("Executing program...")
        exec(program_code, namespace)  # noqa: S102
        print("Execution complete.")

        # --------------------------------------------------------------
        # Step 3: Interactive correction loop
        # --------------------------------------------------------------
        interactive = (args.program is None) and (not args.no_interactive)
        save_path = Path(args.save) if args.save else None

        while interactive:
            try:
                correction = input("\nCorrection (or 'done' to finish): ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if correction.strip().lower() in ("done", "exit", "quit", ""):
                break

            # Capture current EE pose so the LLM can reason about spatial offsets
            current_ee_pose = None
            try:
                current_ee_pose = namespace["get_current_ee_pose"]()
            except Exception as e:
                print(f"  (Could not read EE pose: {e})")

            print(f"\nRefining program with correction: {correction!r}")
            program_code = refine(
                program_code, correction,
                model=args.model, current_ee_pose=current_ee_pose,
            )

            print("\n--- Revised Program ---")
            print(program_code)
            print("--- End ---\n")

            if save_path:
                save_path.write_text(program_code)
                print(f"Saved revised program to {save_path}")

            print("Resetting robot...")
            namespace["reset_robot"]()
            print("Re-executing program...")
            exec(program_code, namespace)  # noqa: S102
            print("Execution complete.")

    finally:
        if client is not None:
            import rclpy
            client.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
