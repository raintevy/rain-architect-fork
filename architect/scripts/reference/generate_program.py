#!/usr/bin/env python3
"""Generate a robot program from a demonstration and/or language instruction.

Reference only — NOT part of the supported release. This is the historical,
non-interactive precursor to ``scripts/architect_cli.py`` (the interactive agentic
CLI, which supersedes it). It targets the demonstration-preprocessing pipeline,
which has been removed from this release, so it is retained for reference and is
not runnable as-is. Use ``scripts/architect_cli.py`` instead.

Supports three modes based on which arguments are provided:
  demo + instruction  ->  demo_language   (trajectory + language)
  demo only           ->  demo_only       (trajectory replay)
  instruction only    ->  language_only   (perception-driven)

Usage:
    # Demo + language
    python scripts/generate_program.py \
        --demo output/22_1.json \
        --instruction "Pick up the cup from the table"

    # Demo only
    python scripts/generate_program.py --demo output/22_1.json

    # Language only
    python scripts/generate_program.py \
        --instruction "Pick up the red cup from the table"
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
from architect.llm.client import call_claude
from architect.llm.prompts import build_correction_prompt, build_prompt


def strip_fences(program: str) -> str:
    """Strip markdown code fences if the model wrapped the output."""
    stripped = program.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return program


def generate(demo: dict | None, instruction: str | None, model: str = ANTHROPIC_MODEL) -> str:
    """Generate a robot program string from a demo dict and/or instruction.

    Args:
        demo: Preprocessed demo dict (from preprocess_demo.py), or None.
        instruction: Natural-language task instruction, or None.
        model: Anthropic model name.

    Returns:
        Generated program as a string.
    """
    # Demo-trajectory generation has historically been Stretch-only (the
    # preprocessor emits Stretch joint configs); resolve the robot config
    # explicitly so the prompts get the right api_spec + display name.
    from architect.robots import get_robot_config
    cfg = get_robot_config("stretch")
    messages = build_prompt(
        demo=demo, instruction=instruction,
        api_spec=cfg.api_spec, robot_label=cfg.display_name,
    )
    return strip_fences(call_claude(messages, model=model))


def refine(
    current_program: str,
    correction: str,
    model: str = ANTHROPIC_MODEL,
    current_ee_pose: dict | None = None,
) -> str:
    """Refine an existing program by applying a natural-language correction.

    Args:
        current_program: The current program code.
        correction: The user's correction instruction.
        model: Anthropic model name.
        current_ee_pose: Current EE pose from the robot (if available).

    Returns:
        The modified program as a string.
    """
    from architect.robots import get_robot_config
    cfg = get_robot_config("stretch")
    messages = build_correction_prompt(
        current_program, correction,
        api_spec=cfg.api_spec, robot_label=cfg.display_name,
        current_ee_pose=current_ee_pose,
    )
    return strip_fences(call_claude(messages, model=model))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        "--output", "-o",
        default=None,
        help="Output file for the generated program. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--model", "-m",
        default=ANTHROPIC_MODEL,
        help=f"Anthropic model to use (default: {ANTHROPIC_MODEL}).",
    )
    args = parser.parse_args()

    if args.demo is None and args.instruction is None:
        parser.error("At least one of --demo or --instruction is required.")

    # Determine mode
    if args.demo is not None and args.instruction is not None:
        mode = "demo_language"
    elif args.demo is not None:
        mode = "demo_only"
    else:
        mode = "language_only"

    # Load preprocessed demo if provided
    demo = None
    if args.demo is not None:
        with open(args.demo) as f:
            demo = json.load(f)

    # Print run info
    print(f"Mode: {mode}")
    if demo is not None:
        print(f"Demo: {demo['source_file']}  ({demo['num_segments']} segments, {demo['total_duration_s']:.1f}s)")
    if args.instruction is not None:
        print(f"Instruction: {args.instruction}")
    print(f"Model: {args.model}")
    print()

    program = generate(demo=demo, instruction=args.instruction, model=args.model)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(program)
        print(f"Wrote generated program to {output_path}")
    else:
        print("--- Generated Program ---")
        print(program)
        print("--- End ---")


if __name__ == "__main__":
    main()
