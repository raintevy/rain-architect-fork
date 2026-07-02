# Reference scripts

`scripts/reference/generate_program.py` and `scripts/reference/run_program.py`
are the historical, non-interactive precursors to `scripts/architect_cli.py`. They
predate the agentic loop and do not support the `--condition` / `--task` flag
set.

They are retained **for reference only** and are **not runnable in this
release**: both were built around the demonstration-preprocessing pipeline
(teleoperated demo recordings → segmented JSON → program), which has been
removed. Use `scripts/architect_cli.py` instead — see the [README](../README.md).

- `generate_program.py` — generated a program from a demo and/or instruction and
  wrote it to a file/stdout.
- `run_program.py` — generated and/or executed a program with an optional
  correction loop.
</content>
