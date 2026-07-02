# ARCHITECT — Language-Guided Robot Program Synthesis

The agent application for the project (see the root [README](../README.md)).
ARCHITECT casts robot policy acquisition as **interactive program synthesis**: an
LLM coding agent turns a language instruction — and any follow-up corrections —
into a modular, executable program composed from the robot's perception and
control *tools*, rather than a black-box policy. Programs run closed-loop against
the live ROS API, with VQA-gated success checks and in-program retry; on failure
the agent localizes the fault and refines the program from a human correction.
Because programs are modular and built from named tools, failures stay isolated
and feedback is localizable, and validated sub-routines accumulate in a
**persistent skill library** that transfers to later tasks (long-term in-context
learning). The result is an interpretable, steerable alternative to end-to-end
vision-language-action policies, robust to how an instruction is phrased.

This is the **main entry point** for the system. It runs in a **second
terminal**, alongside the robot stack:

- **Terminal 1 — robot stack:** the ROS 1 workspace in this repo
  (`roslaunch franka_robot_apis franka_robot_core.launch …`); see the root
  [README](../README.md).
- **Terminal 2 — ARCHITECT:** the agent CLI described here.

## How it works

1. You issue a task instruction in language.
2. The coding agent introspects the tool API and current robot state, then
   synthesizes a program composed from perception/control tools — reusing
   validated helpers/patterns retrieved from the skill library where relevant.
3. During execution, VQA queries gate per-step success and drive in-program
   retry. On a true failure the agent localizes the fault and requests a
   correction; the program is refined and re-run — the human-in-the-loop
   refinement step.

## Highlights

- **Agentic synthesis** — tool-use loop that writes, runs, and repairs programs.
- **Persistent skill library** — Python helpers (`skills/<robot>/library.db`)
  and/or markdown subtask docs (`skills/<robot>/subtasks/*.md`) that accumulate
  across sessions.
- **Four eval conditions** (`ICL`, `FuncReuse`, `SlateGated`, `PatternReuse`)
  selectable via `--condition` for ablation runs.
- **Counterfactual probes** — graduation gate, slate of alternative programs,
  audit penalties, embedding-based retrieval.
- **Closed-loop corrections** — per-call VLM observations, fault-localisation
  summary, and post-execution scoring feed automatic refinement.
- **Task scoring** — `--task <id>` prints a PASS/FAIL line after each run; PASS
  auto-exits, FAIL (with `--closed-loop`) auto-issues a refining correction.
- **Agent-memory features** (opt-in) — structured failure diagnosis + persistent
  failure store, an agent-curated session scratchpad, and hybrid skill-doc
  retrieval. See [`docs/agent_memory.md`](docs/agent_memory.md).

Embeddings (used wherever semantic similarity is needed) run on
`text-embedding-3-small` via either Azure OpenAI (`AZURE_OPENAI_API_KEY`) or
OpenAI (`OPENAI_API_KEY`), auto-selected by which credential is in `.env`.

## Setup

ARCHITECT runs in the **same conda env as the robot stack** (`env_franka`), because the
live path uses `rospy` to call the ROS services. Install the extra Python deps
into that env:

```bash
conda activate env_franka
pip install -r requirements.txt      # anthropic, python-dotenv, rich, numpy
cp .env.example .env                  # then fill in your keys (see below)
```

`.env` keys:

- `ANTHROPIC_API_KEY` — required (Claude).
- `ANTHROPIC_BASE_URL` — optional; set for an Anthropic-compatible endpoint
  (e.g. Azure AI Foundry). Blank uses the default Anthropic API.
- `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`, **or** `OPENAI_API_KEY` —
  optional; enables embedding-based retrieval.

For the full multi-machine layout (GPU model servers + robot control PC), see
[`docs/deployment.md`](docs/deployment.md).

> `--dry-run` needs no robot/ROS and only `ANTHROPIC_API_KEY`, so the agent loop
> can be exercised offline in any Python 3.11+ env with `requirements.txt`.

## Quick start

`run_architect.sh` wraps ROS-workspace sourcing + conda activation + the CLI:

```bash
# Instruction-driven (live robot)
./run_architect.sh "Pick up the red cup"

# Debug a pre-written program (first arg is a flag → no instruction needed)
./run_architect.sh --program /tmp/your_program.py

# Task-driven — instruction auto-fills from tasks.json + PASS/FAIL scoring
./run_architect.sh --task pick_block_on_marker --condition PatternReuse

# Offline smoke test (no robot, no API cost)
./run_architect.sh "Pick up the red cup" --dry-run
```

On `PASS` the session auto-exits; on `FAIL` the correction prompt opens (or, with
`--closed-loop`, a refining correction is issued automatically).

## Conditions matrix

Each condition toggles a set of mechanisms; the matrix is canonical in
`scripts/eval/tasks.json`'s `conditions` block and reused by `--condition`.

| Condition | library | slate | graduation | retrieval | markdown_library | inprogram_retry |
|---|---|---|---|---|---|---|
| **ICL** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **FuncReuse** | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **SlateGated** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **PatternReuse** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

See [`docs/mechanisms.md`](docs/mechanisms.md) for the feature-level architecture.

## Interactive CLI

```bash
python3 scripts/architect_cli.py --instruction "Pick up the red cup"
```

After each execution the CLI surfaces a correction prompt:

| Command | Description |
|---------|-------------|
| *(free text)* | Run an agentic correction loop and re-execute |
| `undo` | Revert the last correction or manual edit |
| `edit` | Open the current program in `$EDITOR` |
| `show` | Print the current program |
| `reset` | Re-execute the current program from the home pose |
| `/library` | Render the persistent skill library as a status table |
| `/graduate <name>` | Force-promote a candidate skill |
| `/forget <name>` | Delete a skill from the store + active namespace |
| `done` | Exit the correction loop |

### Key flags

| Flag | Short | Description |
|---|---|---|
| `--instruction` | `-n` | Natural-language task instruction |
| `--program` | `-p` | Path to a pre-generated `.py` file; skips generation |
| `--save` | `-s` | Save the generated program before execution |
| `--task` | `-t` | Task id from `tasks.json`; attaches its success scorer (PASS/FAIL), auto-fills the instruction |
| `--condition` | `-c` | `ICL` / `FuncReuse` / `SlateGated` / `PatternReuse` |
| `--closed-loop` | | On FAIL, auto-issue a refining correction (until PASS or `--max-auto-corrections`) |
| `--max-auto-corrections` | | Cap on auto-correction rounds (default 2) |
| `--no-vlm-tracing` | | Disable the trace wrapper's per-call VLM observations |
| `--robot` | `-r` | Robot platform: `franka` (default) |
| `--model` | `-m` | Claude model (default from `config.py`) |
| `--dry-run` | | Run without a live robot; API calls print to console |
| `--retrieval hybrid` | | Replace the static skill-doc dump with always-loaded + top-k cosine retrieval |
| `--failure-store` / `--scratchpad` | | Agent-memory features (see `docs/agent_memory.md`) |

Run `python3 scripts/architect_cli.py --help` for the complete list. Feedback-channel
toggles (`--no-fault-summary`, `--no-trace-injection`) are documented in
[`docs/feedback_flags.md`](docs/feedback_flags.md).

## Evaluation harness

`scripts/eval/` is the multi-condition comparison tool. **It does not touch a
real robot** — every perception call is bound to a synthetic `Scene` fixture.

```bash
# Full sweep — all conditions × seeds (sequential under --llm claude)
PYTHONPATH=. python3 -m scripts.eval.sweep --llm claude --output-dir /tmp/eval/run1

# Single trial
PYTHONPATH=. python3 -m scripts.eval.run_trial --task pick_block_on_marker \
    --condition PatternReuse --seed 1 --llm claude --output-dir /tmp/eval/single

# Dry-run smoke (deterministic stub LLM, no API cost)
PYTHONPATH=. python3 -m scripts.eval.sweep --output-dir /tmp/eval/dry
```

`scripts/eval/aggregate.py` computes OOD pass-rate ± bootstrap CI,
useful-disagreement rate, library compactness, per-sequence position curves, and
paired Wilcoxon p-values.

## Project structure

```
architect/
├── config.py                 # LLM / embedding config (endpoints read from env)
├── requirements.txt
├── .env.example
├── run_architect.sh             # convenience wrapper (env setup + CLI)
├── architect/
│   ├── agent/                # agentic session: tool-use loop, runtime synthesis
│   ├── corrections/          # trace, fault summary, subtask docs, LLM scorers
│   ├── llm/                  # Claude client, embeddings, prompt builders
│   ├── probes/               # scene mutator, probe runner, slate sampling, graduation
│   ├── eval/                 # task-level success scorers
│   ├── library/              # SkillStore, relevance filter, context budget, episode log
│   ├── robots/               # Franka config, ROS client, API spec
│   └── ui/                   # slate diff panels
├── scripts/
│   ├── architect_cli.py            # interactive CLI (main entry point)
│   ├── probe_program.py      # probe a program against perturbations
│   ├── eval/                 # sweep, run_trial, aggregate, profile_session, stub_llms
│   └── reference/            # historical non-interactive scripts (reference only)
├── skills/franka/            # library.db, episodes.jsonl, subtasks/*.md, topical guides
└── docs/                     # reference docs (below)
```

## Reference docs

- [`docs/deployment.md`](docs/deployment.md) — multi-machine layout + model servers
- [`docs/mechanisms.md`](docs/mechanisms.md) — feature-level architecture
- [`docs/feedback_flags.md`](docs/feedback_flags.md) — feedback-channel toggles + ablation recipes
- [`docs/agent_memory.md`](docs/agent_memory.md) — failure store / scratchpad / hybrid retrieval
- [`docs/api.md`](docs/api.md) — Franka primitive API reference (what the LLM sees)
- [`docs/legacy_scripts.md`](docs/legacy_scripts.md) — the reference (non-interactive) scripts
</content>
