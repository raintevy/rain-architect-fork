# Feedback-channel flags

The closed-loop correction path uses three independent execution-derived
feedback channels to shape the LLM's next program. Each one is a
separate signal with its own CLI flag, so you can ablate them
individually for testing or measuring channel-level contribution.

## What the three channels carry

| Channel | What the LLM sees | Where it lives |
|---|---|---|
| **Layer B per-call VQA observations** | A contextual VQA query (`"is the gripper holding something?"`, `"where is the gripper now?"`, …) fired automatically after every traced control/perception primitive. Each Q/A pair is attached to its call in the trace. | `architect/corrections/execution_trace.py` — the trace wrapper |
| **Fault-localisation summary** | A 3-line LLM-generated diagnostic produced *before* the correction turn: what the agent was trying to do, what actually happened, one-line hypothesis about the gap. Rendered above the raw trace in the correction prompt. | `architect/agent/session.py:_compute_fault_summary` |
| **Raw execution-trace injection** | A `## Last Execution Trace` section in the correction prompt: the full call sequence + per-step state snapshots (EE pose, joints, gripper width) + program-level VQA Q/A pairs. | `architect/llm/prompts.py:build_agentic_correction_message` |

The `vqa_judge` / `explain_failure` scorers ALSO read the trace, but
that's a separate pathway from the prompt injection. The flags below
only affect what the *correction* LLM call sees; scoring is unaffected.

## The flags

| Flag | Default | When set, suppresses |
|---|---|---|
| `--no-vlm-tracing` | channel on | Layer B per-call VQA observations during execution. State snapshots + call args still land in the trace. |
| `--no-fault-summary` | inherits from `--condition` (on without one) | The 3-line LLM summary above the trace. |
| `--no-trace-injection` | channel on | The raw `## Last Execution Trace` section. Trace is still captured (scoring still works). |

`--no-fault-summary` and `--no-trace-injection` default to `None` in
argparse; they only override `agentic_kwargs` when explicitly passed.
So `--condition PatternReuse` keeps fault-summary on unless you also
pass `--no-fault-summary`. There's no "turn it back on" flag — the
absence of the flag preserves the condition's default.

## Typical usage patterns

### 1. Full feedback (default — what you usually want)

All three channels on. Generated programs see everything.

```bash
./run_architect.sh --condition PatternReuse --closed-loop \
    --instruction "Pick up the red cup"
```

### 2. Quiet trace (VQA backend is flaky / slow / expensive)

Skip the Layer B per-call VQA round-trips. Programs still issue their
own VQA calls (those are program logic, not trace decoration).

```bash
./run_architect.sh --condition PatternReuse --closed-loop \
    --no-vlm-tracing --instruction "Pick up the red cup"
```

### 3. PatternReuse minus fault summary

Useful when you want to measure how much the 3-line summary actually
helps — does the agent already extract the right hypothesis from the
raw trace, or does it need the LLM-prefiltered hint?

```bash
./run_architect.sh --condition PatternReuse --closed-loop \
    --no-fault-summary --instruction "Pick up the red cup"
```

### 4. Closed-loop with NO execution-derived prompt context

Trace is still scored (closed-loop fires on FAIL as usual), but the
refining LLM call sees only the correction text + current program —
nothing about what happened last run. Most pessimistic baseline.

```bash
./run_architect.sh --condition PatternReuse --closed-loop \
    --no-vlm-tracing --no-fault-summary --no-trace-injection \
    --instruction "Pick up the red cup"
```

### 5. Probing what each channel adds (4-cell mini-ablation)

For a single task / instruction, run all four combinations and compare
PASS-rate / corrections-to-PASS:

```bash
# Cell A: everything on
./run_architect.sh --task pick_block_on_marker --condition PatternReuse --closed-loop --save out/A.json

# Cell B: no Layer B
./run_architect.sh --task pick_block_on_marker --condition PatternReuse --closed-loop --no-vlm-tracing --save out/B.json

# Cell C: no fault summary
./run_architect.sh --task pick_block_on_marker --condition PatternReuse --closed-loop --no-fault-summary --save out/C.json

# Cell D: no raw trace
./run_architect.sh --task pick_block_on_marker --condition PatternReuse --closed-loop --no-trace-injection --save out/D.json
```

## Composition notes

- The flags layer additively on `--condition`. `--condition ICL`
  already sets `emit_fault_summary=False` as part of its bundled
  ablation — adding `--no-fault-summary` is a no-op there. The flags
  are designed for the case where you want to ablate a single channel
  *without* changing the condition.
- `--dry-run` separately forces `emit_subtask_docs=False`,
  `emit_scorers=False`, and `library_persistent=False` regardless of
  `--condition` — see `docs/deployment.md` for the dry-run isolation
  rationale. The feedback flags above are orthogonal to that.
- Trace capture itself (the in-memory `ExecutionTrace` populated by
  the wrapper) is always on. The flags only suppress *consumption* of
  the trace by various downstream paths (correction prompt, Layer B
  observations). `vqa_judge` and `explain_failure` always read the
  trace they were given.
