# Mechanisms

This file documents the moving parts behind the four conditions (`ICL`,
`FuncReuse`, `SlateGated`, `PatternReuse`) and what each one contributes
to a correction round. For the high-level "what's this for" framing, see
the README. For the per-condition flag matrix, see
`scripts/eval/tasks.json`.

## Agentic synthesis

Generation and correction both use an agentic tool-use loop. Claude has
access to the following tools during synthesis:

| Tool | Description |
|------|-------------|
| `list_primitives` | List all built-in robot API functions plus any primitives registered this session |
| `read_primitive(name)` | Get the source code of a session-registered primitive |
| `query_robot_state` | Read current EE pose, joint positions, and gripper width from the robot |
| `get_scene_description` | Detect ArUco markers and get a VQA description of the visible scene |
| `write_primitive(name, code, docstring)` | Register a new reusable helper function (requires user approval) |
| `run_ros2_command(command)` | Run a ROS CLI command for environment introspection (requires user approval); accepts `ros2` commands for Stretch, `rosservice`/`rostopic` for Franka |
| `submit_program(code)` | Deliver the final program for user approval and execution |

Typical flow for a new task:

```
Claude calls list_primitives()          → sees all available API functions
Claude calls query_robot_state()        → grounds plan in current robot state
Claude calls get_scene_description()    → finds ArUco markers / objects
Claude calls write_primitive(...)       → proposes a reusable pick_and_place() helper

╭─ New primitive: pick_and_place() ────────────────────────────────────────╮
│  1  def pick_and_place(marker_id, lift_height=0.2):                      │
│  2      markers = detect_markers()                                       │
│  3      ...                                                              │
╰──────────────────────────────────────────────────────────────────────────╯
Register this primitive? [y/n/e(dit)]:

Claude calls submit_program(code)       → final program shown for approval
```

Registered primitives persist for the session and are available to all
subsequent corrections. They are injected into the execution namespace
alongside the built-in robot API.

If the submitted program still calls an undefined function at runtime,
the fallback CaP synthesis (`architect/agent/synthesis.py`) catches the
`NameError` and asks Claude to synthesize the missing function body on
the fly (up to 5 levels deep).

PatternReuse mode (CLI flag `--condition PatternReuse`) overrides this
loop with a system-prompt addendum: no `write_primitive`, inline helpers,
arbitrary Python control flow allowed, and every fragile sub-operation
MUST implement the `for attempt in range(2)` verification + retry
pattern. See `architect/llm/prompts.py` `_PATTERN_REUSE_ADDENDUM`.

## Persistent skill library

Every approved primitive is persisted to a per-robot SQLite database at
`skills/<robot>/library.db`. The library survives across CLI sessions:
when you start a new session against the same robot, previously-approved
helpers are loaded into the execution namespace and surfaced to Claude
as available API.

`/library` in the CLI renders the current state:

```
                           Skill library
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━┳━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ name                  ┃ status    ┃ ✓ ┃ ✗ ┃ last probe ┃ docstring    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━╇━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ detect_red_cube       │ graduated │ 0 │ 0 │ 100% (8/8) │ Detect ...   │
│ grasp_object_at_pose  │ graduated │ 0 │ 0 │ 100% (8/8) │ Open grip... │
│ lift_object           │ graduated │ 0 │ 0 │ 100% (8/8) │ Lift the ... │
│ place_object_on_table │ graduated │ 0 │ 0 │ 100% (8/8) │ Move to ...  │
└───────────────────────┴───────────┴───┴───┴────────────┴──────────────┘
```

Each skill has a `graduation_status` of `candidate`, `graduated`, or
`archived`. Graduated skills are eligible for retrieval into Claude's
system prompt on subsequent turns and across sessions; candidates stay
session-local.

Retrieval is semantic. The relevance filter embeds the active correction
and each candidate's description with `text-embedding-3-small`, then
keeps the top-k matches (cosine ≥ 0.45). A second session that asks
"grab the red block" will find `grasp_object_at_pose` even though "grab"
and "object" never appear in the helper's signature.

## Graduation gate

When `write_primitive` registers a new helper (under the dry-run eval
harness — *not* in the live CLI today, see the gap noted in the README's
conditions table), the C2 graduation gate runs a synthetic test program
against K=8 counterfactual scene perturbations (object translation,
rotation, marker jitter, detection order, adversarial VQA). The skill is
**admitted** to the graduated set only if probe pass-rate ≥ 0.7 AND
every invariant supplied to the gate held on every probe.

Failed gates are non-destructive — the helper stays at `candidate`, the
failed probe-suite is recorded for `/library` to display, and
`/graduate <name>` lets you force-promote if you disagree with the
verdict.

## Slate mode (active counterfactual sampling)

Activated by `--condition SlateGated` (or the legacy `--slate` flag).
Each correction triggers a single LLM call that emits N candidate
programs, each committing to a *different* design choice on what the
correction left under-specified ("approach from +y vs +x", "5cm vs 10cm
offset", "cache detection vs re-detect", "open-loop vs verified retry").

Candidates are ranked by:

```
score = probe_pass_rate × mean_scorer_score − 0.05·high_audit − 0.02·medium_audit
```

If the top candidate's score dominates the runner-up (≥0.85 and margin
≥0.20), the system auto-applies it. Otherwise the slate is surfaced in a
rich diff UI for you to pick A/B/C. Your choice + the rejected
candidates' axis labels are written to `skills/<robot>/episodes.jsonl`
as multi-axis supervision — the eval harness reads this log to compute
the **useful-disagreement rate** (fraction of surfaced slates where the
user's pick was not the model's rank-1).

When a correction text becomes a slate, the LLM is also asked to
operationalize it as a `def score(scene_before, scene_after, call_log) ->
float` plus zero-or-more `def invariant_<n>(...) -> bool` artifacts.
These are persisted to the `scorers` table of the library DB and reused
by future slate rankings + the C2 graduation gate. Stale scorers from
unrelated corrections are filtered out at slate time by an embedding-
based relevance check.

"Verification strategy" is one of the explicit decision axes the slate
generator is told to vary — so at least one candidate per slate uses a
VQA-verified-retry pattern.

## Closed-loop corrections (what makes failures actionable)

When a program runs (live or dry-run), the namespace wrappers in
`architect/corrections/execution_trace.py` capture an ordered timeline of
every traced control + perception call along with:

- args / kwargs passed
- post-call proprioception snapshot (EE pose, joints, gripper width)
- a *contextual* VLM observation — a one-sentence question keyed on the
  call type (e.g. `set_gripper_width(0.0)` asks "did the gripper grasp
  an object?"; `move_ee_to_pose(...)` asks "where is the gripper now
  relative to the target?")
- the call's return value (for perception calls — `get_vqa_response`,
  `detect_objects`, `get_placement_pose` etc. — so post-execution scoring can inspect what
  the program saw)

Wrappers are restored to their originals on exec end (success or
failure) so they don't persist across runs. Disable the per-call VLM
hook via the CLI's `--no-vlm-tracing` flag (state snapshots still get
captured; only the per-call VQA round-trip is skipped).

On the next correction, three signals get prepended to Claude's user
message before the correction text:

1. **`## Fault Localisation Summary`** — a foreground pre-correction LLM
   call (`architect/corrections/fault_localization.py`) reads the trace +
   correction and emits three lines:
   ```
   Failure:              <which call failed and what evidence supports it>
   Likely root cause:    <what upstream choice caused it>
   Correction direction: <what dimension the next program should adjust>
   ```
   Capped at ~500 output tokens, temperature 0. Best-effort: a flaky
   call drops to `None` and the correction proceeds without the
   summary section.
2. **`## Last Execution Trace`** — the structured call timeline, with
   the VLM observations rendered as indented sub-lines. Truncated to ~80
   entries for long programs.
3. **`## Correction`** — the user's free-text correction (unchanged).

After each accepted correction, two background daemon threads fire so
the user's loop is never blocked:

- **Scorer + invariant emission** (`architect/corrections/llm_scorer.py`) —
  the P3 pipeline that operationalises the correction text as scoring
  artifacts.
- **Subtask doc emission** (`architect/corrections/subtask_docs.py`) — one LLM
  call that decides `create` / `update` / `skip` for a
  `skills/<robot>/subtasks/<name>.md` describing the generalised sub-
  operation the correction taught. Under PatternReuse, the create-vs-
  update decision is driven by embedding similarity (cosine ≥ 0.70
  → update existing, else create new). Subtask docs feed into future
  sessions' system prompts as Domain Knowledge.

The system prompt also encourages the **VQA-verified-retry pattern** by
default for fragile sub-operations (grasp / place / articulation):
perform the action, ask `get_vqa_response` a yes/no question keyed on
the intended outcome, retry once on a negative answer, give up after
one retry. The pattern is documented in `skills/<robot>/verification.md`
and surfaced as an explicit slate axis so each slate produces at least
one closed-loop candidate.

### `--closed-loop` auto-correction

When `--task` is set and an execution returns FAIL, `--closed-loop`
makes the CLI automatically issue a refining correction using a
structured diagnostic of *what* failed (cycle count short, missing
final-state VQA, negative VQA answer, etc.). The refined program is
auto-approved and re-executed. Loop exits on PASS or after
`--max-auto-corrections` rounds (default 2), at which point the user is
prompted as usual. The diagnostic comes from
`architect.eval.task_scorers.explain_failure` — testable in isolation by
calling it on a captured trace.

All four closed-loop pieces are opt-out via constructor flags on
`AgenticSession` (`emit_scorers`, `emit_subtask_docs`,
`emit_fault_summary`, `inprogram_retry`) and via the CLI
(`--no-vlm-tracing`, `--closed-loop`). They are on by default under the
appropriate condition.

## Context profiling

`--profile-context` writes one JSONL line per LLM call to
`skills/<robot>/context_profile_<YYYYMMDD_HHMMSS>.jsonl`:

```bash
python3 scripts/architect_cli.py --instruction "..." --profile-context
```

Each line carries per-turn: total bytes / `tokens_est` / per-channel
attribution (system, retrieval addendum, history, current turn) /
clipped + stale bytes saved / read calls per tool / window-fill fraction
/ `response.usage.{input,output}_tokens` (ground truth) / wall-clock
latency.

Aggregate with:

```bash
python3 -m scripts.eval.profile_session \
    --jsonl skills/franka/context_profile_20260514_142022.jsonl
```

…to produce tokens-per-call curve, per-channel byte share, cumulative
clip + stale savings, lost-context-rate proxy (per tool), and a
heuristic-quality audit (`tokens_est / input_tokens` ratio).

## CuRobo pre-execution validation

Every call to `move_ee_to_pose` is intercepted before execution and
passed to a CuRobo feasibility check that verifies the motion plan is
collision-free and kinematically reachable. If the check fails, a
`RuntimeError` is raised and the program halts before any physical
motion occurs.

> **Note:** The validation endpoint is not yet implemented in the CuRobo
> WebSocket service. The current stub always returns `feasible=True`. To
> enable it, add a `validate_only` JSON request flag to the CuRobo
> motion-planning service on the GPU server (see `docs/deployment.md`) that
> calls `motion_gen.plan_single()` but skips the articulation controller
> execution loop.
