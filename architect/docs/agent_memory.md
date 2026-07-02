# Agent memory features

Three opt-in features that augment the agentic correction loop with
failure-aware memory + working-memory + retrieval. All three default to
off; each can be enabled independently or combined.

## Failure diagnosis + failure store

Persistent log of structured failure diagnoses with embedding-based
cross-session lookup. Each FAIL produces a `FailureDiagnosis` (category
from a 12-entry taxonomy, terse signature, root cause, suggested
directions, trace evidence) rendered as a panel above the human-
correction prompt and used as the closed-loop auto-correction body.
With `--failure-store` on, the diagnosis + applied correction + outcome
are persisted to `skills/<robot>/failures.db`; at the next FAIL, top-3
similar past failures (only those marked `fixed`) are surfaced as
"Cross-session failure recall."

**Flags**:

- `--failure-store` — persist failures + similarity lookup. Off by
  default. Implies `--failure-diagnosis` on.
- `--no-failure-diagnosis` — disable the LLM diagnosis call; falls back
  to the older keyword-based `explain_failure` text.

**Example**:

```bash
# Manual correction with a structured diagnosis panel before ARCHITECT › prompt:
./run_architect.sh --condition PatternReuse "Pick up the red cup"

# Closed-loop with cross-session learning + persistence:
./run_architect.sh --condition PatternReuse --closed-loop --failure-store \
    "Pick up the red cup"
```

**What you see on FAIL**:

```
✗ vqa_judge: FAIL (score 0.00)
╭───── Failure diagnosis ─────╮
│ Category: missing_verification
│ Signature: no_vqa_check_after_grasp_and_lift
│ Root cause: The program never verified the grasp before lifting...
│ Suggested directions:
│   - Add a VQA check between gripper close and lift
│   - Wrap the grasp in a retry loop
╰─────────────────────────────╯
╭─── Cross-session failure recall ───╮     ← only with --failure-store
│ - sig: gripper closed on empty space (sim 0.87, outcome: fixed)
│   applied correction: Descend 5cm before close + verify
╰────────────────────────────────────╯
```

**Outcome bookkeeping**: after each correction's next execution, the
previous failure row is marked `fixed` (PASS) or `still_failed` (FAIL).
For the paper's failure-mode aggregation, read `failures.db` directly
or `FailureStore.all()` in code.

**Failure-category taxonomy**: `grasp_failed_no_contact`,
`grasp_failed_wrong_object`, `vqa_false_positive`, `vqa_false_negative`,
`missing_verification`, `motion_infeasible`, `object_not_detected`,
`placement_misaligned`, `premature_release`, `incomplete_sequence`,
`exec_error`, `other`. Add new categories by editing the
`FAILURE_CATEGORIES` tuple in `architect/corrections/failure_diagnosis.py`.

## Session scratchpad (agent-driven working memory)

Following Raschka's two-layer pattern (full transcript + distilled
working memory) for AI coding agents. The agent maintains a small
structured scratchpad across correction turns via a new
`update_scratchpad` tool. Rendered at the **top** of each correction
prompt — above fault summary and trace — so the agent's prior beliefs
guide interpretation of new evidence.

Fields:
- `current_goal` — what the agent is actively trying to solve (possibly
  narrower than the original task)
- `working_hypotheses` — beliefs about cause / fix that haven't been
  confirmed yet
- `invalidated_hypotheses` — disproved theories, so the agent doesn't
  revisit them
- `key_observations` — trace-confirmed facts worth carrying across
  corrections

Lists are deduplicated case-insensitively and capped at 8 entries
(FIFO eviction). Invalidating a hypothesis does a soft substring
match against `working_hypotheses` and moves the match to
`invalidated_hypotheses` — no need for exact-string recall across turns.

**Flag**: `--scratchpad` — opt-in (off by default; adds ~tokens/turn
not worth it for single-correction sessions).

**Example**:

```bash
./run_architect.sh --condition PatternReuse --closed-loop --scratchpad \
    "Pick up the red cup and place it on marker 1"
```

**Persistence**: in-session only. Dropped at `done`. Cross-session
working memory would conflate task-specific state with general robot
knowledge (which belongs in the skill library).

**Agent tool**: `update_scratchpad(current_goal?, add_observations?,
add_working_hypotheses?, invalidate_hypotheses?)`. Only available to
the agent when `--scratchpad` is on; filtered out of the tool list
otherwise so the agent doesn't see a no-op tool.

## Hybrid retrieval

Replaces the static dump of every `skills/<robot>/*.md` doc into the
system prompt with: foundational docs (tagged `always_loaded: true` in
their frontmatter) plus the top-3 task-specific docs retrieved by
embedding cosine against the instruction. Plus a `query_skills` agent
tool for fetching more docs mid-loop.

Concretely, for `"Pick up the red cup"` against the current Franka
library (10 docs), hybrid produces a Domain Knowledge section **66%
the size** of the static version — `vqa.md`, `motion.md`,
`verification.md`, `ros1.md` always in; `grasping.md`,
`sequential_pick_and_place.md`, etc. retrieved by similarity.

**Flags**:

- `--retrieval static|hybrid` — default `static` (preserves current
  behaviour).
- `--retrieval-k N` — k for the retrieval pool, default 3.

**Example**:

```bash
./run_architect.sh --retrieval hybrid --condition PatternReuse \
    "Pick up the red cup"
```

**Always-loaded frontmatter convention** — single HTML comment near
the top of each `.md` file (not rendered in any markdown viewer):

```markdown
<!-- architect-meta
always_loaded: true
description: One-line description for retrieval ranking.
-->
# Visual Question Answering
...
```

Both fields optional. `always_loaded` defaults to `false` (doc enters
the retrieval pool). `description` falls back to (first heading + first
paragraph) capped at 280 chars when omitted.

**Index sidecar**: `skills/<robot>/skill_doc_index.json` caches
embeddings keyed by file mtime so unchanged docs skip the embed round-
trip on next session. Gitignored — machine-derived state. Rebuilds
automatically when a doc's mtime changes.

**Agent tool**: `query_skills(query, k)` — fetches up to 5 more docs
mid-loop ranked by similarity to a query the agent constructs. Useful
when a correction needs a pattern the pre-loaded set didn't include.
Filtered out of the tool list when `--retrieval static`.

**Tagging additional docs as always-loaded**: edit the doc's first
line block to include the frontmatter shown above. The index rebuilds
on next session and re-embeds.

## Composing the features

The flags are orthogonal — combine freely:

```bash
# Full memory stack: persistent failure recall + scratchpad + retrieval
./run_architect.sh --condition PatternReuse --closed-loop \
    --failure-store --scratchpad --retrieval hybrid \
    "Pick up the red cup and place it on marker 1"
```

Per-feature interaction notes:

- `--scratchpad` works in both `--retrieval static` and `--retrieval
  hybrid` — the scratchpad is rendered at the top of correction
  prompts regardless of retrieval mode.
- `--failure-store` works without `--closed-loop` — manual corrections
  still benefit from the diagnosis panel + cross-session recall.
  Closed-loop just adds the auto-correction wrapper on top.
- All three are dry-run-isolated: the failure store falls back to
  `:memory:` under `--dry-run`, the scratchpad never persisted to
  disk in the first place, and the skill_doc_index.json sidecar
  is doc-metadata-only (no execution state) so dry-run can write
  to it safely.
