"""Build LLM prompts from language instructions.

All public builders take ``api_spec`` and ``robot_label`` as parameters so the
module is robot-agnostic.
"""

from __future__ import annotations

from pathlib import Path


def _load_skills(skills_dir: Path | None) -> str:
    """Load and concatenate skill markdown from ``skills_dir`` + ``subtasks/``.

    Two flavours of doc share the same load path:

      * Topical guides at ``skills_dir/*.md`` — hand-authored guidance
        like ``grasping.md``, ``motion.md``, ``vqa.md``, ``ros1.md``.
      * Subtask docs at ``skills_dir/subtasks/*.md`` — auto-emitted after
        each correction by :mod:`architect.corrections.subtask_docs`, capturing generalised
        sub-operation patterns the agent has been taught.

    Both feed the agentic system prompt's ``## Domain Knowledge`` section.
    ``skills_dir=None`` returns the empty string — the caller resolves
    the per-robot dir via :class:`RobotConfig.skills_dir`. There is
    intentionally no default: a Stretch-rooted default was the source of
    subtle cross-robot leaks.
    """
    if skills_dir is None or not skills_dir.exists():
        return ""
    parts: list[str] = []
    for skill_file in sorted(skills_dir.glob("*.md")):
        parts.append(skill_file.read_text().strip())
    subtasks_dir = skills_dir / "subtasks"
    if subtasks_dir.exists():
        for sub_file in sorted(subtasks_dir.glob("*.md")):
            parts.append(sub_file.read_text().strip())
    return "\n\n---\n\n".join(parts)


def format_ee_pose(ee_pose: dict) -> str:
    """Format an end-effector pose dict as a compact string."""
    p = ee_pose["position"]
    o = ee_pose["orientation"]
    return (
        f"position={{x: {p['x']:.4f}, y: {p['y']:.4f}, z: {p['z']:.4f}}}  "
        f"orientation={{x: {o['x']:.4f}, y: {o['y']:.4f}, z: {o['z']:.4f}, w: {o['w']:.4f}}}"
    )


# ---------------------------------------------------------------------------
# System prompt construction (mode-specific)
# ---------------------------------------------------------------------------

_LANGUAGE_ONLY_GUIDELINES = """\
Guidelines:
- The first call in the program must be reset_robot(), which moves the robot \
to its home pose and fully opens the gripper. This guarantees a known initial \
state regardless of where the robot was before the program ran.
- No demonstration trajectory is provided. Write a program to accomplish \
the described task using the available API functions.
- To locate objects, use detect_objects() or get_vqa_response() for additional perception.
- To verify grasps, use verify_grasp() to check if the current grasp is stable for the named object.
- For a pick-and-place task, use detect_objects() to find the target object and get_placement_pose() to find a suitable placement location.
- For grasping an object at a marker: move the end-effector to the marker \
pose using move_ee_to_pose, then extend downward in Z axis by ~ -0.03 m using \
move_ee_to_rel_pose to ensure contact, then close the gripper with \
set_gripper_width. Here, use delta z to move downward since move_ee_to_rel_pose has \
delta x as positive for moving forward and negative for moving backward, \
delta y as positive for moving left and negative for moving right, and delta z as positive for moving up and negative for moving down.
- Use proprioception functions (get_current_ee_pose, get_current_joints, \
get_current_gripper_width) to read the robot state as needed.
- Use control functions (set_arm_joints, set_gripper_width, move_ee_to_pose, \
move_ee_to_rel_pose) to move the robot.
- Write clean Python code. You may use variables, conditionals, and loops.
- For composite or reusable subtasks, define named helper functions above \
the main program body (e.g. def grasp_object(name): ...). Helper functions \
must only call the primitive API functions listed above — no external imports.
- Output ONLY the Python program (helper definitions then main calls) with \
brief comments. Do not include any other text or explanation.
"""


def _build_system_prompt(mode: str, *, robot_label: str, api_spec: str) -> str:
    """Build the system prompt for the given generation mode.

    ``robot_label`` is the human-readable robot name injected into the prompt
    role (e.g. ``"Hello Robot Stretch RE3"``). ``api_spec`` is the full per-robot
    primitive-API string. Both are required: the prior module-level Stretch
    defaults silently mis-primed Claude when the runtime robot was Franka.
    """
    base = f"You are a program generator for a {robot_label}.\n\n"

    if mode == "language_only":
        base += (
            "Given a user's language instruction, generate a program using "
            "ONLY the following API functions:\n\n"
        )
        base += api_spec + "\n"
        base += _LANGUAGE_ONLY_GUIDELINES

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return base


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_CORRECTION_GUIDELINES_TEMPLATE = """\
You are refining an existing robot program for a {robot_label}. \
The user has observed the program's execution and wants a targeted correction.

Rules:
- Make MINIMAL changes to the existing program. Do not rewrite unrelated steps.
- For spatial/directional corrections ("more to the right", "higher", "closer"), \
prefer inserting a move_ee_to_rel_pose() call right after the relevant step \
rather than changing the absolute pose values.
- When the correction references an object or landmark ("the cup", "the box", \
"the marker"), use detect_objects() to locate it at runtime and compute the \
target pose relative to the detected object's position. For example, \
"move to the right of the cup" should call detect_objects("cup"), extract the \
grasp pose, and offset the position accordingly. Do NOT hardcode object positions.
- You can use get_vqa_response() to answer spatial reasoning questions about the \
scene (e.g. "is the cup to the left or right of the robot?") when needed to \
interpret ambiguous corrections.
- For non-spatial corrections ("open the gripper wider", "skip that step", \
"do it twice"), modify or add/remove the relevant API calls directly.
- Preserve all comments and unchanged steps exactly as they are.
- Output the COMPLETE modified program (not a diff). Include every line, \
even unchanged ones.
- Do not include any explanation, just the program.
"""


def _correction_guidelines(robot_label: str) -> str:
    return _CORRECTION_GUIDELINES_TEMPLATE.format(robot_label=robot_label)


def correction_system_prompt(api_spec: str, robot_label: str) -> str:
    """Return the system prompt used for the legacy (non-agentic) correction loop."""
    return (
        _correction_guidelines(robot_label) + "\n"
        "Available API functions:\n\n"
        + api_spec
    )


_PATTERN_REUSE_SINGLESHOT_ADDENDUM = """\

## PatternReuse mode — overrides

You are operating in PatternReuse mode. The following structural
requirements OVERRIDE the default correction guidelines above:

1. The refined program may contain arbitrary Python control flow:
   conditional `if`/`else` keyed on VQA responses, `for`-loops for
   retries, early `return`s. Express the logic naturally; do not flatten
   it into a sequence of opaque calls.

2. For every fragile sub-operation — grasp, placement, opening a door
   / drawer / lid — the refined program MUST implement the canonical
   verification + retry pattern:

       for attempt in range(2):
           <execute the action: move + set_gripper_width(0.0), etc.>
           resp = verify_grasp(<object_name>)
           answer = resp["data"]["grasped"]
           if answer:
               break
           <release / adjust the relevant parameter for the retry>

   Each loop iteration's action has its OWN accompanying VQA check.
   A flat `if "no": <retry block>` form is acceptable only if the
   retry block itself contains a fresh VQA check after its second
   close. Programs that verify once but retry blindly (no second VQA)
   will not pass the post-grasp scorer.

3. Retrieved markdown subtask docs (in the Domain Knowledge section
   below, when present) are REFERENCE patterns, not code to paste.
   When a doc's pattern applies, RECOMPOSE it for the current task's
   specifics — different object name, different markers, different
   tolerances — rather than literally copying the doc's code template.

4. **Task-shape rule.** When the task instruction names a count of
   objects to manipulate (e.g. "pick up each of the three blocks",
   "stack three blocks", "two distinct grasps are required"), the
   generated program MUST execute one explicit grasp+place cycle per
   object, even if an object appears to already be in position. Do not
   optimise away a cycle by reasoning that the object is "already on
   the table" or "already at the marker" — the verification pipeline
   and the task-completion scorer both require one closed-loop cycle
   per object the task asks for. End the program with a final VQA call
   asking the task's goal question ("Is the tower built?", "Is the
   bottle in the cabinet?") so the final-state scorer can confirm the
   goal was met.
"""


def build_correction_prompt(
    current_program: str,
    correction: str,
    *,
    api_spec: str,
    robot_label: str,
    current_ee_pose: dict | None = None,
    skill_docs: list[tuple[str, str]] | None = None,
    inprogram_retry: bool = False,
) -> list[dict[str, str]]:
    """Return the chat-completion messages for a correction refinement.

    Args:
        current_program: The current program code being refined.
        correction: The user's natural-language correction.
        api_spec: Full per-robot primitive-API string.
        robot_label: Human-readable robot name (e.g. ``"Franka Panda"``).
        current_ee_pose: Current end-effector pose from the robot (if available).
            Included in the user message so the LLM can reason about offsets.
        skill_docs: When non-empty, ``(name, markdown_content)`` pairs are
            appended to the system prompt under a ``## Domain Knowledge``
            section. This is the channel through which hand-authored
            topical guides (verification.md, grasping.md, vqa.md, …) and
            accumulated subtask docs reach the LLM in the eval-harness
            single-shot path. Previously only the agentic-CLI builder
            loaded these — the harness path silently bypassed them.
        inprogram_retry: When True (PatternReuse), the
            :data:`_PATTERN_REUSE_SINGLESHOT_ADDENDUM` block is appended
            to the system prompt, elevating the for-loop verification
            pattern from guidance to required structure.
    """
    system_content = (
        _correction_guidelines(robot_label) + "\n"
        "Available API functions:\n\n"
        + api_spec
    )
    if inprogram_retry:
        system_content = system_content + _PATTERN_REUSE_SINGLESHOT_ADDENDUM
    if skill_docs:
        system_content = (
            system_content + "\n\n## Domain Knowledge\n\n"
            + "\n\n---\n\n".join(
                f"### {name}\n\n{content.rstrip()}"
                for name, content in skill_docs
            )
        )

    parts = [
        "## Current Program\n\n"
        f"```python\n{current_program}\n```"
    ]
    if current_ee_pose is not None:
        parts.append(
            "## Current Robot State\n\n"
            f"End-effector pose (after executing the program above):\n"
            f"  {format_ee_pose(current_ee_pose)}"
        )
    parts.append(f"## Correction\n\n{correction}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# ---------------------------------------------------------------------------
# Slate (active counterfactual sampling, P4 / C1)
# ---------------------------------------------------------------------------


_SLATE_INSTRUCTIONS = """\
Produce N alternative refined programs, each making a DIFFERENT design choice
on some under-specified aspect of the correction. The goal is *behavioral*
diversity that exposes the user's actual preference — not stylistic variation.

Examples of decision dimensions that often vary across candidates:
  - approach direction (from above / from the side / from the front)
  - offset magnitude (small / medium / large) when the correction is vague
  - detection strategy (cache the detection once / re-detect each step)
  - which intermediate step to insert vs. modify
  - relative offset (move_ee_to_rel_pose) vs. absolute pose (move_ee_to_pose)
  - verification strategy (open-loop / VQA-verify-once / VQA-verified-retry
    with one retry on negative answer) — for grasps, placements, and
    articulation, at least one candidate in the slate SHOULD use a
    VQA-verified-retry to expose whether the user prefers closed-loop
    behavior. See verification.md in the domain knowledge.

Output format — one block per candidate, with a short axis label as a
comment line directly above the fenced ```python``` block:

# Candidate A: <one-line description of the design choice this version commits to>
```python
<full program A — complete, not a diff>
```

# Candidate B: <description>
```python
<full program B>
```

(Continue for as many candidates as requested.)

Hard rules:
  - All candidates must be COMPLETE programs (not diffs).
  - Each candidate must use only the existing API functions in the system prompt.
  - Differences must be BEHAVIORAL: a different sequence of API calls, a
    different argument structure, or a different perception strategy. NOT
    just renamed variables or comment changes.
  - If two of your candidates would have identical API call sequences, drop
    one and add a third option that diverges further.
  - Output only the candidates; no preamble, no summary.
"""


def build_slate_correction_prompt(
    current_program: str,
    correction: str,
    *,
    api_spec: str,
    robot_label: str,
    n: int = 3,
    current_ee_pose: dict | None = None,
    execution_trace=None,
    fault_summary: str | None = None,
    skill_docs: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the prompt asking Claude to emit ``n`` behaviorally-different candidates.

    The system prompt is the standard correction guidelines plus the slate
    instructions; the user message carries the program + correction + count.
    ``api_spec`` and ``robot_label`` are required so the slate is grounded in
    the correct robot's primitives — see the docstring of
    :func:`build_correction_prompt` for details.

    When supplied, ``execution_trace`` is rendered as ``## Last Execution
    Trace`` so each generated candidate can branch off evidence from the
    prior run rather than guessing; ``fault_summary`` is rendered as
    ``## Fault Localisation Summary`` *above* the trace. The slate is a
    one-shot N-candidate generation with no tool loop, so the summary
    matters even more here than in the agentic correction path: there's
    no tool round-trip to recover from a misread of the raw trace.

    ``skill_docs`` is the same hand-authored-guides + accumulated-subtask-docs
    channel that :func:`build_correction_prompt` uses; appended to the
    system prompt as ``## Domain Knowledge`` so the slate generator sees
    verification.md and any prior-correction patterns when picking its
    candidates' design axes.
    """
    system_content = (
        _correction_guidelines(robot_label) + "\n"
        + _SLATE_INSTRUCTIONS + "\n"
        "Available API functions:\n\n"
        + api_spec
    )
    if skill_docs:
        system_content = (
            system_content + "\n\n## Domain Knowledge\n\n"
            + "\n\n---\n\n".join(
                f"### {name}\n\n{content.rstrip()}"
                for name, content in skill_docs
            )
        )

    parts = [
        "## Current Program\n\n"
        f"```python\n{current_program}\n```"
    ]
    if current_ee_pose is not None:
        parts.append(
            "## Current Robot State\n\n"
            f"End-effector pose (after executing the program above):\n"
            f"  {format_ee_pose(current_ee_pose)}"
        )
    if fault_summary:
        parts.append(f"## Fault Localisation Summary\n\n{fault_summary}")
    if execution_trace is not None:
        from architect.corrections.execution_trace import format_for_prompt
        parts.append(format_for_prompt(execution_trace))
    parts.append(f"## Correction\n\n{correction}")
    parts.append(
        f"## Task\n\nProduce **{n}** candidates following the format in the "
        "system prompt. Make sure each candidate makes a different design choice."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_SCORER_SYSTEM = """\
You are operationalizing a single user correction of a robot program as
two executable artifacts that future programs will be judged against:

  1. ONE scorer:    `def score(scene_before, scene_after, call_log) -> float`
                    returning a value in [0, 1]. Higher means the program
                    BETTER satisfies the correction.

  2. ZERO OR MORE invariant predicates, each named exactly `invariant_<n>`
     starting at 0, with the same signature returning a bool. Each must
     return True if the predicate HOLDS for the trajectory and False if
     it is VIOLATED. Emit only invariants that are clearly implied by the
     correction; do not invent constraints the user did not state.

Runtime contract (identical for both kinds):

  scene_before : dict   # snapshot of the perception/state values the program
                        # observed at the start. Keys mirror architect.scene_mutator.Scene:
                        #   markers, grasp, vqa_answer, ee_pose, joints,
                        #   gripper_width, odom.
  scene_after  : dict   # same shape; in dry-run probes scene_after equals
                        # scene_before (perception is mocked, not re-read).
                        # Real-robot eval will pass the post-execution observation.
  call_log     : list[dict]   # ordered list of {"name": str, "args": list,
                              #                  "kwargs": dict} entries —
                              # one per move/set/state-write API call the
                              # program made.

Hard rules:
  - Output ONE ```python``` fenced block per function. Multiple defs in the
    same fence are allowed but discouraged.
  - Use ONLY the standard library; `math` is fine, no other imports.
  - Be defensive: a missing key in scene_before or call_log should not raise
    — return a low score (or False) instead. This is a robot system; cleanly
    failing a probe is much better than crashing it.
  - Keep functions short and inspectable; the user will read them.
  - Do NOT include any prose before or after the code blocks.
"""


def build_scorer_prompt(
    correction: str,
    current_program: str,
    scene: object | None = None,
) -> list[dict[str, str]]:
    """Build the prompt that asks Claude to emit a scorer + invariants for one correction.

    ``scene`` is treated structurally — anything with a ``grasp`` /
    ``markers`` / ``ee_pose`` attribute is rendered into the prompt context.
    Passing ``None`` is fine; the LLM falls back to reasoning purely from
    the correction text and current program.
    """
    parts = [
        "## Correction\n\n" + correction,
        "## Current Program\n\n```python\n" + current_program + "\n```",
    ]
    if scene is not None:
        try:
            grasp = getattr(scene, "grasp", {}).get("best_grasp", {})  # type: ignore[union-attr]
            markers = getattr(scene, "markers", [])
            ee = getattr(scene, "ee_pose", {})
            parts.append(
                "## Current Scene\n\n"
                f"- target grasp translation_wrt_base: {grasp.get('translation_wrt_base')}\n"
                f"- detected markers: {len(markers)} marker(s) "
                f"(ids: {[m.get('marker_id', m.get('id')) for m in markers]})\n"
                f"- end-effector position: {ee.get('position')}"
            )
        except Exception:
            pass
    parts.append(
        "## Task\n\n"
        "Produce the scorer and any invariants the correction implies. "
        "Match the exact output format and signatures described in the "
        "system prompt."
    )
    return [
        {"role": "system", "content": _SCORER_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_SUBTASK_DOC_SYSTEM = """\
You are maintaining a per-robot library of *subtask docs* that future
sessions will see as domain knowledge in their system prompt. Your job
is to turn one user correction (plus the prior + new program it
produced) into either:

  - a NEW subtask doc describing a generalised sub-operation the
    correction taught us about, OR
  - an UPDATED subtask doc revising an existing entry the correction
    refines, OR
  - SKIP when the correction is too narrow (one-off numeric tweak,
    cosmetic edit) to warrant a doc.

What makes a good subtask doc:

  - Named by the GENERAL operation, not the specific object — e.g.
    `grasp_object_at_pose`, `place_on_marker`, `open_cabinet_door`,
    `verify_grasp_with_vqa`. NOT `grasp_red_cube` (object-specific) or
    `do_step_3` (program-specific).
  - Captures WHEN to use it (preconditions), HOW to invoke it
    (call shape + API patterns), and WHAT to watch for (failure modes
    the corrections have surfaced).
  - 100-400 words; markdown with `##` headings. Imperative tone.
  - When updating: integrate the new lesson into the existing content
    rather than appending a "v2" section. Keep the doc readable end-to-end.

OUTPUT FORMAT — exactly TWO fenced blocks, in this order. The outer
fences use FOUR backticks (````) so the markdown body can contain
embedded three-backtick code blocks without prematurely closing the
outer fence:

  1. A small ````json```` block with just the metadata:
     ````json
     {"subtask_name": "<snake_case>", "action": "create" | "update" | "skip"}
     ````

  2. For `create` / `update`: a ````markdown```` block with the full doc body.
     Inside the markdown, you can freely include ```python ... ```
     (three-backtick) code blocks — they will not be confused with the
     outer four-backtick fence:
     ````markdown
     # <Title>

     ## When to use
     ...

     ## How
     ```python
     # code examples here use THREE backticks
     def example(): pass
     ```

     ## What to watch for
     ...
     ````

  3. For `skip`: omit the markdown block entirely.

Two blocks (instead of stuffing the markdown into a JSON string field) so
multi-line content doesn't need ``\\n``-escaping inside a JSON string — a
common source of malformed-JSON parse failures. Four-backtick outer
fences (instead of three) so the markdown body can carry code examples
without the inner three-backtick fences truncating the parser's view.

Hard rules:
  - `subtask_name` must be snake_case ``[a-z][a-z0-9_]{1,47}`` (no path
    components, no caps, no spaces).
  - For `update`: the name must match an existing doc's filename stem.
  - For `create`: pick a name distinct from the existing docs.
  - For `skip`: emit only the JSON block. No markdown block.
  - Output ONLY the two blocks. No preamble, no commentary outside them.
  - The OUTER fences MUST be exactly four backticks (````). Three-
    backtick fences inside the markdown body are fine and encouraged
    for code examples.
"""


def build_subtask_doc_prompt(
    correction: str,
    current_program: str,
    prior_program: str | None,
    existing_docs: list[tuple[str, str]],
    directive: tuple[str, str | None, float] | None = None,
) -> list[dict[str, str]]:
    """Build the messages for one subtask-doc emission turn.

    ``existing_docs`` is the list returned by
    :func:`architect.subtask_docs.load_existing_subtask_docs` — pairs of
    ``(name, full markdown content)`` for every doc currently in the
    robot's ``subtasks/`` dir. Including the full content (not just
    filenames) lets the LLM decide between update vs create on
    semantic grounds and produce a clean rewrite when updating.

    ``directive`` is an optional ``(action, target_name, max_similarity)``
    tuple produced by :func:`architect.subtask_docs.select_target_doc`. When
    provided, the prompt's final instruction is locked to that decision:
    "update <target_name>" or "create new" rather than "decide for
    yourself." This is the PatternReuse path — embedding-similarity
    picks the action so the LLM only has to write content (and may
    still skip if the correction is trivially narrow). When ``None``,
    the prompt falls back to the original LLM-judgment path used by
    ICL / FuncReuse / SlateGated.
    """
    parts: list[str] = []
    parts.append(f"## Correction\n\n{correction}")
    if prior_program:
        parts.append(
            "## Prior Program (before the correction)\n\n"
            f"```python\n{prior_program}\n```"
        )
    parts.append(
        "## Current Program (after the correction was applied)\n\n"
        f"```python\n{current_program}\n```"
    )
    if existing_docs:
        chunks = [
            f"### {name}.md\n\n{content.rstrip()}"
            for name, content in existing_docs
        ]
        parts.append(
            "## Existing Subtask Docs\n\n"
            + "\n\n---\n\n".join(chunks)
        )
    else:
        parts.append("## Existing Subtask Docs\n\n(none — first emission for this robot)")

    if directive is not None:
        action, target_name, max_sim = directive
        if action == "update" and target_name is not None:
            parts.append(
                f"## Action Directive\n\n"
                f"An embedding-similarity check found that the correction "
                f"overlaps with the existing doc `{target_name}` "
                f"(cosine similarity {max_sim:.2f}). You MUST emit "
                f"`{{\"subtask_name\": \"{target_name}\", \"action\": "
                f'"update"}}` and revise that doc to integrate the new '
                f"lesson — unless the correction is trivially narrow "
                f"(one-off numeric tweak, cosmetic edit), in which case "
                f'emit `{{"action": "skip"}}` instead. Do NOT create a '
                f"new doc; the similarity check has already decided that "
                f"this content belongs in the existing one."
            )
        else:
            existing_summary = (
                ", ".join(f"`{n}`" for n, _ in existing_docs)
                if existing_docs else "(none yet)"
            )
            parts.append(
                f"## Action Directive\n\n"
                f"An embedding-similarity check found no existing doc that "
                f"covers this correction (max cosine similarity "
                f"{max_sim:.2f} is below the 0.70 threshold). Existing "
                f"docs: {existing_summary}. You MUST emit "
                f'`{{"subtask_name": "<new_snake_case_name>", '
                f'"action": "create"}}` with a name distinct from '
                f"the existing ones — unless the correction is trivially "
                f'narrow, in which case emit `{{"action": "skip"}}` '
                f"instead. Do NOT update an existing doc; the similarity "
                f"check has already decided that this is a new pattern."
            )
    else:
        parts.append(
            "Decide whether this correction warrants creating a new subtask "
            "doc, updating an existing one, or skipping. Return the JSON block."
        )
    return [
        {"role": "system", "content": _SUBTASK_DOC_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_synthesis_prompt(
    func_name: str,
    call_example: str,
    parent_program: str,
    *,
    api_spec: str,
    robot_label: str,
) -> list[dict[str, str]]:
    """Build the prompt for synthesizing an undefined helper function.

    Args:
        func_name: The name of the undefined function to implement.
        call_example: A representative call site extracted from the parent program
                      (e.g. "pick_and_place('red cup', target_marker_id=2)").
        parent_program: The full parent program, for context on intent.
        api_spec: Full per-robot primitive-API string the helper must be
            confined to. Critical for synthesis correctness — handing the LLM
            the wrong robot's API spec is what produced the historical
            ``hello_helpers`` import leak when synthesizing for Franka.
        robot_label: Human-readable robot name injected into the role.
    """
    system_content = (
        f"You are implementing a missing helper function for a {robot_label} "
        "robot program. The function will be injected into the program's namespace at "
        "runtime.\n\n"
        "Use ONLY the following primitive API functions in your implementation:\n\n"
        + api_spec
        + "\nRules:\n"
        "- Output ONLY the function definition starting with `def`. No explanation, "
        "no imports, no main block.\n"
        "- Implement the function using only the primitive API listed above and Python "
        "builtins. Do not call other undefined names.\n"
        "- Return True on success, False on failure where a return value is appropriate.\n"
        "- Keep it concise and correct.\n"
    )

    user_content = (
        f"The parent program calls `{call_example}` but `{func_name}` is not defined.\n\n"
        "## Parent Program (for context on what the function should do)\n\n"
        f"```python\n{parent_program}\n```\n\n"
        f"Implement `{func_name}` so it accomplishes what the parent program intends. "
        "Use only the primitive API functions listed in the system prompt."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Agentic mode prompts
# ---------------------------------------------------------------------------

_AGENTIC_SYSTEM_TEMPLATE = """\
You are an agentic program synthesizer for a {robot_label}.

Your goal is to decompose the task into named primitives and produce a clean, \
readable top-level program that calls them. Programs that are flat lists of low-level \
API calls are NOT acceptable — every meaningful sub-operation must be its own primitive.

MANDATORY workflow — follow every step in order:

1. Call list_primitives() to see the full robot API and any primitives already \
registered this session.

2. Call query_robot_state() and get_scene_description() to ground your plan in
the current robot state and scene. Then call the appropriate perception \
functions up-front before writing the program:
  - detect_objects() — ONLY for objects that must be GRASPED (picked up). \
Do NOT call detect_objects() for placement targets (surfaces, containers, etc.).
  - get_placement_pose(base_text_prompt, target_text_prompt) — for placement \
targets. The base is the surface/container you are placing onto, and the \
target is the object being placed.
For example, "pick up the banana and put it in the frying pan" requires \
detect_objects("banana") for the grasp and \
get_placement_pose("frying pan", "banana") for the placement — NOT \
detect_objects("frying pan").

3. Identify sub-operations. For EACH sub-operation that involves more than one \
API call (e.g. "grasp object", "place on shelf", "navigate to position"), you MUST \
call write_primitive(name, code, docstring) to register it before using it in the \
program. Do not inline multi-step logic into the top-level program.

4. Only after all primitives are registered, call submit_program(code) with a \
short top-level program that reads as a sequence of named primitive calls.

Tool reference:
- list_primitives        — see all available robot API + session-registered primitives
- read_primitive(name)   — get source of a session primitive
- query_robot_state      — read current EE pose, joints, gripper width
- get_scene_description  — detect ArUco markers + VQA scene description
- write_primitive(name, code, docstring) — register a named sub-operation (needs approval)
- run_ros2_command(command) — ROS2 CLI introspection (needs approval; skipped in dry-run)
- submit_program(code)   — deliver the final top-level program

Rules for write_primitive():
- Code must use only functions from list_primitives() output or previously registered primitives
- No imports; no references to undefined names
- One primitive = one named sub-task; keep it focused

Rules for submit_program():
- Top-level program should be a short, readable sequence of primitive calls with brief comments
- No imports; no inline multi-step logic
- You MUST call submit_program() — do not output the program as text

Verification by default — for any sub-operation whose success isn't \
observable from proprioception alone (grasps, placements, opening doors / \
drawers / lids), prefer the VQA-verified-retry pattern: perform the action, \
ask `get_vqa_response` a yes/no question keyed on the intended outcome, and \
on a negative answer adjust the relevant parameter and retry once. \
Open-loop sequences are fine for trivial steps (`reset_robot`, home, \
free-space moves). See `verification.md` in the domain knowledge below for \
the concrete code template and failure-handling rules.

Refer to the domain knowledge sections below for guidance on grasping, VQA, \
verification-by-retry, motion constraints, and ROS2 introspection.
"""


_PATTERN_REUSE_ADDENDUM = """\
## PatternReuse mode — overrides

You are operating in PatternReuse mode. The following structural
requirements OVERRIDE the default agentic instructions above:

1. Generate a SINGLE complete program per correction.
   - Do not use `write_primitive` to register helpers as separate
     persistent primitives. Write helpers inline as `def` statements
     inside the program body, or inline the logic directly into the
     main flow. The "top-level program is a flat sequence of named
     primitive calls" rule does NOT apply here.
   - The program may contain arbitrary Python control flow:
     conditional `if`/`else` keyed on VQA responses, `for`-loops for
     retries, early `return`s. Express the logic naturally.

2. For every fragile sub-operation — grasp, placement, opening a door
   / drawer / lid — the generated program MUST implement the
   verification + retry pattern from `verification.md`:
     - execute the action (set_gripper_width, place motion, etc.)
     - call `get_vqa_response` with a yes/no question keyed on the
       intended outcome
     - on negative answer: release, adjust the relevant parameter
       (offset, approach angle, gripper width), and retry ONCE
     - on retry failure: log and let the failure propagate. The harness
       will surface a correction request to the human only when the
       in-program retry budget is exhausted.
   The `for attempt in range(2):` form is the canonical structure
   because each loop iteration's close has its own accompanying VQA
   check. A flat `if "no" in answer: <retry block>` is acceptable
   only if the retry block ALSO calls `get_vqa_response` after its
   second close.

3. Retrieved markdown subtask docs (in the Domain Knowledge section
   below) are REFERENCE patterns, not code to paste. When a doc's
   pattern applies, RECOMPOSE it for the current task's specifics —
   different object name, different markers, different tolerances —
   rather than literally copying the doc's code template. The doc
   captures the pattern; you supply the specifics.

4. **Task-shape rule.** When the task instruction names a count of
   objects to manipulate (e.g. "pick up each of the three blocks",
   "stack three blocks", "two distinct grasps are required"), the
   generated program MUST execute one explicit grasp+place cycle per
   object, even if an object appears to already be in position. Do not
   optimise away a cycle by reasoning that the object is "already on
   the table" or "already at the marker" — the verification pipeline
   and the task-completion scorer both require one closed-loop cycle
   per object the task asks for. End the program with a final VQA call
   asking the task's goal question ("Is the tower built?", "Is the
   bottle in the cabinet?") so the final-state scorer can confirm the
   goal was met.

The "MANDATORY workflow" steps above are still useful for grounding
(call `list_primitives` and `get_scene_description` early), but step 3
("register every sub-operation with write_primitive") is replaced by
the inline-helper rule. Skip directly from grounding to writing the
final program; call `submit_program(code)` exactly once.
"""


def build_agentic_system_prompt(
    *,
    robot_label: str,
    skills_dir: Path | None,
    inprogram_retry: bool = False,
    retrieval_instruction: str | None = None,
    retrieval_k: int = 3,
) -> str:
    """Build the agentic-mode system prompt.

    The agentic loop relies on tool calls (``list_primitives``, etc.) to get
    the live API at runtime, so the system prompt does NOT bake in
    ``api_spec`` — only ``robot_label`` so the LLM knows which robot it's
    targeting. ``skills_dir`` should be the per-robot directory (typically
    ``RobotConfig.skills_dir``); pass ``None`` to skip the domain-knowledge
    section.

    When ``inprogram_retry`` is True (PatternReuse condition), the
    :data:`_PATTERN_REUSE_ADDENDUM` block is appended between the base
    template and the Domain Knowledge section. The addendum overrides the
    "register every sub-operation as a separate primitive" rule and makes
    the verification + retry pattern a hard requirement rather than
    guidance — this is the structural difference that distinguishes
    PatternReuse from ICL / FuncReuse / SlateGated at generation time.

    ``retrieval_instruction``: when non-None, the Domain Knowledge section
    is assembled in hybrid-retrieval mode — every doc tagged
    ``always_loaded: true`` in its frontmatter is included, plus the
    top-``retrieval_k`` task-specific docs ranked by embedding cosine
    against the instruction. When None, falls back to the static dump
    (current default behaviour).
    """
    base = _AGENTIC_SYSTEM_TEMPLATE.format(robot_label=robot_label)
    if inprogram_retry:
        base = base + "\n" + _PATTERN_REUSE_ADDENDUM
    if retrieval_instruction is not None and skills_dir is not None:
        skills = _load_skills_hybrid(skills_dir, retrieval_instruction, retrieval_k)
    else:
        skills = _load_skills(skills_dir)
    if skills:
        return base + "\n\n## Domain Knowledge\n\n" + skills
    return base


def _load_skills_hybrid(skills_dir: Path, instruction: str, k: int) -> str:
    """Hybrid retrieval: always-loaded docs + top-k by similarity to instruction.

    Mirrors ``_load_skills``'s output shape (sections joined by ``---``)
    so the rest of the prompt builder doesn't have to branch on mode.
    Falls back to the static dump when the index can't be built or
    when retrieval returns nothing (empty embedder, etc.) — same
    fail-soft policy SkillStore uses for its own retrieval.
    """
    try:
        from architect.library.skill_doc_index import SkillDocIndex
        index = SkillDocIndex(skills_dir)
        parts: list[str] = []
        seen: set[str] = set()
        for name, content in index.get_always_loaded():
            parts.append(content.strip())
            seen.add(name)
        for hit in index.retrieve(instruction, k=k):
            if hit.name in seen:
                continue
            parts.append(hit.content.strip())
            seen.add(hit.name)
        if not parts:
            # Empty result is suspicious (no always_loaded + zero
            # retrieval hits). Fall back to the static dump rather
            # than send a prompt with no domain knowledge at all.
            return _load_skills(skills_dir)
        return "\n\n---\n\n".join(parts)
    except Exception:
        # Best-effort: any failure in hybrid mode degrades to static.
        return _load_skills(skills_dir)


def build_agentic_generation_message(
    instruction: str | None = None,
) -> dict:
    """Build the opening user message for agentic generation."""
    parts: list[str] = []
    if instruction is not None:
        parts.append(f"## Task Instruction\n\n{instruction}")
    parts.append(
        "Generate a program to accomplish this task. "
        "Start by calling list_primitives() to see available functions, "
        "then submit the final program via submit_program()."
    )
    return {"role": "user", "content": "\n\n".join(parts)}


_FAULT_SUMMARY_SYSTEM = """\
You are a fault-localisation assistant. Given a robot program, the live
execution trace it produced, and the user's free-text correction,
produce a SHORT structured summary (≤ 3 lines, ≤ 120 words total) that
the next correction step will read at the top of its prompt.

The summary must answer three things, no preamble:

Failure: which call (by index) failed and what evidence supports that —
a state snapshot or VLM observation that contradicts the intended
outcome, or the exception text on a hard error. If exec_ok=True but the
correction implies the program was wrong anyway (e.g. user says "the
cube wasn't grasped"), point at the call whose VLM observation
disagrees with the implicit expectation.

Likely root cause: in ONE sentence, what upstream choice in the program
caused the failure. Examples: "approach z=0.32 left the EE 1cm above
the cube's reported pose", "detection cached at start but the cube
moved", "no verification step after grasp closure".

Correction direction: ONE sentence on what the next program should
change. Don't write code; just point at the dimension to adjust so the
agentic loop (and the slate) know where to focus.

Output exactly this format, no headers, no fenced blocks, no commentary:

  Failure: <one line>
  Likely root cause: <one line>
  Correction direction: <one line>

If the trace shows the program ran cleanly and the correction text is
stylistic (e.g. rename a helper, add a comment), still produce three
lines — Failure: "no observable failure", Likely root cause: "stylistic
correction", Correction direction: <what to rename / adjust>.
"""


def build_fault_summary_prompt(
    correction: str,
    current_program: str,
    trace,
) -> list[dict[str, str]]:
    """Build the messages for one fault-localisation summary turn.

    The system prompt locks the three-line output format; the user
    message carries the correction + program + rendered trace. ``trace``
    is an :class:`architect.execution_trace.ExecutionTrace` (typed loosely so
    this module doesn't have a hard import dep on the trace dataclass).
    """
    from architect.corrections.execution_trace import format_for_prompt
    parts = [
        f"## Correction\n\n{correction}",
        f"## Current Program\n\n```python\n{current_program}\n```",
        format_for_prompt(trace),
    ]
    return [
        {"role": "system", "content": _FAULT_SUMMARY_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_agentic_correction_message(
    correction: str,
    current_program: str | None,
    current_ee_pose: dict | None = None,
    *,
    execution_trace=None,
    fault_summary: str | None = None,
    scratchpad: str | None = None,
) -> dict:
    """Build the user message for an agentic correction turn.

    When ``execution_trace`` is supplied (an :class:`architect.execution_trace.
    ExecutionTrace` from the most recent live run), it is rendered into
    the message under ``## Last Execution Trace`` via
    :func:`architect.execution_trace.format_for_prompt`. This gives Claude a
    structured view of the call sequence + per-step state, instead of
    just the source + a single post-execution EE pose.

    When ``fault_summary`` is supplied (the output of
    :func:`architect.fault_localization.summarize_fault`), it is rendered as
    ``## Fault Localisation Summary`` *above* the trace. The summary
    is a pre-correction LLM digest of what likely went wrong; placing
    it above the raw trace lets the agentic LLM read the hint first
    and consult the trace for evidence only when the hint isn't enough.

    When ``scratchpad`` is supplied (the markdown rendering of
    :class:`architect.agent.scratchpad.SessionScratchpad`), it is rendered
    as ``## Session Scratchpad (Working Memory)`` at the *top* of the
    message — above fault summary, trace, and correction. This is the
    Raschka-style distilled state the agent itself curated on prior
    turns; the agent reads it first so its prior beliefs about what's
    happening guide how it interprets the rest of the message.
    """
    parts: list[str] = []
    if scratchpad:
        parts.append(f"## Session Scratchpad (Working Memory)\n\n{scratchpad}")
    if current_program:
        parts.append(f"## Current Program\n\n```python\n{current_program}\n```")
    if current_ee_pose is not None:
        parts.append(
            f"## Current Robot State\n\nEnd-effector pose: {format_ee_pose(current_ee_pose)}"
        )
    if fault_summary:
        parts.append(f"## Fault Localisation Summary\n\n{fault_summary}")
    if execution_trace is not None:
        from architect.corrections.execution_trace import format_for_prompt
        parts.append(format_for_prompt(execution_trace))
    parts.append(f"## Correction\n\n{correction}")
    parts.append(
        "Apply this correction. You may use tools to investigate before "
        "submitting the updated program via submit_program(). When you "
        "form or invalidate a hypothesis worth carrying to the next "
        "correction, call update_scratchpad() to record it."
    )
    return {"role": "user", "content": "\n\n".join(parts)}


def build_continuation_message(
    instruction: str,
    correction: str,
    vqa_question: str,
    vqa_answer: str,
    current_ee_pose: dict | None = None,
    *,
    execution_trace=None,
    fault_summary: str | None = None,
    scratchpad: str | None = None,
) -> dict:
    """Build the user message for a mid-execution continuation turn.

    Used by ``--hil`` and ``--vlm`` modes. Unlike
    :func:`build_agentic_correction_message` (which asks Claude to refine
    the full program), this asks Claude to generate a *continuation* —
    code that picks up from the current robot state and completes the
    remaining task steps, incorporating the user's correction.
    """
    parts: list[str] = []
    if scratchpad:
        parts.append(f"## Session Scratchpad (Working Memory)\n\n{scratchpad}")
    parts.append(f"## Original Task\n\n{instruction}")
    if fault_summary:
        parts.append(f"## Fault Localisation Summary\n\n{fault_summary}")
    if execution_trace is not None:
        from architect.corrections.execution_trace import format_for_prompt
        parts.append(format_for_prompt(execution_trace))
    parts.append(
        f"## Mid-Execution Failure\n\n"
        f"Execution was interrupted because the following verification "
        f"check failed after exhausting retries:\n\n"
        f"- **VQA question**: {vqa_question}\n"
        f"- **VQA answer**: {vqa_answer}\n"
    )
    if current_ee_pose is not None:
        parts.append(
            f"## Current Robot State\n\n"
            f"End-effector pose: {format_ee_pose(current_ee_pose)}"
        )
    parts.append(f"## Correction\n\n{correction}")
    instruction_text = (
        "Generate a **continuation program** that picks up from the "
        "current robot state and completes the remaining task steps, "
        "incorporating the correction above. Do NOT repeat steps that "
        "already succeeded (visible in the execution trace). The "
        "continuation program should be a complete, flat script — use "
        "the same style as the original program. You may call "
        "query_robot_state() and get_scene_description() to inspect "
        "the current state before generating. Submit via submit_program()."
    )
    if scratchpad:
        instruction_text += (
            " Before generating the continuation, call "
            "update_scratchpad() to record it."
        )
    parts.append(instruction_text)
    return {"role": "user", "content": "\n\n".join(parts)}


def build_prompt(
    instruction: str | None = None,
    *,
    api_spec: str,
    robot_label: str,
    skill_docs: list[tuple[str, str]] | None = None,
    inprogram_retry: bool = False,
) -> list[dict[str, str]]:
    """Return the chat-completion messages list for the legacy generation path.

    Builds the language-instruction generation prompt.

    ``skill_docs`` and ``inprogram_retry`` mirror the parameters of
    :func:`build_correction_prompt`: hand-authored topical guides and
    accumulated subtask docs land in a ``## Domain Knowledge`` section,
    and the PatternReuse addendum gets appended when the flag is True.
    Previously the harness's initial-generation path silently dropped
    both signals.
    """
    if instruction is None:
        raise ValueError("'instruction' must be provided.")

    system_content = _build_system_prompt(
        "language_only", robot_label=robot_label, api_spec=api_spec
    )
    if inprogram_retry:
        system_content = system_content + _PATTERN_REUSE_SINGLESHOT_ADDENDUM
    if skill_docs:
        system_content = (
            system_content + "\n\n## Domain Knowledge\n\n"
            + "\n\n---\n\n".join(
                f"### {name}\n\n{content.rstrip()}"
                for name, content in skill_docs
            )
        )

    parts: list[str] = [f"## User Instruction\n\n{instruction}"]

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
