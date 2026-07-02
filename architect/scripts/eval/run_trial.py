#!/usr/bin/env python3
"""Run a single eval trial: one (task, condition, seed) cell of the §8 matrix.

A trial is the unit of measurement in the P7 sweep and the P8 real-robot
run. The harness:

  1. Loads the task config from ``tasks.json``.
  2. Builds an empty per-condition workspace — a fresh SkillStore +
     EpisodeLog scoped to ``(task_id, condition, seed)`` so trials don't
     contaminate each other.
  3. Generates an initial program (LLM call, or stubbed canned response).
  4. Applies each correction in order. Under ``SlateGated`` the slate
     path is used (and the auto-applied or rank-1 candidate is recorded
     to the episode log). Under ``ICL`` / ``FuncReuse`` a single-shot
     correction is applied directly.
  5. After all corrections, probes the final program against each
     held-out perturbation listed in the task and reports the mean OOD
     pass-rate. This is the headline C1+C2 metric of §8.4.

The trial output is a JSON file the aggregator reads. No external state
is mutated; trial workspaces live under ``--output-dir`` and can be
``rm -rf``'d safely.

LLM source is selected by ``--llm``:

  * ``stub``  — deterministic canned responses from
                :mod:`scripts.eval.stub_llms` (used by the P7 dry-run
                sweep so the harness can be exercised without API costs).
  * ``claude`` — ``architect.llm_client.call_claude`` / ``call_claude_with_tools``
                (real API; used in the P8 real-robot eval).

Usage:
    python -m scripts.eval.run_trial --task pick_block_on_marker \\
        --condition Ours --seed 42 --llm stub --output-dir /tmp/eval/
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load .env so ANTHROPIC_API_KEY / AZURE_OPENAI_API_KEY are visible when the
# harness is invoked directly (not via scripts/architect_cli.py, which already does
# this). Best-effort — dotenv is a soft dep; without .env the user is
# expected to export the keys in the shell.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from architect.library.episode_log import EpisodeLog
from architect.eval.task_scorers import build_success_scorer
from architect.probes.graduation import persist_and_admit, try_graduate
from architect.probes.probe_runner import run_probes
from architect.probes.sampling import Slate, build_slate
from architect.probes.scene_mutator import default_scene, mutate
from architect.corrections.llm_scorer import ScorerArtifact, emit_scorer_and_invariants
from architect.library.skill_store import SkillStore


# ---------------------------------------------------------------------------
# Trial config helpers
# ---------------------------------------------------------------------------


def load_tasks(tasks_path: Path) -> dict[str, Any]:
    return json.loads(tasks_path.read_text())


def get_task(tasks: dict, task_id: str) -> dict:
    for t in tasks["tasks"]:
        if t["task_id"] == task_id:
            return t
    raise KeyError(f"Unknown task: {task_id!r}. Available: "
                   f"{[t['task_id'] for t in tasks['tasks']]}")


def get_condition(tasks: dict, condition: str) -> dict:
    if condition not in tasks["conditions"]:
        raise KeyError(f"Unknown condition: {condition!r}. Available: "
                       f"{list(tasks['conditions'])}")
    return tasks["conditions"][condition]


# ---------------------------------------------------------------------------
# Per-condition workspace
# ---------------------------------------------------------------------------


def _trial_paths(
    output_dir: Path,
    task_id: str,
    condition: str,
    seed: int,
    *,
    sequence_id: str | None = None,
    position: int | None = None,
) -> dict[str, Path]:
    """Resolve filesystem paths for one trial.

    When ``sequence_id`` is set, the workspace dir is keyed on
    ``(sequence_id, condition, seed)`` instead of ``(task_id, condition,
    seed)``. Multiple positions in the same sequence therefore share a
    single ``library.db`` + ``episodes.jsonl`` — exactly the setup that
    lets later positions inherit helpers / scorers graduated during
    earlier positions. Trial-record filenames still distinguish
    positions so the aggregator can plot per-position curves.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sequence_id is not None:
        workspace = output_dir / "workspaces" / f"{sequence_id}__{condition}__seed{seed}"
        record_name = (
            f"trial_{sequence_id}__pos{position}__"
            f"{task_id}__{condition}__seed{seed}.json"
        )
    else:
        workspace = output_dir / "workspaces" / f"{task_id}__{condition}__seed{seed}"
        record_name = f"trial_{task_id}__{condition}__seed{seed}.json"

    workspace.mkdir(parents=True, exist_ok=True)
    # `subtasks/` accumulates markdown subtask docs across positions in a
    # sequence cell — the PatternReuse equivalent of the SQLite library
    # for SlateGated. Created lazily on first emission, but the path is
    # stable so prompt-time loaders can probe it cheaply.
    return {
        "workspace":     workspace,
        "library_db":    workspace / "library.db",
        "episodes":      workspace / "episodes.jsonl",
        "subtasks_dir":  workspace / "subtasks",
        "trial_record":  output_dir / record_name,
    }


def _make_store(cond: dict, db_path: Path) -> SkillStore:
    """Persistent SkillStore for conditions with ``library: true``; otherwise
    an in-memory store so the trial leaves no trace and library-size metrics
    correctly read zero."""
    if cond["library"]:
        return SkillStore(db_path=db_path)
    return SkillStore(":memory:")


# ---------------------------------------------------------------------------
# LLM stubs vs. real
# ---------------------------------------------------------------------------


def _make_llm_calls(
    llm: str,
    task_id: str,
    current_program_ref: dict[str, str],
) -> tuple[Callable, Callable]:
    """Return ``(call_claude_stub, call_claude_with_tools_stub)`` for ``llm``.

    ``current_program_ref`` is a mutable single-cell dict so the per-task
    stubs (in :mod:`scripts.eval.stub_llms`) can see the *latest* program
    text when generating correction responses — without it each correction
    would unconditionally regenerate the baseline pickup.
    """
    if llm == "stub":
        from scripts.eval.stub_llms import (
            make_call_claude_stub,
            make_call_claude_with_tools_stub,
        )
        return (
            make_call_claude_stub(task_id, current_program=current_program_ref),
            make_call_claude_with_tools_stub(task_id, current_program=current_program_ref),
        )
    if llm == "claude":
        from architect.llm.client import call_claude, call_claude_with_tools
        return call_claude, call_claude_with_tools
    raise ValueError(f"Unknown llm: {llm!r}")


# ---------------------------------------------------------------------------
# Correction application paths
# ---------------------------------------------------------------------------


def _load_skill_docs(
    robot_name: str,
    *,
    subtasks_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Load all hand-authored topical guides + accumulated subtask docs.

    Returns ``[(name, content), ...]`` ready to pass to
    :func:`architect.prompt_builder.build_correction_prompt`'s ``skill_docs``
    param. The harness previously never loaded any of these in its
    single-shot path; v5 / v6 trials saw an empty system prompt where
    the agentic-CLI users were seeing verification.md, vqa.md, etc.
    Unifying the channels means ICL / FuncReuse / SlateGated all see
    the same domain knowledge that PatternReuse does — fairer
    comparison and matches what the user actually experiences.

    Two sources are merged in order:
      1. ``skills/<robot>/*.md`` — static topical guides committed to
         the repo (grasping.md, motion.md, vqa.md, verification.md, …)
      2. ``<subtasks_dir>/*.md`` — auto-emitted subtask docs from
         prior corrections in this trial cell, when supplied
    """
    from architect.robots import get_robot_config
    docs: list[tuple[str, str]] = []
    try:
        cfg = get_robot_config(robot_name)
        robot_skills_dir = Path(cfg.skills_dir) if cfg.skills_dir else None
    except Exception:
        robot_skills_dir = None
    if robot_skills_dir is not None and robot_skills_dir.exists():
        for p in sorted(robot_skills_dir.glob("*.md")):
            try:
                docs.append((p.stem, p.read_text()))
            except OSError:
                continue
    if subtasks_dir is not None and subtasks_dir.exists():
        from architect.corrections.subtask_docs import load_existing_subtask_docs
        docs.extend(load_existing_subtask_docs(subtasks_dir))
    return docs


def _emit_subtask_doc_to_workspace(
    correction: str,
    program: str,
    prior_program: str | None,
    *,
    subtasks_dir: Path,
    call_claude: Callable,
    model: str,
    use_similarity_directive: bool,
) -> dict[str, Any] | None:
    """Emit one subtask doc per correction (PatternReuse-only).

    Returns a small dict capturing what was persisted (or why nothing
    was) for inclusion in the trial record's ``subtask_docs`` field, or
    ``None`` when the daemon path's exception swallowing would have
    fired. The harness calls this synchronously — emission is part of
    the trial's measurable output, not a background side effect.
    """
    try:
        from architect.corrections.subtask_docs import emit_subtask_doc, persist_subtask_doc
        doc, _raw = emit_subtask_doc(
            correction,
            current_program=program,
            prior_program=prior_program,
            subtasks_dir=subtasks_dir,
            llm_call=call_claude,
            model=model,
            use_similarity_directive=use_similarity_directive,
        )
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    if doc is None:
        return {"status": "parse_failure"}
    if doc.action == "skip":
        return {"status": "skip"}
    try:
        path = persist_subtask_doc(doc, subtasks_dir)
    except Exception as exc:
        return {
            "status": "persist_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "status": doc.action,            # 'create' or 'update'
        "name": doc.name,
        "path": str(path) if path else None,
        "directive": doc.provenance.get("directive"),
    }


def _apply_correction_singleshot(
    correction: str,
    current_program: str,
    call_claude: Callable,
    *,
    api_spec: str = "",
    robot_label: str = "",
    model: str = "stub",
    store: SkillStore | None = None,
    retrieval_enabled: bool = False,
    retrieval_k: int = 3,
    skill_docs: list[tuple[str, str]] | None = None,
    inprogram_retry: bool = False,
) -> str:
    """Single-shot correction: build the prompt + call the LLM + return the program.

    Used by ICL / FuncReuse / PatternReuse. The slate path lives in
    :func:`_apply_correction_slate`.

    When ``retrieval_enabled`` is True and ``store`` is provided, the prompt's
    system message is augmented with a ``## Relevant Skills`` block listing the
    top-k graduated skills matching the correction. This is what differentiates
    FuncReuse from ICL — without it both baselines silently produce
    identical outputs.

    ``skill_docs`` is the markdown-domain-knowledge channel — hand-authored
    topical guides + accumulated subtask docs. ``inprogram_retry`` activates
    the PatternReuse system-prompt addendum that elevates the for-loop
    verification pattern from guidance to required structure.
    """
    from architect.library.context import format_retrieved_skills, retrieve_relevant_skills
    from architect.llm.prompts import build_correction_prompt

    messages = build_correction_prompt(
        current_program=current_program,
        correction=correction,
        api_spec=api_spec,
        robot_label=robot_label,
        skill_docs=skill_docs,
        inprogram_retry=inprogram_retry,
    )
    if retrieval_enabled and store is not None:
        retrieved = retrieve_relevant_skills(
            store, correction, k=retrieval_k, status="graduated",
        )
        if retrieved:
            system_msg = messages[0]
            system_msg["content"] = (
                system_msg["content"] + "\n\n" + format_retrieved_skills(retrieved)
            )
    raw = call_claude(messages, model=model)
    # Strip fences if present.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _apply_correction_slate(
    correction: str,
    current_program: str,
    *,
    store: SkillStore,
    call_claude: Callable,
    api_spec: str,
    robot_label: str,
    robot_name: str,
    n: int,
    seed: int,
    episode_log: EpisodeLog,
    model: str,
    skill_docs: list[tuple[str, str]] | None = None,
) -> tuple[str, Slate]:
    """Slate-based correction (C1).

    Stored graduated scorers + invariants are loaded and passed to
    :func:`build_slate`'s ranker so the C1 likelihood signal compounds
    with the C3 preference memory. Auto-applied slates *and* surfaced
    slates are both recorded to the episode log so the
    useful-disagreement-rate denominator is well-defined.
    """
    scorers = tuple(
        ScorerArtifact(
            name=r["name"], kind=r["kind"], code=r["code"],
            preamble=r.get("preamble", ""), description=r["description"],
            provenance=r.get("provenance") or {},
        )
        for r in store.list_scorers(kind="scorer")
    )
    invariants = tuple(
        ScorerArtifact(
            name=r["name"], kind=r["kind"], code=r["code"],
            preamble=r.get("preamble", ""), description=r["description"],
            provenance=r.get("provenance") or {},
        )
        for r in store.list_scorers(kind="invariant")
    )

    slate = build_slate(
        correction=correction,
        current_program=current_program,
        api_spec=api_spec,
        robot_label=robot_label,
        n=n,
        robot_name=robot_name,
        scorers=scorers,
        invariants=invariants,
        seed=seed,
        llm_call=call_claude,
        model=model,
        skill_docs=skill_docs,
    )

    if not slate.candidates:
        # Slate emission failed; fall back to single-shot.
        new_program = _apply_correction_singleshot(
            correction, current_program, call_claude,
            api_spec=api_spec, robot_label=robot_label,
            model=model,
        )
        return new_program, slate

    # For the eval harness we *always* pick the top candidate. A real
    # interactive session would surface the slate to the user — that
    # behavior is exercised by architect_cli.py, not the eval. We still record
    # the slate to the episode log so the useful-disagreement-rate metric
    # can be computed against future user-study runs.
    picked_index = 0
    episode_log.record_slate(
        slate, picked_index=picked_index, auto_applied=slate.dominant,
    )
    return slate.candidates[picked_index].code, slate


# ---------------------------------------------------------------------------
# Register + (optionally) graduate the helpers that appear in a program
# ---------------------------------------------------------------------------


def _register_and_gate_helpers(
    program: str,
    *,
    store: SkillStore,
    register: bool,
    graduate: bool,
    robot_name: str,
) -> tuple[int, int]:
    """Find top-level ``def``s in ``program``; register and optionally gate them.

    Returns ``(n_registered, n_graduated)``. Helpers already in the store are
    re-registered (preserving status). When ``graduate`` is True, each helper
    runs through ``try_graduate`` + ``persist_and_admit`` — i.e. the P5 probe
    suite. Under ICL the store is in-memory so nothing persists; under
    FuncReuse the store keeps the row but ``graduate=False`` leaves it
    at ``candidate``; under SlateGated the row flips to ``graduated``
    only if the gate passes.

    Top-level imports from ``program`` are prepended to each helper's stored
    ``code`` so the helper is self-contained: the gate's synthetic test
    program (``<def>\\n<name>(STUB, ...)``) and any future
    :meth:`PrimitiveRegistry.inject_into_namespace` both see the imports the
    helper relies on. Without this, helpers using ``time.sleep`` /
    ``math.sqrt`` / etc. fail the gate with ``NameError`` even when they're
    structurally correct (caught in the first --llm claude sweep:
    stack_two_blocks seed 6,7 lost graduation on a helper that called
    ``time.sleep``).
    """
    if not register:
        return 0, 0
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return 0, 0

    import_lines = [
        ast.unparse(node) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imports_preamble = "\n".join(import_lines)

    n_reg = 0
    n_grad = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        def_code = ast.unparse(node)
        # Inline the imports so each stored skill is self-contained. Duplicate
        # `import time` lines across multiple helpers are harmless (imports
        # are idempotent in Python).
        full_code = (
            f"{imports_preamble}\n\n{def_code}" if imports_preamble else def_code
        )
        docstring = ast.get_docstring(node) or node.name
        store.register(name=node.name, code=full_code, docstring=docstring,
                       description=docstring)
        n_reg += 1
        if not graduate:
            continue
        try:
            result = try_graduate(
                name=node.name, code=full_code, robot_name=robot_name,
            )
            persist_and_admit(store, result)
            if result.admitted:
                n_grad += 1
        except Exception:
            # A buggy primitive must not kill the trial; the gate's verdict
            # is False by default and the row stays at candidate.
            pass
    return n_reg, n_grad


# ---------------------------------------------------------------------------
# Scorer / invariant emission (only fired under conditions that store them)
# ---------------------------------------------------------------------------


def _emit_and_persist_scorers(
    correction: str,
    new_program: str,
    *,
    turn_index: int,
    store: SkillStore,
    call_claude: Callable,
) -> tuple[int, int]:
    """Emit scorer + invariants for ``correction`` and persist them.

    Returns ``(n_scorers, n_invariants)`` actually persisted (a parse
    failure yields zeros). Names are suffixed with the turn index so a
    multi-correction trial doesn't collide on the ``UNIQUE(name)``
    constraint.
    """
    scorer, invariants = emit_scorer_and_invariants(
        correction,
        current_program=new_program,
        llm_call=call_claude,
        name_prefix=f"t{turn_index}_",
    )
    n_s = n_i = 0
    if scorer is not None:
        store.register_scorer(
            name=scorer.name, code=scorer.code, description=scorer.description,
            kind="scorer", preamble=scorer.preamble, provenance=scorer.provenance,
        )
        n_s += 1
    for inv in invariants:
        store.register_scorer(
            name=inv.name, code=inv.code, description=inv.description,
            kind="invariant", preamble=inv.preamble, provenance=inv.provenance,
        )
        n_i += 1
    return n_s, n_i


# ---------------------------------------------------------------------------
# Held-out OOD evaluation
# ---------------------------------------------------------------------------


def _evaluate_ood(
    program: str,
    perturbations: list[dict],
    *,
    robot_name: str,
    seed: int,
    k: int = 4,
    success_scorer: Callable | None = None,
) -> list[dict]:
    """Run the program against each held-out perturbation; return per-axis pass-rates.

    When ``success_scorer`` is supplied (built from the task's pre-registered
    ``success_scorer`` spec via :func:`architect.eval_scorers.build_success_scorer`),
    it gates each probe: a probe ``passed`` only if exec succeeded AND the
    success scorer returned ≥ ``scorer_threshold``. Without this the OOD
    pass-rate saturates at 100% on any non-crashing program (the probe
    runner can't tell whether the program actually completed the task).
    """
    base = default_scene(robot_name)
    rng = random.Random(seed)
    scorers = (success_scorer,) if success_scorer is not None else ()
    out: list[dict] = []
    for pert in perturbations:
        ood_scene = mutate(base, axis=pert["axis"], magnitude=pert["magnitude"], rng=rng)
        report = run_probes(
            program, base_scene=ood_scene, robot_name=robot_name, k=k, seed=seed,
            scorers=scorers,
        )
        out.append({
            "axis":      pert["axis"],
            "magnitude": pert["magnitude"],
            "pass_rate": report.pass_rate,
            "k":         k,
            "audit_severity": report.audit_severity(),
        })
    return out


# ---------------------------------------------------------------------------
# Trial orchestration
# ---------------------------------------------------------------------------


def run_trial(
    *,
    task_id: str,
    condition: str,
    seed: int,
    tasks_path: Path,
    output_dir: Path,
    llm: str = "stub",
    api_spec: str | None = None,
    robot_label: str | None = None,
    robot_name: str = "franka",
    slate_n: int = 3,
    sequence_id: str | None = None,
    position_in_sequence: int | None = None,
) -> dict:
    """Execute one trial; persist the trial record; return it as a dict.

    ``api_spec`` and ``robot_label`` default to ``None`` and are derived from
    the per-robot ``RobotConfig`` so that callers (e.g. ``sweep.py``) don't
    have to plumb them through. Passing strings explicitly still wins. The
    derived path is the *only* sane default for ``--llm claude``: without it
    Claude gets an empty API surface and hallucinates non-existent helpers
    like ``close_gripper()``.
    """
    tasks = load_tasks(tasks_path)
    task = get_task(tasks, task_id)
    cond = get_condition(tasks, condition)
    paths = _trial_paths(
        output_dir, task_id, condition, seed,
        sequence_id=sequence_id, position=position_in_sequence,
    )

    # Derive the API surface / robot identity Claude needs in the prompt.
    if api_spec is None or robot_label is None:
        from architect.robots import get_robot_config
        cfg = get_robot_config(robot_name)
        api_spec = api_spec if api_spec is not None else cfg.api_spec
        robot_label = robot_label if robot_label is not None else cfg.display_name

    store = _make_store(cond, paths["library_db"])
    episode_log = EpisodeLog(paths["episodes"])

    # Mutable single-cell so the stub LLM can see the latest program text.
    current_program_ref: dict[str, str] = {"program": ""}
    call_claude_fn, _call_claude_with_tools_fn = _make_llm_calls(
        llm, task_id, current_program_ref,
    )

    # Model identifier passed to call_claude. Under ``--llm stub`` this is
    # ignored (the stub doesn't dispatch on model). Under ``--llm claude`` it
    # selects the deployed Azure / Anthropic model name from config.
    if llm == "stub":
        model_name = "stub"
    else:
        from config import ANTHROPIC_MODEL
        model_name = ANTHROPIC_MODEL

    # 1. Initial generation. Build a proper system prompt that gives the
    # LLM the per-robot API spec + identity — without this, real Claude
    # hallucinates non-existent helpers (close_gripper, robot_api module,
    # list-shaped poses, etc.) and every trial scores 0% OOD. The stub
    # LLM ignores both args so this is a no-op for --llm stub trials.
    #
    # Skill docs are loaded here so initial generation sees verification.md
    # and any accumulated subtask docs from earlier positions in the same
    # sequence cell. Before this commit the harness silently skipped them.
    from architect.llm.prompts import build_prompt
    inprogram_retry = bool(cond.get("inprogram_retry", False))
    markdown_library_enabled = bool(cond.get("markdown_library", False))
    skill_docs = _load_skill_docs(
        robot_name,
        subtasks_dir=paths["subtasks_dir"] if markdown_library_enabled else None,
    )
    gen_messages = build_prompt(
        instruction=task["instruction"],
        api_spec=api_spec,
        robot_label=robot_label,
        skill_docs=skill_docs,
        inprogram_retry=inprogram_retry,
    )
    initial = call_claude_fn(
        messages=gen_messages,
        model=model_name,
    )
    current_program_ref["program"] = initial
    program_history = [initial]
    slate_records: list[dict[str, Any]] = []
    subtask_doc_records: list[dict[str, Any]] = []
    scorers_persisted = 0
    invariants_persisted = 0
    helpers_registered = 0
    helpers_graduated = 0

    # 1b. Register + (optionally) gate any helpers defined in the initial
    # program. ICL's in-memory store discards immediately; FuncReuse
    # stores at candidate; SlateGated runs the probe suite and admits
    # on pass.
    n_r, n_g = _register_and_gate_helpers(
        initial,
        store=store,
        register=cond.get("library", False),
        graduate=cond.get("graduation", False),
        robot_name=robot_name,
    )
    helpers_registered += n_r
    helpers_graduated += n_g

    # Early-exit threshold: stop applying corrections once held-out OOD
    # pass-rate clears this bar. Pre-registered per-task with a global
    # fallback in tasks.json so corrections_applied is a real measurement
    # of "corrections-to-success" — not just the length of the script.
    early_exit_threshold = task.get(
        "early_exit_threshold",
        tasks.get("early_exit_threshold", 1.0),
    )
    # Task-level success scorer: gates each probe so OOD pass-rate isn't
    # saturated by exec-only success. Pre-registered in tasks.json. Falls
    # back to None (exec-only check) when the task didn't specify one.
    success_scorer = build_success_scorer(task.get("success_scorer"))
    final_ood: list[dict] = []
    final_ood_pass_rate = 0.0
    corrections_applied = 0
    early_exit_triggered = False

    # 2. Apply each correction.
    for turn_index, correction in enumerate(task["corrections"], start=1):
        # Skill docs are reloaded per turn so any subtask docs emitted at
        # earlier positions in the sequence (or earlier corrections in
        # this position) are visible to the next LLM call. Used by both
        # the slate and singleshot paths.
        skill_docs_turn = _load_skill_docs(
            robot_name,
            subtasks_dir=paths["subtasks_dir"] if markdown_library_enabled else None,
        )
        if cond["slate"]:
            new_program, slate = _apply_correction_slate(
                correction,
                current_program=current_program_ref["program"],
                store=store,
                call_claude=call_claude_fn,
                api_spec=api_spec,
                robot_label=robot_label,
                robot_name=robot_name,
                n=slate_n,
                seed=seed + turn_index,
                episode_log=episode_log,
                model=model_name,
                skill_docs=skill_docs_turn,
            )
            slate_records.append({
                "turn":          turn_index,
                "n_candidates":  len(slate.candidates),
                "dominant":      slate.dominant,
                "rank1_score":   slate.candidates[0].score if slate.candidates else 0.0,
                "axis_labels":   [c.axis_label for c in slate.candidates],
            })
        else:
            new_program = _apply_correction_singleshot(
                correction, current_program_ref["program"], call_claude_fn,
                api_spec=api_spec, robot_label=robot_label,
                model=model_name,
                store=store,
                retrieval_enabled=cond.get("retrieval", False),
                skill_docs=skill_docs_turn,
                inprogram_retry=inprogram_retry,
            )

        prior_program_for_turn = current_program_ref["program"]
        current_program_ref["program"] = new_program
        program_history.append(new_program)

        # PatternReuse: emit a subtask doc per correction, foreground so
        # the eval can see it land before the next turn's prompt is built.
        # Other conditions skip this — they don't have the markdown
        # subtask doc library mechanism. The emission uses
        # embedding-similarity to decide create-vs-update (see
        # architect.subtask_docs.select_target_doc) when ``inprogram_retry`` is
        # also on, which it is for PatternReuse.
        if markdown_library_enabled:
            subtask_record = _emit_subtask_doc_to_workspace(
                correction, new_program, prior_program_for_turn,
                subtasks_dir=paths["subtasks_dir"],
                call_claude=call_claude_fn,
                model=model_name,
                use_similarity_directive=inprogram_retry,
            )
            if subtask_record is not None:
                subtask_record["turn"] = turn_index
                subtask_doc_records.append(subtask_record)

        # Register + gate any helpers defined in the post-correction program.
        n_r, n_g = _register_and_gate_helpers(
            new_program,
            store=store,
            register=cond.get("library", False),
            graduate=cond.get("graduation", False),
            robot_name=robot_name,
        )
        helpers_registered += n_r
        helpers_graduated += n_g

        # Emit scorers + invariants only when the condition retains them
        # (graduation / slate ranking depend on them; ICL doesn't).
        if cond["graduation"] or cond["slate"]:
            n_s, n_i = _emit_and_persist_scorers(
                correction, new_program,
                turn_index=turn_index,
                store=store,
                call_claude=call_claude_fn,
            )
            scorers_persisted += n_s
            invariants_persisted += n_i

        corrections_applied = turn_index
        # Probe the post-correction program against the held-out perturbations
        # and re-use the report as the final OOD measurement if we early-exit.
        # Without this, corrections_applied is always len(corrections) and
        # the "corrections-to-success" metric is a constant.
        final_ood = _evaluate_ood(
            new_program, task["perturbations"],
            robot_name=robot_name, seed=seed,
            success_scorer=success_scorer,
        )
        final_ood_pass_rate = (
            sum(r["pass_rate"] for r in final_ood) / len(final_ood)
            if final_ood else 0.0
        )
        if final_ood_pass_rate >= early_exit_threshold:
            early_exit_triggered = True
            break

    # 3. Held-out OOD evaluation: only re-probe if the loop didn't already
    # leave a probe report (i.e. zero corrections in the task script).
    if not final_ood:
        final_ood = _evaluate_ood(
            current_program_ref["program"],
            task["perturbations"],
            robot_name=robot_name,
            seed=seed,
            success_scorer=success_scorer,
        )
        final_ood_pass_rate = (
            sum(r["pass_rate"] for r in final_ood) / len(final_ood)
            if final_ood else 0.0
        )
    ood = final_ood
    ood_pass_rate = final_ood_pass_rate

    # 4. Library snapshot.
    library_snapshot = {
        "total":      len(store.list_names()),
        "graduated":  len(store.list_names(status="graduated")),
        "candidate":  len(store.list_names(status="candidate")),
        "scorers":    scorers_persisted,
        "invariants": invariants_persisted,
    }

    # 5. Episode log + useful-disagreement (per-trial).
    ud_rate = episode_log.useful_disagreement_rate()

    record: dict[str, Any] = {
        "schema":               1,
        "task_id":              task_id,
        "sequence_id":          sequence_id,
        "position_in_sequence": position_in_sequence,
        "condition":            condition,
        "condition_config":     cond,
        "seed":                 seed,
        "llm":                  llm,
        "robot_name":           robot_name,
        "corrections":          list(task["corrections"]),
        "corrections_applied":  corrections_applied,
        "early_exit_triggered": early_exit_triggered,
        "early_exit_threshold": early_exit_threshold,
        "final_program":        current_program_ref["program"],
        "ood_pass_rate":        ood_pass_rate,
        "ood_per_perturbation": ood,
        "library":              library_snapshot,
        "useful_disagreement_rate": ud_rate,
        "slate_records":        slate_records,
        "subtask_doc_records":  subtask_doc_records,
    }

    paths["trial_record"].write_text(json.dumps(record, indent=2, default=str))
    store.close()
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True,
                        help="task_id from tasks.json")
    parser.add_argument("--condition", required=True,
                        choices=("ICL", "FuncReuse", "SlateGated", "PatternReuse"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tasks", type=Path,
                        default=Path(__file__).parent / "tasks.json",
                        help="Path to tasks.json (default: scripts/eval/tasks.json)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write trial JSONs + per-trial workspaces.")
    parser.add_argument("--llm", default="stub", choices=("stub", "claude"))
    parser.add_argument("--robot", default="franka", choices=("franka",))
    parser.add_argument("--slate-n", type=int, default=3)
    parser.add_argument("--sequence-id", default=None,
                        help="Sequence id this trial is a position of. Workspace "
                             "is keyed on (sequence_id, condition, seed) so prior "
                             "positions' library + episodes persist into this one.")
    parser.add_argument("--position-in-sequence", type=int, default=None,
                        help="0-based position; only meaningful with --sequence-id.")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip the trial-summary line on stdout.")
    args = parser.parse_args()

    record = run_trial(
        task_id=args.task, condition=args.condition, seed=args.seed,
        tasks_path=args.tasks, output_dir=args.output_dir,
        llm=args.llm, robot_name=args.robot, slate_n=args.slate_n,
        sequence_id=args.sequence_id,
        position_in_sequence=args.position_in_sequence,
    )

    if not args.quiet:
        ud = record["useful_disagreement_rate"]
        ud_str = f"{ud:.0%}" if ud is not None else "n/a"
        print(
            f"  {args.task:<28s} {args.condition:<6s} seed={args.seed:<3d} "
            f"OOD={record['ood_pass_rate']:.0%}  "
            f"lib={record['library']['graduated']}/{record['library']['total']}  "
            f"UD={ud_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
