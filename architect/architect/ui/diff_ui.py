"""Side-by-side slate display + user pick (P4 / C1).

Renders a slate of candidate programs as labelled rich panels with
score, probe pass-rate, and audit-flag summary; then prompts the user
to pick one. The pick (and the rejected candidates' axis labels) are
the multi-axis supervision signal recorded by :mod:`architect.library.episode_log`.

Console output is via the same ``rich.Console`` instance the CLI uses
elsewhere, so colors, line wrapping, and width respect the user's
terminal.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from architect.probes.audit import severity_summary
from architect.probes.sampling import Candidate, Slate


# ---------------------------------------------------------------------------
# Pick result
# ---------------------------------------------------------------------------


@dataclass
class PickResult:
    """What the user did when shown the slate.

    ``index`` is ``None`` if the user cancelled / abandoned. The list of
    rejected axis labels is what the episode log persists as supervision
    for the un-picked alternatives.
    """

    index: int | None
    picked: Candidate | None
    rejected_axes: list[str]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


# Letters used to label candidates in the UI. Matches the LLM prompt's
# "Candidate A / B / C" convention so the displayed letters line up with
# the LLM's own labels.
_LETTERS = "ABCDEFGH"


def _candidate_summary(idx: int, c: Candidate, *, show_diff_against: str | None = None) -> Panel:
    letter = _LETTERS[idx] if idx < len(_LETTERS) else str(idx + 1)
    sev = severity_summary(c.audit_flags)
    pass_rate = c.probe_report.pass_rate if c.probe_report is not None else 0.0
    n_probes = c.probe_report.k if c.probe_report is not None else 0

    header = Table.grid(padding=(0, 1))
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text(c.axis_label, style="bold cyan"),
        Text(f"score {c.score:.2f}", style="bold green"),
    )
    inv_rate = (
        c.probe_report.invariant_pass_rate if c.probe_report is not None else 1.0
    )
    header.add_row(
        Text(
            f"probes {pass_rate:.0%} ({n_probes})  •  "
            f"invariants {inv_rate:.0%}  •  "
            f"audit: {sev['high']}H {sev['medium']}M",
            style="dim",
        ),
        Text(
            f"pass={c.score_breakdown.get('pass_rate', 0):.2f}  "
            f"scorer={c.score_breakdown.get('scorer_mean', 0):.2f}  "
            f"−audit={c.score_breakdown.get('audit_penalty', 0):.2f}",
            style="dim",
        ),
    )

    breakdown = _render_breakdown(c.score_breakdown)

    body: object
    if show_diff_against is not None:
        body = _render_diff(show_diff_against, c.code)
    else:
        body = Syntax(c.code, "python", theme="monokai", line_numbers=False)

    inner = Table.grid()
    inner.add_column()
    inner.add_row(header)
    if breakdown is not None:
        inner.add_row(breakdown)
    inner.add_row(body)

    return Panel(
        inner,
        title=f"[bold]Candidate {letter}[/bold]",
        border_style="cyan" if idx == 0 else "blue",
    )


def _render_breakdown(score_breakdown: dict) -> Text | None:
    """Render the per-scorer / per-invariant breakdown when there's signal worth showing.

    Returns ``None`` (and the row is omitted) when the breakdown carries
    nothing actionable: no scorers ran, every invariant held, etc. The goal
    is to keep the candidate panel quiet on the happy path and surface
    attribution only when a probe-time signal moved the candidate's score.
    """
    per_scorer: dict[str, float] = score_breakdown.get("per_scorer", {}) or {}
    per_inv: dict[str, float] = (
        score_breakdown.get("per_invariant_violation_rate", {}) or {}
    )
    inv_failures = {name: rate for name, rate in per_inv.items() if rate > 0}

    if not per_scorer and not inv_failures:
        return None

    text = Text()
    if per_scorer:
        text.append("scorers: ", style="dim")
        chunks = [
            f"{name}={mean:.2f}"
            for name, mean in sorted(per_scorer.items(), key=lambda kv: -kv[1])
        ]
        text.append("  ".join(chunks), style="dim cyan")
    if inv_failures:
        if per_scorer:
            text.append("\n")
        text.append("invariant violations: ", style="dim")
        chunks = [
            f"{name} ({rate:.0%})"
            for name, rate in sorted(inv_failures.items(), key=lambda kv: -kv[1])
        ]
        text.append("  ".join(chunks), style="dim red")
    return text


def _render_diff(old: str, new: str) -> Text:
    """Compact unified diff old → new, color-coded for the CLI."""
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="before",
        tofile="after",
        lineterm="",
        n=2,
    ))
    if not diff:
        return Text("(no changes)", style="dim")
    text = Text()
    for line in diff:
        if line.startswith(("---", "+++")):
            text.append(line + "\n", style="bold")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        else:
            text.append(line + "\n", style="dim")
    return text


def render_slate(
    slate: Slate,
    console: Console,
    *,
    current_program: str | None = None,
    show_diff: bool = True,
) -> None:
    """Print a slate to ``console``: one panel per candidate, top-ranked first.

    When ``show_diff`` is True (default) and ``current_program`` is provided,
    the body of each panel is the unified diff against the current program
    so the user sees what *changes* the candidate proposes rather than the
    full code. Set ``show_diff=False`` to print the full code instead.
    """
    if not slate.candidates:
        console.print("[yellow]No candidates produced. Falling back to single-shot correction.[/yellow]")
        return

    diff_against = current_program if show_diff else None
    panels = [
        _candidate_summary(i, c, show_diff_against=diff_against)
        for i, c in enumerate(slate.candidates)
    ]

    console.print()
    console.rule(f"[bold]Slate ({len(slate.candidates)} candidates)[/bold]")
    if slate.dominant:
        console.print("[dim]Top candidate dominates — auto-apply available.[/dim]")
    for p in panels:
        console.print(p)


def prompt_pick(
    slate: Slate,
    console: Console,
    *,
    current_program: str | None = None,
) -> PickResult:
    """Show the slate, then ask which candidate to apply.

    Accepted inputs at the prompt:

      ``A``..``Z``  pick that candidate (case-insensitive)
      ``1``..``N``  pick that candidate by 1-based index
      ``?``         re-print the slate as full code (instead of diff)
      ``q`` / EOF   cancel; returns :class:`PickResult` with ``index=None``

    The returned :class:`PickResult` carries the picked candidate and the
    axis labels of every *rejected* candidate, which the episode log
    persists as supervision for the user's preference.
    """
    if not slate.candidates:
        return PickResult(index=None, picked=None, rejected_axes=[])

    render_slate(slate, console, current_program=current_program, show_diff=True)
    n = len(slate.candidates)
    prompt = (
        f"\n[bold]Pick one[/bold] "
        f"({'/'.join(_LETTERS[i] for i in range(n))}, "
        f"[dim]?=show full code, q=cancel[/dim]) › "
    )

    while True:
        console.print(prompt, end="")
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return PickResult(index=None, picked=None, rejected_axes=[])

        if not raw or raw.lower() in ("q", "quit", "cancel", "x"):
            return PickResult(index=None, picked=None, rejected_axes=[])

        if raw == "?":
            render_slate(slate, console, current_program=current_program, show_diff=False)
            continue

        idx: int | None = None
        if raw.isdigit():
            i = int(raw) - 1
            if 0 <= i < n:
                idx = i
        elif len(raw) == 1 and raw.upper() in _LETTERS[:n]:
            idx = _LETTERS.index(raw.upper())

        if idx is None:
            console.print(f"[yellow]didn't understand {raw!r}; try {'/'.join(_LETTERS[:n])} or q[/yellow]")
            continue

        picked = slate.candidates[idx]
        rejected = [c.axis_label for j, c in enumerate(slate.candidates) if j != idx]
        return PickResult(index=idx, picked=picked, rejected_axes=rejected)
