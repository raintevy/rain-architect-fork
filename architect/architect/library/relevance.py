"""Lexical relevance filter for stored scorers and invariants.

When a user issues a new correction, ``agent.correct_with_slate`` pulls
*every* scorer + invariant from the SkillStore and feeds them to
``rank_candidates``. That produces cross-talk: scorers built for one
correction silently penalise candidates for unrelated corrections (a
side-approach scorer rates a "set down the cube" candidate as ~0.1
because it sees no lateral motion). Invariants are worse — they were
historically AND-gated into the probe ``passed`` flag, so a single
violation from a stale invariant zeroed the candidate's pass rate.

This module provides a cheap, dependency-free filter: each artifact's
``description`` (the original correction text the artifact was built
for) is compared against the active correction using Jaccard overlap of
*content words*. Artifacts below the threshold are skipped for this
slate.

The trade-off is the usual lexical-similarity one: synonyms miss, and
domain jargon may need a custom stopword list. The upgrade path is
sentence embeddings (cosine similarity), drop-in compatible with the
``filter_artifacts`` signature — see the design doc's C1 / C3 section.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal

from architect.corrections.llm_scorer import ScorerArtifact


_Backend = Literal["auto", "lexical", "embedding"]


# Default thresholds per backend. Cosine over modern embeddings lives in a
# much narrower band than Jaccard over content-word sets, so the embedding
# threshold is calibrated separately. These are starting points — tune via
# the eval harness once enough corrections have accumulated.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "lexical": 0.20,
    "embedding": 0.45,
}


# Tracks whether we've already warned about the embedding backend being
# unavailable. Avoids spamming the console on every slate when Azure is
# unreachable for a whole session.
_embedding_unavailable_warned: bool = False


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


# Small focused stopword set. We deliberately keep this list short — over-
# aggressive stopwording strips out genuine correction signal ("not above",
# "move slower"). Tune up if the eval harness shows the filter is too loose.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "as", "at", "be", "but", "by", "did", "do",
    "for", "from", "if", "in", "is", "it", "of", "on", "or", "so",
    "than", "that", "the", "then", "this", "those", "to", "with",
    # Correction-domain noise: programs and corrections talk about themselves
    # constantly, so these add no discriminative signal.
    "correction", "program", "robot", "step", "make", "should", "want",
})

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


def content_words(text: str) -> set[str]:
    """Lowercase, tokenise, drop stopwords and 1-2 char tokens.

    The 1-2 char filter discards "no" / "is" / "x" / "z" tokens that pop
    up in correction text but rarely carry distinguishing meaning. Axis
    references like "+y" / "-z" survive via the surrounding context words
    ("approach", "lateral", "horizontal", etc.).
    """
    return {
        w.lower() for w in _WORD_RE.findall(text)
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets in [0, 1]; 0 when both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Public filter
# ---------------------------------------------------------------------------


def relevance_score(active_correction: str, artifact: ScorerArtifact) -> float:
    """Compute the relevance score the filter uses to keep / skip an artifact.

    Defined as the Jaccard overlap between the active correction's
    content-word set and the artifact's description's content-word set.
    Provenance is not consulted — the description is what the user sees
    in the SkillStore and what they'd reason about manually.
    """
    return jaccard(
        content_words(active_correction),
        content_words(artifact.description),
    )


def _embedding_scores(
    active_correction: str,
    artifacts: list[ScorerArtifact],
) -> list[float]:
    """Cosine-similarity score per artifact via Azure embeddings.

    One batched API call covers the active correction and every artifact
    description. Returns parallel scores in ``artifacts`` order. Raises
    :class:`architect.embeddings.EmbeddingsUnavailable` on backend failure; the
    auto-mode dispatcher catches and falls back to lexical.
    """
    from architect.llm.embeddings import cosine, get_embeddings

    if not artifacts:
        return []
    texts = [active_correction] + [a.description for a in artifacts]
    vectors = get_embeddings(texts)
    active_vec = vectors[0]
    return [cosine(active_vec, v) for v in vectors[1:]]


def filter_artifacts(
    active_correction: str,
    artifacts: Iterable[ScorerArtifact],
    *,
    backend: _Backend = "auto",
    min_overlap: float | None = None,
    console=None,
) -> tuple[list[ScorerArtifact], list[tuple[ScorerArtifact, float]]]:
    """Split ``artifacts`` into ``(kept, skipped_with_score)`` by relevance.

    ``backend`` selects the similarity function:

    * ``"lexical"``  — Jaccard over content words; threshold defaults to 0.20.
    * ``"embedding"`` — Azure OpenAI embeddings + cosine; threshold
      defaults to 0.45. Raises if the embedding backend can't service the
      request (no API key, deployment missing, network error).
    * ``"auto"`` — try embeddings, fall back to lexical on
      :class:`architect.embeddings.EmbeddingsUnavailable`. Emits a one-time
      warning per process to ``console`` (if provided) so users notice
      they're running on the weaker backend.

    ``min_overlap=None`` (the default) uses the backend's calibrated
    threshold from ``_DEFAULT_THRESHOLDS``. Passing an explicit float
    overrides it — set to ``-1`` (or any value ≤ minimum) to disable
    filtering entirely while keeping the score column populated.

    Returns parallel lists so the caller can log "filtered N→M scorers
    (skipped: c1_score [0.05], c2_score [0.00])" with the actual scores
    that drove each decision.
    """
    artifacts = list(artifacts)
    if not artifacts:
        return [], []

    chosen_backend: str = backend
    scores: list[float]
    if backend == "auto":
        try:
            scores = _embedding_scores(active_correction, artifacts)
            chosen_backend = "embedding"
        except Exception as exc:
            global _embedding_unavailable_warned
            if console is not None and not _embedding_unavailable_warned:
                console.print(
                    f"[dim]\\[relevance] embedding backend unavailable "
                    f"({type(exc).__name__}); falling back to lexical. "
                    f"Subsequent fallbacks will be silent.[/dim]"
                )
                _embedding_unavailable_warned = True
            scores = [relevance_score(active_correction, a) for a in artifacts]
            chosen_backend = "lexical"
    elif backend == "embedding":
        scores = _embedding_scores(active_correction, artifacts)
    elif backend == "lexical":
        scores = [relevance_score(active_correction, a) for a in artifacts]
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    threshold = (
        min_overlap if min_overlap is not None
        else _DEFAULT_THRESHOLDS[chosen_backend]
    )

    kept: list[ScorerArtifact] = []
    skipped: list[tuple[ScorerArtifact, float]] = []
    for art, score in zip(artifacts, scores):
        if score >= threshold:
            kept.append(art)
        else:
            skipped.append((art, score))
    return kept, skipped
