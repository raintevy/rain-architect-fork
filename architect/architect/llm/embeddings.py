"""OpenAI / Azure OpenAI embeddings for the slate-time relevance filter.

The relevance filter (``architect.library.relevance``) calls :func:`get_embeddings`
once per slate, passing the active correction plus every stored
scorer/invariant description, and uses cosine similarity to keep only
the artifacts whose meaning overlaps the correction's.

Backend selection
-----------------

The client auto-picks between two backends based on which credential is
present in the environment:

* ``AZURE_OPENAI_API_KEY`` set → :class:`openai.AzureOpenAI` against the
  ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_VERSION`` configured in
  ``config.py``. ``OPENAI_EMBEDDING_MODEL`` is treated as a *deployment
  name* on the tenant.

* Only ``OPENAI_API_KEY`` set → :class:`openai.OpenAI` against the
  standard ``api.openai.com`` endpoint. ``OPENAI_EMBEDDING_MODEL`` is
  treated as a standard model name.

Both backends accept ``text-embedding-3-small`` at the same price, so a
shared default keeps the config flat. If neither key is set,
:class:`EmbeddingsUnavailable` is raised and callers
(``filter_artifacts(backend="auto")``) fall back to the lexical filter
with a one-time console warning.

Caching
-------

Embeddings are cached in-memory keyed by exact text. The cache is per-
process; it survives across slates in the same CLI session but does
*not* persist across runs. Artifact descriptions are stable (they're the
correction text the artifact was built for), so re-embedding the same
library across many slates is a one-API-call-per-description cost,
amortised by the cache.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Iterable


class EmbeddingsUnavailable(RuntimeError):
    """Raised when the embedding backend can't service a request.

    Distinct from generic ``Exception`` so callers can match it precisely
    and choose to fall back to lexical similarity rather than abort. The
    error message names the underlying cause (no API key, deployment
    missing, network error) so the one-time warning is actionable.
    """


# ---------------------------------------------------------------------------
# Module-level cache + client
# ---------------------------------------------------------------------------


from collections import OrderedDict

# In-memory text → embedding cache, bounded so long-running sweeps don't
# accumulate every unique correction / description ever embedded. 5000
# entries × 1536 floats × 8 bytes ≈ 60 MB — generous for a single sweep
# but won't grow without bound. Keys are texts; eviction is FIFO via
# OrderedDict.move_to_end (used as LRU).
_CACHE_MAX_ENTRIES: int = 5000
_cache: "OrderedDict[str, list[float]]" = OrderedDict()
_cache_lock = threading.Lock()
_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazy embeddings client. Picks Azure or plain OpenAI by env credential.

    Raises :class:`EmbeddingsUnavailable` if neither ``AZURE_OPENAI_API_KEY``
    nor ``OPENAI_API_KEY`` is set, or if the ``openai`` package can't be
    imported.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not azure_key and not openai_key:
            raise EmbeddingsUnavailable(
                "neither AZURE_OPENAI_API_KEY nor OPENAI_API_KEY is set"
            )
        try:
            import openai  # noqa: F401  (import surface check before branching)
        except Exception as exc:
            raise EmbeddingsUnavailable(
                f"failed to import openai package: {exc!r}"
            ) from exc
        if azure_key:
            from openai import AzureOpenAI
            from config import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT
            _client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=azure_key,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        else:
            from openai import OpenAI
            _client = OpenAI(api_key=openai_key)
        return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_embedding(text: str, *, model: str | None = None) -> list[float]:
    """Embed one piece of text. Convenience wrapper over :func:`get_embeddings`."""
    return get_embeddings([text], model=model)[0]


def get_embeddings(
    texts: Iterable[str],
    *,
    model: str | None = None,
) -> list[list[float]]:
    """Embed a batch of texts and return their vectors in input order.

    Texts already present in the in-memory cache are not re-requested. The
    Azure API is called at most once per invocation, batching every
    cache-miss text into a single request. The returned list mirrors the
    order of ``texts`` (including duplicates).

    Raises :class:`EmbeddingsUnavailable` if the Azure deployment can't
    be reached. The caller decides whether to fall back to lexical
    matching or propagate the failure.
    """
    if model is None:
        from config import OPENAI_EMBEDDING_MODEL
        model = OPENAI_EMBEDDING_MODEL

    text_list = list(texts)
    # Identify which inputs we need to fetch fresh, deduplicated.
    missing: list[str] = []
    seen: set[str] = set()
    with _cache_lock:
        for t in text_list:
            if t not in _cache and t not in seen:
                missing.append(t)
                seen.add(t)

    if missing:
        client = _get_client()
        try:
            response = client.embeddings.create(model=model, input=missing)
        except Exception as exc:
            raise EmbeddingsUnavailable(
                f"embedding call failed ({type(client).__name__}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Response data is in the same order as the input list. Insert
        # under the lock and evict LRU entries beyond the cap.
        with _cache_lock:
            for text, item in zip(missing, response.data):
                _cache[text] = list(item.embedding)
                _cache.move_to_end(text)
            while len(_cache) > _CACHE_MAX_ENTRIES:
                _cache.popitem(last=False)  # FIFO/LRU eviction

    with _cache_lock:
        # Touch every retrieved entry so popular texts stay hot.
        result = []
        for t in text_list:
            _cache.move_to_end(t)
            result.append(_cache[t])
        return result


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; returns 0.0 if either vector is zero."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom > 0 else 0.0


def is_available() -> bool:
    """True iff the embedding client can be initialised (env var present)."""
    try:
        _get_client()
        return True
    except EmbeddingsUnavailable:
        return False
