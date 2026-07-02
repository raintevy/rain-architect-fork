"""Registry of session primitives, backed by an optional persistent SkillStore.

The public API matches the original in-memory implementation so existing
callers (``architect.agent.AgenticSession``, ``architect.agent_tools.ToolExecutor``,
``scripts.architect_cli``) need no changes. When constructed with no ``store``,
an in-memory SQLite SkillStore is used and behavior is equivalent to the
old in-memory dict — primitives vanish at process exit. Pass a path-backed
:class:`SkillStore` (per-robot ``library.db``) for cross-session persistence.
"""

from __future__ import annotations

from architect.library.skill_store import SkillStore


class PrimitiveRegistry:
    """Holds helper functions synthesized or approved during a session.

    Each entry stores the complete function definition so it can be injected
    into ``exec()`` namespaces and included in the API spec shown to Claude.
    """

    def __init__(self, store: SkillStore | None = None) -> None:
        self._store = store if store is not None else SkillStore()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, name: str, code: str, docstring: str) -> None:
        """Register a primitive.

        Args:
            name: Python function name.
            code: Complete function definition starting with ``def``.
            docstring: One-line description for the API spec.
        """
        self._store.register(name, code, docstring)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_source(self, name: str) -> str | None:
        """Return the source code for a registered primitive, or None."""
        return self._store.get_source(name)

    def list_names(self) -> list[str]:
        return self._store.list_names()

    def to_spec_string(self) -> str:
        """Format registered primitives as an API spec section."""
        return self._store.to_spec_string()

    # ------------------------------------------------------------------
    # Namespace injection
    # ------------------------------------------------------------------

    def inject_into_namespace(self, namespace: dict) -> None:
        """Exec each registered function into *namespace* in registration order."""
        for entry in self._store.get_all():
            try:
                exec(entry["code"], namespace)  # noqa: S102
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def store(self) -> SkillStore:
        """Underlying :class:`SkillStore`. Useful for retrieval, outcome
        recording, and graduation-status changes that the registry does not
        proxy."""
        return self._store

    def close(self) -> None:
        self._store.close()
