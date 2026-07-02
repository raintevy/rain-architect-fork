"""Mid-execution interruption for --hil and --vlm modes.

When a generated program's in-program VQA retry exhausts (two consecutive
"no" answers), a ``MidExecInterrupt`` is raised *inside* ``exec(program)``.
``execute_with_synthesis`` catches it and returns the partial trace so the
CLI can ask for a correction (human or auto) and generate a continuation
program from the current robot state.
"""

from __future__ import annotations

from typing import Any, Callable


class MidExecInterrupt(Exception):
    """Raised mid-execution when VQA retry exhausts under --hil / --vlm."""

    def __init__(
        self,
        vqa_question: str,
        vqa_answer: str,
        consecutive_no_count: int = 2,
    ) -> None:
        self.vqa_question = vqa_question
        self.vqa_answer = vqa_answer
        self.consecutive_no_count = consecutive_no_count
        super().__init__(
            f"VQA retry exhausted ({consecutive_no_count} consecutive 'no'): "
            f"{vqa_question!r} → {vqa_answer!r}"
        )


def wrap_vqa_for_midexec(namespace: dict) -> None:
    """Replace ``get_vqa_response`` in *namespace* with an intercepting wrapper.

    The wrapper tracks consecutive "no" answers. On the first "no" it
    returns normally (letting the in-program retry handle it). On the
    second consecutive "no" it raises :class:`MidExecInterrupt`.

    A "yes" or ambiguous answer resets the counter.

    Call :func:`unwrap_vqa_for_midexec` before executing a continuation
    program so the counter starts fresh.
    """
    original: Callable[..., Any] | None = namespace.get("get_vqa_response")
    if original is None:
        return
    # Guard against double-wrapping.
    if getattr(original, "__midexec_wrapped__", False):
        return

    # Mutable state shared by the wrapper closure.
    state = {"consecutive_no": 0, "last_question": None}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)

        # Extract the answer string.
        try:
            answer = result["data"]["answer"].lower()
        except (KeyError, TypeError, AttributeError):
            # Non-standard return — treat as ambiguous, reset counter.
            state["consecutive_no"] = 0
            return result

        # Determine the question text for display purposes.
        question = args[0] if args else kwargs.get("prompt", "<unknown>")

        if "no" in answer:
            state["consecutive_no"] += 1
            state["last_question"] = question
            if state["consecutive_no"] >= 2:
                raise MidExecInterrupt(
                    vqa_question=str(question),
                    vqa_answer=answer,
                    consecutive_no_count=state["consecutive_no"],
                )
        else:
            # "yes" or ambiguous → block succeeded or soft-success.
            state["consecutive_no"] = 0

        return result

    wrapped.__midexec_wrapped__ = True  # type: ignore[attr-defined]
    wrapped.__midexec_original__ = original  # type: ignore[attr-defined]
    namespace["get_vqa_response"] = wrapped


def unwrap_vqa_for_midexec(namespace: dict) -> None:
    """Restore the original ``get_vqa_response`` if it was wrapped."""
    current = namespace.get("get_vqa_response")
    if current is not None and getattr(current, "__midexec_wrapped__", False):
        namespace["get_vqa_response"] = current.__midexec_original__  # type: ignore[union-attr]
