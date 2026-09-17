"""Shared Anthropic API client for Claude calls via Azure AI Foundry."""

from __future__ import annotations

import os
import threading

from anthropic import Anthropic


# Module-level singleton. Previously _make_client() ran on every call_claude
# / call_claude_with_tools call, instantiating a fresh AnthropicFoundry
# (which holds an HTTP session + connection pool) every time. A 120-trial
# real-LLM sweep can issue 2000+ calls; each leaked client accumulates
# until the process is OOM-killed (caught during the v3 sweep). Caching
# the client once per process keeps memory flat over the run.
_client: Anthropic | None = None
_client_lock = threading.Lock()


def _make_client() -> Anthropic:
    """Return the process-wide AnthropicFoundry client; build it on first call."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from config import ANTHROPIC_BASE_URL  # late import — avoids circular import at module load
        _client = Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=ANTHROPIC_BASE_URL,
        )
        return _client


def call_claude(
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """Send messages to Claude via Azure AI Foundry and return the response text.

    The first message may have role "system"; it is extracted and passed as
    Claude's top-level system parameter.
    """
    client = _make_client()

    system = None
    user_messages = messages
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        user_messages = messages[1:]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=user_messages,
    )
    return response.content[0].text


def call_claude_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
):
    """Send messages to Claude with tool definitions; return the raw response.

    Unlike call_claude(), this returns the full response object so the caller
    can inspect stop_reason and iterate over content blocks (TextBlock /
    ToolUseBlock).

    The first message may have role "system"; it is extracted and passed as
    Claude's top-level system parameter.
    """
    client = _make_client()

    system = None
    user_messages = messages
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        user_messages = messages[1:]

    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=tools,
        messages=user_messages,
    )
