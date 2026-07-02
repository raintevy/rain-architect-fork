"""Project-wide configuration constants.

Deployment-specific endpoints are read from the environment (see ``.env.example``)
so no personal/organization values are baked into the source.
"""

import os

# --- LLM (Claude) ---
ANTHROPIC_MODEL = "claude-opus-4-6"
#: Base URL for the Anthropic-compatible endpoint (e.g. an Azure AI Foundry
#: deployment). Set ANTHROPIC_BASE_URL in your .env; leave unset to use the
#: default Anthropic API.
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")

# --- Azure OpenAI (used for embeddings; optional) ---
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --- Embeddings (for slate-time relevance filtering / hybrid retrieval) ---
#: Works as either an Azure deployment name (when AZURE_OPENAI_API_KEY is set)
#: or a standard OpenAI model name (when only OPENAI_API_KEY is set).
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
