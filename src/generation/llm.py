"""LiteLLM client — unified LLM interface.

Wraps LiteLLM so we can call any model (Claude, GPT-4, etc.) through
BaseLLM.generate(). LiteLLM handles the multi-provider abstraction.

Langfuse integration: if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
set in the environment (via .env.local), every LLM call is automatically
traced via LiteLLM's callback system. No code changes needed per-call —
it's fire-and-forget logging to Langfuse Cloud.
"""

import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

from src.base.interfaces import BaseLLM

# Lazy initialization flag — ensures dotenv + callbacks are set up once.
_initialized = False


def _ensure_initialized() -> None:
    """Load .env.local and configure Langfuse callbacks on first call.

    Follows the lazy initialization pattern from project 3: dotenv is
    loaded just-in-time when the first LLM call happens, not at import
    time. This avoids module-level side effects and ensures .env.local
    is found regardless of when the module is imported.
    """
    global _initialized
    if _initialized:
        return

    for candidate in [Path(".env.local"), Path("../.env.local")]:
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate))
            break

    # Enable Langfuse tracing if credentials are present.
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]

    _initialized = True


class LiteLLMClient(BaseLLM):
    """LLM client using LiteLLM for multi-provider support.

    Args:
        model: LiteLLM model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        _ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if metadata:
            kwargs["metadata"] = metadata

        response = litellm.completion(**kwargs)

        return response.choices[0].message.content
