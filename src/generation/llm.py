"""LiteLLM client — unified LLM interface.

Wraps LiteLLM so we can call any model (Claude, GPT-4, etc.) through
BaseLLM.generate(). LiteLLM handles the multi-provider abstraction.

Langfuse integration: if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
set in the environment, every LLM call is automatically traced via
LiteLLM's callback system. No code changes needed per-call — it's
fire-and-forget logging to Langfuse Cloud.
"""

import os

import litellm

from src.base.interfaces import BaseLLM

# Enable Langfuse tracing if credentials are present.
# This is a module-level side effect — runs once on import.
# If keys are missing, LLM calls work normally with no tracing.
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


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
