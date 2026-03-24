#!/usr/bin/env python3
"""Light test: verify LLM + Langfuse tracing works.

Run: python scripts/test_langfuse.py
Then check Langfuse dashboard for the trace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.llm import LiteLLMClient

llm = LiteLLMClient(model="gpt-4o-mini")
print("Calling gpt-4o-mini...")
response = llm.generate(
    "Say 'hello' in one word.",
    metadata={"trace_name": "langfuse-test"},
)
print(f"Response: {response}")
print("\nCheck Langfuse dashboard for the trace.")
