#!/usr/bin/env python3
"""
Streaming chat completion example.

Demonstrates real-time token-by-token output from the local vLLM server.

Usage:
    python examples/streaming_example.py
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    exit(1)

PORT = os.environ.get("VLLM_PORT", "8000")

client = OpenAI(
    base_url=f"http://localhost:{PORT}/v1",
    api_key="local-vllm",
)

# Auto-detect model
models = client.models.list()
model_id = models.data[0].id
print(f"Model: {model_id} | Streaming enabled")
print("─" * 60)

# Stream the response
stream = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": "Explain how a transformer neural network works, step by step."},
    ],
    max_tokens=512,
    temperature=0.7,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        sys.stdout.write(chunk.choices[0].delta.content)
        sys.stdout.flush()

print("\n" + "─" * 60)
print("Stream complete!")
