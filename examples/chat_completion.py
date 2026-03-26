#!/usr/bin/env python3
"""
Basic chat completion example using the OpenAI Python SDK.

Connects to the local vLLM server and sends a simple chat request.

Usage:
    python examples/chat_completion.py
    VLLM_PORT=8001 python examples/chat_completion.py
"""

import os

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    exit(1)

PORT = os.environ.get("VLLM_PORT", "8000")
BASE_URL = f"http://localhost:{PORT}/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="local-vllm",  # vLLM doesn't require a real API key
)

# Auto-detect model name
models = client.models.list()
model_id = models.data[0].id
print(f"Using model: {model_id}")
print(f"Endpoint:    {BASE_URL}")
print()

# Send a chat completion request
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that reverses a linked list. Include type hints."},
    ],
    max_tokens=512,
    temperature=0.7,
)

# Print the response
print("─" * 60)
print(response.choices[0].message.content)
print("─" * 60)
print(f"\nTokens: prompt={response.usage.prompt_tokens}, "
      f"completion={response.usage.completion_tokens}, "
      f"total={response.usage.total_tokens}")
