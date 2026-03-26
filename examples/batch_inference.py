#!/usr/bin/env python3
"""
Batch inference example — process multiple prompts concurrently.

Sends multiple chat completion requests in parallel using asyncio,
demonstrating vLLM's ability to batch requests efficiently.

Usage:
    python examples/batch_inference.py
"""

import asyncio
import os
import time

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    exit(1)

PORT = os.environ.get("VLLM_PORT", "8000")

client = AsyncOpenAI(
    base_url=f"http://localhost:{PORT}/v1",
    api_key="local-vllm",
)

PROMPTS = [
    "Write a one-line Python function to check if a string is a palindrome.",
    "What is the time complexity of quicksort? Answer in one sentence.",
    "Explain the difference between a stack and a queue in 2 sentences.",
    "Write a bash one-liner to find the largest file in the current directory.",
    "What does the `yield` keyword do in Python? One sentence.",
]


async def complete(prompt: str, model: str) -> dict:
    """Send a single chat completion request."""
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content.strip()
    return {
        "prompt": prompt[:50],
        "response": content[:100],
        "tokens": response.usage.total_tokens,
        "latency": f"{elapsed:.2f}s",
    }


async def main():
    # Auto-detect model
    models = await client.models.list()
    model_id = models.data[0].id

    print(f"Model: {model_id}")
    print(f"Batch size: {len(PROMPTS)} prompts")
    print("=" * 60)

    # Send all requests concurrently
    start = time.time()
    results = await asyncio.gather(*[complete(p, model_id) for p in PROMPTS])
    total = time.time() - start

    # Print results
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['prompt']}...")
        print(f"    → {r['response']}...")
        print(f"    ({r['tokens']} tokens, {r['latency']})")

    print(f"\n{'=' * 60}")
    print(f"Total time: {total:.2f}s for {len(PROMPTS)} requests")
    print(f"Throughput: {len(PROMPTS)/total:.1f} req/s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
