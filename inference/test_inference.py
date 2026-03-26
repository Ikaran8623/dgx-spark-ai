#!/usr/bin/env python3
"""
Smoke test for the vLLM API endpoint.

Tests both /v1/models and /v1/chat/completions to verify the server
is running and generating responses correctly.

Usage:
    python3 test_inference.py                    # Default: localhost:8000
    python3 test_inference.py --port 8001        # Custom port
    python3 test_inference.py --base-url http://remote:8000/v1
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Test vLLM API endpoint")
    parser.add_argument("--base-url", default=None, help="Full base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")), help="vLLM port")
    parser.add_argument("--host", default="localhost", help="vLLM host")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"

    print("=" * 60)
    print(f"  vLLM API Smoke Test — {base_url}")
    print("=" * 60)

    # ─── Test 1: List models ─────────────────────────────────────
    print("\n[1/3] GET /v1/models")
    try:
        r = requests.get(f"{base_url}/models", timeout=10)
        r.raise_for_status()
        models = r.json()
        model_id = models["data"][0]["id"]
        print(f"  ✓ Server is up — serving: {model_id}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print(f"\n  Is vLLM running? Start it with: make serve")
        sys.exit(1)

    # ─── Test 2: Chat completion ─────────────────────────────────
    print("\n[2/3] POST /v1/chat/completions")
    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=60,
        )
        r.raise_for_status()
        elapsed = time.time() - start
        data = r.json()
        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        print(f"  ✓ Response: {content.strip()[:80]}")
        print(f"    Tokens: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}")
        print(f"    Latency: {elapsed:.2f}s")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # ─── Test 3: Streaming ───────────────────────────────────────
    print("\n[3/3] POST /v1/chat/completions (streaming)")
    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 64,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=60,
            stream=True,
        )
        r.raise_for_status()

        chunks = 0
        content_parts = []
        first_token_time = None
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks += 1
                if first_token_time is None:
                    first_token_time = time.time() - start
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    pass

        elapsed = time.time() - start
        content = "".join(content_parts)
        print(f"  ✓ Streamed {chunks} chunks")
        print(f"    Response: {content.strip()[:80]}")
        print(f"    TTFT: {first_token_time:.2f}s | Total: {elapsed:.2f}s")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
