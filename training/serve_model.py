#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible API server for fine-tuned models.

Uses transformers + FastAPI to serve a fine-tuned model with the standard
OpenAI chat completions API format. Useful for testing before deploying to vLLM.

Usage:
    python serve_model.py output/merged
    python serve_model.py output/merged my-model-name
    PORT=8001 python serve_model.py output/merged
"""

import json
import os
import sys
import time
import uuid

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./output/merged"
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "fine-tuned-model"
PORT = int(os.environ.get("PORT", "8000"))

print(f"Loading model from {MODEL_PATH}...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
print(f"Model loaded: {MODEL_NAME}")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Fine-tuned Model Server")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    # Format as ChatML
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    result_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if stream:
        async def generate():
            chunk = {
                "id": result_id, "object": "chat.completion.chunk", "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    return {
        "id": result_id, "object": "chat.completion", "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(response_ids),
            "total_tokens": inputs["input_ids"].shape[1] + len(response_ids),
        }
    }


if __name__ == "__main__":
    print(f"Starting server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
