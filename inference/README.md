# Inference Module

Serve LLMs locally on the NVIDIA DGX Spark using [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible API.

## Scripts

| Script | Purpose |
|--------|---------|
| `start_vllm_gptoss.sh` | Start vLLM with GPT-OSS 120B (full DGX Spark setup) |
| `start_vllm.sh` | Start vLLM with any HuggingFace model |
| `stop_vllm.sh` | Gracefully stop all vLLM processes + release GPU |
| `health_check.sh` | End-to-end pipeline health check (GPU → API → Cline) |
| `switch_model.sh` | Hot-swap to a different model |
| `test_inference.py` | API smoke tests (models, chat, streaming) |

## Quick Start

```bash
# Start GPT-OSS 120B
make serve

# Or start any model
bash inference/start_vllm.sh Qwen/Qwen2.5-7B-Instruct

# Check health
make health

# Stop
make stop
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model to serve |
| `VLLM_HOST` | `0.0.0.0` | Listen host |
| `VLLM_PORT` | `8000` | Listen port |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max context length |
| `VLLM_VENV_DIR` | `../vllm-env` | Python virtual environment path |

## API Endpoints

Once running, the server exposes the standard OpenAI API:

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Memory Requirements

| Model | VRAM/Memory | Context Length |
|-------|-------------|----------------|
| GPT-OSS 120B (mxfp4) | ~65 GB | 32K tokens |
| Qwen2.5-7B-Instruct | ~15 GB | 32K tokens |
| Llama-3.1-8B-Instruct | ~17 GB | 8K tokens |

The DGX Spark has 128 GB of unified CPU+GPU memory, so GPT-OSS 120B fits comfortably.
