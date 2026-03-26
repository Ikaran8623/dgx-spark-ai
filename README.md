# dgx-spark-ai

**Run GPT-OSS 120B locally on NVIDIA DGX Spark. Fine-tune your own models. Use Cline CLI with zero cloud dependency.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.18.0-green)](https://github.com/vllm-project/vllm)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900)](https://developer.nvidia.com/cuda-toolkit)

---

## What is this?

A complete toolkit for running LLM inference and training on the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) personal AI supercomputer:

- **🚀 Inference** — Serve GPT-OSS 120B (or any HuggingFace model) via vLLM with an OpenAI-compatible API
- **🧠 Training** — Fine-tune models with QLoRA using Unsloth (2x faster) or standard PEFT/TRL
- **🔧 Cline Integration** — Auto-configure [Cline CLI](https://github.com/cline/cline) for local AI-powered coding
- **🛡️ Production-ready** — Systemd services, health watchdog, auto-restart on failure

## Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA GB10 Grace Blackwell (sm_121a) |
| **Memory** | 128 GB unified CPU+GPU |
| **CUDA** | 13.0 |
| **Architecture** | ARM64 (aarch64) |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USER/dgx-spark-ai.git
cd dgx-spark-ai

# 2. Create Python environment (if you don't have one)
make setup-venv

# 3. Start vLLM with GPT-OSS 120B
make serve

# 4. Test the API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello!"}]}'

# 5. Connect Cline CLI
make setup-cline
cline "Say hello"
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    NVIDIA DGX Spark                          │
│                                                              │
│  ┌─────────────┐     ┌──────────────────────────────────┐   │
│  │  Cline CLI  │────▶│  vLLM Server (:8000)             │   │
│  │  (coding)   │     │  ├─ OpenAI-compatible API         │   │
│  └─────────────┘     │  ├─ GPT-OSS 120B (mxfp4, ~65GB)  │   │
│                      │  └─ or any HuggingFace model      │   │
│  ┌─────────────┐     └──────────────────────────────────┘   │
│  │  Your App   │────▶  /v1/chat/completions                 │
│  │  (any SDK)  │       /v1/models                           │
│  └─────────────┘                                             │
│                      ┌──────────────────────────────────┐   │
│  ┌─────────────┐     │  Training Pipeline               │   │
│  │  Dataset    │────▶│  ├─ QLoRA (Unsloth / PEFT)       │   │
│  │  (JSONL)    │     │  ├─ Merge LoRA → Standalone       │   │
│  └─────────────┘     │  └─ Export GGUF → Ollama          │   │
│                      └──────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Systemd Services                                     │   │
│  │  ├─ vllm-server.service    (auto-start on boot)      │   │
│  │  ├─ vllm-watchdog.timer    (health check / 2 min)    │   │
│  │  └─ vllm-watchdog.service  (restart if unhealthy)    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Modules

### [`inference/`](inference/) — Serve Models with vLLM

```bash
make serve              # Start GPT-OSS 120B
make serve-force        # Force-restart (kills existing)
make stop               # Graceful shutdown
make health             # Full pipeline health check
make check              # Quick API check
```

Scripts for starting, stopping, health-checking, and switching between models. Supports any HuggingFace model — not just GPT-OSS.

### [`training/`](training/) — Fine-tune with QLoRA

```bash
make train DATASET=data.jsonl                    # Unsloth (2x faster)
make train-peft DATASET=data.jsonl               # Standard PEFT/TRL
make merge-lora ADAPTER=output/final OUTPUT=merged  # Merge LoRA
make export-gguf OUTPUT_DIR=output               # Export to GGUF
```

Two fine-tuning pipelines, LoRA merging, and GGUF export for Ollama. Pre-built YAML configs for different model sizes.

### [`cline/`](cline/) — Cline CLI Integration

```bash
make setup-cline        # Auto-configure Cline for local vLLM
```

One command to point Cline at your local vLLM server. Supports auto-detection of model name.

### [`systemd/`](systemd/) — Production Services

```bash
make install-services   # Install + enable auto-start
make status             # Check service status
make uninstall-services # Remove services
```

Systemd user services for auto-start, auto-restart, and a health watchdog that checks vLLM every 2 minutes.

### [`examples/`](examples/) — Usage Examples

- [`chat_completion.py`](examples/chat_completion.py) — Basic OpenAI SDK usage
- [`streaming_example.py`](examples/streaming_example.py) — Real-time streaming
- [`batch_inference.py`](examples/batch_inference.py) — Concurrent batch processing

### [`docs/`](docs/) — Documentation

- [**Lesson: vLLM + GPT-OSS on DGX Spark**](docs/LESSON_VLLM_DGX_SPARK.md) — Complete tutorial
- [**Training Guide**](docs/TRAINING_GUIDE.md) — End-to-end fine-tuning walkthrough
- [**Architecture**](docs/ARCHITECTURE.md) — System design deep-dive
- [**Troubleshooting**](docs/TROUBLESHOOTING.md) — Common issues and fixes

## Make Targets

```
$ make help

╔══════════════════════════════════════════════════════════════╗
║  dgx-spark-ai — vLLM + Training on NVIDIA DGX Spark        ║
╚══════════════════════════════════════════════════════════════╝

  serve                Start vLLM server with GPT-OSS 120B
  serve-force          Force-restart vLLM server (kills existing)
  stop                 Stop all vLLM processes
  stop-force           Force-kill all vLLM processes
  health               Run full pipeline health check
  check                Quick check if vLLM is responding
  test-api             Run API smoke tests
  train                Fine-tune with QLoRA via Unsloth (DATASET=path required)
  train-peft           Fine-tune with standard PEFT/TRL (DATASET=path required)
  merge-lora           Merge LoRA adapter into base model
  export-gguf          Export model to GGUF format
  setup-cline          Configure Cline CLI to use local vLLM
  install-services     Install systemd user services
  uninstall-services   Remove systemd user services
  status               Show status of all services
  setup-venv           Create Python venv and install vLLM
  setup-training-deps  Install training dependencies
```

## Model Support

| Model | Parameters | Memory | Context | Notes |
|-------|-----------|--------|---------|-------|
| **openai/gpt-oss-120b** | 120B (mxfp4) | ~65 GB | 32K | Default — best quality |
| Qwen/Qwen2.5-7B-Instruct | 7B | ~15 GB | 32K | Fast, good for testing |
| Qwen/Qwen2.5-Coder-32B | 32B | ~24 GB | 32K | Great for code |
| meta-llama/Llama-3.1-8B | 8B | ~17 GB | 8K | General purpose |

Switch models with: `bash inference/switch_model.sh MODEL_NAME`

## Prerequisites

- NVIDIA DGX Spark (or similar Grace Blackwell system)
- Ubuntu 24.04+ with CUDA 13.0
- Python 3.12+
- [Cline CLI](https://github.com/cline/cline) (for AI coding integration)
- 60+ GB free disk space (for model weights)

## License

[MIT](LICENSE)
