# Training Module

Fine-tune open-source LLMs on the NVIDIA DGX Spark using QLoRA (Quantized Low-Rank Adaptation). Two pipelines are available:

| Script | Engine | Speed | Notes |
|--------|--------|-------|-------|
| `fine_tune.py` | Unsloth | ~2x faster | Recommended — optimized kernels for GB10 |
| `fine_tune_peft.py` | PEFT/TRL | Standard | Fallback — no Unsloth dependency |

## Quick Start

```bash
# Install training dependencies
make setup-training-deps

# Fine-tune with Unsloth (recommended)
make train DATASET=path/to/data.jsonl

# Or with standard PEFT/TRL
make train-peft DATASET=path/to/data.jsonl

# Merge LoRA adapter into standalone model
make merge-lora ADAPTER=output/final OUTPUT=output/merged

# Export to GGUF for Ollama
make export-gguf OUTPUT_DIR=output
```

## Dataset Format

Training data should be in **ChatML JSONL** format — one JSON object per line:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a Python hello world"}, {"role": "assistant", "content": "print('Hello, world!')"}]}
{"messages": [{"role": "user", "content": "What is recursion?"}, {"role": "assistant", "content": "Recursion is when a function calls itself..."}]}
```

## Training Configs

Pre-built configurations in `configs/`:

| Config | Model Size | LoRA Rank | Epochs | Use Case |
|--------|-----------|-----------|--------|----------|
| `qlora_default.yaml` | 32B | 64 | 3 | Production fine-tuning |
| `qlora_aggressive.yaml` | 7B | 16 | 1 | Quick prototyping |

## Pipeline Overview

```
Dataset (JSONL)
    ↓
[fine_tune.py]     QLoRA training → LoRA adapter
    ↓
[merge_lora.py]    Merge adapter + base model → Standalone model
    ↓
[export_gguf.sh]   Convert to GGUF → Ollama-ready
    ↓
[serve_model.py]   Quick OpenAI-compatible API server
  or
[vLLM]             Production serving via inference module
```

## Memory Usage on DGX Spark (128 GB)

| Model | Quantization | Training Memory | Notes |
|-------|-------------|-----------------|-------|
| Qwen2.5-7B | 4-bit NF4 | ~12 GB | Fast iteration |
| Qwen2.5-32B | 4-bit NF4 | ~24 GB | Higher quality |
| Llama-3.1-70B | 4-bit NF4 | ~45 GB | Fits in unified memory |

The DGX Spark's 128 GB unified memory means even 70B parameter models can be fine-tuned with QLoRA.
