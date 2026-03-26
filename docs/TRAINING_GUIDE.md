# Training Guide: Fine-tuning LLMs on DGX Spark

> **Goal:** Take a base model (e.g., Qwen2.5-Coder-32B), fine-tune it on your own data using QLoRA, and deploy it for local inference — all on the DGX Spark.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Prepare Your Dataset](#step-1-prepare-your-dataset)
4. [Step 2: Choose Your Training Pipeline](#step-2-choose-your-training-pipeline)
5. [Step 3: Run Training](#step-3-run-training)
6. [Step 4: Evaluate the Model](#step-4-evaluate-the-model)
7. [Step 5: Merge and Deploy](#step-5-merge-and-deploy)
8. [Step 6: Export to GGUF (Optional)](#step-6-export-to-gguf-optional)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Common Issues](#common-issues)

---

## Overview

We use **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune large models efficiently:

- Base model loaded in **4-bit quantization** (saves ~75% memory)
- Only **0.05-0.2%** of parameters are trainable (LoRA adapters)
- **2x faster** with Unsloth's optimized kernels for Blackwell GPU
- **128 GB unified memory** means we can fine-tune models up to 70B parameters

## Prerequisites

```bash
# Install training dependencies
make setup-training-deps

# Or manually
pip install -r training/requirements.txt
```

Verify GPU access:
```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GB10 (or similar)
```

## Step 1: Prepare Your Dataset

Training data uses the **ChatML JSONL** format — one conversation per line:

```jsonl
{"messages": [{"role": "system", "content": "You are an expert Python developer."}, {"role": "user", "content": "How do I read a CSV file?"}, {"role": "assistant", "content": "Use pandas:\n\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```"}]}
{"messages": [{"role": "user", "content": "Write a function to sort a dictionary by values"}, {"role": "assistant", "content": "```python\ndef sort_dict_by_values(d: dict, reverse: bool = False) -> dict:\n    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))\n```"}]}
```

### Guidelines

- **Minimum**: 50-100 examples for noticeable effect
- **Sweet spot**: 500-5,000 examples for good quality
- **Format**: Always include `messages` array with `role` and `content`
- **System prompt**: Optional but recommended for consistent behavior
- **Quality > Quantity**: 100 high-quality examples beat 10,000 noisy ones

### Creating Training Data

You can generate training data from various sources:

```bash
# From existing code repos
find . -name "*.py" -exec sh -c 'echo "{\"messages\": [{\"role\": \"user\", \"content\": \"Explain this code\"}, {\"role\": \"assistant\", \"content\": $(cat {} | jq -Rs .)}]}"' \;

# From a frontier model (knowledge distillation)
# See ai-bootlegging approach for automated pipelines
```

## Step 2: Choose Your Training Pipeline

| Pipeline | Script | Speed | Best For |
|----------|--------|-------|----------|
| **Unsloth** (recommended) | `fine_tune.py` | ~2x faster | Most use cases |
| **Standard PEFT** | `fine_tune_peft.py` | Baseline | Compatibility edge cases |

**Use Unsloth** unless you have a specific reason not to. It provides custom Blackwell GPU kernels that dramatically speed up training.

## Step 3: Run Training

### Quick Start (Unsloth)

```bash
# Default settings (32B model, rank 64, 3 epochs)
make train DATASET=my_data.jsonl

# Custom settings
python3 training/fine_tune.py \
    --dataset my_data.jsonl \
    --model unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit \
    --epochs 3 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --max-seq-length 4096 \
    --output-dir ./output
```

### Quick Start (PEFT/TRL)

```bash
# More conservative defaults (32B model, rank 16, 1 epoch)
make train-peft DATASET=my_data.jsonl

# Custom settings
python3 training/fine_tune_peft.py \
    --dataset my_data.jsonl \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --epochs 1 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --batch-size 1 \
    --max-seq-length 2048
```

### What Happens During Training

1. **Model loading** (~2-5 min): Downloads and loads the 4-bit quantized model
2. **LoRA injection**: Adds small trainable adapters to attention + MLP layers
3. **Training**: Iterates over your dataset, updating only LoRA weights
4. **Checkpointing**: Saves progress every N steps
5. **Final save**: Saves the LoRA adapter to `output/final/`

Monitor training:
```bash
# Watch the training output for:
# - Training loss (should decrease)
# - Learning rate (follows cosine schedule)
# - Samples/second (throughput)
```

## Step 4: Evaluate the Model

Quick interactive test:

```bash
# Serve the fine-tuned model
python3 training/serve_model.py output/final

# In another terminal, test it
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "fine-tuned-model", "messages": [{"role": "user", "content": "Your test prompt here"}]}'
```

## Step 5: Merge and Deploy

After fine-tuning, merge the LoRA adapter into the base model for easier deployment:

```bash
# Merge LoRA + base model → standalone model
make merge-lora ADAPTER=output/final OUTPUT=output/merged

# Serve with vLLM (production)
bash inference/start_vllm.sh output/merged

# Or serve with the lightweight server (testing)
python3 training/serve_model.py output/merged
```

## Step 6: Export to GGUF (Optional)

Export to GGUF format for use with Ollama or llama.cpp:

```bash
# Export with Q4_K_M quantization (good balance of size/quality)
make export-gguf OUTPUT_DIR=output

# Or with different quantization
make export-gguf OUTPUT_DIR=output QUANT=Q8_0

# Load in Ollama
ollama create my-model -f Modelfile
```

## Hyperparameter Tuning

### LoRA Parameters

| Parameter | Small datasets (<500) | Large datasets (>2000) |
|-----------|----------------------|----------------------|
| `lora_rank` | 16 | 64 |
| `lora_alpha` | 32 | 128 |
| Rule of thumb | `alpha = 2 × rank` | `alpha = 2 × rank` |

Higher rank = more trainable parameters = more capacity but slower training.

### Training Parameters

| Parameter | Conservative | Aggressive |
|-----------|-------------|------------|
| `epochs` | 1 | 3-5 |
| `learning_rate` | 1e-4 | 5e-4 |
| `batch_size` | 1 | 2-4 |
| `gradient_accumulation` | 8 | 2-4 |
| `warmup_ratio` | 0.1 | 0.03 |

**Effective batch size** = `batch_size × gradient_accumulation` (aim for 8-16).

### Memory vs Speed Tradeoffs

| Setting | More Memory | Less Memory |
|---------|------------|-------------|
| `max_seq_length` | 4096 | 2048 |
| `batch_size` | 4 | 1 |
| `gradient_checkpointing` | Off | On (Unsloth) |
| `packing` | On | Off |

## Common Issues

### CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix**: Reduce `batch_size`, `max_seq_length`, or `lora_rank`. Enable gradient checkpointing.

### Training Loss Not Decreasing
**Fix**: Increase `learning_rate` (try 5e-4), add more epochs, or check dataset quality.

### Training Loss Oscillating Wildly
**Fix**: Decrease `learning_rate` (try 1e-4), increase `warmup_ratio`, increase `gradient_accumulation`.

### Slow Training
**Fix**: Use Unsloth (`fine_tune.py`) instead of PEFT. Enable `packing=True`. Increase `batch_size`.

### Model Generates Repetitive/Degraded Output
**Fix**: You may be overtraining. Reduce epochs or add a validation set. Check that your dataset doesn't have repeated examples.
