# Architecture

## System Overview

`dgx-spark-ai` runs entirely on the NVIDIA DGX Spark — a personal AI supercomputer with an NVIDIA GB10 Grace Blackwell GPU and 128 GB of unified CPU+GPU memory.

### Why DGX Spark?

The GB10's **128 GB unified memory** is the key enabler. Unlike discrete GPUs where VRAM is separate (and usually 24-80 GB), the DGX Spark shares memory between CPU and GPU. This means:

- **GPT-OSS 120B** (mxfp4 quantized, ~65 GB) fits with room to spare
- **QLoRA fine-tuning** of 70B models is possible (quantized weights + LoRA adapters)
- No need for multi-GPU setups, model parallelism, or offloading

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     NVIDIA DGX Spark                            │
│                     Ubuntu 24.04 / CUDA 13.0                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  vLLM Server (port 8000)                                │   │
│  │                                                         │   │
│  │  ┌───────────────┐  ┌────────────────────────────────┐ │   │
│  │  │ OpenAI API    │  │  Model Engine                   │ │   │
│  │  │               │  │                                 │ │   │
│  │  │ /v1/models    │  │  ┌───────────────────────────┐ │ │   │
│  │  │ /v1/chat/     │──│  │ GPT-OSS 120B (mxfp4)     │ │ │   │
│  │  │   completions │  │  │ ~65GB unified memory      │ │ │   │
│  │  │ /v1/          │  │  │ Blackwell sm_121a kernels │ │ │   │
│  │  │   completions │  │  └───────────────────────────┘ │ │   │
│  │  └───────┬───────┘  └────────────────────────────────┘ │   │
│  │          │                                              │   │
│  │  ┌───────┴───────┐                                     │   │
│  │  │ PagedAttention │  ← Efficient KV-cache management   │   │
│  │  │ Continuous     │  ← Dynamic batching of requests     │   │
│  │  │ Batching       │                                     │   │
│  │  └───────────────┘                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│          ▲         ▲          ▲                                 │
│          │         │          │                                 │
│   ┌──────┴──┐ ┌────┴───┐ ┌───┴─────┐                         │
│   │ Cline   │ │ Your   │ │ curl/   │                         │
│   │ CLI     │ │ App    │ │ httpie  │                         │
│   └─────────┘ └────────┘ └─────────┘                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Systemd Services                                       │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐│   │
│  │  │ vllm-server      │  │ vllm-watchdog.timer (2min)   ││   │
│  │  │ (auto-restart)   │  │ → vllm-watchdog.service      ││   │
│  │  │ RestartSec=15    │  │ → curl health check          ││   │
│  │  └──────────────────┘  └──────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Pipeline

### vLLM Configuration

The vLLM server is configured specifically for the DGX Spark:

| Setting | Value | Reason |
|---------|-------|--------|
| `--enforce-eager` | Enabled | Disables CUDA graphs to save memory for large models |
| `--max-model-len 32768` | 32K tokens | Balances context length with available memory |
| `--trust-remote-code` | Enabled | Required for GPT-OSS model architecture |
| `--chat-template` | Harmony | GPT-OSS uses the Harmony chat template format |

### CUDA / Library Setup

The DGX Spark uses CUDA 13.0 with the Blackwell `sm_121a` architecture. The startup scripts configure:

1. **LD_LIBRARY_PATH** — Points to NVIDIA libraries in the venv (`nvidia-*` packages) and PyTorch
2. **TRITON_PTXAS_PATH** — Uses system `/usr/local/cuda/bin/ptxas` for Blackwell kernel compilation
3. **TIKTOKEN paths** — For the Harmony tokenizer used by GPT-OSS

### Memory Layout (GPT-OSS 120B)

```
128 GB Unified Memory
├── Model weights (mxfp4):     ~65 GB
├── KV cache (32K context):    ~20 GB
├── CUDA workspace:            ~5 GB
├── Python/OS overhead:        ~8 GB
└── Available:                 ~30 GB
```

## Training Pipeline

```
                    ┌─────────────────┐
                    │  Training Data  │
                    │  (ChatML JSONL) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  QLoRA Training │
                    │                 │
                    │  Base model     │──▶ 4-bit quantized (NF4)
                    │  + LoRA adapters│──▶ rank 16-64, ~0.1% params
                    │                 │
                    │  Unsloth (fast) │
                    │  or PEFT (std)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LoRA Adapter   │  Small (~100MB-1GB)
                    │  (output/final) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ Serve with │  │ Merge into │  │ Export to  │
     │ vLLM       │  │ standalone │  │ GGUF for   │
     │ (+ adapter)│  │ model      │  │ Ollama     │
     └────────────┘  └────────────┘  └────────────┘
```

### QLoRA on DGX Spark

QLoRA (Quantized Low-Rank Adaptation) enables fine-tuning models that would otherwise not fit in memory:

1. **Base model** is loaded in 4-bit (NF4) quantization — 32B model uses ~24GB
2. **LoRA adapters** add trainable parameters on top — typically 0.05-0.2% of total
3. **Gradient checkpointing** trades compute for memory — enables larger batch sizes
4. **Unsloth optimization** provides custom kernels that are 2x faster on Blackwell

## Service Architecture

### Startup Flow

```
systemd timer → check health → restart if unhealthy

Boot
 ├── vllm-server.service starts
 │   ├── start_vllm_gptoss.sh
 │   │   ├── Check for existing processes
 │   │   ├── Activate venv
 │   │   ├── Set CUDA/library paths
 │   │   └── exec vllm serve ...
 │   └── Restart on failure (15s delay, max 5/10min)
 │
 └── vllm-watchdog.timer starts (after 5 min)
     └── Every 2 minutes:
         ├── curl localhost:8000/v1/models
         ├── If healthy: log and exit
         ├── If <5min uptime: skip (model still loading)
         └── If unhealthy: systemctl restart vllm-server
```

### Conflict Prevention

The startup script (`start_vllm_gptoss.sh`) handles the #1 cause of CUDA OOM errors — multiple vLLM instances:

1. **Check existing server** — If vLLM is already healthy, just report success
2. **Find zombie processes** — Detect stale `vllm serve` and `VLLM::EngineCore` processes
3. **Check port conflicts** — Verify port 8000 isn't taken by another service
4. **Force mode** — `--force` flag kills everything and starts fresh
