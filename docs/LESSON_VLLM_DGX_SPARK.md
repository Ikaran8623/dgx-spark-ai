# Lesson: Running vLLM + GPT-OSS on NVIDIA DGX Spark with Cline CLI

> **Goal:** Serve OpenAI's GPT-OSS 120B model locally on an NVIDIA DGX Spark using vLLM, then connect Cline CLI to use it as your AI coding assistant — all running on-device with zero cloud dependency.

---

## Table of Contents

1. [Prerequisites & Hardware Overview](#1-prerequisites--hardware-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start (TL;DR)](#3-quick-start-tldr)
4. [Step 1: Understanding the DGX Spark](#step-1-understanding-the-dgx-spark)
5. [Step 2: Setting Up the vLLM Environment](#step-2-setting-up-the-vllm-environment)
6. [Step 3: Starting the vLLM Server](#step-3-starting-the-vllm-server)
7. [Step 4: Testing the API](#step-4-testing-the-api)
8. [Step 5: Connecting Cline CLI](#step-5-connecting-cline-cli)
9. [Step 6: Using Cline with Local Inference](#step-6-using-cline-with-local-inference)
10. [Step 7: Health Checks & Monitoring](#step-7-health-checks--monitoring)
11. [Stopping & Restarting](#stopping--restarting)
12. [Troubleshooting](#troubleshooting)
13. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)
14. [How It Works (Deep Dive)](#how-it-works-deep-dive)

---

## 1. Prerequisites & Hardware Overview

### Hardware: NVIDIA DGX Spark

| Spec | Value |
|------|-------|
| **GPU** | NVIDIA GB10 (Grace-Blackwell architecture) |
| **CUDA Capability** | 12.1 (Blackwell, sm_121a) |
| **Memory** | ~128 GB unified (CPU + GPU shared) |
| **CPU** | ARM64 (Grace, aarch64) |
| **CUDA Version** | 13.0 |

The DGX Spark uses a **unified memory architecture** — the CPU and GPU share the same physical memory pool. This is what allows us to run a 65 GB model (mxfp4 quantized) that would normally require a much larger dedicated GPU.

### Software Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| Ubuntu | 24.04 LTS | Operating system |
| Python | 3.12 | Runtime for vLLM |
| vLLM | 0.18.0 | Model serving engine |
| PyTorch | 2.10.0+cu130 | ML framework |
| CUDA | 13.0 | GPU compute |
| Node.js | 22+ | Cline CLI runtime |
| Cline CLI | 2.11.0 | AI coding assistant |

### The Model: GPT-OSS 120B

- **Full name:** `openai/gpt-oss-120b`
- **Parameters:** 120 billion
- **Quantization:** mxfp4 (microscaling FP4) — reduces ~240 GB FP16 model to ~65 GB
- **Disk size:** ~65 GB cached in `~/.cache/huggingface/hub/`
- **Tokenizer:** Harmony (custom OpenAI tokenizer)
- **Capabilities:** Chat, reasoning, code generation, tool use

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   DGX Spark                         │
│                                                     │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │  Cline CLI   │  HTTP   │     vLLM Server      │  │
│  │  (terminal)  │ ──────► │   localhost:8000      │  │
│  │              │         │                      │  │
│  │  Provider:   │         │  /v1/models          │  │
│  │  "openai"    │         │  /v1/chat/completions│  │
│  │  (compatible)│         │  /v1/completions     │  │
│  └──────────────┘         └──────────┬───────────┘  │
│                                      │              │
│                              ┌───────▼───────┐      │
│                              │   GPT-OSS     │      │
│                              │   120B model  │      │
│                              │   (mxfp4)     │      │
│                              │   ~65 GB      │      │
│                              └───────┬───────┘      │
│                                      │              │
│                              ┌───────▼───────┐      │
│                              │  NVIDIA GB10  │      │
│                              │  Unified Mem  │      │
│                              │  ~128 GB      │      │
│                              └───────────────┘      │
└─────────────────────────────────────────────────────┘
```

**Data flow:**
1. You type in Cline CLI (or run `cline "your prompt"`)
2. Cline sends an OpenAI-compatible HTTP request to `http://localhost:8000/v1/chat/completions`
3. vLLM tokenizes the input with the Harmony tokenizer
4. The GPU runs inference on GPT-OSS 120B
5. vLLM streams tokens back to Cline
6. Cline displays the response and can execute code, edit files, etc.

**Key insight:** vLLM exposes an **OpenAI-compatible API**, which means any tool that works with the OpenAI API can work with your local model — including Cline, which has a built-in "OpenAI Compatible" provider.

---

## 3. Quick Start (TL;DR)

If everything is already installed and you just want to get running:

```bash
# 1. Start the vLLM server (kills any existing instance)
bash ~/Desktop/Workspace/start_vllm_gptoss.sh --force

# 2. Wait ~60 seconds for the model to load, then in another terminal:

# 3. Configure Cline to use local vLLM
bash ~/Desktop/Workspace/setup_cline_local.sh

# 4. Verify everything works
bash ~/Desktop/Workspace/health_check.sh

# 5. Use Cline!
cline "Hello! What GPU am I running on?"
```

---

## Step 1: Understanding the DGX Spark

### Check Your Hardware

Open a terminal and verify your GPU:

```bash
nvidia-smi
```

You should see something like:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0  On |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Check Available Memory

```bash
free -h
```

You need at least ~70 GB free for the model weights plus vLLM overhead. The DGX Spark's 128 GB unified memory is sufficient, but you should close unnecessary applications.

### Understanding Unified Memory

Unlike traditional setups where the GPU has its own dedicated VRAM, the DGX Spark's GB10 shares memory with the CPU. This means:

- ✅ You can load models larger than typical GPU VRAM
- ✅ No CPU↔GPU memory transfer bottleneck  
- ⚠️ Everything competes for the same memory pool
- ⚠️ Running two models simultaneously will OOM (this is the #1 gotcha!)

---

## Step 2: Setting Up the vLLM Environment

> **Note:** If you're on a pre-configured DGX Spark, this is likely already done. Skip to [Step 3](#step-3-starting-the-vllm-server).

### Create a Python Virtual Environment

```bash
cd ~/Desktop/Workspace
python3 -m venv vllm-env
source vllm-env/bin/activate
```

### Install vLLM

```bash
pip install vllm
```

This installs vLLM along with PyTorch, CUDA bindings, and all dependencies. On the DGX Spark (aarch64), this uses the ARM64-optimized build.

### Download the Model

```bash
# This downloads ~65 GB of model weights
huggingface-cli download openai/gpt-oss-120b
```

The model will be cached at `~/.cache/huggingface/hub/models--openai--gpt-oss-120b/`.

### Verify Installation

```bash
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
# Should print: vLLM 0.18.0
```

---

## Step 3: Starting the vLLM Server

### Using the Startup Script

The easiest way to start the server:

```bash
bash ~/Desktop/Workspace/start_vllm_gptoss.sh
```

This script handles:
- ✅ Activating the Python virtual environment
- ✅ Setting up NVIDIA library paths for CUDA 13.0
- ✅ Configuring the Harmony tokenizer paths
- ✅ Detecting and warning about process conflicts
- ✅ Locating the chat template for proper message formatting
- ✅ Launching vLLM with optimized settings for the GB10

### What the Flags Mean

```bash
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \           # Listen on all interfaces
    --port 8000 \               # API port
    --trust-remote-code \       # Required for GPT-OSS custom architecture
    --enforce-eager \           # Disable CUDA graphs (saves memory)
    --max-model-len 32768 \     # 32K context window
    --chat-template chat_template.jinja  # Harmony chat format
```

| Flag | Why |
|------|-----|
| `--enforce-eager` | CUDA graphs consume extra GPU memory. On the GB10 with a 65 GB model, we need every byte. Eager mode is slower but fits. |
| `--max-model-len 32768` | Limits context to 32K tokens. Longer contexts need more KV cache memory. |
| `--trust-remote-code` | GPT-OSS uses a custom `GptOssForCausalLM` architecture not built into vLLM. |
| `--chat-template` | GPT-OSS uses a "Harmony" format with analysis/commentary/final channels. |

### Startup Time

Expect **60-90 seconds** for the model to load. You'll see:
1. Model architecture resolution
2. Safetensor file parsing (15 files)
3. Weight loading into GPU memory
4. Scheduler and engine initialization
5. "INFO: Started server process" — **this means it's ready**

### If It Fails: Common Startup Issues

**CUDA OOM (Out of Memory)**
```
torch.AcceleratorError: CUDA error: out of memory
```
→ Another process is using GPU memory. Fix: `bash stop_vllm.sh --force`

**Port already in use**
```
OSError: [Errno 98] Address already in use
```
→ Another server is on port 8000. Fix: `bash start_vllm_gptoss.sh --force`

---

## Step 4: Testing the API

Once vLLM is running, test it from another terminal:

### List Models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

Expected output:
```json
{
    "object": "list",
    "data": [
        {
            "id": "openai/gpt-oss-120b",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 32768
        }
    ]
}
```

### Send a Chat Message

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hello! What are you?"}],
        "max_tokens": 100
    }'
```

### Understanding GPT-OSS Responses

GPT-OSS uses a **reasoning model** architecture. Responses may include:
- **`reasoning`** field — the model's internal chain-of-thought
- **`content`** field — the final answer

```json
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Hello! I'm GPT-OSS, an open-source language model...",
            "reasoning": "The user is greeting me and asking what I am..."
        }
    }]
}
```

> **Note:** `http://localhost:8000/` returns `{"detail":"Not Found"}` — this is normal! vLLM is a pure API server with no web UI. The endpoints are under `/v1/`.

---

## Step 5: Connecting Cline CLI

### Automatic Setup

The easiest way:

```bash
bash ~/Desktop/Workspace/setup_cline_local.sh
```

This script:
1. Verifies vLLM is running
2. Auto-detects the model name from the API
3. Runs `cline auth` to configure the "OpenAI Compatible" provider
4. Verifies the configuration

### Manual Setup

If you prefer to do it manually:

```bash
cline auth \
    --provider openai \
    --apikey "local-vllm" \
    --modelid "openai/gpt-oss-120b" \
    --baseurl "http://localhost:8000/v1"
```

**What each flag means:**

| Flag | Value | Why |
|------|-------|-----|
| `--provider openai` | "OpenAI Compatible" provider | vLLM speaks the OpenAI API protocol |
| `--apikey "local-vllm"` | Any non-empty string | vLLM doesn't require auth, but Cline needs a value |
| `--modelid` | `openai/gpt-oss-120b` | Must match the model ID from `/v1/models` |
| `--baseurl` | `http://localhost:8000/v1` | Your local vLLM endpoint (note the `/v1` suffix!) |

### Verify the Configuration

Check what Cline is configured to use:

```bash
cline config
```

Look for:
- `actModeApiProvider: openai`
- `actModeOpenAiModelId: openai/gpt-oss-120b`
- `openAiBaseUrl: http://localhost:8000/v1`

---

## Step 6: Using Cline with Local Inference

### One-Shot Commands

```bash
# Ask a question
cline "What is the capital of France?"

# Generate code
cline "Write a Python function to calculate fibonacci numbers"

# Work with files in current directory
cline "Read the README.md and summarize it"
```

### Interactive Mode

```bash
cline
```

This opens the full Cline TUI (terminal UI) with:
- Chat interface
- File editing capabilities
- Command execution
- Plan and Act modes

### Tips for GPT-OSS on DGX Spark

1. **Be patient** — Local inference is slower than cloud APIs. GPT-OSS 120B on GB10 generates roughly 5-15 tokens/second depending on context length.

2. **Keep context short** — Longer conversations consume more KV cache memory. If responses slow down or you get OOM errors, start a new conversation.

3. **Use Plan mode** — For complex tasks, use `cline` in Plan mode first (type `/plan` in the TUI) to think through the approach before executing.

4. **One model at a time** — The DGX Spark can only run one large model. Don't try to start Ollama or another vLLM instance simultaneously.

---

## Step 7: Health Checks & Monitoring

### Run the Health Check

```bash
bash ~/Desktop/Workspace/health_check.sh
```

This checks:
- ✅ GPU detected and functional
- ✅ vLLM process running
- ✅ Port 8000 listening
- ✅ Model responding to API calls
- ✅ Chat completion working
- ✅ Cline configured correctly

### Monitor GPU Usage

```bash
# One-time snapshot
nvidia-smi

# Continuous monitoring (updates every 2 seconds)
watch -n 2 nvidia-smi
```

### Check vLLM Logs

If vLLM was started in the foreground (from the startup script), logs appear in the same terminal. Look for:
- `INFO: Started server process` — Server is ready
- `INFO ... Avg generation throughput` — Periodic performance stats

---

## Stopping & Restarting

### Stop All vLLM Processes

```bash
bash ~/Desktop/Workspace/stop_vllm.sh
```

This gracefully stops all vLLM processes and waits for GPU memory to be freed.

For immediate shutdown:
```bash
bash ~/Desktop/Workspace/stop_vllm.sh --force
```

### Restart with Fresh State

```bash
bash ~/Desktop/Workspace/start_vllm_gptoss.sh --force
```

The `--force` flag stops any existing instances before starting.

### Full Pipeline Restart

```bash
# Stop everything
bash ~/Desktop/Workspace/stop_vllm.sh --force

# Wait for GPU to clear
sleep 5

# Start fresh
bash ~/Desktop/Workspace/start_vllm_gptoss.sh

# (In another terminal, once vLLM is ready)
bash ~/Desktop/Workspace/setup_cline_local.sh
bash ~/Desktop/Workspace/health_check.sh
```

---

## Troubleshooting

### "CUDA error: out of memory"

**Cause:** Another process is using GPU memory. The GB10's unified memory is shared, and you can only run one large model.

**Fix:**
```bash
# See what's using the GPU
nvidia-smi

# Kill all vLLM processes
bash stop_vllm.sh --force

# Check for other CUDA processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Force restart
bash start_vllm_gptoss.sh --force
```

### "Port 8000 already in use"

**Cause:** Another service is listening on port 8000.

**Fix:**
```bash
# See what's on port 8000
ss -tlnp | grep :8000

# Option A: Kill it and use port 8000
bash start_vllm_gptoss.sh --force

# Option B: Use a different port
VLLM_PORT=8001 bash start_vllm_gptoss.sh
# Then: bash setup_cline_local.sh --port 8001
```

### "CUDA capability 12.1 ... Maximum supported is (12.0)"

**Cause:** PyTorch version doesn't fully support Blackwell (sm_121a) yet. This is a **warning**, not an error — it still works.

**Fix:** Ignore the warning. If it causes actual failures, update PyTorch:
```bash
source vllm-env/bin/activate
pip install --upgrade torch
```

### Cline Shows "Not authenticated"

**Cause:** Cline CLI isn't configured with a provider.

**Fix:**
```bash
bash setup_cline_local.sh
# Or manually:
cline auth -p openai -k "local-vllm" -m "openai/gpt-oss-120b" -b "http://localhost:8000/v1"
```

### Slow Generation Speed

**Cause:** Normal for a 120B parameter model on consumer-adjacent hardware.

**Tips:**
- Use `--max-model-len 8192` for shorter context (faster KV cache operations)
- Close other GPU-using applications (Firefox, VS Code with GPU acceleration)
- Use shorter prompts and conversations
- Consider using a smaller model for rapid iteration tasks

### vLLM Crashes After Running for Hours

**Cause:** Memory leak or fragmentation over long sessions.

**Fix:**
```bash
bash start_vllm_gptoss.sh --force  # Clean restart
```

Consider setting up the systemd watchdog for automatic restarts:
```bash
# See existing service files
cat ~/Desktop/Workspace/vllm-server.service
cat ~/Desktop/Workspace/vllm-watchdog.service
```

---

## Quick Reference Cheat Sheet

```
╔══════════════════════════════════════════════════════════════╗
║  vLLM + GPT-OSS + Cline on DGX Spark — Cheat Sheet        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  START SERVER                                                ║
║    bash start_vllm_gptoss.sh           # Normal start       ║
║    bash start_vllm_gptoss.sh --force   # Kill + restart     ║
║    bash start_vllm_gptoss.sh --check   # Check if healthy   ║
║                                                              ║
║  STOP SERVER                                                 ║
║    bash stop_vllm.sh                   # Graceful stop      ║
║    bash stop_vllm.sh --force           # Immediate kill     ║
║                                                              ║
║  CONFIGURE CLINE                                             ║
║    bash setup_cline_local.sh           # Auto-configure     ║
║    cline auth -p openai -k local-vllm \                     ║
║      -m openai/gpt-oss-120b \                               ║
║      -b http://localhost:8000/v1       # Manual configure   ║
║                                                              ║
║  HEALTH CHECK                                                ║
║    bash health_check.sh                # Full pipeline test ║
║                                                              ║
║  USE CLINE                                                   ║
║    cline "your prompt"                 # One-shot           ║
║    cline                               # Interactive TUI    ║
║                                                              ║
║  TEST API DIRECTLY                                           ║
║    curl localhost:8000/v1/models                             ║
║    curl localhost:8000/v1/chat/completions \                 ║
║      -H 'Content-Type: application/json' \                  ║
║      -d '{"model":"openai/gpt-oss-120b", ...}'              ║
║                                                              ║
║  MONITOR                                                     ║
║    nvidia-smi                          # GPU status         ║
║    watch -n 2 nvidia-smi              # Live GPU monitor    ║
║    ps aux | grep vllm                  # vLLM processes     ║
║                                                              ║
║  KEY PATHS                                                   ║
║    ~/Desktop/Workspace/vllm-env/       # Python venv        ║
║    ~/.cache/huggingface/hub/           # Model weights      ║
║    ~/.cline/data/globalState.json      # Cline config       ║
║                                                              ║
║  PORTS                                                       ║
║    8000 — vLLM OpenAI-compatible API                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## How It Works (Deep Dive)

### Why vLLM?

vLLM (Virtual Large Language Model) is the leading open-source inference engine for LLMs. Key features:

- **PagedAttention** — Manages GPU memory for KV cache like an OS manages virtual memory, reducing waste by up to 50%
- **Continuous Batching** — Processes multiple requests simultaneously, maximizing GPU utilization
- **OpenAI-compatible API** — Drop-in replacement for OpenAI's API, works with any OpenAI client
- **Chunked Prefill** — Breaks long prompts into chunks to maintain responsiveness
- **mxfp4 Quantization Support** — Runs GPT-OSS 120B in microscaling FP4 (4-bit) format

### Why GPT-OSS?

GPT-OSS (Open Source Software) is OpenAI's first fully open-weight model release:

- **120B parameters** — Larger than Llama 3.1 70B, competitive with much larger models
- **Open weights** — No API key needed, runs entirely on your hardware
- **mxfp4 quantized** — Ships in 4-bit format, designed for efficient inference
- **Reasoning capabilities** — Built-in chain-of-thought with analysis/commentary/final channels
- **Code generation** — Strong coding abilities, suitable for powering Cline

### Why Cline CLI?

Cline is an AI coding assistant that:

- Runs in the terminal (no VS Code required)
- Can edit files, run commands, browse the web
- Supports 44+ LLM providers including "OpenAI Compatible" (our vLLM endpoint)
- Has Plan/Act modes for structured problem-solving
- Stores configuration in `~/.cline/data/`

### The Full Request Lifecycle

```
1. User types: cline "Write a hello world script"

2. Cline CLI reads ~/.cline/data/globalState.json
   → actModeApiProvider: "openai"
   → openAiBaseUrl: "http://localhost:8000/v1"
   → actModeOpenAiModelId: "openai/gpt-oss-120b"

3. Cline constructs an OpenAI-format request:
   POST http://localhost:8000/v1/chat/completions
   {
     "model": "openai/gpt-oss-120b",
     "messages": [
       {"role": "system", "content": "You are Cline, a coding assistant..."},
       {"role": "user", "content": "Write a hello world script"}
     ],
     "stream": true
   }

4. vLLM receives the request:
   a. Tokenizes with Harmony tokenizer
   b. Applies chat template (analysis/commentary/final channels)
   c. Schedules for GPU inference
   d. Runs through GPT-OSS 120B forward pass
   e. Samples output tokens

5. vLLM streams SSE (Server-Sent Events) back:
   data: {"choices":[{"delta":{"content":"Here"}}]}
   data: {"choices":[{"delta":{"content":"'s"}}]}
   data: {"choices":[{"delta":{"content":" a"}}]}
   ...

6. Cline displays the streaming response and executes any actions
   (file creation, command execution, etc.)
```

---

## Next Steps

Once you have the basic pipeline working, explore:

- **Custom system prompts** — Edit Cline's rules via `cline config` → Rules tab
- **MCP servers** — Add tools to Cline (web search, database access, etc.)
- **Different models** — Try other models that fit in ~65 GB (adjust the startup script)
- **Systemd services** — Use the included `.service` files for auto-start on boot
- **Inference Hub** — Route requests through the included `inference-hub` for multi-user billing

---

*This lesson was created for the NVIDIA DGX Spark (GB10) running Ubuntu 24.04, vLLM 0.18.0, and Cline CLI 2.11.0. Last updated: March 2026.*
