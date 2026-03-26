# Troubleshooting

Common issues and their solutions when running vLLM and training on the DGX Spark.

---

## Quick Diagnostics

```bash
# Run the full health check first
make health

# Check GPU status
nvidia-smi

# Check for running vLLM processes
pgrep -fa "vllm serve"

# Check what's on port 8000
ss -tlnp | grep 8000

# Check systemd service logs
journalctl --user -u vllm-server --no-pager -n 50
```

---

## Inference Issues

### CUDA Out of Memory (OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB
```

**Causes & Fixes:**

1. **Multiple vLLM instances running** (most common)
   ```bash
   # Kill all vLLM processes
   make stop-force
   # Then restart clean
   make serve
   ```

2. **Context length too large**
   ```bash
   # Reduce max context
   VLLM_MAX_MODEL_LEN=16384 make serve
   ```

3. **Another Python process using GPU**
   ```bash
   # Check what's using GPU memory
   nvidia-smi
   # Kill the offending process
   kill -9 <PID>
   ```

### vLLM Won't Start

**"Virtual environment not found"**
```bash
make setup-venv
```

**"Port 8000 is already in use"**
```bash
# Find what's using the port
ss -tlnp | grep 8000
# Kill it, or use a different port
VLLM_PORT=8001 make serve
```

**"chat_template.jinja not found"**
```bash
# Re-download the model
huggingface-cli download openai/gpt-oss-120b
```

### API Returns Errors

**502/503 errors right after start**
- The model is still loading. GPT-OSS 120B takes 3-5 minutes to load.
- Wait and retry, or check: `make check`

**Empty responses**
- Check the chat template is loaded correctly
- Try a direct curl test:
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
  ```

### Slow Inference

- **First request is slow** — This is normal. vLLM compiles CUDA kernels on the first request. Subsequent requests are much faster.
- **All requests slow** — Check GPU utilization with `nvidia-smi`. If GPU usage is low, the model may not be loaded properly.
- **Enable continuous batching** — vLLM does this by default. Send multiple concurrent requests for best throughput.

---

## Training Issues

### CUDA OOM During Training

```bash
# Reduce batch size
python3 training/fine_tune.py --dataset data.jsonl --batch-size 1

# Reduce sequence length
python3 training/fine_tune.py --dataset data.jsonl --max-seq-length 2048

# Use smaller LoRA rank
python3 training/fine_tune.py --dataset data.jsonl --lora-rank 16 --lora-alpha 32
```

### Unsloth Import Error

```
ModuleNotFoundError: No module named 'unsloth'
```

```bash
# Install training dependencies
make setup-training-deps
```

If Unsloth won't install, use the PEFT pipeline instead:
```bash
make train-peft DATASET=data.jsonl
```

### Training Loss is NaN

- Reduce learning rate: `--learning-rate 1e-4`
- Check dataset for corrupted entries
- Ensure dataset is valid JSONL (one JSON object per line)

---

## Cline Integration Issues

### "vLLM server is not responding"

```bash
# Start vLLM first
make serve
# Wait for model to load (3-5 min for GPT-OSS 120B)
make check
# Then configure Cline
make setup-cline
```

### Cline Ignores Local Config

If another Cline session is running, it may overwrite your config when it saves state.

```bash
# Close all Cline sessions, then:
make setup-cline
```

### Cline Works But Responses Are Bad

- GPT-OSS 120B with `temperature=0` can be overly deterministic
- Try adjusting in Cline settings or adding context to your prompts
- For coding tasks, GPT-OSS 120B generally performs well

---

## Systemd Service Issues

### Service Won't Start

```bash
# Check what's wrong
systemctl --user status vllm-server
journalctl --user -u vllm-server --no-pager -n 20

# Common fix: reinstall services
make uninstall-services
make install-services
```

### Service Keeps Restarting

```bash
# Check logs for the error
journalctl --user -u vllm-server -f

# Common causes:
# 1. OOM → reduce model size or context length
# 2. Missing dependencies → make setup-venv
# 3. Port conflict → check with ss -tlnp | grep 8000
```

### Watchdog Keeps Restarting a Healthy Server

The watchdog waits 5 minutes before first health check to allow model loading. If your model takes longer:

```bash
# Edit the watchdog script
# Change UPTIME_SECS threshold from 300 to 600 (10 minutes)
vim systemd/vllm-watchdog.sh
# Reinstall
make install-services
```

---

## General

### Check Available Memory

```bash
# System memory
free -h

# GPU memory
nvidia-smi

# Disk space (for model weights)
df -h ~/.cache/huggingface/
```

### Reset Everything

```bash
# Stop all services and processes
make uninstall-services
make stop-force

# Clear vLLM cache
rm -rf ~/.cache/vllm/

# Reinstall
make setup-venv
make install-services
make serve
```
