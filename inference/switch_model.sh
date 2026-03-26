#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Switch vLLM to a different model                            ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash switch_model.sh openai/gpt-oss-120b
#   bash switch_model.sh Qwen/Qwen2.5-7B-Instruct
#   bash switch_model.sh meta-llama/Llama-3.1-8B-Instruct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: bash switch_model.sh MODEL_NAME"
    echo ""
    echo "Stop the current vLLM server and restart with a different model."
    echo ""
    echo "Examples:"
    echo "  bash switch_model.sh openai/gpt-oss-120b"
    echo "  bash switch_model.sh Qwen/Qwen2.5-7B-Instruct"
    exit 0
fi

MODEL="$1"
shift
EXTRA_ARGS=("$@")

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}=== Switching vLLM to ${MODEL} ===${NC}"
echo ""

# 1. Check if model is downloaded (for HuggingFace models)
MODEL_CACHE_NAME=$(echo "$MODEL" | tr '/' '--')
MODEL_DIR=$(find ~/.cache/huggingface/hub/models--${MODEL_CACHE_NAME}/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -n "$MODEL_DIR" ]; then
    MODEL_SIZE=$(du -sh ~/.cache/huggingface/hub/models--${MODEL_CACHE_NAME} 2>/dev/null | cut -f1)
    echo -e "${GREEN}✓${NC} Model found in cache (${MODEL_SIZE})"
else
    echo "  Model not found in local cache — will be downloaded on first start"
fi
echo ""

# 2. Stop current server
echo "Stopping current vLLM server..."
bash "${SCRIPT_DIR}/stop_vllm.sh" --force 2>/dev/null || true
sleep 2
echo ""

# 3. Restart with new model
echo "Starting vLLM with ${MODEL}..."

# Use GPT-OSS script if it's a GPT-OSS model
if echo "$MODEL" | grep -qi "gpt-oss"; then
    VLLM_MODEL="${MODEL}" bash "${SCRIPT_DIR}/start_vllm_gptoss.sh" "${EXTRA_ARGS[@]}"
else
    bash "${SCRIPT_DIR}/start_vllm.sh" "${MODEL}" "${EXTRA_ARGS[@]}"
fi
