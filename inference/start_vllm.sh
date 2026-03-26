#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Generic vLLM Server Startup for DGX Spark                  ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Start vLLM with any model. For GPT-OSS specific setup, use start_vllm_gptoss.sh.
#
# Usage:
#   bash start_vllm.sh MODEL_NAME [VLLM_ARGS...]
#   bash start_vllm.sh Qwen/Qwen2.5-7B-Instruct
#   bash start_vllm.sh meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
#
# Environment variables:
#   VLLM_HOST          Listen host (default: 0.0.0.0)
#   VLLM_PORT          Listen port (default: 8000)
#   VLLM_VENV_DIR      Path to venv (default: ../vllm-env)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VLLM_VENV_DIR:-${REPO_DIR}/vllm-env}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[vllm]${NC} $*"; }
ok()   { echo -e "${GREEN}[vllm]${NC} $*"; }
err()  { echo -e "${RED}[vllm]${NC} $*" >&2; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: bash start_vllm.sh MODEL_NAME [VLLM_ARGS...]"
    echo ""
    echo "Start vLLM with any HuggingFace model on DGX Spark."
    echo ""
    echo "Examples:"
    echo "  bash start_vllm.sh Qwen/Qwen2.5-7B-Instruct"
    echo "  bash start_vllm.sh meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192"
    echo "  bash start_vllm.sh openai/gpt-oss-120b --enforce-eager --max-model-len 32768"
    echo ""
    echo "For GPT-OSS with full setup, use: bash start_vllm_gptoss.sh"
    exit 0
fi

MODEL="$1"
shift
EXTRA_ARGS=("$@")

# ─── Activate Virtual Environment ────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    err "Virtual environment not found at ${VENV_DIR}"
    err "Create it with: make setup-venv"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# ─── Set Up Environment ──────────────────────────────────────────────────────
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE="${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/nvidia"
NVIDIA_LIBS=$(find "${NVIDIA_BASE}" -name "lib" -type d 2>/dev/null | tr '\n' ':')
TORCH_LIBS="${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${TORCH_LIBS}:${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  vLLM Server — ${MODEL}${NC}"
echo -e "${BOLD}============================================${NC}"
echo "  Host:  ${HOST}"
echo "  Port:  ${PORT}"
echo "  API:   http://${HOST}:${PORT}"
echo "============================================"
echo ""

# ─── Start vLLM ──────────────────────────────────────────────────────────────
log "Starting vLLM with ${MODEL}..."
exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
