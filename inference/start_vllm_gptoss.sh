#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  vLLM Server Startup — GPT-OSS on DGX Spark (NVIDIA GB10)  ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Serves OpenAI's GPT-OSS 120B model via an OpenAI-compatible API.
# Handles conflict detection, GPU memory checks, and auto-recovery.
#
# Usage:
#   bash start_vllm_gptoss.sh              # Start (fails if already running)
#   bash start_vllm_gptoss.sh --force      # Kill existing, then start fresh
#   bash start_vllm_gptoss.sh --check      # Just check if server is healthy
#
# Environment variables (all optional):
#   VLLM_MODEL         Model name (default: openai/gpt-oss-120b)
#   VLLM_HOST          Listen host (default: 0.0.0.0)
#   VLLM_PORT          Listen port (default: 8000)
#   VLLM_MAX_MODEL_LEN Max context length (default: 32768)
#   VLLM_VENV_DIR      Path to venv (default: ../vllm-env)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VLLM_VENV_DIR:-${REPO_DIR}/vllm-env}"
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

FORCE=false
CHECK_ONLY=false
HEALTH_TIMEOUT=120  # seconds to wait for vLLM to become healthy after launch

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[vllm-gptoss]${NC} $*"; }
warn() { echo -e "${YELLOW}[vllm-gptoss]${NC} $*"; }
ok()   { echo -e "${GREEN}[vllm-gptoss]${NC} $*"; }
err()  { echo -e "${RED}[vllm-gptoss]${NC} $*" >&2; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=true ;;
        --check|-c) CHECK_ONLY=true ;;
        --help|-h)
            echo "Usage: bash start_vllm_gptoss.sh [OPTIONS] [-- VLLM_ARGS...]"
            echo ""
            echo "Start the vLLM server with GPT-OSS 120B on DGX Spark."
            echo ""
            echo "Options:"
            echo "  --force, -f    Kill any existing vLLM processes before starting"
            echo "  --check, -c    Just check if the server is healthy, don't start"
            echo "  --help, -h     Show this help"
            echo ""
            echo "Any additional arguments after -- are passed to vllm serve."
            exit 0
            ;;
        --) shift; PASSTHROUGH_ARGS=("$@"); break ;;
        *)  PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

# ─── Pre-flight Checks ───────────────────────────────────────────────────────

# Check if vLLM is already serving on our port and healthy
check_existing_server() {
    local response
    response=$(curl -s --connect-timeout 3 "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "")
    if echo "$response" | grep -q '"object":"list"'; then
        return 0  # Server is up and healthy
    fi
    return 1
}

# Check if anything is listening on our port
check_port_in_use() {
    ss -tlnp 2>/dev/null | grep -q ":${PORT} " && return 0
    return 1
}

# Find any running vLLM processes
find_vllm_pids() {
    local pids=""
    pids+=" $(pgrep -f 'vllm serve' 2>/dev/null || true)"
    pids+=" $(pgrep -f 'VLLM::EngineCore' 2>/dev/null || true)"
    echo "$pids" | tr ' ' '\n' | sort -un | grep -v '^$' || true
}

# ─── Health Check Mode ───────────────────────────────────────────────────────
if $CHECK_ONLY; then
    if check_existing_server; then
        ok "vLLM server is healthy on port ${PORT}"
        curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    else
        err "vLLM server is NOT responding on port ${PORT}"
        exit 1
    fi
fi

# ─── Conflict Detection ──────────────────────────────────────────────────────

# Check if a healthy server already exists
if check_existing_server; then
    if $FORCE; then
        warn "Existing vLLM server detected on port ${PORT} — stopping it (--force)"
        bash "${SCRIPT_DIR}/stop_vllm.sh" --force
        sleep 3
    else
        ok "vLLM server is already running and healthy on port ${PORT}!"
        echo ""
        echo "  Model endpoint: http://localhost:${PORT}/v1/chat/completions"
        echo ""
        echo "  To restart:  bash $0 --force"
        echo "  To stop:     bash ${SCRIPT_DIR}/stop_vllm.sh"
        exit 0
    fi
fi

# Check for zombie vLLM processes (running but not serving)
existing_pids=$(find_vllm_pids)
if [ -n "$existing_pids" ]; then
    if $FORCE; then
        warn "Found stale vLLM processes — cleaning up (--force)"
        bash "${SCRIPT_DIR}/stop_vllm.sh" --force
        sleep 3
    else
        err "Found existing vLLM processes that are not serving properly:"
        for pid in $existing_pids; do
            ps -p "$pid" -o pid,user,etime,args 2>/dev/null || true
        done
        echo ""
        err "Run with --force to kill them and start fresh:"
        err "  bash $0 --force"
        exit 1
    fi
fi

# Check if port is taken by something else
if check_port_in_use; then
    if $FORCE; then
        warn "Port ${PORT} is in use by another process — attempting to free it"
        local_pid=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
        if [ -n "$local_pid" ]; then
            warn "Killing PID $local_pid on port ${PORT}"
            kill -9 "$local_pid" 2>/dev/null || true
            sleep 2
        fi
    else
        err "Port ${PORT} is already in use:"
        ss -tlnp 2>/dev/null | grep ":${PORT} " || true
        echo ""
        err "Run with --force to kill the occupying process, or choose a different port:"
        err "  VLLM_PORT=8001 bash $0"
        exit 1
    fi
fi

# ─── Activate Virtual Environment ────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    err "Virtual environment not found at ${VENV_DIR}"
    err "Create it with: make setup-venv"
    err "  or: python3 -m venv ${VENV_DIR} && ${VENV_DIR}/bin/pip install vllm"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# ─── Set Up Environment ──────────────────────────────────────────────────────

# NVIDIA library paths (auto-detect Python version)
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE="${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/nvidia"
NVIDIA_LIBS=$(find "${NVIDIA_BASE}" -name "lib" -type d 2>/dev/null | tr '\n' ':')
TORCH_LIBS="${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${TORCH_LIBS}:${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

# Tiktoken encoding for openai_harmony (Harmony tokenizer)
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-encodings/"
export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken-rs-cache"

# System CUDA ptxas for Blackwell (sm_121a) support
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  vLLM Server — GPT-OSS 120B${NC}"
echo -e "${BOLD}============================================${NC}"
echo "  Model:          ${MODEL}"
echo "  Host:           ${HOST}"
echo "  Port:           ${PORT}"
echo "  Max Model Len:  ${MAX_MODEL_LEN}"
echo "  API URL:        http://${HOST}:${PORT}"
echo "============================================"
echo ""
echo "Once running, test with:"
echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""

# ─── Locate Chat Template ────────────────────────────────────────────────────
CHAT_TEMPLATE=$(find ~/.cache/huggingface/hub/models--openai--gpt-oss-120b -name "chat_template.jinja" 2>/dev/null | head -1)
if [ -z "$CHAT_TEMPLATE" ]; then
    warn "chat_template.jinja not found, chat completions may not work correctly"
fi

# ─── Start vLLM Server ───────────────────────────────────────────────────────
# --enforce-eager: disable CUDA graphs to save memory for large model
# --max-model-len: limit context to fit in memory alongside model weights
# --trust-remote-code: required for custom model architectures
# --chat-template: use the Harmony chat template for proper message formatting
#
# Note: gpt-oss-120b uses a custom Harmony channel format (analysis/commentary/final)
# that is incompatible with vLLM's OpenAI tool-call parser. Tool calling is
# handled natively by the model's chat template.

log "Starting vLLM server..."
exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
    "${PASSTHROUGH_ARGS[@]}"
