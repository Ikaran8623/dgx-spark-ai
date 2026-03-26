#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  End-to-end health check: GPU → vLLM → API → Cline         ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Checks:
#   1. GPU status (nvidia-smi)
#   2. vLLM process running
#   3. Port 8000 listening
#   4. /v1/models returns the expected model
#   5. Chat completion generates a response
#   6. Cline CLI configured for local endpoint
#
# Usage:
#   bash health_check.sh          # Run all checks
#   bash health_check.sh --quiet  # Only show failures

set -uo pipefail

PORT="${VLLM_PORT:-8000}"
BASE_URL="http://localhost:${PORT}/v1"
QUIET=false

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

PASS="${GREEN}✓ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"
WARN="${YELLOW}⚠ WARN${NC}"

TOTAL=0
PASSED=0
FAILED=0
WARNINGS=0

check() {
    TOTAL=$((TOTAL + 1))
    local name="$1"
    shift

    if "$@" >/dev/null 2>&1; then
        PASSED=$((PASSED + 1))
        if ! $QUIET; then
            echo -e "  ${PASS}  ${name}"
        fi
        return 0
    else
        FAILED=$((FAILED + 1))
        echo -e "  ${FAIL}  ${name}"
        return 1
    fi
}

info() {
    if ! $QUIET; then
        echo -e "         ${DIM}$*${NC}"
    fi
}

section() {
    if ! $QUIET; then
        echo ""
        echo -e "${BOLD}$*${NC}"
    fi
}

# ─── Parse Arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --quiet|-q) QUIET=true ;;
        --help|-h)
            echo "Usage: bash health_check.sh [--quiet]"
            echo ""
            echo "Run end-to-end health checks on the vLLM + Cline pipeline."
            echo ""
            echo "Options:"
            echo "  --quiet, -q    Only show failures"
            echo "  --help, -h     Show this help"
            exit 0
            ;;
    esac
done

# ─── Banner ──────────────────────────────────────────────────────────────────
if ! $QUIET; then
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  vLLM + Cline Pipeline Health Check     ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
fi

# ─── 1. GPU Checks ───────────────────────────────────────────────────────────
section "🖥  GPU"

check "nvidia-smi is available" command -v nvidia-smi

check "GPU is detected" bash -c "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 'nvidia\|gb10'"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo "?")
info "GPU: ${GPU_NAME} | Temp: ${GPU_TEMP}°C"

# ─── 2. vLLM Process ─────────────────────────────────────────────────────────
section "⚙  vLLM Process"

check "vLLM serve process is running" pgrep -f "vllm serve"

VLLM_PID=$(pgrep -f "vllm serve" -o 2>/dev/null || echo "none")
if [ "$VLLM_PID" != "none" ]; then
    VLLM_USER=$(ps -p "$VLLM_PID" -o user= 2>/dev/null || echo "?")
    VLLM_UPTIME=$(ps -p "$VLLM_PID" -o etime= 2>/dev/null || echo "?")
    info "PID: ${VLLM_PID} | User: ${VLLM_USER} | Uptime: ${VLLM_UPTIME}"
fi

check "EngineCore worker is running" pgrep -f "VLLM::EngineCore"

check "Port ${PORT} is listening" bash -c "ss -tlnp 2>/dev/null | grep -q ':${PORT} '"

# ─── 3. API Health ───────────────────────────────────────────────────────────
section "🌐  API"

check "API responds on ${BASE_URL}/models" bash -c "curl -s --connect-timeout 5 '${BASE_URL}/models' | grep -q '\"object\":\"list\"'"

MODEL_NAME=$(curl -s --connect-timeout 5 "${BASE_URL}/models" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
if [ -n "$MODEL_NAME" ]; then
    info "Serving model: ${MODEL_NAME}"
    check "Model is GPT-OSS" bash -c "echo '${MODEL_NAME}' | grep -qi 'gpt-oss'"
else
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    echo -e "  ${FAIL}  Could not detect model name"
fi

# ─── 4. Chat Completion ──────────────────────────────────────────────────────
section "💬  Inference"

if [ -n "$MODEL_NAME" ]; then
    CHAT_RESPONSE=$(curl -s --connect-timeout 10 --max-time 30 \
        "${BASE_URL}/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: health check ok\"}], \"max_tokens\": 20}" \
        2>/dev/null || echo "")

    check "Chat completion returns a response" bash -c "echo '${CHAT_RESPONSE}' | grep -q '\"choices\"'"

    CONTENT=$(echo "$CHAT_RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']
    print(c.get('content') or c.get('reasoning') or 'generated')
except:
    print('')
" 2>/dev/null || echo "")

    if [ -n "$CONTENT" ]; then
        info "Response: ${CONTENT:0:60}..."
    fi

    TOKENS=$(echo "$CHAT_RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    u = d['usage']
    print(f\"prompt={u['prompt_tokens']} completion={u['completion_tokens']} total={u['total_tokens']}\")
except:
    print('')
" 2>/dev/null || echo "")

    if [ -n "$TOKENS" ]; then
        info "Tokens: ${TOKENS}"
    fi
else
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    echo -e "  ${FAIL}  Skipped — no model detected"
fi

# ─── 5. Cline CLI Configuration ──────────────────────────────────────────────
section "🔧  Cline CLI"

check "cline command is available" command -v cline

GLOBAL_STATE="${HOME}/.cline/data/globalState.json"
if [ -f "$GLOBAL_STATE" ]; then
    check "globalState.json exists" test -f "$GLOBAL_STATE"

    PROVIDER=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('actModeApiProvider',''))" 2>/dev/null || echo "")
    check "Act mode provider is 'openai'" bash -c "[ '${PROVIDER}' = 'openai' ]"
    info "Provider: ${PROVIDER}"

    CONFIGURED_URL=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('openAiBaseUrl',''))" 2>/dev/null || echo "")
    check "Base URL points to local vLLM" bash -c "echo '${CONFIGURED_URL}' | grep -q 'localhost:${PORT}'"
    info "Base URL: ${CONFIGURED_URL}"

    CONFIGURED_MODEL=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('actModeOpenAiModelId',''))" 2>/dev/null || echo "")
    info "Configured model: ${CONFIGURED_MODEL}"

    PLAN_PROVIDER=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('planModeApiProvider',''))" 2>/dev/null || echo "")
    check "Plan mode provider is 'openai'" bash -c "[ '${PLAN_PROVIDER}' = 'openai' ]"
else
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    echo -e "  ${FAIL}  globalState.json not found at ${GLOBAL_STATE}"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}────────────────────────────────────────────${NC}"
echo -e "${BOLD}  Results: ${PASSED}/${TOTAL} passed${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}${FAILED} check(s) failed${NC}"
fi

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "  ${GREEN}${BOLD}All checks passed! Pipeline is healthy.${NC}"
    echo -e "  ${DIM}Run 'cline \"Say hello\"' to test end-to-end.${NC}"
else
    echo ""
    echo -e "  ${YELLOW}Some checks failed. Common fixes:${NC}"
    echo -e "  ${DIM}• Start vLLM:      make serve${NC}"
    echo -e "  ${DIM}• Configure Cline:  make setup-cline${NC}"
    echo -e "  ${DIM}• Force restart:    make serve-force${NC}"
fi

echo -e "${BOLD}────────────────────────────────────────────${NC}"
echo ""

exit $FAILED
