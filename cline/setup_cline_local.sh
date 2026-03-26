#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Configure Cline CLI to use local vLLM on DGX Spark         ║
# ╚══════════════════════════════════════════════════════════════╝
#
# This script:
#   1. Verifies vLLM is running and serving on localhost
#   2. Auto-detects the model name from the /v1/models endpoint
#   3. Configures Cline CLI to use it as an OpenAI-compatible provider
#   4. Verifies the configuration was applied correctly
#
# Usage:
#   bash setup_cline_local.sh                  # Auto-detect everything
#   bash setup_cline_local.sh --port 8001      # Use a different port
#   bash setup_cline_local.sh --model mymodel  # Override model name

set -euo pipefail

PORT="${VLLM_PORT:-8000}"
MODEL_OVERRIDE=""
BASE_URL=""
API_KEY="local-vllm"  # vLLM doesn't require a real key, but Cline needs a non-empty value

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[setup-cline]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup-cline]${NC} $*"; }
ok()   { echo -e "${GREEN}[setup-cline]${NC} $*"; }
err()  { echo -e "${RED}[setup-cline]${NC} $*" >&2; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p) PORT="$2"; shift 2 ;;
        --model|-m) MODEL_OVERRIDE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash setup_cline_local.sh [OPTIONS]"
            echo ""
            echo "Configure Cline CLI to use the local vLLM endpoint."
            echo ""
            echo "Options:"
            echo "  --port, -p PORT     vLLM port (default: 8000, or \$VLLM_PORT)"
            echo "  --model, -m MODEL   Override model name (auto-detected by default)"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

BASE_URL="http://localhost:${PORT}/v1"

# ─── Step 1: Verify vLLM is Running ──────────────────────────────────────────
log "Checking vLLM server at ${BASE_URL}..."

MODELS_RESPONSE=$(curl -s --connect-timeout 5 "${BASE_URL}/models" 2>/dev/null || echo "")

if [ -z "$MODELS_RESPONSE" ] || ! echo "$MODELS_RESPONSE" | grep -q '"object":"list"'; then
    err "vLLM server is not responding at ${BASE_URL}"
    err ""
    err "Please start it first:"
    err "  make serve"
    exit 1
fi

ok "vLLM server is running"

# ─── Step 2: Auto-detect Model Name ──────────────────────────────────────────
if [ -n "$MODEL_OVERRIDE" ]; then
    MODEL="$MODEL_OVERRIDE"
    log "Using specified model: ${MODEL}"
else
    if command -v jq &>/dev/null; then
        MODEL=$(echo "$MODELS_RESPONSE" | jq -r '.data[0].id' 2>/dev/null || echo "")
    else
        MODEL=$(echo "$MODELS_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
    fi

    if [ -z "$MODEL" ] || [ "$MODEL" = "null" ]; then
        err "Could not auto-detect model name from vLLM"
        err "Please specify manually: bash $0 --model YOUR_MODEL_NAME"
        exit 1
    fi

    ok "Auto-detected model: ${MODEL}"
fi

# ─── Step 3: Configure Cline CLI ─────────────────────────────────────────────
log "Configuring Cline CLI..."
log "  Provider:  openai (OpenAI Compatible)"
log "  Base URL:  ${BASE_URL}"
log "  Model:     ${MODEL}"
log "  API Key:   ${API_KEY} (dummy — vLLM doesn't require auth)"

GLOBAL_STATE="${HOME}/.cline/data/globalState.json"
SECRETS_FILE="${HOME}/.cline/data/secrets.json"

# Check if another Cline session is running
CLINE_RUNNING=false
if pgrep -f "cline" | grep -v $$ | grep -v "setup_cline" >/dev/null 2>&1; then
    CLINE_RUNNING=true
    warn "A Cline CLI session is currently running."
    warn "Config changes may be overwritten when that session saves state."
    warn "For best results, close all Cline sessions first, then re-run this script."
fi

# Method 1: Direct JSON patching
if [ -f "$GLOBAL_STATE" ]; then
    log "Patching globalState.json directly..."
    python3 -c "
import json

with open('$GLOBAL_STATE', 'r') as f:
    config = json.load(f)

config['actModeApiProvider'] = 'openai'
config['planModeApiProvider'] = 'openai'
config['actModeOpenAiModelId'] = '$MODEL'
config['planModeOpenAiModelId'] = '$MODEL'
config['openAiBaseUrl'] = '$BASE_URL'

config['actModeOpenAiModelInfo'] = {
    'name': 'GPT-OSS 120B (local vLLM)',
    'maxTokens': 8192,
    'contextWindow': 32768,
    'supportsImages': False,
    'supportsPromptCache': False,
    'inputPrice': 0,
    'outputPrice': 0,
    'temperature': 0,
    'supportsStreaming': True,
    'description': 'OpenAI GPT-OSS 120B served locally via vLLM on DGX Spark'
}
config['planModeOpenAiModelInfo'] = config['actModeOpenAiModelInfo'].copy()

with open('$GLOBAL_STATE', 'w') as f:
    json.dump(config, f, indent=2)

print('OK')
" 2>&1

    if [ $? -eq 0 ]; then
        ok "globalState.json patched"
    else
        err "Failed to patch globalState.json"
    fi
else
    warn "globalState.json not found — creating via cline auth"
fi

# Method 2: Set API key in secrets.json
if [ -f "$SECRETS_FILE" ]; then
    log "Setting API key in secrets.json..."
    python3 -c "
import json

with open('$SECRETS_FILE', 'r') as f:
    secrets = json.load(f)

secrets['openAiApiKey'] = '$API_KEY'

with open('$SECRETS_FILE', 'w') as f:
    json.dump(secrets, f, indent=2)

print('OK')
" 2>&1

    if [ $? -eq 0 ]; then
        ok "API key set in secrets.json"
    else
        warn "Failed to set API key — try: cline auth -p openai -k $API_KEY -m $MODEL -b $BASE_URL"
    fi
fi

# Method 3: Also try cline auth as backup
if command -v cline &>/dev/null && ! $CLINE_RUNNING; then
    log "Running cline auth for complete setup..."
    cline auth \
        --provider openai \
        --apikey "${API_KEY}" \
        --modelid "${MODEL}" \
        --baseurl "${BASE_URL}" 2>&1 | while IFS= read -r line; do
        log "  cline: $line"
    done
fi

ok "Cline CLI configured"

# ─── Step 4: Verify Configuration ────────────────────────────────────────────
log "Verifying configuration..."

if [ ! -f "$GLOBAL_STATE" ]; then
    warn "Could not find globalState.json to verify — config may still be correct"
else
    if command -v jq &>/dev/null; then
        PROVIDER=$(jq -r '.actModeApiProvider // empty' "$GLOBAL_STATE" 2>/dev/null)
        CONFIGURED_URL=$(jq -r '.openAiBaseUrl // empty' "$GLOBAL_STATE" 2>/dev/null)
        CONFIGURED_MODEL=$(jq -r '.actModeOpenAiModelId // empty' "$GLOBAL_STATE" 2>/dev/null)
    else
        PROVIDER=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('actModeApiProvider',''))" 2>/dev/null)
        CONFIGURED_URL=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('openAiBaseUrl',''))" 2>/dev/null)
        CONFIGURED_MODEL=$(python3 -c "import json; d=json.load(open('$GLOBAL_STATE')); print(d.get('actModeOpenAiModelId',''))" 2>/dev/null)
    fi

    ERRORS=0

    if [ "$PROVIDER" = "openai" ]; then
        ok "  Provider:  openai ✓"
    else
        err "  Provider:  expected 'openai', got '${PROVIDER}'"
        ERRORS=$((ERRORS + 1))
    fi

    if [ "$CONFIGURED_URL" = "$BASE_URL" ]; then
        ok "  Base URL:  ${CONFIGURED_URL} ✓"
    else
        err "  Base URL:  expected '${BASE_URL}', got '${CONFIGURED_URL}'"
        ERRORS=$((ERRORS + 1))
    fi

    if [ "$CONFIGURED_MODEL" = "$MODEL" ]; then
        ok "  Model:     ${CONFIGURED_MODEL} ✓"
    else
        err "  Model:     expected '${MODEL}', got '${CONFIGURED_MODEL}'"
        ERRORS=$((ERRORS + 1))
    fi

    if [ $ERRORS -gt 0 ]; then
        warn "Configuration verification had ${ERRORS} issue(s)"
        warn "You may need to configure manually: cline auth"
    fi
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${GREEN}  Cline CLI → Local vLLM${NC}"
echo -e "${BOLD}============================================${NC}"
echo "  Provider:   OpenAI Compatible"
echo "  Endpoint:   ${BASE_URL}"
echo "  Model:      ${MODEL}"
echo "============================================"
echo ""
echo "Test it with:"
echo "  cline \"Say hello\""
echo ""
echo "Or start an interactive session:"
echo "  cline"
echo ""
