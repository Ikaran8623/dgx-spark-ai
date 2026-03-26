#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Stop all vLLM processes and release GPU memory              ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash stop_vllm.sh          # Graceful shutdown
#   bash stop_vllm.sh --force  # Immediate SIGKILL (no grace period)

set -euo pipefail

FORCE=false
GRACE_TIMEOUT=10
GPU_WAIT_TIMEOUT=15

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[stop_vllm]${NC} $*"; }
warn() { echo -e "${YELLOW}[stop_vllm]${NC} $*"; }
ok()   { echo -e "${GREEN}[stop_vllm]${NC} $*"; }
err()  { echo -e "${RED}[stop_vllm]${NC} $*" >&2; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=true ;;
        --help|-h)
            echo "Usage: bash stop_vllm.sh [--force]"
            echo ""
            echo "Stops all vLLM processes and waits for GPU memory release."
            echo ""
            echo "Options:"
            echo "  --force, -f    Skip graceful shutdown, send SIGKILL immediately"
            echo "  --help, -h     Show this help"
            exit 0
            ;;
    esac
done

# ─── Find vLLM Processes ─────────────────────────────────────────────────────
find_vllm_pids() {
    local pids=""
    pids+=" $(pgrep -f 'vllm serve' 2>/dev/null || true)"
    pids+=" $(pgrep -f 'VLLM::EngineCore' 2>/dev/null || true)"
    pids+=" $(pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null || true)"
    echo "$pids" | tr ' ' '\n' | sort -un | grep -v '^$' || true
}

# ─── Kill Processes ───────────────────────────────────────────────────────────
kill_gracefully() {
    local pid=$1
    local cmdline
    cmdline=$(ps -p "$pid" -o args= 2>/dev/null | head -c 80 || echo "unknown")

    if $FORCE; then
        log "Killing PID $pid (SIGKILL): $cmdline"
        kill -9 "$pid" 2>/dev/null || true
        return
    fi

    log "Stopping PID $pid (SIGTERM): $cmdline"
    kill -15 "$pid" 2>/dev/null || true

    local waited=0
    while [ $waited -lt $GRACE_TIMEOUT ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        warn "PID $pid didn't exit gracefully after ${GRACE_TIMEOUT}s, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

# ─── Wait for GPU Release ────────────────────────────────────────────────────
wait_gpu_release() {
    log "Waiting for GPU memory to be released..."
    local waited=0
    while [ $waited -lt $GPU_WAIT_TIMEOUT ]; do
        local gpu_procs
        gpu_procs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || echo "")

        if [ -z "$gpu_procs" ] || ! echo "$gpu_procs" | grep -qi "vllm\|python"; then
            ok "GPU memory released"
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    warn "Some GPU processes may still be running after ${GPU_WAIT_TIMEOUT}s timeout"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    return 1
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    log "Finding vLLM processes..."

    local pids
    pids=$(find_vllm_pids)

    if [ -z "$pids" ]; then
        ok "No vLLM processes found — nothing to stop"
        exit 0
    fi

    local count
    count=$(echo "$pids" | wc -l)
    log "Found $count vLLM process(es) to stop"

    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill_gracefully "$pid"
        fi
    done

    # Verify all dead
    local remaining
    remaining=$(find_vllm_pids)
    if [ -n "$remaining" ]; then
        warn "Some processes survived, force-killing..."
        for pid in $remaining; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2
    fi

    wait_gpu_release

    remaining=$(find_vllm_pids)
    if [ -z "$remaining" ]; then
        ok "All vLLM processes stopped successfully"
    else
        err "WARNING: Some vLLM processes may still be running:"
        for pid in $remaining; do
            ps -p "$pid" -o pid,user,args 2>/dev/null || true
        done
        exit 1
    fi
}

main
