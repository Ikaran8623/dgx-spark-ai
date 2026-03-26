#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  vLLM Watchdog — Auto-restart unhealthy servers              ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Called by vllm-watchdog.timer every 2 minutes.
# Checks if vLLM is responding and restarts it if not.
# Skips restart if the server just started (model loading takes time).

VLLM_URL="http://localhost:8000/v1/models"
TIMEOUT=30

# Only check if the service is supposed to be running
if ! systemctl --user is-active --quiet vllm-server; then
    echo "[watchdog] vllm-server is not active, skipping health check"
    exit 0
fi

echo "[watchdog] Checking vLLM health at ${VLLM_URL}..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT}" "${VLLM_URL}" 2>/dev/null)

if [ "${HTTP_CODE}" = "200" ]; then
    echo "[watchdog] vLLM is healthy (HTTP ${HTTP_CODE})"
    exit 0
fi

# Don't restart if the service just started (model loading takes several minutes)
ACTIVE_ENTER=$(systemctl --user show vllm-server --property=ActiveEnterTimestamp --value 2>/dev/null)
if [ -n "${ACTIVE_ENTER}" ]; then
    START_EPOCH=$(date -d "${ACTIVE_ENTER}" +%s 2>/dev/null)
    NOW_EPOCH=$(date +%s)
    UPTIME_SECS=$(( NOW_EPOCH - START_EPOCH ))

    if [ "${UPTIME_SECS}" -lt 300 ]; then
        echo "[watchdog] vLLM not responding (HTTP ${HTTP_CODE}) but only running for ${UPTIME_SECS}s — likely still loading model, skipping restart"
        exit 0
    fi
fi

echo "[watchdog] vLLM is UNHEALTHY (HTTP ${HTTP_CODE}) — restarting vllm-server..."
systemctl --user restart vllm-server
echo "[watchdog] vllm-server restart triggered"
