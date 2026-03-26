#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Install/Uninstall systemd user services for vLLM            ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash install_services.sh              # Install and enable
#   bash install_services.sh --uninstall  # Remove services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYSTEMD_DIR="${HOME}/.config/systemd/user"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[systemd]${NC} $*"; }
ok()   { echo -e "${GREEN}[systemd]${NC} $*"; }
warn() { echo -e "${YELLOW}[systemd]${NC} $*"; }
err()  { echo -e "${RED}[systemd]${NC} $*" >&2; }

SERVICES=(
    "vllm-server.service"
    "vllm-watchdog.service"
    "vllm-watchdog.timer"
)

# ─── Uninstall ────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--uninstall" ]; then
    echo -e "${BOLD}Uninstalling vLLM systemd services...${NC}"
    echo ""

    # Stop services
    systemctl --user stop vllm-watchdog.timer 2>/dev/null || true
    systemctl --user stop vllm-server 2>/dev/null || true

    # Disable
    systemctl --user disable vllm-watchdog.timer 2>/dev/null || true
    systemctl --user disable vllm-server 2>/dev/null || true

    # Remove files
    for svc in "${SERVICES[@]}"; do
        if [ -f "${SYSTEMD_DIR}/${svc}" ]; then
            rm "${SYSTEMD_DIR}/${svc}"
            ok "Removed ${svc}"
        fi
    done

    # Remove watchdog script symlink
    if [ -L "${SYSTEMD_DIR}/vllm-watchdog.sh" ] || [ -f "${SYSTEMD_DIR}/vllm-watchdog.sh" ]; then
        rm "${SYSTEMD_DIR}/vllm-watchdog.sh" 2>/dev/null || true
    fi

    systemctl --user daemon-reload
    ok "Services uninstalled"
    exit 0
fi

# ─── Install ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}Installing vLLM systemd services...${NC}"
echo ""
echo "  Repo:     ${REPO_DIR}"
echo "  Systemd:  ${SYSTEMD_DIR}"
echo ""

# Create systemd user directory
mkdir -p "${SYSTEMD_DIR}"

# Update service files to point to this repo's actual location
log "Generating service files..."

# vllm-server.service
cat > "${SYSTEMD_DIR}/vllm-server.service" << EOF
[Unit]
Description=vLLM Inference Server (GPT-OSS 120B on DGX Spark)
After=network-online.target
Wants=network-online.target
StartLimitBurst=5
StartLimitIntervalSec=600

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/inference/start_vllm_gptoss.sh
Restart=always
RestartSec=15
Environment=VLLM_HOST=0.0.0.0
Environment=VLLM_PORT=8000

[Install]
WantedBy=default.target
EOF
ok "Installed vllm-server.service"

# vllm-watchdog.service
cat > "${SYSTEMD_DIR}/vllm-watchdog.service" << EOF
[Unit]
Description=vLLM Health Check Watchdog
After=vllm-server.service

[Service]
Type=oneshot
ExecStart=${REPO_DIR}/systemd/vllm-watchdog.sh
EOF
ok "Installed vllm-watchdog.service"

# vllm-watchdog.timer
cp "${SCRIPT_DIR}/vllm-watchdog.timer" "${SYSTEMD_DIR}/vllm-watchdog.timer"
ok "Installed vllm-watchdog.timer"

# Make watchdog script executable
chmod +x "${SCRIPT_DIR}/vllm-watchdog.sh"
chmod +x "${REPO_DIR}/inference/start_vllm_gptoss.sh"
chmod +x "${REPO_DIR}/inference/stop_vllm.sh"

# Reload systemd
systemctl --user daemon-reload
ok "Daemon reloaded"

# Enable services
systemctl --user enable vllm-server
ok "Enabled vllm-server (starts on login)"

systemctl --user enable vllm-watchdog.timer
ok "Enabled vllm-watchdog.timer (health checks every 2 min)"

# Enable lingering so services run even when not logged in
if command -v loginctl &>/dev/null; then
    loginctl enable-linger "$(whoami)" 2>/dev/null || true
    ok "Enabled user lingering (services persist after logout)"
fi

echo ""
echo -e "${BOLD}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Services installed successfully!${NC}"
echo -e "${BOLD}════════════════════════════════════════════${NC}"
echo ""
echo "  Start now:     systemctl --user start vllm-server"
echo "  Check status:  systemctl --user status vllm-server"
echo "  View logs:     journalctl --user -u vllm-server -f"
echo "  Uninstall:     bash $0 --uninstall"
echo ""
