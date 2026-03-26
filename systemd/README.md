# Systemd Services

Auto-start vLLM on boot and keep it healthy with a watchdog timer.

## Services

| Service | Type | Purpose |
|---------|------|---------|
| `vllm-server.service` | Long-running | Main vLLM inference server |
| `vllm-watchdog.service` | Oneshot | Health check (triggered by timer) |
| `vllm-watchdog.timer` | Timer | Runs health check every 2 minutes |

## Install

```bash
make install-services
# or
bash systemd/install_services.sh
```

This will:
1. Copy service files to `~/.config/systemd/user/`
2. Enable auto-start on login
3. Enable lingering (services run even when logged out)

## Manage

```bash
# Start/stop
systemctl --user start vllm-server
systemctl --user stop vllm-server
systemctl --user restart vllm-server

# Status
systemctl --user status vllm-server
systemctl --user status vllm-watchdog.timer

# Logs
journalctl --user -u vllm-server -f
journalctl --user -u vllm-watchdog -f
```

## Uninstall

```bash
make uninstall-services
# or
bash systemd/install_services.sh --uninstall
```
