# ╔══════════════════════════════════════════════════════════════╗
# ║  dgx-spark-ai — vLLM + Training on NVIDIA DGX Spark        ║
# ╚══════════════════════════════════════════════════════════════╝

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Configuration ────────────────────────────────────────────
VLLM_PORT   ?= 8000
VLLM_HOST   ?= 0.0.0.0
VLLM_MODEL  ?= openai/gpt-oss-120b
VENV_DIR    ?= vllm-env

# ─── Inference ────────────────────────────────────────────────
.PHONY: serve stop health check test-api

serve: ## Start vLLM server with GPT-OSS 120B
	@echo "Starting vLLM server..."
	VLLM_MODEL=$(VLLM_MODEL) VLLM_HOST=$(VLLM_HOST) VLLM_PORT=$(VLLM_PORT) \
		bash inference/start_vllm_gptoss.sh

serve-force: ## Force-restart vLLM server (kills existing)
	@echo "Force-restarting vLLM server..."
	VLLM_MODEL=$(VLLM_MODEL) VLLM_HOST=$(VLLM_HOST) VLLM_PORT=$(VLLM_PORT) \
		bash inference/start_vllm_gptoss.sh --force

stop: ## Stop all vLLM processes
	bash inference/stop_vllm.sh

stop-force: ## Force-kill all vLLM processes
	bash inference/stop_vllm.sh --force

health: ## Run full pipeline health check
	bash inference/health_check.sh

check: ## Quick check if vLLM is responding
	@bash inference/start_vllm_gptoss.sh --check

test-api: ## Run API smoke tests
	@python3 examples/chat_completion.py

# ─── Training ─────────────────────────────────────────────────
.PHONY: train train-peft merge-lora export-gguf

train: ## Fine-tune with QLoRA via Unsloth (DATASET=path required)
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make train DATASET=path/to/data.jsonl"; \
		echo "  Optional: MODEL=... EPOCHS=... LR=... LORA_RANK=..."; \
		exit 1; \
	fi
	python3 training/fine_tune.py \
		--dataset $(DATASET) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(LR),--learning-rate $(LR)) \
		$(if $(LORA_RANK),--lora-rank $(LORA_RANK))

train-peft: ## Fine-tune with standard PEFT/TRL (DATASET=path required)
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make train-peft DATASET=path/to/data.jsonl"; \
		exit 1; \
	fi
	python3 training/fine_tune_peft.py --dataset $(DATASET)

merge-lora: ## Merge LoRA adapter into base model (ADAPTER=path OUTPUT=path)
	@if [ -z "$(ADAPTER)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make merge-lora ADAPTER=path/to/adapter OUTPUT=path/to/merged"; \
		exit 1; \
	fi
	python3 training/merge_lora.py --adapter $(ADAPTER) --output $(OUTPUT)

export-gguf: ## Export model to GGUF format (OUTPUT_DIR=path)
	bash training/export_gguf.sh $(OUTPUT_DIR) $(QUANT)

# ─── Cline Integration ───────────────────────────────────────
.PHONY: setup-cline

setup-cline: ## Configure Cline CLI to use local vLLM
	bash cline/setup_cline_local.sh

# ─── Systemd Services ────────────────────────────────────────
.PHONY: install-services uninstall-services status

install-services: ## Install systemd user services (auto-start on boot)
	bash systemd/install_services.sh

uninstall-services: ## Remove systemd user services
	bash systemd/install_services.sh --uninstall

status: ## Show status of all services
	@echo "─── vLLM Server ───"
	@systemctl --user status vllm-server --no-pager 2>/dev/null || echo "  Not installed"
	@echo ""
	@echo "─── Watchdog Timer ───"
	@systemctl --user status vllm-watchdog.timer --no-pager 2>/dev/null || echo "  Not installed"

# ─── Environment Setup ───────────────────────────────────────
.PHONY: setup-venv setup-training-deps

setup-venv: ## Create Python venv and install vLLM
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r inference/requirements.txt
	@echo ""
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

setup-training-deps: ## Install training dependencies (into active venv)
	pip install -r training/requirements.txt

# ─── Help ─────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  dgx-spark-ai — vLLM + Training on NVIDIA DGX Spark        ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
