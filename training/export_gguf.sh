#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Export fine-tuned model to GGUF format for Ollama           ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash export_gguf.sh <output-dir> [quantization]
#   bash export_gguf.sh output Q4_K_M
#   bash export_gguf.sh output Q8_0

set -euo pipefail

OUTPUT_DIR="${1:-output}"
QUANT="${2:-Q4_K_M}"
FINAL_DIR="${OUTPUT_DIR}/final"
GGUF_DIR="${OUTPUT_DIR}/gguf"

echo "╔══════════════════════════════════════════╗"
echo "║  Export to GGUF                          ║"
echo "╚══════════════════════════════════════════╝"
echo "  Input:  ${FINAL_DIR}"
echo "  Output: ${GGUF_DIR}"
echo "  Quant:  ${QUANT}"
echo ""

if [ ! -d "${FINAL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${FINAL_DIR}"
    echo "Run fine_tune.py first."
    exit 1
fi

# Activate venv if present
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/vllm-env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

mkdir -p "${GGUF_DIR}"

echo "Step 1: Merge LoRA adapters with base model..."
python3 -c "
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='${FINAL_DIR}',
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
)

merged_dir = '${OUTPUT_DIR}/merged'
model.save_pretrained_merged(merged_dir, tokenizer, save_method='merged_16bit')
print(f'Merged model saved to: {merged_dir}')
"

echo ""
echo "Step 2: Convert to GGUF format with ${QUANT} quantization..."
python3 -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='${OUTPUT_DIR}/merged',
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=False,
)

model.save_pretrained_gguf(
    '${GGUF_DIR}',
    tokenizer,
    quantization_method='${QUANT}',
)
print('GGUF export complete!')
"

# Find and rename the output GGUF file
GGUF_FILE=$(find "${GGUF_DIR}" -name "*.gguf" -type f | head -1)
if [ -n "${GGUF_FILE}" ]; then
    cp "${GGUF_FILE}" "${OUTPUT_DIR}/model.gguf"
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Export Complete!                        ║"
    echo "╚══════════════════════════════════════════╝"
    echo "  GGUF model: ${OUTPUT_DIR}/model.gguf"
    echo "  Size: $(du -h "${OUTPUT_DIR}/model.gguf" | cut -f1)"
    echo ""
    echo "To load in Ollama:"
    echo "  ollama create my-model -f Modelfile"
    echo ""
    echo "Modelfile contents:"
    echo "  FROM ${OUTPUT_DIR}/model.gguf"
    echo "  SYSTEM You are an expert coding assistant."
else
    echo "ERROR: No GGUF file found in ${GGUF_DIR}"
    exit 1
fi
