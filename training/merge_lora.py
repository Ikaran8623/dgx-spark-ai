#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for standalone serving.

After fine-tuning, the output is a small LoRA adapter that must be used with the
base model. This script merges them into a single standalone model.

Usage:
    python merge_lora.py --adapter output/final --output output/merged
    python merge_lora.py --adapter output/final-lora --output output/merged
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    args = parser.parse_args()

    print("Loading libraries...")
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Get base model name from adapter config
    config = PeftConfig.from_pretrained(args.adapter)
    base_model_name = config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    print(f"Adapter:    {args.adapter}")

    print(f"\nLoading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    print("\nDone! Merged model saved.")
    print(f"\nNext steps:")
    print(f"  • Export GGUF:    bash training/export_gguf.sh {os.path.dirname(args.output)}")
    print(f"  • Serve with vLLM: bash inference/start_vllm.sh {args.output}")


if __name__ == "__main__":
    main()
