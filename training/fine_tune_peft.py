#!/usr/bin/env python3
"""
QLoRA Fine-Tuning with Standard PEFT/TRL on DGX Spark

Alternative to fine_tune.py that uses standard PEFT/TRL (no Unsloth dependency).
Slightly slower but more compatible with edge cases.

Usage:
    python fine_tune_peft.py --dataset data.jsonl
    python fine_tune_peft.py --dataset data.jsonl --model Qwen/Qwen2.5-Coder-32B-Instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with QLoRA via PEFT/TRL")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  QLoRA Fine-Tuning on DGX Spark (PEFT/TRL)")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LoRA:     rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  LR:       {args.learning_rate}")
    print("=" * 60)

    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    print("\nLoading libraries...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu}")
    else:
        print("WARNING: No GPU detected.")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    examples = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = ""
            for msg in data["messages"]:
                role, content = msg["role"], msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            examples.append({"text": text})
    dataset = Dataset.from_list(examples)
    print(f"Dataset: {len(dataset)} examples")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading model: {args.model} (4-bit quantized)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print("Starting QLoRA training on DGX Spark...")
    print("=" * 60)
    stats = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Loss:    {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.1f}s")
    print("=" * 60)

    final_dir = os.path.join(args.output_dir, "final-lora")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA adapter saved to: {final_dir}")

    with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
        json.dump({
            "loss": stats.training_loss,
            "runtime_seconds": stats.metrics["train_runtime"],
            "model": args.model,
            "lora_rank": args.lora_rank,
            "dataset_size": len(dataset),
        }, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
