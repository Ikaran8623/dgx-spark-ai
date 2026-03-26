#!/usr/bin/env python3
"""
QLoRA Fine-Tuning with Unsloth on DGX Spark

Fine-tune any Qwen/Llama/Mistral model using QLoRA with Unsloth for 2x faster
training on the NVIDIA GB10 (Grace Blackwell) GPU.

Usage:
    python fine_tune.py --dataset data.jsonl
    python fine_tune.py --dataset data.jsonl --model unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit
    python fine_tune.py --dataset data.jsonl --config configs/qlora_default.yaml

Dataset format (ChatML JSONL):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune with QLoRA via Unsloth")
    parser.add_argument("--dataset", required=True, help="Path to ChatML JSONL training data")
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", default="./output", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    return parser.parse_args()


def load_dataset_from_jsonl(filepath: str):
    """Load ChatML format JSONL into HuggingFace Dataset."""
    from datasets import Dataset

    examples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(data)

    return Dataset.from_list(examples)


def format_chatml(example):
    """Format a ChatML example into the training text format."""
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}


def main():
    args = parse_args()

    print("=" * 60)
    print("  QLoRA Fine-Tuning on DGX Spark (Unsloth)")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LoRA:     rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  LR:       {args.learning_rate}")
    print(f"  Batch:    {args.batch_size} × {args.gradient_accumulation} accumulation")
    print(f"  Seq len:  {args.max_seq_length}")
    print("=" * 60)

    # Validate dataset exists
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    # Import ML libraries (slow imports, do after arg parsing)
    print("\nLoading libraries...")
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install -r training/requirements.txt")
        sys.exit(1)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")

    # Load model with 4-bit quantization
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    print(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Load and format dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset_from_jsonl(args.dataset)
    dataset = dataset.map(format_chatml)
    print(f"Dataset size: {len(dataset)} examples")

    # Split into train/eval
    if len(dataset) > 10:
        split = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none",
        lr_scheduler_type="cosine",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=True,  # Pack multiple examples per sequence for efficiency
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer_stats = trainer.train()

    # Log results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Training loss:     {trainer_stats.training_loss:.4f}")
    print(f"  Training runtime:  {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Samples/second:    {trainer_stats.metrics['train_samples_per_second']:.2f}")
    print("=" * 60)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving model to: {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training stats
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "loss": trainer_stats.training_loss,
            "runtime_seconds": trainer_stats.metrics["train_runtime"],
            "samples_per_second": trainer_stats.metrics["train_samples_per_second"],
            "epochs": args.epochs,
            "dataset_size": len(dataset),
            "model": args.model,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        }, f, indent=2)

    print(f"Stats saved to: {stats_path}")
    print("\nDone! Next steps:")
    print(f"  • Merge LoRA:   python training/merge_lora.py --adapter {final_dir} --output output/merged")
    print(f"  • Export GGUF:  bash training/export_gguf.sh output")


if __name__ == "__main__":
    main()
