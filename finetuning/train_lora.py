"""Run a 20-step Qwen LoRA smoke test without saving model checkpoints.

This script logs metrics to Weights & Biases but uses a temporary output
directory and disables checkpoint/model saving. It requires a CUDA GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "finetuning" / "data" / "train.jsonl"
VAL_PATH = PROJECT_ROOT / "finetuning" / "data" / "val.jsonl"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_TEST_STEPS = 20
RUN_NAME = "qwen2-5-1-5b-lora-smoke-test"


def format_for_sft(tokenizer, example: dict[str, str]) -> dict[str, str]:
    """Turn one instruction/response pair into Qwen chat-template text."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set in the project root .env")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this LoRA smoke test.")
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise RuntimeError("Run finetuning/prepare_dataset.py before training.")

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_name = torch.cuda.get_device_name(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {device_name} ({total_memory_gb:.1f} GB)")
    print(f"Smoke test: {SMOKE_TEST_STEPS} training steps; no checkpoint will be saved.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    datasets = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )
    datasets = datasets.map(
        lambda example: format_for_sft(tokenizer, example),
        remove_columns=datasets["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(
        prefix="feedback-lens-smoke-", ignore_cleanup_errors=True
    ) as temporary_dir:
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "feedback-lens"),
            name=RUN_NAME,
            dir=temporary_dir,
            config={
                "model": MODEL_ID,
                "max_steps": SMOKE_TEST_STEPS,
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
            },
        )
        try:
            training_args = SFTConfig(
                output_dir=temporary_dir,
                max_steps=SMOKE_TEST_STEPS,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=10,
                save_strategy="no",
                report_to=["wandb"],
                run_name=RUN_NAME,
                bf16=use_bf16,
                fp16=not use_bf16,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_length=512,
                dataset_text_field="text",
                use_liger_kernel=False,
            )
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["validation"],
                processing_class=tokenizer,
                peft_config=peft_config,
            )
            trainer.train()

            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory allocated: {peak_memory_gb:.2f} GB")
            wandb.log({"gpu/peak_memory_allocated_gb": peak_memory_gb})
        finally:
            wandb_run.finish()


if __name__ == "__main__":
    main()
