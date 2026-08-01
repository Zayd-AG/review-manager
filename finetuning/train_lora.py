"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA and save the best adapter."""

from __future__ import annotations

import math
import os
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
TRAINING_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
RUN_NAME = "qwen2-5-1-5b-feedback-lens-lora-3ep-v1"
CHECKPOINT_ROOT = PROJECT_ROOT / "finetuning" / "checkpoints"
ADAPTER_OUTPUT_DIR = CHECKPOINT_ROOT / RUN_NAME


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
        raise RuntimeError("A CUDA GPU is required for LoRA training.")
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise RuntimeError("Run finetuning/prepare_dataset.py before training.")

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_name = torch.cuda.get_device_name(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {device_name} ({total_memory_gb:.1f} GB)")
    print(f"Full training run: {TRAINING_EPOCHS} epochs")

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
    steps_per_epoch = math.ceil(
        len(datasets["train"])
        / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    print(
        f"Training examples: {len(datasets['train'])}; "
        f"validation examples: {len(datasets['validation'])}"
    )
    print(
        f"Effective batch size: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}; "
        f"approximately {steps_per_epoch * TRAINING_EPOCHS} optimizer steps"
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
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "feedback-lens"),
        name=RUN_NAME,
        dir=str(CHECKPOINT_ROOT),
        config={
            "model": MODEL_ID,
            "epochs": TRAINING_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": (
                TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "max_length": 512,
            "learning_rate": 2e-4,
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
        },
        tags=["full-training", "qwen2.5-1.5b", "lora", "v1"],
    )
    try:
        training_args = SFTConfig(
            output_dir=str(ADAPTER_OUTPUT_DIR),
            num_train_epochs=TRAINING_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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

        # load_best_model_at_end leaves the best epoch's adapter in memory.
        trainer.save_model(str(ADAPTER_OUTPUT_DIR))
        tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        best_eval_loss = trainer.state.best_metric
        print(f"Best validation loss: {best_eval_loss:.4f}")
        print(f"Peak GPU memory allocated: {peak_memory_gb:.2f} GB")
        print(f"LoRA adapter saved to: {ADAPTER_OUTPUT_DIR}")
        wandb.log(
            {
                "gpu/peak_memory_allocated_gb": peak_memory_gb,
                "best_eval_loss": best_eval_loss,
            }
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
