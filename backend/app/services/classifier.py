"""Lazy local inference for the fine-tuned LoRA review classifier."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.prepare_dataset import TASK_DESCRIPTION
from labeling.prompts import CATEGORIES, SEVERITIES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = (
    PROJECT_ROOT
    / "finetuning"
    / "checkpoints"
    / "qwen2-5-1-5b-feedback-lens-lora-3ep-v1"
)


class FineTunedClassifier:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        if not (ADAPTER_PATH / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device = torch.device("cuda")
        else:
            dtype = torch.float32
            device = torch.device("cpu")

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.to(device)
        self.model.eval()

    def classify(self, text: str) -> dict[str, str]:
        with self.lock:
            self.load()
            assert self.model is not None and self.tokenizer is not None
            messages = [{"role": "user", "content": f"{TASK_DESCRIPTION}\n\nReview:\n{text}"}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            return parse_label(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def parse_label(response_text: str) -> dict[str, str]:
    text = response_text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", maxsplit=1)[-1][:-3].strip()
    label = json.loads(text)
    if set(label) != {"category", "severity", "justification"}:
        raise ValueError("Model response did not match the required label schema")
    if label["category"] not in CATEGORIES or label["severity"] not in SEVERITIES:
        raise ValueError("Model response contains an unknown category or severity")
    if not isinstance(label["justification"], str) or not label["justification"].strip():
        raise ValueError("Model response has no justification")
    return {key: str(value) for key, value in label.items()}


classifier = FineTunedClassifier()
