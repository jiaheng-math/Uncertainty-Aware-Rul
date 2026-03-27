from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chat_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int = 768
    temperature: float = 0.35
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class TimeOmniAdapter:
    def __init__(self, model_dir: str | None, generation_config: GenerationConfig) -> None:
        self.model_dir = model_dir
        self.generation_config = generation_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_dir)

    def load(self) -> None:
        if not self.enabled:
            return
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def generate(self, question: str, system_prompt: str) -> str | None:
        if not self.enabled:
            return None
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = build_chat_prompt(question, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        use_sampling = self.generation_config.temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=use_sampling,
                temperature=self.generation_config.temperature if use_sampling else None,
                top_p=self.generation_config.top_p if use_sampling else None,
                repetition_penalty=self.generation_config.repetition_penalty,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
