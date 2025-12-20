import os
from typing import Optional

GPT_KEY = ""
GPT_URL = ""

# ============== 抽象基类 ==============
class BaseLLM:
    def __init__(self) -> None:
        # Track latest and cumulative token usage for transparency across agents.
        self.last_usage = None
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_usage(self, usage: dict) -> None:
        """Persist usage stats without changing return signature."""
        if not usage:
            self.last_usage = None
            return

        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        for key, value in self.last_usage.items():
            if key in self.total_usage:
                self.total_usage[key] += value or 0

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement generate()")


import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError, Timeout

class ClosedSourceLLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(api_key=GPT_KEY, base_url=GPT_URL)
        self.model = model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((APIError, RateLimitError, Timeout, OSError))
    )
    def generate(self, messages, **kwargs) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # response_format={
                #     'type': 'json_object'
                # },
                temperature=kwargs.get("temperature", 0),
                top_p=1
            )
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            if usage:
                self._record_usage(
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                )
            else:
                self._record_usage(None)
            return content
        except Exception as e:
            print(f"[WARN] 请求失败: {e}, 将重试...")
            raise


class LocalLLM(BaseLLM):
    def __init__(self, model_path: str, device: Optional[str] = "cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.device = device

    def generate(self, prompt: str, **kwargs) -> str:
        import torch
        messages=[
        {'role': 'user', 'content': prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        generated = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        try:
            prompt_tokens = inputs.shape[-1]
            completion_tokens = outputs.shape[-1] - inputs.shape[-1]
            self._record_usage(
                {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                }
            )
        except Exception:
            self._record_usage(None)

        return generated


def LLMFactory(llm_type: str, **kwargs) -> BaseLLM:

    if llm_type == "closed":
        return ClosedSourceLLM(model=kwargs.get("model", "gpt-4o-mini"))
    elif llm_type == "local":
        return LocalLLM(model_path=kwargs.get("model_path"), device=kwargs.get("device", "cuda"))
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")
