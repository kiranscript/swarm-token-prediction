"""Shared model setup and timing helpers."""
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EOS = {151645, 151643}


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def validate_request(ids, max_new, device, capacity=None, extra=0):
    if not isinstance(max_new, int) or max_new < 1:
        raise ValueError("max_new must be a positive integer")
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("Expected one nonempty prompt: input_ids shape [1, length]")
    if ids.dtype != torch.long or ids.device != torch.device(device):
        raise ValueError("input_ids must be int64 on the model's device")
    if capacity is not None and ids.shape[1] + max_new + extra > capacity:
        raise ValueError("Prompt and generation exceed the preallocated cache; increase cache_size")


def _result(toks, t0, t1, t2, forwards):
    return {"tokens": toks, "margins": [], "forwards": forwards,
            "prefill_s": t1 - t0, "decode_s": t2 - t1,
            "total_s": t2 - t0, "nodes_run": forwards}


def load_model(device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device).eval()
    model.requires_grad_(False)
    return tokenizer, model


def encode_chat(tokenizer, text, device):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def env_info(device):
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": str(device),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info
