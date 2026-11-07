"""Exact fp32 Qwen2 decoding with CUDA graphs and Token Recycling."""
from ._common import MODEL_ID, encode_chat, load_model

__all__ = ["MODEL_ID", "create_decoder", "encode_chat", "load_model"]


def create_decoder(model, cache_size=1600, method="recycling", *, cuda_graph=True,
                   root_only=False):
    """Build a batch-one Qwen2 decoder. Disable graphs only for CPU correctness checks.

    The recycling table persists between generate calls. Call reset_table() to
    start a fresh sequence of prompts. Decoder instances are not thread-safe.
    """
    import torch
    from .fast_decode import Core, FastGreedy
    from .greedy import Greedy
    from .token_recycling import TokenRecycling, load_tree

    if method not in {"greedy", "fast-greedy", "recycling"}:
        raise ValueError(f"Unknown method: {method}")
    if root_only and method != "recycling":
        raise ValueError("root_only is a Token Recycling benchmark control")
    if model.config.model_type != "qwen2":
        raise ValueError("This decoder supports Qwen2 models only")
    parameter = next(model.parameters())
    device = parameter.device
    if parameter.dtype != torch.float32:
        raise ValueError("This implementation is validated for float32 only")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Call model.eval().requires_grad_(False) before creating a decoder")
    if method == "greedy":
        return Greedy(model, device)
    if not isinstance(cache_size, int) or cache_size < 64:
        raise ValueError("cache_size must be an integer of at least 64")
    if cuda_graph and device.type != "cuda":
        raise ValueError("CUDA graphs require a CUDA device; use cuda_graph=False for CPU smoke checks")
    # Capture and replay on the model's CUDA device, including multi-GPU hosts.
    from contextlib import nullcontext
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context, torch.inference_mode():
        core = Core(model, cache_size)
        if method == "fast-greedy":
            return FastGreedy(model, device, core, cuda_graph=cuda_graph)
        decoder = TokenRecycling(model, device, core, load_tree(device),
                                cuda_graph=cuda_graph, control=root_only)
        decoder.reset_table()
        return decoder
