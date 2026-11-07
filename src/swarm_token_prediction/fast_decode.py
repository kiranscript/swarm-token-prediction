"""Preallocated Qwen2 KV cache and CUDA-graph greedy decoding."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from ._common import EOS, sync, validate_request, _result

def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def _rms(x, ln):
    dt = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon)
    return ln.weight * h.to(dt)

class Core(nn.Module):
    """The decoder stack on a preallocated KV buffer kv[layer, k/v, 1, kv_heads, M, head_dim]."""

    def __init__(self, model, M):
        super().__init__()
        inner, cfg = model.model, model.config
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        self.embed, self.layers, self.norm, self.lm_head = inner.embed_tokens, inner.layers, inner.norm, model.lm_head
        self.H, self.Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.G, self.M, self.scale = self.H // self.Hkv, M, self.D ** -0.5
        pos = torch.arange(M, device=dev)[None]
        cos, sin = inner.rotary_emb(torch.empty(1, dtype=dt, device=dev), pos)
        self.register_buffer("cos", cos[0], persistent=False)       # [M, D], computed by the stock rotary module
        self.register_buffer("sin", sin[0], persistent=False)
        self.register_buffer("ar", torch.arange(M, device=dev), persistent=False)
        self.register_buffer("kv", torch.zeros(len(self.layers), 2, 1, self.Hkv, M, self.D, dtype=dt, device=dev),
                             persistent=False)

    def forward(self, tok, pos, mask, S, slots=None):
        """tok [1, q] long, pos [q] long (rotary positions), mask [q, S] bool or None, S: keys attended.
        slots [q] long: where the new keys go in the buffer (default: pos).

        Writes the new keys and values into the buffer at `slots`. Returns logits [1, q, V] for every row.
        """
        q = tok.shape[1]
        slots = pos if slots is None else slots
        H, Hkv, D, G = self.H, self.Hkv, self.D, self.G
        x = self.embed(tok)
        cos = self.cos.index_select(0, pos)[None].unsqueeze(1)       # [1, 1, q, D]
        sin = self.sin.index_select(0, pos)[None].unsqueeze(1)
        m = None
        if mask is not None:
            m = mask[None, None] if G == 1 else mask.repeat(G, 1)[None, None]   # rows ordered (group, query)
        for l, layer in enumerate(self.layers):
            a = layer.self_attn
            h = _rms(x, layer.input_layernorm)
            qh = a.q_proj(h).view(1, q, H, D).transpose(1, 2)
            kh = a.k_proj(h).view(1, q, Hkv, D).transpose(1, 2)
            vh = a.v_proj(h).view(1, q, Hkv, D).transpose(1, 2)
            qh = qh * cos + _rot(qh) * sin
            kh = kh * cos + _rot(kh) * sin
            kv = self.kv[l]
            kv[0].index_copy_(2, slots, kh)
            kv[1].index_copy_(2, slots, vh)
            K, V = kv[0][:, :, :S], kv[1][:, :, :S]
            qg = qh.reshape(1, Hkv, G * q, D)                         # heads h = kv * G + g sit together
            o = F.scaled_dot_product_attention(qg, K, V, attn_mask=m, scale=self.scale)
            o = o.reshape(1, H, q, D).transpose(1, 2).reshape(1, q, H * D)
            x = x + a.o_proj(o)
            x = x + layer.mlp(_rms(x, layer.post_attention_layernorm))
        return self.lm_head(self.norm(x))

    def step_static(self, cur, pos, out, idx):
        """One greedy token with fixed shapes: attends the whole buffer under a mask. All state stays on the device."""
        mask = (self.ar <= pos)[None]                                 # [1, M]
        logits = self.forward(cur, pos, mask, self.M)
        nxt = logits[0, -1].argmax(-1, keepdim=True)                  # [1]
        cur.copy_(nxt.view(1, 1))
        out.index_copy_(0, idx, nxt)
        idx.add_(1)
        pos.add_(1)

    def load_prefix(self, cache, n):
        for l, layer in enumerate(cache.layers):
            self.kv[l, 0, :, :, :n].copy_(layer.keys)
            self.kv[l, 1, :, :, :n].copy_(layer.values)

class FastGreedy:
    """Fixed-shape greedy decoding with one CUDA graph replay per token."""

    @torch.inference_mode()
    def __init__(self, model, device, core, cuda_graph=True):
        self.model, self.device, self.core = model, device, core
        self.cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.out = torch.zeros(1, dtype=torch.long, device=device)
        self.idx = torch.zeros(1, dtype=torch.long, device=device)
        fn = core.step_static
        self.fn, self.graph = fn, None
        self.args = (self.cur, self.pos, self.out, self.idx)
        if cuda_graph:
            for _ in range(3):                                       # Warm up before graph capture.
                self.idx.zero_()
                self.fn(*self.args)
            sync(device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.idx.zero_()
                    self.fn(*self.args)
            torch.cuda.current_stream().wait_stream(s)
            sync(device)
            self.graph = torch.cuda.CUDAGraph()
            self.idx.zero_()
            with torch.cuda.graph(self.graph):
                self.fn(*self.args)
            sync(device)

    def _step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.fn(*self.args)

    @torch.inference_mode()
    def generate(self, ids, max_new):
        validate_request(ids, max_new, self.device, self.core.M)
        m, dev, core = self.model, self.device, self.core
        n = ids.shape[1]
        sync(dev)
        t0 = time.perf_counter()
        cache = DynamicCache()
        out = m(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        first = out.logits[0, -1].argmax()
        toks = [int(first)]
        core.load_prefix(cache, n)
        self.cur.copy_(first.view(1, 1))
        self.pos.fill_(n)
        sync(dev)
        t1 = time.perf_counter()
        while len(toks) < max_new and toks[-1] not in EOS:
            self.idx.zero_()
            self._step()
            toks.append(self.out.item())
        sync(dev)
        return _result(toks, t0, t1, time.perf_counter(), len(toks) - 1)
