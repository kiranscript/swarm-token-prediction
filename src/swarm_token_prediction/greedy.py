"""Stock Hugging Face forward loop used as the exactness and timing reference."""
import time

import torch
from transformers import DynamicCache

from ._common import EOS, sync, validate_request


class Greedy:
    def __init__(self, model, device):
        self.model, self.device = model, device

    @torch.inference_mode()
    def generate(self, ids, max_new, margins=False):
        validate_request(ids, max_new, self.device)
        m = self.model
        sync(self.device)
        t0 = time.perf_counter()
        cache = DynamicCache()
        out = m(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1]
        cur = logits.argmax().view(1, 1)
        tok = int(cur)
        toks, marg = [tok], []
        if margins:
            t2 = torch.topk(logits.float(), 2).values
            marg.append(float(t2[0] - t2[1]))
        sync(self.device)
        t1 = time.perf_counter()
        while len(toks) < max_new and toks[-1] not in EOS:
            out = m(input_ids=cur, past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1]
            cur = logits.argmax().view(1, 1)
            toks.append(int(cur))
            if margins:
                t2 = torch.topk(logits.float(), 2).values
                marg.append(float(t2[0] - t2[1]))
        sync(self.device)
        t2_ = time.perf_counter()
        return {"tokens": toks, "margins": marg, "forwards": len(toks) - 1,
                "prefill_s": t1 - t0, "decode_s": t2_ - t1, "total_s": t2_ - t0, "nodes_run": len(toks) - 1}
