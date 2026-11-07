"""Thirty-node Token Recycling: table proposals, tree verification, exact acceptance."""
import json
import time
from importlib.resources import files

import numpy as np
import torch
from transformers import DynamicCache

from ._common import EOS, sync, validate_request, _result


def load_tree(device):
    data = json.loads(files(__package__).joinpath("data/tree.json").read_text())
    return Tree(data["paths"], device)


class Tree:
    """A static draft tree. Node 0 is the root; nodes are ordered by depth."""

    def __init__(self, paths, device):
        order = sorted(range(len(paths)), key=lambda i: (len(paths[i]), i))
        paths = [tuple(paths[i]) for i in order]
        self.paths, self.n, self.q = paths, len(paths), len(paths) + 1
        q = self.q
        idx = {p: i + 1 for i, p in enumerate(paths)}
        idx[()] = 0
        parent = [0] + [idx[p[:-1]] for p in paths]
        depth = [0] + [len(p) for p in paths]
        rank = [0] + [p[-1] for p in paths]
        anc = np.zeros((q, q), dtype=bool)
        for i in range(q):
            j = i
            while True:
                anc[i, j] = True
                if j == 0:
                    break
                j = parent[j]
        self.parent_l, self.depth_l = parent, depth
        self.children = [[] for _ in range(q)]
        for i in range(1, q):
            self.children[parent[i]].append(i)
        self.max_depth = max(depth)
        self.levels = []                                   # contiguous (start, end) per depth
        for d in range(1, self.max_depth + 1):
            ids = [i for i in range(q) if depth[i] == d]
            self.levels.append((ids[0], ids[-1] + 1))
        self.device = device
        self.parent = torch.tensor(parent, device=device)
        self.rank = torch.tensor(rank, device=device)
        self.depth = torch.tensor(depth, device=device)
        self.anc = torch.tensor(anc, device=device)


class TokenRecycling:
    """Token Recycling (rule R0) on the fast stack: a whole round is one CUDA graph, with one host sync per round.

    Builds on the Token Recycling algorithm: build the tree from the table, tree attention, greedy acceptance,
    table update from every node, keep only the accepted path in the KV buffer. Node i is written to slot base + i
    and rotated at position base + depth[i].
    """

    @torch.inference_mode()
    def __init__(self, model, device, core, tree, cuda_graph=True, control=False):
        # control: build and run the whole tree but accept only the root (cannot gain anything)
        self.model, self.device, self.core, self.tree, self.control = model, device, core, tree, control
        self.V, self.K = model.config.vocab_size, 8
        q, D = tree.q, tree.max_depth
        self.q, self.D = q, D
        maxc = max(len(c) for c in tree.children)
        cm = torch.full((q, max(maxc, 1)), -1, dtype=torch.long)
        for i, ch in enumerate(tree.children):
            for j, c in enumerate(ch):
                cm[i, j] = c
        self.cm = cm.to(device)
        self.root = torch.zeros(1, dtype=torch.long, device=device)
        self.base = torch.zeros(1, dtype=torch.long, device=device)
        self.out = torch.zeros(D + 2, dtype=torch.long, device=device)
        self.table = torch.zeros((self.V, self.K), dtype=torch.long, device=device)
        self.ar_q = torch.arange(q, device=device)
        self.ar_d = torch.arange(D + 1, device=device)
        fn = self._round
        self.fn, self.graph = fn, None
        if cuda_graph:
            self.base.fill_(8)
            self.fn()
            fd = self.device
            sync(fd)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.base.fill_(8)
                    self.fn()
            torch.cuda.current_stream().wait_stream(s)
            sync(fd)
            self.graph = torch.cuda.CUDAGraph()
            self.base.fill_(8)
            with torch.cuda.graph(self.graph):
                self.fn()
            sync(fd)

    @torch.inference_mode()
    def reset_table(self):
        self.table.zero_()

    def _round(self):
        t, core, q, D = self.tree, self.core, self.q, self.D
        tok = torch.empty(q, dtype=torch.long, device=self.device)
        tok[0] = self.root[0]
        for s, e in t.levels:
            tok[s:e] = self.table[tok[t.parent[s:e]], t.rank[s:e]]
        base = self.base
        rel = core.ar - base                                          # [M]
        inside = (rel >= 0) & (rel < q)
        mask = (rel < 0)[None, :] | (t.anc[:, rel.clamp(0, q - 1)] & inside[None, :])      # [q, M]
        rpos = base + t.depth
        slots = base + self.ar_q
        logits = core(tok[None], rpos, mask, core.M, slots)[0]       # [q, V]
        pred = logits.argmax(-1)
        # greedy acceptance, one level at a time. `cur` stays a shape-[1] tensor and is used only through
        # index_select / gather: indexing with a 0-dim CUDA tensor would sync the host, which CUDA-graph capture forbids.
        cur = torch.zeros(1, dtype=torch.long, device=self.device)
        alive = torch.full((1,), not self.control, dtype=torch.bool, device=self.device)
        path = torch.zeros(D + 1, dtype=torch.long, device=self.device)
        for d in range(1, D + 1):
            cand = self.cm.index_select(0, cur).view(-1)                   # children of cur, -1 padded
            pred_cur = pred.index_select(0, cur)                          # [1]
            ok = (cand >= 0) & (tok[cand.clamp(min=0)] == pred_cur) & alive
            has = ok.any().view(1)
            nxt = cand.gather(0, ok.to(torch.int32).argmax().view(1))     # first matching child
            cur = torch.where(has, nxt, cur)
            alive = has
            path[d:d + 1] = torch.where(has, nxt, torch.zeros_like(nxt))
        n = (path > 0).sum().view(1)
        pred_last = pred.index_select(0, cur)
        # table update from every node (rule R0: duplicates resolve to the last node)
        topk = torch.topk(logits, self.K, dim=-1).indices
        best = torch.full((self.V,), -1, dtype=torch.long, device=self.device)
        best.scatter_reduce_(0, tok, self.ar_q, "amax", include_self=True)
        self.table[tok] = topk[best[tok]]
        # keep only the accepted path in the KV buffer (padded entries copy the root, into slots that stay masked)
        src = base + path
        kv = core.kv
        kv.index_copy_(4, base + self.ar_d, kv.index_select(4, src))
        newtok = torch.cat([tok[path[1:]], torch.zeros(1, dtype=torch.long, device=self.device)])
        newtok = newtok.index_put((n,), pred_last)
        self.out[0:1] = n
        self.out[1:] = newtok
        self.root.copy_(pred_last)
        base.add_(n + 1)

    @torch.inference_mode()
    def generate(self, ids, max_new):
        validate_request(ids, max_new, self.device, self.core.M, extra=self.q)
        m, dev, core = self.model, self.device, self.core
        n0 = ids.shape[1]
        sync(dev)
        t0 = time.perf_counter()
        cache = DynamicCache()
        out = m(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        first = out.logits[0, -1].argmax()
        toks = [int(first)]
        core.load_prefix(cache, n0)
        self.root.copy_(first.view(1))
        self.base.fill_(n0)
        sync(dev)
        t1 = time.perf_counter()
        rounds = 0
        while len(toks) < max_new and toks[-1] not in EOS:
            if self.graph is not None:
                self.graph.replay()
            else:
                self.fn()
            res = self.out.tolist()                                  # the one host sync per round
            rounds += 1
            new = res[1:2 + res[0]]
            cut = next((i for i, x in enumerate(new) if x in EOS), None)
            if cut is not None:
                new = new[:cut + 1]
            toks.extend(new[:max(max_new - len(toks), 0)])
        sync(dev)
        r = _result(toks[:max_new], t0, t1, time.perf_counter(), rounds)
        r["nodes_run"] = rounds * self.q
        return r

