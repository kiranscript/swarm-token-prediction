"""Reproduce the selected fp32 benchmark on a CUDA GPU."""
import argparse
import json
import random
from pathlib import Path

import torch

from swarm_token_prediction import create_decoder, encode_chat, load_model
from swarm_token_prediction._common import env_info
from swarm_token_prediction.fast_decode import Core, FastGreedy
from swarm_token_prediction.token_recycling import TokenRecycling, load_tree
from report import CONFIGS, ROOT, summarize, write_report


@torch.inference_mode()
def run(out):
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA. For a CPU check, run examples/generate.py --device cpu --max-new 8 --check")
    out.mkdir(parents=True, exist_ok=False)
    tokenizer, model = load_model("cuda")
    device = next(model.parameters()).device
    prompts = json.loads((ROOT / "benchmarks/data/prompts.json").read_text())["prompts"]
    warm = [p for p in prompts if p["split"] == "warmup"]
    scored = [p for p in prompts if p["split"] == "dev"]
    max_new, repeats, cache_size = 128, 3, 1600
    enc = {p["id"]: encode_chat(tokenizer, p["text"], device) for p in prompts}
    if max(e.shape[1] for e in enc.values()) + max_new + 31 > cache_size:
        raise ValueError("Prompt lengths no longer fit the recorded 1600-slot cache")
    settings = {"model": model.config._name_or_path,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "env": env_info(device), "dtype": "float32", "batch_size": 1,
                "cache_size": cache_size, "max_new": max_new, "repeats": repeats,
                "prompt_ids": [p["id"] for p in scored],
                "warmup_ids": [p["id"] for p in warm], "configs": CONFIGS}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    reference = create_decoder(model, method="greedy")
    refs = {p["id"]: reference.generate(enc[p["id"]], max_new, margins=True) for p in scored}
    # Share the KV buffer between sequential configurations, as in the source run.
    core, tree = Core(model, cache_size), load_tree(device)
    runners = {"hf": reference, "hf2": reference,
               "fgreedy": FastGreedy(model, device, core),
               "fastTR:30": TokenRecycling(model, device, core, tree),
               "fastTRctl:30": TokenRecycling(model, device, core, tree, control=True)}
    for decoder in runners.values():
        if hasattr(decoder, "reset_table"):
            decoder.reset_table()
        decoder.generate(enc[scored[0]["id"]], 8)
    rows = []
    with (out / "timings.jsonl").open("w") as file:
        for rep in range(repeats):
            order = list(CONFIGS)
            random.Random(f"exp8tree:{rep}").shuffle(order)
            for name in order:
                decoder = runners[name]
                if hasattr(decoder, "reset_table"):
                    decoder.reset_table()
                    for p in warm:
                        decoder.generate(enc[p["id"]], max_new)
                for p in scored:
                    value = decoder.generate(enc[p["id"]], max_new)
                    ref = refs[p["id"]]
                    matches = value["tokens"] == ref["tokens"]
                    row = {"rep": rep, "cfg": name, "id": p["id"], "group": p["group"],
                           "tokens": len(value["tokens"]), "forwards": value["forwards"],
                           "nodes_run": value["nodes_run"], "decode_s": value["decode_s"],
                           "total_s": value["total_s"], "match": matches, "tie": False}
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    rows.append(row)
                    if not matches:
                        (out / "mismatch.json").write_text(json.dumps(
                            {"id": p["id"], "cfg": name, "reference": ref["tokens"],
                             "actual": value["tokens"]}, indent=2) + "\n")
                        raise ValueError(f"Token mismatch: {name}, prompt {p['id']}")
                print(f"Repeat {rep + 1}/{repeats}: {name} complete", flush=True)
    result = summarize(rows, settings["prompt_ids"], repeats)
    context = (f"Qwen2.5-0.5B-Instruct · {settings['env']['gpu']} · fp32 · batch 1 · "
               "60 dev prompts · up to 128 tokens · 3 repeats · decode only")
    print(write_report(result, out, context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/runs/reproduction")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out to preserve the previous run")
    run(args.out)


if __name__ == "__main__":
    main()
