"""Validate saved timing records and regenerate the result table and chart."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("hf", "hf2", "fgreedy", "fastTR:30", "fastTRctl:30")
LABELS = {"hf": "Stock greedy", "fgreedy": "CUDA-graph greedy",
          "fastTR:30": "CUDA graphs + Token Recycling"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(rows, ids, repeats, bootstrap=None):
    """Ratios of summed per-prompt means; each bootstrap resamples paired prompts."""
    bootstrap = bootstrap or {"seed": 0, "resamples": 1000,
                              "stream_offsets": {"hf": 0, "hf2": 2, "fgreedy": 4,
                                                 "fastTR:30": 5, "fastTRctl:30": 7}}
    if not ids or len(set(ids)) != len(ids) or repeats < 1:
        raise ValueError("Expected unique prompt IDs and at least one repeat")
    by = {}
    for row in rows:
        key = (row["cfg"], row["id"], row["rep"])
        if key in by:
            raise ValueError(f"Duplicate timing row: {key}")
        if not np.isfinite(row["decode_s"]) or row["decode_s"] <= 0 or row["tokens"] < 2:
            raise ValueError(f"Invalid decode measurement: {key}")
        by[key] = row
    expected = {(c, i, r) for c in CONFIGS for i in ids for r in range(repeats)}
    if set(by) != expected:
        raise ValueError("Incomplete or unexpected configuration/prompt/repeat grid")
    for i in ids:
        counts = {by[c, i, r]["tokens"] for c in CONFIGS for r in range(repeats)}
        if len(counts) != 1:
            raise ValueError(f"Configurations generated different token counts for prompt {i}")
    dec = {c: np.array([np.mean([by[c, i, r]["decode_s"] for r in range(repeats)])
                       for i in ids]) for c in CONFIGS}
    result = {}
    for c in CONFIGS:
        mine = [by[c, i, r] for r in range(repeats) for i in ids]
        b, n = bootstrap["resamples"], len(ids)
        rng = np.random.default_rng(bootstrap["seed"])
        # The original report shared one RNG across configurations.
        for _ in range(bootstrap["stream_offsets"][c]):
            rng.integers(0, n, (b, n))

        def interval(numerator):
            indices = rng.integers(0, n, (b, n))
            ratios = dec[numerator][indices].sum(1) / dec[c][indices].sum(1)
            return np.percentile(ratios, [2.5, 97.5]).tolist()

        entry = {
            "ms_per_token": sum(r["decode_s"] for r in mine) * 1000 / sum(r["tokens"] - 1 for r in mine),
            "vs_hf": float(dec["hf"].sum() / dec[c].sum()),
            "vs_hf_ci": interval("hf"),
            "exact": sum(r["match"] is True for r in mine) / len(mine),
            "per_repeat": [sum(by["hf", i, r]["decode_s"] for i in ids) /
                           sum(by[c, i, r]["decode_s"] for i in ids) for r in range(repeats)],
            "tokens_per_forward": sum(by[c, i, 0]["tokens"] - 1 for i in ids) /
                                  sum(by[c, i, 0]["forwards"] for i in ids),
        }
        if c != "fgreedy":
            entry["vs_fgreedy"] = float(dec["fgreedy"].sum() / dec[c].sum())
            entry["vs_fgreedy_ci"] = interval("fgreedy")
        result[c] = entry
    checks = {"exact": all(e["exact"] == 1 for e in result.values()),
              "aa_ok": abs(result["hf2"]["vs_hf"] - 1) <= 0.03,
              "control_ok": result["fastTRctl:30"]["vs_fgreedy"] < 1.03}
    return {"prompts": len(ids), "repeats": repeats, "configs": result, "checks": checks}


def verify_archive(root=ROOT):
    manifest = json.loads((root / "results/fp32/provenance.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived evidence changed: {name}")
    return manifest


def verify_summary(result, original):
    for name, entry in result["configs"].items():
        for key in ("ms_per_token", "vs_hf", "vs_hf_ci", "per_repeat", "tokens_per_forward",
                    "vs_fgreedy", "vs_fgreedy_ci"):
            if key in entry and not np.allclose(entry[key], original["configs"][name][key],
                                               rtol=1e-12, atol=1e-12):
                raise ValueError(f"Recomputed {name}/{key} differs from the original summary")


def write_report(result, out, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = result["configs"]
    lines = ["| Decoder | ms/token | Speedup | 95% interval | Exact output |",
             "|---|---:|---:|---:|---:|"]
    for name, label in LABELS.items():
        e = data[name]
        lo, hi = e["vs_hf_ci"]
        lines.append(f"| {label} | {e['ms_per_token']:.1f} | {e['vs_hf']:.2f}× | "
                     f"{lo:.2f}–{hi:.2f} | {e['exact']:.0%} |")
    lines += ["", f"A/A: {data['hf2']['vs_hf']:.3f}. Root-only control versus fast greedy: "
              f"{data['fastTRctl:30']['vs_fgreedy']:.3f}×.", "", context]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({**result, "context": context}, indent=2) + "\n")
    if not all(result["checks"].values()):
        (out / "decoding.svg").unlink(missing_ok=True)
        raise ValueError(f"Benchmark validation failed: {result['checks']}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "swarm-token-prediction"})
    fig, ax = plt.subplots(figsize=(10, 3.6))
    values = [data[c]["vs_hf"] for c in LABELS]
    errors = np.array([[v - data[c]["vs_hf_ci"][0], data[c]["vs_hf_ci"][1] - v]
                       for c, v in zip(LABELS, values)]).T
    ax.barh(range(3), values, height=0.5, xerr=errors, capsize=4,
            color=["#94a3b8", "#3b82f6", "#0d9488"])
    ax.set_yticks(range(3), list(LABELS.values()))
    ax.invert_yaxis()
    for i, (c, v) in enumerate(zip(LABELS, values)):
        ax.text(data[c]["vs_hf_ci"][1] + 0.07, i,
                f"{v:.2f}×  ·  {data[c]['ms_per_token']:.1f} ms/token", va="center", fontsize=10)
    ax.set_xlim(0, 3.65)
    ax.set_xticks([0, 1, 2, 3], ["0", "1×", "2×", "3×"])
    ax.set_xlabel("Decode speedup over stock greedy · 95% paired bootstrap intervals")
    ax.set_title("Exact decoding with CUDA graphs and Token Recycling", loc="left", pad=18, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, 0.025, context, fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "decoding.svg", metadata={"Date": None})
    plt.close(fig)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    manifest = verify_archive()
    prompts = json.loads((ROOT / "benchmarks/data/prompts.json").read_text())["prompts"]
    ids = [p["id"] for p in prompts if p["split"] == "dev"]
    result = summarize(read_rows(ROOT / "results/fp32/timings.jsonl"), ids,
                       manifest["benchmark"]["repeats"], manifest["bootstrap"])
    verify_summary(result, json.loads((ROOT / "results/fp32/original_summary.json").read_text()))
    context = "Qwen2.5-0.5B-Instruct · T4 · fp32 · batch 1 · 60 dev prompts · up to 128 tokens · 3 repeats · decode only"
    print(write_report(result, args.out, context))


if __name__ == "__main__":
    main()
