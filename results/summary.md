| Decoder | ms/token | Speedup | 95% interval | Exact output |
|---|---:|---:|---:|---:|
| Stock greedy | 32.9 | 1.00× | 1.00–1.00 | 100% |
| CUDA-graph greedy | 17.7 | 1.86× | 1.85–1.86 | 100% |
| CUDA graphs + Token Recycling | 13.9 | 2.37× | 2.27–2.46 | 100% |

A/A: 0.995. Root-only control versus fast greedy: 0.652×.

Qwen2.5-0.5B-Instruct · T4 · fp32 · batch 1 · 60 dev prompts · up to 128 tokens · 3 repeats · decode only
