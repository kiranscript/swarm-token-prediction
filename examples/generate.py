"""Generate with the 30-node decoder. CPU mode is for correctness checks."""
import argparse

from swarm_token_prediction import create_decoder, encode_chat, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--check", action="store_true", help="Check output against stock greedy")
    args = parser.parse_args()
    if args.max_new < 1:
        parser.error("--max-new must be positive")
    tokenizer, model = load_model(args.device)
    ids = encode_chat(tokenizer, args.prompt, args.device)
    cache_size = max(64, ((ids.shape[1] + args.max_new + 160 + 63) // 64) * 64)
    decoder = create_decoder(model, cache_size, cuda_graph=args.device == "cuda")
    result = decoder.generate(ids, args.max_new)
    print(tokenizer.decode(result["tokens"], skip_special_tokens=True))
    if args.check:
        reference = create_decoder(model, method="greedy").generate(ids, args.max_new)
        if result["tokens"] != reference["tokens"]:
            raise SystemExit("FAIL: token IDs differ from stock greedy")
        print("Exact match with stock greedy.")


if __name__ == "__main__":
    main()
