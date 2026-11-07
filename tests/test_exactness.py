"""Exercise the extracted decoding logic without downloading a model."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from swarm_token_prediction import create_decoder
from swarm_token_prediction._common import EOS


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=128, hidden_size=56, intermediate_size=112,
                         num_hidden_layers=2, num_attention_heads=14,
                         num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=False, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


@pytest.mark.parametrize("method", ["fast-greedy", "recycling"])
def test_matches_stock_across_prompts_and_table_reset(model, method):
    reference = create_decoder(model, method="greedy")
    decoder = create_decoder(model, 128, method=method, cuda_graph=False)
    for n, limit in [(7, 1), (11, 24), (3, 16), (20, 9)]:
        ids = torch.arange(5, 5 + n).view(1, -1)
        actual = decoder.generate(ids, limit)
        assert actual["tokens"] == reference.generate(ids, limit)["tokens"]
        assert len(actual["tokens"]) == limit
        assert actual["prefill_s"] >= 0 and actual["decode_s"] >= 0
    if method == "recycling":
        decoder.reset_table()
        assert not decoder.table.any()
        ids = torch.tensor([[4, 6, 9]])
        assert decoder.generate(ids, 17)["tokens"] == reference.generate(ids, 17)["tokens"]


def test_recycling_accepts_multiple_tokens(model):
    # Force a deterministic successor so this covers acceptance and KV compaction,
    # rather than only the root/fallback path of an untrained random model.
    config = model.config
    uniform = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        uniform.lm_head.weight.zero_()
    ids = torch.tensor([[1, 2, 3]])
    reference = create_decoder(uniform, method="greedy").generate(ids, 19)
    decoder = create_decoder(uniform, 128, cuda_graph=False)
    actual = decoder.generate(ids, 19)
    assert actual["tokens"] == reference["tokens"] == [0] * 19
    assert actual["forwards"] < reference["forwards"]
    control = create_decoder(uniform, 128, cuda_graph=False, root_only=True).generate(ids, 19)
    assert control["tokens"] == reference["tokens"]
    assert control["forwards"] == reference["forwards"]


@pytest.mark.parametrize("method", ["greedy", "fast-greedy", "recycling"])
def test_stops_at_eos(model, method):
    ids = torch.tensor([[3, 6, 10]])
    reference = create_decoder(model, method="greedy").generate(ids, 16)["tokens"]
    decoder = create_decoder(model, 128, method=method, cuda_graph=False)
    previous = EOS.copy()
    try:
        # Cover EOS from prefill and from a later decode step.
        for token in (reference[0], reference[3]):
            EOS.clear()
            EOS.add(token)
            expected = reference[:reference.index(token) + 1]
            assert decoder.generate(ids, 16)["tokens"] == expected
    finally:
        EOS.clear()
        EOS.update(previous)


def test_input_and_capacity_errors(model):
    decoder = create_decoder(model, 64, cuda_graph=False)
    for ids, count in [(torch.tensor([[1]]), 0), (torch.ones(2, 3, dtype=torch.long), 8),
                       (torch.ones(1, 30, dtype=torch.long), 8)]:
        with pytest.raises(ValueError):
            decoder.generate(ids, count)
    with pytest.raises(ValueError, match="CUDA"):
        create_decoder(model)
