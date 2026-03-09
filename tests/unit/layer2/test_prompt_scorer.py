"""Tests for prompt scorer model."""

from __future__ import annotations

import torch

from brand_conscience.models.prompt_scorer.architecture import PromptScorerNet
from brand_conscience.models.prompt_scorer.tokenizer import PromptTokenizer


def test_prompt_scorer_output_shape():
    model = PromptScorerNet(vocab_size=100, d_model=64, n_heads=2, n_layers=2)
    input_ids = torch.randint(0, 100, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    output = model(input_ids, mask)
    assert output.shape == (2,)
    assert (output >= 0).all()
    assert (output <= 1).all()


def test_tokenizer_encode():
    tokenizer = PromptTokenizer()
    result = tokenizer.encode("hello world")
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape == (256,)


def test_tokenizer_batch_encode():
    tokenizer = PromptTokenizer()
    result = tokenizer.encode_batch(["hello", "world"])
    assert result["input_ids"].shape == (2, 256)


def test_tokenizer_build_vocab():
    tokenizer = PromptTokenizer()
    tokenizer.build_vocab(["hello world", "hello there", "world peace"], min_freq=1)
    assert tokenizer.vocab_size > 2
    assert "hello" in tokenizer.vocab
