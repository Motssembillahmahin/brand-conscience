"""Simple prompt tokenizer for the prompt scorer model."""

from __future__ import annotations

import re
from typing import Any

import torch


class PromptTokenizer:
    """Simple word-level tokenizer for ad prompts.

    Uses a basic vocabulary built from training data. Falls back to <UNK>
    for out-of-vocabulary words.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, vocab: dict[str, int] | None = None, max_len: int = 256) -> None:
        self.max_len = max_len
        self.vocab = vocab or {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens."""
        text = text.lower().strip()
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        """Encode a single text string.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors of shape (max_len,).
        """
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, self.UNK_ID) for t in tokens[: self.max_len]]

        # Pad
        pad_len = self.max_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.PAD_ID] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def encode_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Encode a batch of texts.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors of shape (batch, max_len).
        """
        encoded = [self.encode(t) for t in texts]
        return {
            "input_ids": torch.stack([e["input_ids"] for e in encoded]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encoded]),
        }

    def build_vocab(self, texts: list[str], min_freq: int = 2) -> None:
        """Build vocabulary from a list of texts.

        Args:
            texts: Training texts to build vocab from.
            min_freq: Minimum frequency for a token to be included.
        """
        freq: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                freq[token] = freq.get(token, 0) + 1

        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        idx = 2
        for token, count in sorted(freq.items()):
            if count >= min_freq:
                self.vocab[token] = idx
                idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, path: str, max_len: int = 256) -> PromptTokenizer:
        """Load vocabulary from file."""
        import json
        from pathlib import Path

        vocab = json.loads(Path(path).read_text())
        return cls(vocab=vocab, max_len=max_len)
