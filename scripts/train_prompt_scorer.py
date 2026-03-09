"""Train the prompt scorer model.

Usage:
    uv run python scripts/train_prompt_scorer.py --data data/prompt_performance.json --output model_checkpoints/prompt_scorer.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.models.prompt_scorer.architecture import PromptScorerNet
from brand_conscience.models.prompt_scorer.tokenizer import PromptTokenizer

logger = get_logger(__name__)


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="JSON with prompt-performance pairs")
@click.option("--output", required=True, type=click.Path(), help="Output checkpoint path")
@click.option("--epochs", default=50, help="Training epochs")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--vocab-output", default=None, help="Output path for vocabulary")
def train(data: str, output: str, epochs: int, lr: float, batch_size: int, vocab_output: str | None) -> None:
    """Train prompt scorer on prompt-performance pairs."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)

    with open(data) as f:
        dataset = json.load(f)

    prompts = [d["prompt"] for d in dataset]
    scores = [d["score"] for d in dataset]

    # Build tokenizer
    tokenizer = PromptTokenizer()
    tokenizer.build_vocab(prompts, min_freq=2)
    logger.info("vocab_built", size=tokenizer.vocab_size)

    if vocab_output:
        tokenizer.save(vocab_output)

    # Encode
    encoded = tokenizer.encode_batch(prompts)
    targets = torch.tensor(scores, dtype=torch.float32)

    # Split train/val
    n_val = max(1, len(prompts) // 10)
    train_dataset = TensorDataset(
        encoded["input_ids"][n_val:], encoded["attention_mask"][n_val:], targets[n_val:]
    )
    val_dataset = TensorDataset(
        encoded["input_ids"][:n_val], encoded["attention_mask"][:n_val], targets[:n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptScorerNet(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for ids, mask, target in train_loader:
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            pred = model(ids, mask)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids, mask, target in val_loader:
                ids, mask, target = ids.to(device), mask.to(device), target.to(device)
                pred = model(ids, mask)
                val_loss += criterion(pred, target).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / max(len(val_loader), 1)
        logger.info("epoch", epoch=epoch + 1, train_loss=avg_train, val_loss=avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)

    logger.info("training_complete", best_val_loss=best_val_loss, output=output)


if __name__ == "__main__":
    train()
