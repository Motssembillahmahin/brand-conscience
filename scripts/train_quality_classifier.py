"""Train the quality classifier model.

Usage:
    uv run python scripts/train_quality_classifier.py \
        --embeddings data/embeddings.pt \
        --output model_checkpoints/quality_classifier.pt
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brand_conscience.common.config import load_settings
from brand_conscience.common.logging import configure_logging, get_logger
from brand_conscience.models.quality_classifier.architecture import QualityClassifierNet

logger = get_logger(__name__)


def _performance_to_label(perf: dict) -> int:
    """Map performance metrics to quality tier label.

    0=excellent, 1=good, 2=acceptable, 3=reject
    """
    ctr = perf.get("ctr", 0.0)
    if ctr >= 0.03:
        return 0
    elif ctr >= 0.02:
        return 1
    elif ctr >= 0.01:
        return 2
    else:
        return 3


@click.command()
@click.option(
    "--embeddings",
    required=True,
    type=click.Path(exists=True),
    help="Embeddings from bootstrap script",
)
@click.option("--output", required=True, type=click.Path(), help="Output checkpoint path")
@click.option("--epochs", default=30, help="Training epochs")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--batch-size", default=64, help="Batch size")
def train(embeddings: str, output: str, epochs: int, lr: float, batch_size: int) -> None:
    """Train quality classifier on CLIP embeddings."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)

    data = torch.load(embeddings, weights_only=False)
    clip_embeddings = data["embeddings"]
    labels = torch.tensor(
        [_performance_to_label(p) for p in data["performance"]],
        dtype=torch.long,
    )

    logger.info("data_loaded", n_samples=len(labels), embedding_dim=clip_embeddings.shape[1])

    # Split
    n_val = max(1, len(labels) // 10)
    train_ds = TensorDataset(clip_embeddings[n_val:], labels[n_val:])
    val_ds = TensorDataset(clip_embeddings[:n_val], labels[:n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QualityClassifierNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            logits = model(emb)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for emb, label in val_loader:
                emb, label = emb.to(device), label.to(device)
                logits = model(emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == label).sum().item()
                total += len(label)

        val_acc = correct / max(total, 1)
        logger.info(
            "epoch",
            epoch=epoch + 1,
            train_loss=train_loss / len(train_loader),
            val_acc=val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)

    logger.info("training_complete", best_val_acc=best_val_acc, output=output)


if __name__ == "__main__":
    train()
