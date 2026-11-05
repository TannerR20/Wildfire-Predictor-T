"""Train a GAT on the training nodes and save the best validation checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import DATASET_NAMES, load_dataset
from evaluate import evaluate_split
from model import GAT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="Cora")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("gat.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=8, help="Features per head")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if min(args.epochs, args.patience, args.hidden_channels, args.heads) < 1:
        parser.error("Epochs, patience, hidden channels, and heads must be positive.")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Use dropout in [0, 1), a positive learning rate, and nonnegative weight decay.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph, in_channels, num_classes = load_dataset(args.data_root, args.dataset)
    graph = graph.to(device)

    model_config = {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "hidden_channels": args.hidden_channels,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    model = GAT(**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # The full graph supplies messages, but only training labels affect
        # the gradient. This is transductive node classification.
        logits = model(graph.x, graph.edge_index)
        train_mask = graph.train_mask
        loss = F.cross_entropy(logits[train_mask], graph.y[train_mask])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
        loss.backward()
        optimizer.step()

        val_loss, val_accuracy = evaluate_split(model, graph, graph.val_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "dataset": args.dataset,
                    "epoch": epoch,
                },
                args.checkpoint,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | train loss: {loss.item():.4f} | "
                f"val loss: {val_loss:.4f} | val accuracy: {val_accuracy:.2%}"
            )
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_epoch == 0:
        raise RuntimeError("No checkpoint saved: validation loss never improved.")
    print(f"Saved epoch {best_epoch} (val loss {best_val_loss:.4f}) to {args.checkpoint}")


if __name__ == "__main__":
    main()
