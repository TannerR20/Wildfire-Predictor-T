"""Restore a trained GAT and report its held-out test metrics."""

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data

from dataset import load_dataset
from model import GAT


@torch.no_grad()
def evaluate_split(model: GAT, graph: Data, mask: Tensor) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy for the selected nodes."""
    model.eval()  # Disable both feature dropout and attention dropout.
    logits = model(graph.x, graph.edge_index)[mask]
    labels = graph.y[mask]
    loss = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("gat.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run train.py first.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    graph, _, _ = load_dataset(args.data_root, checkpoint["dataset"])
    graph = graph.to(device)
    model = GAT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, accuracy = evaluate_split(model, graph, graph.test_mask)
    print(f'{checkpoint["dataset"]} test split | checkpoint epoch: {checkpoint["epoch"]}')
    print(f"Test loss: {loss:.4f} | test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
