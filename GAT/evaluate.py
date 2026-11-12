"""Evaluate saved checkpoints without refitting preprocessing or choosing a threshold."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from torch_geometric.loader import DataLoader

try:
    from .dataset import read_rows, GridTopology, SnapshotDataset, npz_paths, inspect_npz, NPZSnapshotDataset
    from .model import WildfireGAT
except ImportError:
    from dataset import read_rows, GridTopology, SnapshotDataset, npz_paths, inspect_npz, NPZSnapshotDataset
    from model import WildfireGAT


@torch.no_grad()
def predict(model, dataset, device, batch_size=32):
    model.eval()
    probabilities = np.empty(len(dataset.frame), dtype=float)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        rows = batch.row_id.numpy()
        batch = batch.to(device)
        selected = rows >= 0
        scores = model(batch.x, batch.edge_index).sigmoid().cpu().numpy()
        probabilities[rows[selected]] = scores[selected]
    return probabilities


def metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=float)
    mask = np.isfinite(labels)
    y, p = labels[mask].astype(int), np.asarray(probabilities)[mask]
    if not len(y):
        raise ValueError('Evaluation requires labeled rows.')
    predicted = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {'labeled_rows': len(y), 'prevalence': float(y.mean()),
            'average_precision': float(average_precision_score(y, p)) if both else None,
            'roc_auc': float(roc_auc_score(y, p)) if both else None,
            'threshold': threshold,
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--grid', help='Required only for CSV checkpoints.')
    parser.add_argument('--output', default='artifacts/gat/evaluation')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--allow-edgeless', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if checkpoint.get('input_format', 'csv') == 'npz':
        metadata, _ = inspect_npz(npz_paths(args.data))
        dataset = NPZSnapshotDataset([p for p,_ in metadata], checkpoint['preprocessor'], args.allow_edgeless)
        frame = dataset.frame
    else:
        if not args.grid:
            parser.error('--grid is required for CSV checkpoints.')
        frame = read_rows(args.data)
        topology = GridTopology(args.grid, checkpoint['k'], checkpoint['cell_size'])
        dataset = SnapshotDataset(frame, checkpoint['preprocessor'], topology, args.allow_edgeless)
    model = WildfireGAT(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    probabilities = predict(model, dataset, device, args.batch_size)
    result = metrics(frame.fire_next_7d, probabilities, args.threshold)
    result['graph'] = dataset.stats
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result, indent=2))
    predictions = frame[['grid_id', 'prediction_date', 'fire_next_7d']].copy()
    predictions['fire_score'] = probabilities
    predictions.to_csv(output/'predictions.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
