"""Train on date snapshots; purge overlapping label horizons and early-stop on AP."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

try:
    from .dataset import (read_rows, select_features, fit_preprocessor,
                          GridTopology, SnapshotDataset, npz_paths, inspect_npz,
                          fit_npz_preprocessor, NPZSnapshotDataset)
    from .model import WildfireGAT
    from .evaluate import predict, metrics
except ImportError:
    from dataset import (read_rows, select_features, fit_preprocessor,
                         GridTopology, SnapshotDataset, npz_paths, inspect_npz,
                          fit_npz_preprocessor, NPZSnapshotDataset)
    from model import WildfireGAT
    from evaluate import predict, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--grid', help='Required for CSV input; unused for prebuilt NPZ graphs.')
    parser.add_argument('--output', default='artifacts/gat')
    parser.add_argument('--features', nargs='+', help='Explicit predictor allowlist (recommended).')
    parser.add_argument('--k', type=int, default=1, help='Connect cells within k grid steps.')
    parser.add_argument('--cell-size', type=float, default=5000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=1, help='Graphs per batch; start with 1 for large NPZs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon-days', type=int, default=7)
    parser.add_argument('--allow-edgeless', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim, args.heads, args.horizon_days) < 1:
        parser.error('Epochs, patience, batch size, dimensions and horizon must be positive.')
    if not 0 <= args.dropout < 1 or args.lr <= 0:
        parser.error('Require 0 <= dropout < 1 and positive learning rate.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npz_mode = Path(args.train).is_dir() or Path(args.train).suffix.lower() == '.npz'
    if npz_mode != (Path(args.validation).is_dir() or Path(args.validation).suffix.lower() == '.npz'):
        raise ValueError('Training and validation must use the same input format.')
    if npz_mode:
        train_meta, train_names = inspect_npz(npz_paths(args.train), 'train')
        valid_meta, valid_names = inspect_npz(npz_paths(args.validation), 'validation')
        if train_meta[-1][1] >= valid_meta[0][1]:
            raise ValueError('All training dates must precede validation dates.')
        cutoff = valid_meta[0][1] - pd.Timedelta(days=args.horizon_days)
        retained = [(path,date) for path,date in train_meta if date < cutoff]
        print(f'Purged {len(train_meta)-len(retained)} training graphs at split boundary.', flush=True)
        if not retained:
            raise ValueError('No training graphs remain after horizon purge.')
        features = args.features or [c for c in train_names if c in valid_names
                    and c not in {'grid_id','global_node_idx','label_node_idx','fire_next_7d'}
                    and not c.endswith('_id') and 'next_' not in c.lower() and 'future' not in c.lower()]
        if not features or len(features) != len(set(features)):
            raise ValueError('Need a nonempty list of unique predictors.')
        if not set(features).issubset(train_names) or not set(features).issubset(valid_names):
            raise ValueError('Selected features must exist in training and validation NPZs.')
        prep = fit_npz_preprocessor([p for p,_ in retained], features)
        training = NPZSnapshotDataset([p for p,_ in retained], prep, args.allow_edgeless)
        validation = NPZSnapshotDataset([p for p,_ in valid_meta], prep, args.allow_edgeless)
        train, valid = training.frame, validation.frame
        train_date_max, validation_date_min = retained[-1][1], valid_meta[0][1]
        if args.grid:
            print('--grid is unused: NPZ edge_index defines the topology.', flush=True)
    else:
        if not args.grid:
            parser.error('--grid is required with CSV input.')
        train, valid = read_rows(args.train), read_rows(args.validation)
        if train.prediction_date.max() >= valid.prediction_date.min():
            raise ValueError('All training dates must precede validation dates.')
        cutoff = valid.prediction_date.min() - pd.Timedelta(days=args.horizon_days)
        before = len(train)
        train = train.loc[train.prediction_date < cutoff].reset_index(drop=True)
        print(f'Purged {before-len(train)} training rows at split boundary.', flush=True)
        if train.empty:
            raise ValueError('No training rows remain after label-horizon purge.')
        for name, frame in [('training', train), ('validation', valid)]:
            if set(frame.fire_next_7d.dropna().unique()) != {0, 1}:
                raise ValueError(f'{name} needs both positive and negative labeled examples.')
        features = args.features or select_features(train, valid)
        if len(features) != len(set(features)):
            raise ValueError('Duplicate feature names.')
        prep = fit_preprocessor(train, features)
        topology = GridTopology(args.grid, args.k, args.cell_size)
        training = SnapshotDataset(train, prep, topology, args.allow_edgeless)
        validation = SnapshotDataset(valid, prep, topology, args.allow_edgeless)
        train_date_max, validation_date_min = train.prediction_date.max(), valid.prediction_date.min()
    for name, frame in [('training', train), ('validation', valid)]:
        if set(frame.fire_next_7d.dropna().unique()) != {0, 1}:
            raise ValueError(f'{name} needs both positive and negative labeled examples.')
    forbidden = [c for c in features if c in {'fire_next_7d','grid_id','prediction_date','global_node_idx','label_node_idx'}
                 or 'next_' in c.lower() or 'future' in c.lower()]
    if forbidden:
        raise ValueError(f'Outcome/identifier columns cannot be predictors: {forbidden}')
    print('Features:', features, flush=True)
    print('Training graph:', training.stats, '\nValidation graph:', validation.stats, flush=True)
    config = {'input_dim': 2*len(features), 'hidden_dim': args.hidden_dim,
              'heads': args.heads, 'dropout': args.dropout}
    model = WildfireGAT(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # No automatic class weighting: negative sampling already changes prevalence.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output/'best_model.pt'
    best, stale, history = -float('inf'), 0, []
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = count = 0
        for batch in loader:
            batch = batch.to(device)
            mask = batch.label_mask
            n = int(mask.sum())
            if not n:
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index)[mask], batch.y[mask])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()*n
            count += n
        probabilities = predict(model, validation, device, args.batch_size)
        scores = metrics(valid.fire_next_7d, probabilities)
        ap = scores['average_precision']
        history.append({'epoch': epoch, 'train_loss': total_loss/count, **scores})
        print(f'Epoch {epoch}: loss={total_loss/count:.5f}, validation AP={ap:.5f}', flush=True)
        if ap > best:
            best, stale = ap, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                        'model_config': config, 'preprocessor': prep,
                        'input_format': 'npz' if npz_mode else 'csv',
                        'k': None if npz_mode else args.k, 'cell_size': args.cell_size, 'epoch': epoch,
                        'validation_metrics': scores, 'seed': args.seed,
                        'train_date_max': str(train_date_max),
                        'validation_date_min': str(validation_date_min)}, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            break
    (output/'history.json').write_text(json.dumps(history, indent=2))
    (output/'preprocessor.json').write_text(json.dumps(prep, indent=2))
    (output/'graph_stats.json').write_text(json.dumps({'train': training.stats, 'validation': validation.stats}, indent=2))
    print(f'Saved best checkpoint: {checkpoint_path} (validation AP={best:.5f})')


if __name__ == '__main__':
    main()
