"""Export a static k-ring graph and optionally inspect daily feature coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .dataset import GridTopology, read_rows
except ImportError:
    from dataset import GridTopology, read_rows


def build_graph(grid_path, k=1, cell_size=5000):
    """Return ordered grid IDs, a [2,E] int64 edge_index, and graph statistics.

    Both directions are included for each spatial connection; self loops are
    omitted because GATConv adds them. k means within k Chebyshev grid steps.
    Node indices follow the input grid's row order, never its numeric IDs.
    """
    topology = GridTopology(grid_path, k=k, cell_size=cell_size)
    node_ids = list(topology.neighbors)
    if not node_ids:
        raise ValueError('Grid CSV contains no nodes.')
    node_index = {gid: i for i, gid in enumerate(node_ids)}
    edges = [(node_index[gid], node_index[neighbor])
             for gid in node_ids for neighbor in topology.neighbors[gid]]
    edge_index = (np.asarray(edges, dtype=np.int64).T.copy() if edges
                  else np.empty((2, 0), dtype=np.int64))
    degrees = np.bincount(edge_index[0], minlength=len(node_ids))
    stats = {
        'nodes': len(node_ids), 'directed_edges': edge_index.shape[1],
        'undirected_connections': edge_index.shape[1] // 2,
        'isolated_nodes': int((degrees == 0).sum()),
        'min_degree': int(degrees.min()), 'max_degree': int(degrees.max()),
        'mean_degree': float(degrees.mean()), 'k': k, 'cell_size_m': cell_size,
        'neighborhood': 'cumulative_chebyshev', 'self_loops': False,
    }
    return node_ids, edge_index, stats


def snapshot_coverage(data_path, node_ids, edge_index):
    """Report omissions without inventing geometry, features or negative labels."""
    frame = read_rows(data_path)
    node_index = {gid: i for i, gid in enumerate(node_ids)}
    missing = set(frame.grid_id) - set(node_index)
    known = frame[frame.grid_id.isin(node_index)]
    # Global adjacency is used only for diagnostics. Actual model snapshots must
    # remap their observed cells to local indices (SnapshotDataset does this).
    neighbors = [[] for _ in node_ids]
    for source, target in edge_index.T:
        neighbors[source].append(target)
    edges = connected = possible = 0
    for _, group in known.groupby('prediction_date'):
        present = {node_index[gid] for gid in group.grid_id}
        for source in present:
            count = sum(target in present for target in neighbors[source])
            edges += count
            connected += int(count > 0)
            possible += len(neighbors[source])
    return {
        'rows': len(frame), 'snapshots': int(frame.prediction_date.nunique()),
        'labeled_rows': int(frame.fire_next_7d.notna().sum()),
        'missing_grid_ids': len(missing),
        'missing_grid_id_examples': [str(gid) for gid in sorted(missing)[:10]],
        'rows_missing_geometry': len(frame) - len(known),
        'directed_same_date_edges': edges,
        'connected_fraction_of_known_rows': connected / len(known) if len(known) else 0.0,
        'observed_neighbor_fraction_within_supplied_grid': edges / possible if possible else 0.0,
        'usable_geometry': not bool(missing),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--cell-size', type=float, default=5000)
    parser.add_argument('--output', default='artifacts/gat/graph_k1')
    parser.add_argument('--data', nargs='+', help='Optional train/validation CSVs for coverage checks.')
    args = parser.parse_args()
    if args.k < 1 or not np.isfinite(args.cell_size) or args.cell_size <= 0:
        parser.error('k and cell-size must be positive and finite.')
    node_ids, edge_index, stats = build_graph(args.grid, args.k, args.cell_size)
    if args.data:
        stats['datasets'] = {
            str(path): snapshot_coverage(path, node_ids, edge_index) for path in args.data
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # No pickle/object arrays: np.load(..., allow_pickle=False) is sufficient.
    np.savez_compressed(output/'graph.npz', edge_index=edge_index)
    pd.DataFrame({'node_index': range(len(node_ids)), 'grid_id': node_ids}).to_csv(
        output/'nodes.csv', index=False)
    (output/'graph_stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f'Saved graph.npz, nodes.csv and graph_stats.json in {output}')
    if stats.get('datasets') and any(not d['usable_geometry'] for d in stats['datasets'].values()):
        print('WARNING: Missing geometry prevents training on these inputs. Export the full grid.')
    if not edge_index.shape[1]:
        print('WARNING: No spatial edges; this graph cannot provide neighbor attention.')


if __name__ == '__main__':
    main()
