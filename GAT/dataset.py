"""One graph per date; only observed cells become nodes. Unknown labels are masked."""
import warnings

import numpy as np
import pandas as pd
import torch
from shapely import wkb
from torch.utils.data import Dataset
from torch_geometric.data import Data

TARGET = 'fire_next_7d'
EXCLUDE = {TARGET, 'grid_id', 'prediction_date', 'weather_station_id', 'station_id'}


def read_rows(path):
    frame = pd.read_csv(path)
    required = {'grid_id', 'prediction_date', TARGET}
    if not required.issubset(frame):
        raise ValueError(f'{path}: required columns: {sorted(required)}')
    if frame.empty or frame.grid_id.isna().any():
        raise ValueError('Input must contain rows with nonmissing grid IDs.')
    frame['prediction_date'] = pd.to_datetime(frame.prediction_date, errors='raise').dt.normalize()
    if frame.prediction_date.isna().any():
        raise ValueError('Missing prediction dates.')
    if frame.duplicated(['grid_id', 'prediction_date']).any():
        raise ValueError('Duplicate (grid_id, prediction_date) rows.')
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors='raise')
    if not frame[TARGET].dropna().isin([0, 1]).all():
        raise ValueError('Labels must be 0, 1, or missing (unknown).')
    return frame.sort_values(['prediction_date', 'grid_id']).reset_index(drop=True)


def select_features(train, valid):
    # Never automatically include future outcome columns, even if numeric.
    features = [c for c in train.select_dtypes(include='number')
                if c in valid and c not in EXCLUDE and 'next_' not in c.lower()
                and 'future' not in c.lower() and not c.lower().endswith('_id')]
    if not features:
        raise ValueError('No shared numeric predictors.')
    return features


def numeric(frame, features):
    missing = set(features) - set(frame)
    if missing:
        raise ValueError(f'Missing predictors: {sorted(missing)}')
    return frame[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(frame, features):
    values = numeric(frame, features)
    medians = values.median().fillna(0)
    filled = values.fillna(medians)
    means = filled.mean()
    scales = filled.std(ddof=0).replace(0, 1).fillna(1)
    return {'features': features, 'medians': medians.tolist(),
            'means': means.tolist(), 'scales': scales.tolist()}


def transform(frame, prep):
    values = numeric(frame, prep['features']).to_numpy(dtype=np.float64)
    missing = np.isnan(values)
    values = np.where(missing, np.asarray(prep['medians']), values)
    values = (values - np.asarray(prep['means'])) / np.asarray(prep['scales'])
    # A missingness indicator lets the model distinguish imputation from observation.
    result = np.concatenate([values, missing.astype(float)], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed features.')
    return result


class GridTopology:
    """Chebyshev k-hop neighborhood on a regular projected 5 km lattice."""
    def __init__(self, path, k=1, cell_size=5000):
        if k < 1 or cell_size <= 0:
            raise ValueError('k and cell_size must be positive.')
        grid = pd.read_csv(path)
        if 'grid_id' not in grid or grid.grid_id.isna().any() or grid.grid_id.duplicated().any():
            raise ValueError('Grid requires unique nonmissing grid_id values.')
        if {'x', 'y'}.issubset(grid):
            xy = grid[['x', 'y']].to_numpy(dtype=float)
        elif 'centroid_5070' in grid or 'geometry' in grid:
            column = 'centroid_5070' if 'centroid_5070' in grid else 'geometry'
            from shapely import get_srid
            geometries = [wkb.loads(value, hex=True) for value in grid[column]]
            if any(get_srid(g) != 5070 for g in geometries):
                raise ValueError('Geometry must carry EPSG:5070 SRID; or provide projected x/y.')
            xy = np.array([(g.centroid.x, g.centroid.y) for g in geometries])
        else:
            raise ValueError('Grid needs x/y in projected meters, or EPSG:5070 EWKB geometry.')
        lattice = (xy - xy.min(axis=0)) / cell_size
        if not np.isfinite(lattice).all() or not np.allclose(lattice, np.round(lattice), atol=1e-5):
            raise ValueError('Coordinates do not form the configured regular grid.')
        cells = [tuple(v) for v in np.round(lattice).astype(np.int64)]
        if len(set(cells)) != len(cells):
            raise ValueError('Multiple IDs occupy the same grid cell.')
        positions = dict(zip(grid.grid_id, cells))
        lookup = dict(zip(cells, grid.grid_id))
        self.neighbors = {
            gid: [lookup[(x+dx, y+dy)]
                  for dx in range(-k, k+1) for dy in range(-k, k+1)
                  if (dx or dy) and (x+dx, y+dy) in lookup]
            for gid, (x, y) in positions.items()
        }


class SnapshotDataset(Dataset):
    def __init__(self, frame, prep, topology, allow_edgeless=False):
        missing = set(frame.grid_id) - set(topology.neighbors)
        if missing:
            raise ValueError(f'{len(missing)} cell IDs missing from grid CSV; export complete geometry. '
                             f'Examples: {list(sorted(missing))[:5]}')
        self.frame = frame.reset_index(drop=True)
        self.x = transform(self.frame, prep)
        self.topology = topology
        self.groups = list(self.frame.groupby('prediction_date', sort=True).indices.items())
        self.edges = []
        connected = total_nodes = total_edges = possible_edges = 0
        for _, indices in self.groups:
            ids = self.frame.iloc[indices].grid_id.tolist()
            local = {gid: i for i, gid in enumerate(ids)}
            edges = [(local[gid], local[neighbor]) for gid in ids
                     for neighbor in topology.neighbors[gid] if neighbor in local]
            edge = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
            self.edges.append(edge)
            connected += len(set(edge[0].tolist()))
            total_nodes += len(ids)
            total_edges += len(edges)
            possible_edges += sum(len(topology.neighbors[gid]) for gid in ids)
        if not self.frame[TARGET].notna().any():
            raise ValueError('No labeled nodes.')
        self.stats = {'snapshots': len(self.groups), 'node_observations': total_nodes,
                      'directed_spatial_edges': total_edges,
                      'connected_node_fraction': connected / total_nodes,
                      'observed_neighbor_fraction': total_edges / possible_edges if possible_edges else 0.0}
        if total_edges == 0 and not allow_edgeless:
            raise ValueError('No spatial edges exist in these date snapshots. Export same-date neighbors; '
                             'use --allow-edgeless only for a nonspatial diagnostic.')
        if self.stats['connected_node_fraction'] < 0.5:
            warnings.warn(f'Sparse graph coverage: {self.stats}. Missing cells are omitted, not fabricated.')

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        _, indices = self.groups[index]
        labels = self.frame.iloc[indices][TARGET].to_numpy(dtype=float)
        mask = np.isfinite(labels)
        return Data(x=torch.from_numpy(self.x[indices]), edge_index=self.edges[index],
                    y=torch.tensor(np.nan_to_num(labels), dtype=torch.float32),
                    label_mask=torch.tensor(mask), row_id=torch.tensor(indices, dtype=torch.long))


def npz_paths(path):
    """A daily NPZ file or a directory containing daily NPZs, recursively."""
    from pathlib import Path
    path = Path(path)
    paths = sorted(path.rglob('*.npz')) if path.is_dir() else [path]
    if not paths or any(p.suffix.lower() != '.npz' for p in paths):
        raise ValueError(f'Expected an NPZ file or nonempty NPZ directory: {path}')
    return paths


def read_npz(path):
    """Validate local indices. Never use global_node_idx as a tensor index."""
    with np.load(path, allow_pickle=False) as archive:
        required = {'x', 'edge_index', 'grid_id', 'label_node_idx', 'y',
                    'feature_names', 'prediction_date', 'split'}
        if not required.issubset(archive.files):
            raise ValueError(f'{path}: missing arrays {sorted(required-set(archive.files))}')
        data = {key: archive[key] for key in required}
    x, edges, labels, ids, y = (data[k] for k in ('x','edge_index','label_node_idx','grid_id','y'))
    if x.ndim != 2 or len(x) == 0 or x.dtype.kind not in 'fiu':
        raise ValueError(f'{path}: x must be a nonempty numeric [nodes, features] array.')
    names = data['feature_names']
    if names.ndim != 1 or names.dtype.kind not in 'US' or len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError(f'{path}: feature_names must uniquely name every x column.')
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(x) or pd.isna(ids).any():
        raise ValueError(f'{path}: grid_id must uniquely identify each node.')
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu':
        raise ValueError(f'{path}: edge_index must be integer [2, edges].')
    if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError(f'{path}: edge_index contains out-of-range local node indices.')
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or len(np.unique(labels)) != len(labels):
        raise ValueError(f'{path}: label_node_idx must be unique integer indices.')
    if labels.size and (labels.min() < 0 or labels.max() >= len(x)):
        raise ValueError(f'{path}: label_node_idx contains out-of-range local indices.')
    if y.shape != labels.shape or y.dtype.kind not in 'fiu' or not np.isin(y[~np.isnan(y)], [0,1]).all():
        raise ValueError(f'{path}: y must match label_node_idx and contain 0/1 or NaN.')
    if data['prediction_date'].shape != () or data['split'].shape != ():
        raise ValueError(f'{path}: prediction_date and split must be scalar arrays.')
    date = pd.Timestamp(str(data['prediction_date'].item())).normalize()
    if pd.isna(date):
        raise ValueError(f'{path}: missing prediction date.')
    data['date'] = date
    data['split_name'] = str(data['split'].item()).lower()
    return data


def inspect_npz(paths, expected_split=None):
    """Check schemas and split tags before fitting; retain only small metadata."""
    aliases = {'train': {'train','training'}, 'validation': {'validation','validate','valid','val'}}
    metadata = []
    names = None
    for path in paths:
        data = read_npz(path)
        current = data['feature_names'].tolist()
        if names is not None and current != names:
            raise ValueError(f'{path}: inconsistent feature_names/order across daily graphs.')
        names = current
        if expected_split and data['split_name'] not in aliases[expected_split]:
            raise ValueError(f'{path}: split={data["split_name"]!r}, expected {expected_split}.')
        metadata.append((path, data['date']))
    if len({date for _, date in metadata}) != len(metadata):
        raise ValueError('Expected one NPZ graph per prediction date; duplicate dates found.')
    return sorted(metadata, key=lambda item: item[1]), names


def fit_npz_preprocessor(paths, features):
    """Streaming training-only mean imputation/scaling over all context nodes.

    Merge per-file moments to avoid retaining every training graph in memory.
    Missingness indicators are appended by transform(), as for CSV data.
    """
    count = np.zeros(len(features), dtype=np.float64)
    mean = np.zeros(len(features), dtype=np.float64)
    m2 = np.zeros(len(features), dtype=np.float64)
    for path in paths:
        data = read_npz(path)
        names = data['feature_names'].tolist()
        values = data['x'][:, [names.index(c) for c in features]].astype(np.float64)
        for j in range(len(features)):
            column = values[:, j]
            column = column[np.isfinite(column)]
            n = len(column)
            if not n:
                continue
            delta = column.mean() - mean[j]
            total = count[j] + n
            m2[j] += ((column-column.mean())**2).sum() + delta**2 * count[j] * n / total
            mean[j] += delta*n/total
            count[j] = total
    missing = [features[j] for j in range(len(features)) if count[j] == 0]
    if missing:
        warnings.warn(f'Entirely missing training predictors; filling with zero: {missing}')
    variance = np.divide(m2, count, out=np.zeros_like(m2), where=count > 0)
    scales = np.sqrt(variance)
    scales[scales == 0] = 1
    return {'features': features, 'medians': mean.tolist(), 'means': mean.tolist(),
            'scales': scales.tolist(), 'imputation': 'training_mean',
            'all_missing_training_features': missing}


class NPZSnapshotDataset(Dataset):
    """Lazy daily graphs: all context nodes participate, only supplied labels train."""
    def __init__(self, paths, prep, allow_edgeless=False):
        self.paths = list(paths)
        self.prep = prep
        self.offsets = []
        frames = []
        nodes = edges = connected = missing = elements = 0
        for path in self.paths:
            data = read_npz(path)
            names = data['feature_names'].tolist()
            if not set(prep['features']).issubset(names):
                raise ValueError(f'{path}: missing predictors required by checkpoint.')
            indices = data['label_node_idx']
            self.offsets.append(sum(len(f) for f in frames))
            frames.append(pd.DataFrame({'grid_id': data['grid_id'][indices],
                                        'prediction_date': data['date'],
                                        TARGET: data['y']}))
            n = len(data['x'])
            nodes += n
            spatial = data['edge_index'][:, data['edge_index'][0] != data['edge_index'][1]]
            edges += spatial.shape[1]
            connected += len(np.unique(spatial))
            selected = data['x'][:, [names.index(c) for c in prep['features']]]
            missing += int((~np.isfinite(selected)).sum())
            elements += selected.size
        if not frames:
            raise ValueError('No NPZ graphs remain.')
        self.frame = pd.concat(frames, ignore_index=True)
        if self.frame.duplicated(['grid_id','prediction_date']).any():
            raise ValueError('Duplicate labeled cell/date keys across NPZ files.')
        if not self.frame[TARGET].notna().any():
            raise ValueError('No known labels in NPZ dataset.')
        self.stats = {'snapshots': len(self.paths), 'node_observations': nodes,
                      'prediction_rows': len(self.frame), 'directed_spatial_edges': edges,
                      'connected_node_fraction': connected/nodes,
                      'missing_feature_fraction': missing/elements}
        if edges == 0 and not allow_edgeless:
            raise ValueError('No spatial edges in NPZ graphs; use --allow-edgeless only for diagnostics.')
        if missing:
            warnings.warn(f'NPZ missing/nonfinite feature fraction: {missing/elements:.2%}; using training imputation.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = read_npz(self.paths[index])
        n = len(data['x'])
        values = pd.DataFrame(data['x'], columns=data['feature_names'])
        x = transform(values, self.prep)
        selected = data['label_node_idx']
        labels = np.zeros(n, dtype=np.float32)
        labels[selected] = np.nan_to_num(data['y'], nan=0)
        mask = np.zeros(n, dtype=bool)
        mask[selected] = np.isfinite(data['y'])
        # row_id does not contain 'index', so PyG will not shift it during batching.
        rows = np.full(n, -1, dtype=np.int64)
        rows[selected] = np.arange(len(selected)) + self.offsets[index]
        return Data(x=torch.from_numpy(x), edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
                    y=torch.from_numpy(labels), label_mask=torch.from_numpy(mask),
                    row_id=torch.from_numpy(rows))
