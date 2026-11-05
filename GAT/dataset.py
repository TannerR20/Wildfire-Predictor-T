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
