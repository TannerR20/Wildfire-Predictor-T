CREATE INDEX fire_targets_grid_date_idx
ON processed.fire_targets (grid_id, target_date);

ANALYZE processed.fire_targets;