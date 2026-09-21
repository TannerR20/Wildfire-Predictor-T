CREATE INDEX fires_assigned_grid_id_idx
ON processed.fires_assigned (grid_id);

CREATE INDEX fires_assigned_fire_date_idx
ON processed.fires_assigned (fire_date);

ANALYZE processed.fires_assigned;