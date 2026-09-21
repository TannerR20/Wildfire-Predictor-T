ALTER TABLE processed.grid_5km ADD PRIMARY KEY (grid_id);
CREATE INDEX grid_5km_geom_idx ON processed.grid_5km USING gist (geometry);