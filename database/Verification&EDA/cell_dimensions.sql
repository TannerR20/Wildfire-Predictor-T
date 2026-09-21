SELECT
    MIN(ST_XMax(geometry) - ST_XMin(geometry)) AS min_width_m,
    MAX(ST_XMax(geometry) - ST_XMin(geometry)) AS max_width_m,
    MIN(ST_YMax(geometry) - ST_YMin(geometry)) AS min_height_m,
    MAX(ST_YMax(geometry) - ST_YMin(geometry)) AS max_height_m
FROM processed.grid_5km;