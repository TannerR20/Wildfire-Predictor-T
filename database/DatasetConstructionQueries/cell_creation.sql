-- Create the 5km x 5km cells across the lower 48 U.S. states

DROP TABLE IF EXISTS processed.grid_5km;

CREATE TABLE processed.grid_5km AS
WITH states_5070 AS (
    -- Keep states as individual rows so PostGIS can check bounding boxes first
    SELECT ST_Transform("geometry", 5070) AS geometry
    FROM raw.states
    WHERE "STUSPS" NOT IN ('AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP')
),

bounds AS (
    SELECT 
        FLOOR(MIN(ST_XMin(geometry)) / 5000.0) * 5000 AS min_x,
        FLOOR(MIN(ST_YMin(geometry)) / 5000.0) * 5000 AS min_y,
        CEIL(MAX(ST_XMax(geometry)) / 5000.0) * 5000 AS max_x,
        CEIL(MAX(ST_YMax(geometry)) / 5000.0) * 5000 AS max_y
    FROM states_5070
),

grid AS (
    SELECT 
        x,
        y,
        ST_Point(x + 2500, y + 2500, 5070) AS centroid
    FROM bounds,
    generate_series(min_x::bigint, max_x::bigint - 5000, 5000) AS x,
    generate_series(min_y::bigint, max_y::bigint - 5000, 5000) AS y
)

SELECT 
    ROW_NUMBER() OVER (ORDER BY g.x, g.y) AS grid_id,
    g.x,
    g.y,
    ST_MakeEnvelope(g.x, g.y, g.x + 5000, g.y + 5000, 5070) AS geometry
FROM grid AS g
WHERE EXISTS (
    SELECT 1 
    FROM states_5070 AS s 
    WHERE ST_Intersects(s.geometry, g.centroid)
);