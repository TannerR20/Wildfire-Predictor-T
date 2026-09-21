DROP TABLE IF EXISTS processed.fires_assigned;

CREATE TABLE processed.fires_assigned AS
SELECT
    f.*,
    ST_Transform(f.geometry, 5070) AS geom_5070,
    ("DISCOVERY_DATE" AT TIME ZONE 'UTC')::date AS fire_date,
    (FLOOR(ST_X(ST_Transform(f.geometry, 5070)) / 5000.0) * 5000)::bigint AS grid_x,
    (FLOOR(ST_Y(ST_Transform(f.geometry, 5070)) / 5000.0) * 5000)::bigint AS grid_y,
    g.grid_id
FROM processed.fires_filtered AS f
LEFT JOIN processed.grid_5km AS g
    ON g.x = (FLOOR(ST_X(ST_Transform(f.geometry, 5070)) / 5000.0) * 5000)::bigint
   AND g.y = (FLOOR(ST_Y(ST_Transform(f.geometry, 5070)) / 5000.0) * 5000)::bigint
WHERE f.geometry IS NOT NULL;