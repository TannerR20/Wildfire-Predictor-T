DROP TABLE IF EXISTS processed.fire_targets;

CREATE TABLE processed.fire_targets AS
SELECT 
    grid_id,
    target_date::date,
    1 AS fire_next_7d
FROM (
    SELECT DISTINCT
        grid_id,
        fire_date - i AS target_date
    FROM processed.fires_assigned
    CROSS JOIN generate_series(1, 7) AS i
    WHERE grid_id IS NOT NULL
) sub;