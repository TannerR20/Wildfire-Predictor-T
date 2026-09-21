SELECT 
    MIN(fire_date) AS min_fire_date,
    MAX(fire_date) AS max_fire_date,
    MAX(fire_date) - MIN(fire_date) AS total_fire_days,
    COUNT(DISTINCT fire_date) AS unique_fire_days
FROM processed.fires_assigned;