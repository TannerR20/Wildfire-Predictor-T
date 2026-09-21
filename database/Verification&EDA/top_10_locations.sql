SELECT 
    g.grid_id,
    COUNT(f.*) AS fire_count,
    -- Convert 5070 (meters) to 4326 (lat/lon)
    ST_Y(ST_Transform(ST_Centroid(g.geometry), 4326)) AS latitude,
    ST_X(ST_Transform(ST_Centroid(g.geometry), 4326)) AS longitude
FROM processed.fires_assigned AS f
JOIN processed.grid_5km AS g ON f.grid_id = g.grid_id
GROUP BY g.grid_id, g.geometry
ORDER BY fire_count DESC
LIMIT 10;