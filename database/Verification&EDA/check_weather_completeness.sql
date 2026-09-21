SELECT 
    -- Update "weather_year" if your column is "year" or "observation_date"
    observation_year,
    COUNT(*) AS total_records,
    COUNT(DISTINCT grid_id) AS unique_cells,
    
    -- 1. Spatial Coverage Check (Target: 312,333 cells per year)
    (312333 - COUNT(DISTINCT grid_id)) AS missing_grid_cells,

    -- 2. Audit for NULLs across all columns in the table
    COUNT(*) FILTER (WHERE t.* IS NULL) AS completely_null_rows

FROM processed.grid_weather_stations_yearly t
GROUP BY observation_year
ORDER BY observation_year ASC;