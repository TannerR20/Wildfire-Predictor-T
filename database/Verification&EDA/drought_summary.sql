SELECT 
    'Old Table (drought_weekly)' AS dataset,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT grid_id) AS unique_grid_cells,
    MIN(drought_date) AS start_date,
    MAX(drought_date) AS end_date,
    COUNT(*) FILTER (WHERE drought_value IS NULL) AS null_values
FROM processed.drought_weekly

UNION ALL

SELECT 
    'New Table (drought_weekly_new)' AS dataset,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT grid_id) AS unique_grid_cells,
    MIN(drought_date) AS start_date,
    MAX(drought_date) AS end_date,
    COUNT(*) FILTER (WHERE drought_value IS NULL) AS null_values
FROM processed.drought_weekly_new;