SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT grid_id) AS unique_cells,
    MIN(drought_date) AS min_date,
    MAX(drought_date) AS max_date
FROM processed.drought_weekly_new;