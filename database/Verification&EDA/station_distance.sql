SELECT 
	--observation_year,
    COUNT(*) AS total_cells,
    COUNT(DISTINCT station_id) AS unique_stations_assigned,
    
    ROUND(AVG(distance_km)::numeric, 2) AS avg_km,
	
    ROUND(MIN(distance_km)::numeric, 2) AS min_km,
    ROUND(percentile_cont(0.01) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p1_km,
    ROUND(percentile_cont(0.05) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p5_km,
    ROUND(percentile_cont(0.10) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p10_km,
    ROUND(percentile_cont(0.25) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p25_km,
    ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS median_p50_km,
    ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p75_km,
    ROUND(percentile_cont(0.90) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p90_km,
    ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p95_km,
    ROUND(percentile_cont(0.99) WITHIN GROUP (ORDER BY distance_km)::numeric, 2) AS p99_km,
	ROUND(MAX(distance_km)::numeric, 2) AS max_km
FROM processed.grid_weather_stations_yearly
--GROUP BY observation_year;