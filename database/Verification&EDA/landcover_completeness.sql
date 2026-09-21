SELECT 
    landcover_year,
    COUNT(*) AS total_records,
    COUNT(DISTINCT grid_id) AS unique_cells,
    
    -- Record completeness
    (312333 - COUNT(DISTINCT grid_id)) AS missing_cells_count,

    -- Row-level percentage sum check 
    COUNT(*) FILTER (
        WHERE (forest_pct + shrub_pct + grassland_pct + cropland_pct + 
               pasture_pct + developed_pct + wetland_pct + barren_pct + water_ice_pct) 
               NOT BETWEEN 99.90 AND 100.10
    ) AS sum_out_of_bounds_count,

    -- Individual column value bound check
    COUNT(*) FILTER (
        WHERE forest_pct < 0 OR forest_pct > 100 OR
              shrub_pct < 0 OR shrub_pct > 100 OR
              grassland_pct < 0 OR grassland_pct > 100 OR
              cropland_pct < 0 OR cropland_pct > 100 OR
              pasture_pct < 0 OR pasture_pct > 100 OR
              developed_pct < 0 OR developed_pct > 100 OR
              wetland_pct < 0 OR wetland_pct > 100 OR
              barren_pct < 0 OR barren_pct > 100 OR
              water_ice_pct < 0 OR water_ice_pct > 100
    ) AS percentage_range_violations,

    -- Null value check
    COUNT(*) FILTER (
        WHERE forest_pct IS NULL OR shrub_pct IS NULL OR 
              grassland_pct IS NULL OR cropland_pct IS NULL OR 
              pasture_pct IS NULL OR developed_pct IS NULL OR 
              wetland_pct IS NULL OR barren_pct IS NULL OR water_ice_pct IS NULL
    ) AS null_value_count

FROM processed.historical_landcover
GROUP BY landcover_year
ORDER BY landcover_year ASC;