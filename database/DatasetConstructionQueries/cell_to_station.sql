CREATE EXTENSION IF NOT EXISTS btree_gist;

-- 1. Build the yearly complete stations once
DROP TABLE IF EXISTS temp_yearly_complete_stations;
CREATE TEMP TABLE temp_yearly_complete_stations AS
WITH daily_filtered AS (
    SELECT 
        station_id,
        date::date AS obs_date,
        EXTRACT(YEAR FROM date::date)::INT AS observation_year,
        element
    FROM raw.weather_daily
    WHERE element IN ('TMAX', 'TMIN', 'PRCP')
      AND date >= '1992-01-01'          -- optional but helpful
),
station_element_counts AS (
    SELECT 
        station_id,
        observation_year,
        element,
        COUNT(DISTINCT obs_date) AS active_days
    FROM daily_filtered
    GROUP BY station_id, observation_year, element
    HAVING COUNT(DISTINCT obs_date) >= 300
)
SELECT 
    s.station_id,
    c.observation_year,
    s.geometry_5070
FROM raw.weather_stations s
JOIN (
    SELECT station_id, observation_year
    FROM station_element_counts
    GROUP BY station_id, observation_year
    HAVING COUNT(DISTINCT element) = 3
) c ON s.station_id = c.station_id;

CREATE INDEX temp_yearly_stat_composite_idx 
ON temp_yearly_complete_stations USING gist(observation_year, geometry_5070);
ANALYZE temp_yearly_complete_stations;

-- Safety check
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM temp_yearly_complete_stations) = 0 THEN
        RAISE EXCEPTION 'No complete stations found.';
    END IF;
END $$;

-- 2. Create empty result table
DROP TABLE IF EXISTS processed.grid_weather_stations_yearly;
CREATE TABLE processed.grid_weather_stations_yearly (
    grid_id          integer,
    observation_year integer,
    grid_x           integer,
    grid_y           integer,
    station_id       text,          -- adjust type if needed
    distance_meters  numeric(10,2),
    distance_km      numeric(8,2)
);

-- 3. Process one year at a time (much friendlier on 16 GB)
DO $$
DECLARE
    yr integer;
BEGIN
    FOR yr IN 
        SELECT DISTINCT observation_year 
        FROM temp_yearly_complete_stations 
        ORDER BY 1
    LOOP
        RAISE NOTICE 'Processing year % ...', yr;

        INSERT INTO processed.grid_weather_stations_yearly
        SELECT 
            g.grid_id,
            yr AS observation_year,
            g.x AS grid_x,
            g.y AS grid_y,
            nearest.station_id,
            ROUND(nearest.dist_meters::numeric, 2),
            ROUND((nearest.dist_meters / 1000.0)::numeric, 2)
        FROM processed.grid_5km g
        CROSS JOIN LATERAL (
            SELECT 
                cs.station_id,
                ST_Point(g.x + 2500, g.y + 2500, 5070) <-> cs.geometry_5070 AS dist_meters
            FROM temp_yearly_complete_stations cs
            WHERE cs.observation_year = yr
            ORDER BY ST_Point(g.x + 2500, g.y + 2500, 5070) <-> cs.geometry_5070
            LIMIT 1
        ) nearest;

        RAISE NOTICE 'Finished year %', yr;
    END LOOP;
END $$;

-- 4. Final index + cleanup
CREATE INDEX idx_grid_weather_yearly 
ON processed.grid_weather_stations_yearly (grid_id, observation_year);

DROP TABLE IF EXISTS temp_yearly_complete_stations;