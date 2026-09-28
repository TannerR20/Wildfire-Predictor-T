SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';
SET max_parallel_workers_per_gather = 4;

DROP TABLE IF EXISTS processed.weather_daily_clean;

CREATE TABLE processed.weather_daily_clean AS
SELECT
    station_id,
    date,

    ROUND(AVG(CASE WHEN element = 'PRCP' THEN value END)::NUMERIC / 10.0, 2) AS prcp_mm,
    ROUND(AVG(CASE WHEN element = 'TMAX' THEN value END)::NUMERIC / 10.0, 2) AS tmax_c,
    ROUND(AVG(CASE WHEN element = 'TMIN' THEN value END)::NUMERIC / 10.0, 2) AS tmin_c

FROM raw.weather_daily
WHERE (quality_flag IS NULL OR quality_flag = '' OR quality_flag = '0')
  AND element IN ('PRCP', 'TMAX', 'TMIN')
GROUP BY
    station_id,
    date;

CREATE UNIQUE INDEX weather_daily_clean_pk
ON processed.weather_daily_clean (station_id, date);

ANALYZE processed.weather_daily_clean;