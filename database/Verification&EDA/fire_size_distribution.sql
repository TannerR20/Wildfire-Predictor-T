WITH sized_fires AS (
    SELECT
        "FIRE_SIZE",
        CASE
            WHEN "FIRE_SIZE" < 0.5 THEN '< 0.5 acres'
            WHEN "FIRE_SIZE" < 1 THEN '0.5–<1 acre'
            WHEN "FIRE_SIZE" < 10 THEN '1–<10 acres'
            WHEN "FIRE_SIZE" < 100 THEN '10–<100 acres'
            WHEN "FIRE_SIZE" < 1000 THEN '100–<1,000 acres'
            WHEN "FIRE_SIZE" >= 1000 THEN '>= 1,000 acres'
            ELSE 'Missing'
        END AS fire_size_category
    FROM raw.fires
)
SELECT
    fire_size_category,
    COUNT(*) AS fire_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS fire_percent
FROM sized_fires
GROUP BY fire_size_category
ORDER BY
    CASE fire_size_category
        WHEN '< 0.5 acres' THEN 1
        WHEN '0.5–<1 acre' THEN 2
        WHEN '1–<10 acres' THEN 3
        WHEN '10–<100 acres' THEN 4
        WHEN '100–<1,000 acres' THEN 5
        WHEN '>= 1,000 acres' THEN 6
        ELSE 7
    END;