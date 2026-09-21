SELECT
    "NWCG_GENERAL_CAUSE" AS fire_cause,
    COUNT(*) AS fire_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS fire_percent
FROM raw.fires
GROUP BY "NWCG_GENERAL_CAUSE"
ORDER BY fire_count DESC;