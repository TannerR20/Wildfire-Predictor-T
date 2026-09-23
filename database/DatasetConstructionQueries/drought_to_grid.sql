DO $$
DECLARE
    yr INT;
BEGIN
    SET work_mem = '1GB';
    SET synchronous_commit = OFF;

    FOR yr IN 2005..2026 LOOP
        RAISE NOTICE 'Processing year %...', yr;

        INSERT INTO processed.drought_weekly_new (grid_id, drought_date, drought_value)
        WITH year_drought AS MATERIALIZED (
            -- Force independent, fast index scan on date
            SELECT grid_id, drought_date, drought_value
            FROM processed.drought_weekly
            WHERE drought_date >= MAKE_DATE(yr, 1, 1) 
              AND drought_date <  MAKE_DATE(yr + 1, 1, 1)
        )
        SELECT DISTINCT ON (m.new_grid_id, d.drought_date)
            m.new_grid_id AS grid_id,
            d.drought_date,
            d.drought_value
        FROM year_drought d
        JOIN processed.grid_5km_landcover_mapping m 
            ON d.grid_id = m.old_grid_id
        ORDER BY 
            m.new_grid_id, 
            d.drought_date, 
            m.overlap_m2 DESC, 
            m.old_grid_id
        ON CONFLICT (grid_id, drought_date) DO NOTHING;
        
        COMMIT;
    END LOOP;

    SET synchronous_commit = ON;
END $$;