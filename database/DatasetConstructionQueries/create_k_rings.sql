DO $$
DECLARE
    t_start timestamptz := clock_timestamp();
    t_step timestamptz;

    -- Batching variables
    v_min_grid BIGINT;
    v_max_grid BIGINT;
    v_batch_size BIGINT := 1000;
    v_curr_start BIGINT;
    v_curr_end BIGINT;
    v_total_grids BIGINT;
    v_processed_grids BIGINT := 0;
BEGIN

    RAISE NOTICE '[%] Starting batch processing pipeline...',
        clock_timestamp() - t_start;


    SET LOCAL work_mem = '1GB';
    SET LOCAL max_parallel_workers_per_gather = 4;
    SET LOCAL max_parallel_maintenance_workers = 4;
    SET LOCAL enable_nestloop = off;


    --BUILD BASE GRID/YEAR METRICS

    t_step := clock_timestamp();

    RAISE NOTICE '[%] Creating temp_base_metrics...',
        clock_timestamp() - t_start;

    DROP TABLE IF EXISTS temp_base_metrics;

    CREATE TEMP TABLE temp_base_metrics
    ON COMMIT DROP
    AS
    SELECT
        g.grid_id,
        y.year,

 
        -- Local population
        COALESCE(
            p.population_count,
            0
        ) AS local_pop_count,

        COALESCE(
            p.population_count,
            0
        ) / 25.0 AS local_pop_density,


        -- Local land cover
        COALESCE(
            lc.water_ice_pct,
            0
        ) AS water_ice_pct_local,

        COALESCE(
            lc.developed_pct,
            0
        ) AS developed_pct_local,

        COALESCE(
            lc.barren_pct,
            0
        ) AS barren_pct_local,

        COALESCE(
            lc.forest_pct,
            0
        ) AS forest_pct_local,

        COALESCE(
            lc.shrub_pct,
            0
        ) AS shrub_pct_local,

        COALESCE(
            lc.grassland_pct,
            0
        ) AS grassland_pct_local,

        COALESCE(
            lc.pasture_pct,
            0
        ) AS pasture_pct_local,

        COALESCE(
            lc.cropland_pct,
            0
        ) AS cropland_pct_local,

        COALESCE(
            lc.wetland_pct,
            0
        ) AS wetland_pct_local

    FROM processed.grid_5km g

    CROSS JOIN generate_series(
        2000,
        2023
    ) AS y(year)


    -- CURRENT-GRID HISTORICAL LAND COVER

    LEFT JOIN processed.historical_landcover_new lc
        ON g.grid_id = lc.grid_id
       AND y.year = lc.landcover_year


    -- CURRENT-GRID POPULATION
    LEFT JOIN public.annual_population_5km p
        ON g.grid_id = p.grid_id
       AND y.year = p.year;


    RAISE NOTICE '[%] temp_base_metrics created in %',
    	clock_timestamp() - t_start,
    	clock_timestamp() - t_step;

    RAISE NOTICE '[%] Step 1 duration: %',
        clock_timestamp() - t_start,
        clock_timestamp() - t_step;



    -- VALIDATE BASE TABLE BEFORE EXPENSIVE RING PROCESSING
    t_step := clock_timestamp();

    RAISE NOTICE '[%] Validating temp_base_metrics...',
        clock_timestamp() - t_start;

    RAISE NOTICE
        '[%] Base rows: %, nonzero population rows: %, nonzero developed rows: %, nonzero forest rows: %',
        clock_timestamp() - t_start,
        (
            SELECT COUNT(*)
            FROM temp_base_metrics
        ),
        (
            SELECT COUNT(*)
            FROM temp_base_metrics
            WHERE local_pop_count > 0
        ),
        (
            SELECT COUNT(*)
            FROM temp_base_metrics
            WHERE developed_pct_local > 0
        ),
        (
            SELECT COUNT(*)
            FROM temp_base_metrics
            WHERE forest_pct_local > 0
        );

    -- Fail early if the new land-cover table did not match.
    -- This prevents another multi-hour run producing zeros.

    IF NOT EXISTS (
        SELECT 1
        FROM temp_base_metrics
        WHERE developed_pct_local > 0
           OR forest_pct_local > 0
           OR cropland_pct_local > 0
           OR grassland_pct_local > 0
    ) THEN

        RAISE EXCEPTION
            'No nonzero land-cover values were found in temp_base_metrics. Check processed.historical_landcover_new before continuing.';

    END IF;


    RAISE NOTICE '[%] Base validation passed.',
        clock_timestamp() - t_start;



    -- INDEX BASE TABLE
    t_step := clock_timestamp();

    RAISE NOTICE '[%] Indexing temp_base_metrics...',
        clock_timestamp() - t_start;

    CREATE INDEX idx_temp_base_grid_id
        ON temp_base_metrics (grid_id);

    CREATE INDEX idx_temp_base_grid_year
        ON temp_base_metrics (grid_id, year);

    ANALYZE temp_base_metrics;



    -- INDEX K-RING TABLES
    CREATE INDEX IF NOT EXISTS idx_krings_center
        ON processed.grid_5km_krings (center_grid_id);

    CREATE INDEX IF NOT EXISTS idx_krings_neighbor
        ON processed.grid_5km_krings (neighbor_grid_id);

    ANALYZE processed.grid_5km_krings;


    RAISE NOTICE '[%] Base indexing complete. Duration: %',
        clock_timestamp() - t_start,
        clock_timestamp() - t_step;



    -- INITIALIZE OUTPUT TABLE
    t_step := clock_timestamp();

    RAISE NOTICE '[%] Initializing processed.annual_grid_features...',
        clock_timestamp() - t_start;

    DROP TABLE IF EXISTS processed.annual_grid_features;

    CREATE TABLE processed.annual_grid_features
    (

        -- IDENTIFIERS

        grid_id INT,
        year INT,

        -- LOCAL POPULATION

        pop_count_5km_local NUMERIC,
        pop_density_5k_local NUMERIC,

        -- LOCAL LAND COVER

        water_ice_pct_local NUMERIC,
        developed_pct_local NUMERIC,
        barren_pct_local NUMERIC,
        forest_pct_local NUMERIC,
        shrub_pct_local NUMERIC,
        grassland_pct_local NUMERIC,
        pasture_pct_local NUMERIC,
        cropland_pct_local NUMERIC,
        wetland_pct_local NUMERIC,

        -- POPULATION RINGS

        pop_count_5km_ring1 NUMERIC,
        pop_density_5k_ring1 NUMERIC,

        pop_count_5km_ring2 NUMERIC,
        pop_density_5k_ring2 NUMERIC,

        pop_count_5km_ring3 NUMERIC,
        pop_density_5k_ring3 NUMERIC,

        pop_count_5km_ring4 NUMERIC,
        pop_density_5k_ring4 NUMERIC,

        pop_count_5km_ring5 NUMERIC,
        pop_density_5k_ring5 NUMERIC,

        -- LAND COVER RING 1

        water_ice_pct_ring1 NUMERIC,
        developed_pct_ring1 NUMERIC,
        barren_pct_ring1 NUMERIC,
        forest_pct_ring1 NUMERIC,
        shrub_pct_ring1 NUMERIC,
        grassland_pct_ring1 NUMERIC,
        pasture_pct_ring1 NUMERIC,
        cropland_pct_ring1 NUMERIC,
        wetland_pct_ring1 NUMERIC,

        -- LAND COVER RING 2

        water_ice_pct_ring2 NUMERIC,
        developed_pct_ring2 NUMERIC,
        barren_pct_ring2 NUMERIC,
        forest_pct_ring2 NUMERIC,
        shrub_pct_ring2 NUMERIC,
        grassland_pct_ring2 NUMERIC,
        pasture_pct_ring2 NUMERIC,
        cropland_pct_ring2 NUMERIC,
        wetland_pct_ring2 NUMERIC,

        -- LAND COVER RING 3

        water_ice_pct_ring3 NUMERIC,
        developed_pct_ring3 NUMERIC,
        barren_pct_ring3 NUMERIC,
        forest_pct_ring3 NUMERIC,
        shrub_pct_ring3 NUMERIC,
        grassland_pct_ring3 NUMERIC,
        pasture_pct_ring3 NUMERIC,
        cropland_pct_ring3 NUMERIC,
        wetland_pct_ring3 NUMERIC,

        -- LAND COVER RING 4

        water_ice_pct_ring4 NUMERIC,
        developed_pct_ring4 NUMERIC,
        barren_pct_ring4 NUMERIC,
        forest_pct_ring4 NUMERIC,
        shrub_pct_ring4 NUMERIC,
        grassland_pct_ring4 NUMERIC,
        pasture_pct_ring4 NUMERIC,
        cropland_pct_ring4 NUMERIC,
        wetland_pct_ring4 NUMERIC,


        -- LAND COVER RING 5

        water_ice_pct_ring5 NUMERIC,
        developed_pct_ring5 NUMERIC,
        barren_pct_ring5 NUMERIC,
        forest_pct_ring5 NUMERIC,
        shrub_pct_ring5 NUMERIC,
        grassland_pct_ring5 NUMERIC,
        pasture_pct_ring5 NUMERIC,
        cropland_pct_ring5 NUMERIC,
        wetland_pct_ring5 NUMERIC
    );


    -- DETERMINE GRID ID RANGE

    SELECT
        MIN(grid_id),
        MAX(grid_id),
        COUNT(*)
    INTO
        v_min_grid,
        v_max_grid,
        v_total_grids
    FROM processed.grid_5km;


    RAISE NOTICE
        '[%] Found % total grid cells. ID range: % to %',
        clock_timestamp() - t_start,
        v_total_grids,
        v_min_grid,
        v_max_grid;



    -- PROCESS GRID BATCHES

    v_curr_start := v_min_grid;

    WHILE v_curr_start <= v_max_grid LOOP

        v_curr_end :=
            v_curr_start + v_batch_size - 1;

        t_step := clock_timestamp();


        -- BATCH INSERT

        INSERT INTO processed.annual_grid_features
        (
            grid_id,
            year,

            pop_count_5km_local,
            pop_density_5k_local,

            water_ice_pct_local,
            developed_pct_local,
            barren_pct_local,
            forest_pct_local,
            shrub_pct_local,
            grassland_pct_local,
            pasture_pct_local,
            cropland_pct_local,
            wetland_pct_local,

            pop_count_5km_ring1,
            pop_density_5k_ring1,
            pop_count_5km_ring2,
            pop_density_5k_ring2,
            pop_count_5km_ring3,
            pop_density_5k_ring3,
            pop_count_5km_ring4,
            pop_density_5k_ring4,
            pop_count_5km_ring5,
            pop_density_5k_ring5,

            water_ice_pct_ring1,
            developed_pct_ring1,
            barren_pct_ring1,
            forest_pct_ring1,
            shrub_pct_ring1,
            grassland_pct_ring1,
            pasture_pct_ring1,
            cropland_pct_ring1,
            wetland_pct_ring1,

            water_ice_pct_ring2,
            developed_pct_ring2,
            barren_pct_ring2,
            forest_pct_ring2,
            shrub_pct_ring2,
            grassland_pct_ring2,
            pasture_pct_ring2,
            cropland_pct_ring2,
            wetland_pct_ring2,

            water_ice_pct_ring3,
            developed_pct_ring3,
            barren_pct_ring3,
            forest_pct_ring3,
            shrub_pct_ring3,
            grassland_pct_ring3,
            pasture_pct_ring3,
            cropland_pct_ring3,
            wetland_pct_ring3,

            water_ice_pct_ring4,
            developed_pct_ring4,
            barren_pct_ring4,
            forest_pct_ring4,
            shrub_pct_ring4,
            grassland_pct_ring4,
            pasture_pct_ring4,
            cropland_pct_ring4,
            wetland_pct_ring4,

            water_ice_pct_ring5,
            developed_pct_ring5,
            barren_pct_ring5,
            forest_pct_ring5,
            shrub_pct_ring5,
            grassland_pct_ring5,
            pasture_pct_ring5,
            cropland_pct_ring5,
            wetland_pct_ring5
        )

        WITH batch_base AS
        (
            SELECT
                *
            FROM temp_base_metrics
            WHERE grid_id BETWEEN
                    v_curr_start
                AND v_curr_end
        ),

        batch_krings AS
        (
            SELECT

                k.center_grid_id AS grid_id,
                bm_n.year,

                -- POPULATION RINGS

                SUM(
                    bm_n.local_pop_count
                ) FILTER (
                    WHERE k.k_ring = 1
                ) AS pop_count_5km_ring1,

                AVG(
                    bm_n.local_pop_density
                ) FILTER (
                    WHERE k.k_ring = 1
                ) AS pop_density_5k_ring1,


                SUM(
                    bm_n.local_pop_count
                ) FILTER (
                    WHERE k.k_ring = 2
                ) AS pop_count_5km_ring2,

                AVG(
                    bm_n.local_pop_density
                ) FILTER (
                    WHERE k.k_ring = 2
                ) AS pop_density_5k_ring2,


                SUM(
                    bm_n.local_pop_count
                ) FILTER (
                    WHERE k.k_ring = 3
                ) AS pop_count_5km_ring3,

                AVG(
                    bm_n.local_pop_density
                ) FILTER (
                    WHERE k.k_ring = 3
                ) AS pop_density_5k_ring3,


                SUM(
                    bm_n.local_pop_count
                ) FILTER (
                    WHERE k.k_ring = 4
                ) AS pop_count_5km_ring4,

                AVG(
                    bm_n.local_pop_density
                ) FILTER (
                    WHERE k.k_ring = 4
                ) AS pop_density_5k_ring4,


                SUM(
                    bm_n.local_pop_count
                ) FILTER (
                    WHERE k.k_ring = 5
                ) AS pop_count_5km_ring5,

                AVG(
                    bm_n.local_pop_density
                ) FILTER (
                    WHERE k.k_ring = 5
                ) AS pop_density_5k_ring5,


                -- LAND COVER RING 1

                AVG(bm_n.water_ice_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS water_ice_pct_ring1,

                AVG(bm_n.developed_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS developed_pct_ring1,

                AVG(bm_n.barren_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS barren_pct_ring1,

                AVG(bm_n.forest_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS forest_pct_ring1,

                AVG(bm_n.shrub_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS shrub_pct_ring1,

                AVG(bm_n.grassland_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS grassland_pct_ring1,

                AVG(bm_n.pasture_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS pasture_pct_ring1,

                AVG(bm_n.cropland_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS cropland_pct_ring1,

                AVG(bm_n.wetland_pct_local)
                    FILTER (WHERE k.k_ring = 1)
                    AS wetland_pct_ring1,


                -- LAND COVER RING 2

                AVG(bm_n.water_ice_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS water_ice_pct_ring2,

                AVG(bm_n.developed_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS developed_pct_ring2,

                AVG(bm_n.barren_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS barren_pct_ring2,

                AVG(bm_n.forest_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS forest_pct_ring2,

                AVG(bm_n.shrub_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS shrub_pct_ring2,

                AVG(bm_n.grassland_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS grassland_pct_ring2,

                AVG(bm_n.pasture_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS pasture_pct_ring2,

                AVG(bm_n.cropland_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS cropland_pct_ring2,

                AVG(bm_n.wetland_pct_local)
                    FILTER (WHERE k.k_ring = 2)
                    AS wetland_pct_ring2,


                -- LAND COVER RING 3

                AVG(bm_n.water_ice_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS water_ice_pct_ring3,

                AVG(bm_n.developed_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS developed_pct_ring3,

                AVG(bm_n.barren_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS barren_pct_ring3,

                AVG(bm_n.forest_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS forest_pct_ring3,

                AVG(bm_n.shrub_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS shrub_pct_ring3,

                AVG(bm_n.grassland_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS grassland_pct_ring3,

                AVG(bm_n.pasture_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS pasture_pct_ring3,

                AVG(bm_n.cropland_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS cropland_pct_ring3,

                AVG(bm_n.wetland_pct_local)
                    FILTER (WHERE k.k_ring = 3)
                    AS wetland_pct_ring3,



                -- LAND COVER RING 4

                AVG(bm_n.water_ice_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS water_ice_pct_ring4,

                AVG(bm_n.developed_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS developed_pct_ring4,

                AVG(bm_n.barren_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS barren_pct_ring4,

                AVG(bm_n.forest_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS forest_pct_ring4,

                AVG(bm_n.shrub_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS shrub_pct_ring4,

                AVG(bm_n.grassland_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS grassland_pct_ring4,

                AVG(bm_n.pasture_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS pasture_pct_ring4,

                AVG(bm_n.cropland_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS cropland_pct_ring4,

                AVG(bm_n.wetland_pct_local)
                    FILTER (WHERE k.k_ring = 4)
                    AS wetland_pct_ring4,



                -- LAND COVER RING 5=

                AVG(bm_n.water_ice_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS water_ice_pct_ring5,

                AVG(bm_n.developed_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS developed_pct_ring5,

                AVG(bm_n.barren_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS barren_pct_ring5,

                AVG(bm_n.forest_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS forest_pct_ring5,

                AVG(bm_n.shrub_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS shrub_pct_ring5,

                AVG(bm_n.grassland_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS grassland_pct_ring5,

                AVG(bm_n.pasture_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS pasture_pct_ring5,

                AVG(bm_n.cropland_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS cropland_pct_ring5,

                AVG(bm_n.wetland_pct_local)
                    FILTER (WHERE k.k_ring = 5)
                    AS wetland_pct_ring5


            FROM processed.grid_5km_krings k

            JOIN temp_base_metrics bm_n
                ON k.neighbor_grid_id = bm_n.grid_id

            WHERE k.center_grid_id BETWEEN
                    v_curr_start
                AND v_curr_end

            GROUP BY
                k.center_grid_id,
                bm_n.year
        )

        SELECT

            bm.grid_id,
            bm.year,

            bm.local_pop_count,
            bm.local_pop_density,

            bm.water_ice_pct_local,
            bm.developed_pct_local,
            bm.barren_pct_local,
            bm.forest_pct_local,
            bm.shrub_pct_local,
            bm.grassland_pct_local,
            bm.pasture_pct_local,
            bm.cropland_pct_local,
            bm.wetland_pct_local,


            -- POPULATION RINGS

            COALESCE(
                kra.pop_count_5km_ring1,
                0
            ),

            COALESCE(
                kra.pop_density_5k_ring1,
                0
            ),

            COALESCE(
                kra.pop_count_5km_ring2,
                0
            ),

            COALESCE(
                kra.pop_density_5k_ring2,
                0
            ),

            COALESCE(
                kra.pop_count_5km_ring3,
                0
            ),

            COALESCE(
                kra.pop_density_5k_ring3,
                0
            ),

            COALESCE(
                kra.pop_count_5km_ring4,
                0
            ),

            COALESCE(
                kra.pop_density_5k_ring4,
                0
            ),

            COALESCE(
                kra.pop_count_5km_ring5,
                0
            ),

            COALESCE(
                kra.pop_density_5k_ring5,
                0
            ),


            -- LAND COVER RING 1

            COALESCE(kra.water_ice_pct_ring1, 0),
            COALESCE(kra.developed_pct_ring1, 0),
            COALESCE(kra.barren_pct_ring1, 0),
            COALESCE(kra.forest_pct_ring1, 0),
            COALESCE(kra.shrub_pct_ring1, 0),
            COALESCE(kra.grassland_pct_ring1, 0),
            COALESCE(kra.pasture_pct_ring1, 0),
            COALESCE(kra.cropland_pct_ring1, 0),
            COALESCE(kra.wetland_pct_ring1, 0),

            -- LAND COVER RING 2

            COALESCE(kra.water_ice_pct_ring2, 0),
            COALESCE(kra.developed_pct_ring2, 0),
            COALESCE(kra.barren_pct_ring2, 0),
            COALESCE(kra.forest_pct_ring2, 0),
            COALESCE(kra.shrub_pct_ring2, 0),
            COALESCE(kra.grassland_pct_ring2, 0),
            COALESCE(kra.pasture_pct_ring2, 0),
            COALESCE(kra.cropland_pct_ring2, 0),
            COALESCE(kra.wetland_pct_ring2, 0),


            -- LAND COVER RING 3

            COALESCE(kra.water_ice_pct_ring3, 0),
            COALESCE(kra.developed_pct_ring3, 0),
            COALESCE(kra.barren_pct_ring3, 0),
            COALESCE(kra.forest_pct_ring3, 0),
            COALESCE(kra.shrub_pct_ring3, 0),
            COALESCE(kra.grassland_pct_ring3, 0),
            COALESCE(kra.pasture_pct_ring3, 0),
            COALESCE(kra.cropland_pct_ring3, 0),
            COALESCE(kra.wetland_pct_ring3, 0),

            -- LAND COVER RING 4

            COALESCE(kra.water_ice_pct_ring4, 0),
            COALESCE(kra.developed_pct_ring4, 0),
            COALESCE(kra.barren_pct_ring4, 0),
            COALESCE(kra.forest_pct_ring4, 0),
            COALESCE(kra.shrub_pct_ring4, 0),
            COALESCE(kra.grassland_pct_ring4, 0),
            COALESCE(kra.pasture_pct_ring4, 0),
            COALESCE(kra.cropland_pct_ring4, 0),
            COALESCE(kra.wetland_pct_ring4, 0),

            -- LAND COVER RING 5

            COALESCE(kra.water_ice_pct_ring5, 0),
            COALESCE(kra.developed_pct_ring5, 0),
            COALESCE(kra.barren_pct_ring5, 0),
            COALESCE(kra.forest_pct_ring5, 0),
            COALESCE(kra.shrub_pct_ring5, 0),
            COALESCE(kra.grassland_pct_ring5, 0),
            COALESCE(kra.pasture_pct_ring5, 0),
            COALESCE(kra.cropland_pct_ring5, 0),
            COALESCE(kra.wetland_pct_ring5, 0)

        FROM batch_base bm

        LEFT JOIN batch_krings kra
            ON bm.grid_id = kra.grid_id
           AND bm.year = kra.year;



        -- BATCH PROGRESS
        v_processed_grids :=
            v_processed_grids + v_batch_size;

        RAISE NOTICE
            '[%] Processed grid IDs % to % | Batch: % | Total: %',
            clock_timestamp() - t_start,
            v_curr_start,
            LEAST(v_curr_end, v_max_grid),
            clock_timestamp() - t_step,
            clock_timestamp() - t_start;

        v_curr_start :=
            v_curr_end + 1;

    END LOOP;

    -- STEP 4: FINAL INDEX
    t_step := clock_timestamp();

    RAISE NOTICE '[%] Creating final index...',
        clock_timestamp() - t_start;

    CREATE UNIQUE INDEX idx_annual_features_grid_year
        ON processed.annual_grid_features (
            grid_id,
            year
        );

    ANALYZE processed.annual_grid_features;


    -- STEP 5: FINAL VALIDATION
    RAISE NOTICE '[%] Final row count: %',
        clock_timestamp() - t_start,
        (
            SELECT COUNT(*)
            FROM processed.annual_grid_features
        );


    RAISE NOTICE '[%] Nonzero local population rows: %',
        clock_timestamp() - t_start,
        (
            SELECT COUNT(*)
            FROM processed.annual_grid_features
            WHERE pop_count_5km_local > 0
        );


    RAISE NOTICE '[%] Nonzero local developed rows: %',
        clock_timestamp() - t_start,
        (
            SELECT COUNT(*)
            FROM processed.annual_grid_features
            WHERE developed_pct_local > 0
        );


    RAISE NOTICE '[%] Nonzero local forest rows: %',
        clock_timestamp() - t_start,
        (
            SELECT COUNT(*)
            FROM processed.annual_grid_features
            WHERE forest_pct_local > 0
        );


    RAISE NOTICE '[%] ALL STEPS COMPLETE. Total elapsed: %',
        clock_timestamp() - t_start,
        clock_timestamp() - t_start;

END $$;