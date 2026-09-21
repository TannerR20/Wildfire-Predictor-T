SELECT
    (SELECT COUNT(*)
     FROM processed.fires_filtered) AS filtered_fires,

    (SELECT COUNT(*)
     FROM processed.fires_assigned) AS assigned_fires;