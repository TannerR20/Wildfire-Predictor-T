-- Removes fires with size less than or equal to 0.5 acres and ones of categories that do not distinguish fire risk.

DROP TABLE IF EXISTS processed.fires_filtered;

CREATE TABLE processed.fires_filtered AS
SELECT *
FROM raw.fires
WHERE "FIRE_SIZE" > 0.5
  AND "NWCG_GENERAL_CAUSE" NOT IN (
      'Recreation and ceremony',
      'Debris and open burning',
      'Missing data/not specified/undetermined'
  );