ALTER TABLE model.train_master
ADD COLUMN day_of_year_sin double precision,
ADD COLUMN day_of_year_cos double precision;

UPDATE model.train_master
SET
    day_of_year_sin = SIN(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25),
    day_of_year_cos = COS(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25);

ALTER TABLE model.validation_master
ADD COLUMN day_of_year_sin double precision,
ADD COLUMN day_of_year_cos double precision;

UPDATE model.validation_master
SET
    day_of_year_sin = SIN(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25),
    day_of_year_cos = COS(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25);

ALTER TABLE model.test_master
ADD COLUMN day_of_year_sin double precision,
ADD COLUMN day_of_year_cos double precision;

UPDATE model.test_master
SET
    day_of_year_sin = SIN(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25),
    day_of_year_cos = COS(2 * PI() * EXTRACT(DOY FROM target_date) / 365.25);