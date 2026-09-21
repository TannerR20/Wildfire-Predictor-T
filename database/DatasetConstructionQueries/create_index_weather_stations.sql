CREATE INDEX IF NOT EXISTS weather_stations_geom5070_idx 
ON raw.weather_stations USING gist (geometry_5070);