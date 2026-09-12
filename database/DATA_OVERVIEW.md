NOTE: These are not the complete datasets but onlys subsets (20000 rows) as githb limits to 25 MB uploads.

raw:
 - Fires: Raw data from https://research.fs.usda.gov/firelab/products/dataandtools/us-wildfires
 - states: Contains geographic boundary coordinates and administrative data for U.S. states and territories. It was used to select only the lower 48 states.
 - weather_at_stations: The weather at specific weather stations
 - landcover: the raw landcover data (this data is mapped to grid_id so isn't really raw)


Processed:
- us_boundary: For the contiguous U.S.
- grid: The us 5km grid of the U.S using the us_boundary dataset
- Drought_weekly: The raw data used a different cordinate system (ESRI) which was converted to the grid using PostGIS and extracting the values at the exact center of each 5 km cell.
- Current_drought: Created historical drought features
- land_cover: The raw data orriginally came from https://www.mrlc.gov/data/project/annual-nlcd. This website contained landcover data for every year in a grid containing 30m x 30m cells. I only downloaded every 4th year and used the most recent available data (due to the size of the datasets).  I converted thir columns into more model friendly columns.  I converted the 30-meter pixels within each 5 km cell into percentage-area coverage across nine consolidated categories.
- weather_station_grid*: map of grid_ids to nearest weather station.  This dataset is not used because it results in a lot of missing data.
- weather_complete_features: Includes max min temp, historical percipitation, 7 day max min and # of dry days.
- grdid_nearest_20_stations: map of grid_ids to nearest weather stations with complete data.  Looked at the nearest 20 weather stations for each cell and used the closest one with completed weather.  There are a handfull of missing values (~36/6000000), this can be fixed either by increasing the number of stations that are looked at.
- Fire_grid: Mapped the fires to their grid Note: this dataset currently has fires that were controlled/cerimonial burns, this will be updated

Model:
- Current_fire_history: added historical fire history features
- Current_drought: Created historical drought features
- train_positive: Used to get all positive values in the train dataset (Jan 1 2000 to Oct 13, 2016)
- train_negative: Randomly selective negative values.  X * num_of positives
- train_master: Final training set
- _positive: Used to get all positive values in the train dataset (Oct 14 2000 to Jul 31, 2021)
- validation_negative: Randomly selective negative values.  X * num_of positives 
- validation_master: Final validation set
- Prediction_view: Generates full database (all dates/cells).

Upcoming changes:
- Limit to uncontrolled fires.
- Tables for different spacial scales 10km, 25km, 50km.  This columns will be added to the model.
- Create test set.
