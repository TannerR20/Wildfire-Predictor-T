"""
Aggregate LandScan Global annual population rasters to the existing
5 km x 5 km PostgreSQL/PostGIS grid.

Input:
    C:\\Tre\\Desktop\\2026 Fall Semester\\Database systems\\WildfireProject\\data\\Population\\downloads

Expected input:
    One ZIP file per year, 2000-2024.
    Each ZIP should contain the annual LandScan Global GeoTIFF.

Existing database table:
    processed.grid_5km

Output table:
    processed.population

Output columns:
    grid_id
    population_year
    population_count
    population_density_km2

Population is aggregated by SUM because LandScan Global raster
values represent population counts per source pixel.

Requirements:
    pip install rasterio geopandas psycopg2-binary pandas numpy
"""

from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import shutil
import re
import csv
import os

os.environ.pop("PROJ_DATA", None)
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)


import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds

RASTERIO_DIR = Path(rasterio.__file__).resolve().parent
RASTERIO_PROJ_DATA = RASTERIO_DIR / "proj_data"

print("=" * 70)
print("GEOSPATIAL ENVIRONMENT")
print("=" * 70)
print("Rasterio directory:")
print(RASTERIO_DIR)
print()
print("Rasterio PROJ directory:")
print(RASTERIO_PROJ_DATA)
print()
print("PROJ database exists:")
print((RASTERIO_PROJ_DATA / "proj.db").exists())
print()

if (RASTERIO_PROJ_DATA / "proj.db").exists():
    os.environ["PROJ_DATA"] = str(RASTERIO_PROJ_DATA)
    os.environ["PROJ_LIB"] = str(RASTERIO_PROJ_DATA)
else:
    from pyproj import datadir

    os.environ["PROJ_DATA"] = datadir.get_data_dir()
    os.environ["PROJ_LIB"] = datadir.get_data_dir()

import psycopg2



DOWNLOAD_DIR = Path(
    r"C:\Tre\Desktop\2026 Fall Semester\Database systems\WildfireProject\data\Population\downloads"
)


# PostgreSQL connection

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "P1ttsburgh$",
}


GRID_TABLE = "processed.grid_5km"
GRID_ID_COLUMN = "grid_id"
GEOMETRY_COLUMN = "geometry"


OUTPUT_TABLE = "processed.population"


START_YEAR = 2000
END_YEAR = 2024

YEARS = list(range(START_YEAR, END_YEAR + 1))

# Existing grid resolution
GRID_RESOLUTION_METERS = 5000.0

# Number of rasterization/reprojection threads
NUM_THREADS = min(os.cpu_count() or 4, 8)

# Temporary processing directory
TEMP_DIR = DOWNLOAD_DIR.parent / "temp_population"


def get_connection():
    """Create PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def find_zip_files():
    """
    Find one ZIP file for each year from 2000-2024.

    The year is extracted from the ZIP filename.
    """

    zip_files = {}

    for path in DOWNLOAD_DIR.glob("*.zip"):

        match = re.search(r"(20\d{2})", path.name)

        if not match:
            continue

        year = int(match.group(1))

        if START_YEAR <= year <= END_YEAR:

            if year in zip_files:
                raise RuntimeError(
                    f"Multiple ZIP files found for {year}:\n"
                    f"  {zip_files[year]}\n"
                    f"  {path}\n\n"
                    f"Keep only one LandScan ZIP per year or modify "
                    f"find_zip_files()."
                )

            zip_files[year] = path

    missing = [
        year
        for year in YEARS
        if year not in zip_files
    ]

    if missing:
        raise RuntimeError(
            "Missing LandScan ZIP files for years:\n"
            + ", ".join(map(str, missing))
        )

    return zip_files


def find_tiff_in_zip(zip_path, year):
    """
    Find the LandScan GeoTIFF inside a ZIP.

    Preference:
      1. TIFF containing the year in filename
      2. Largest TIFF if there is only one/multiple candidates
    """

    with ZipFile(zip_path, "r") as z:

        tif_members = [
            info
            for info in z.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith((".tif", ".tiff"))
        ]

        if not tif_members:
            raise RuntimeError(
                f"No GeoTIFF found inside:\n{zip_path}"
            )

        year_matches = [
            info
            for info in tif_members
            if str(year) in Path(info.filename).name
        ]

        if len(year_matches) == 1:
            return year_matches[0].filename

        if len(year_matches) > 1:
            # Choose largest year-matching TIFF
            year_matches.sort(
                key=lambda x: x.file_size,
                reverse=True
            )
            return year_matches[0].filename

        # No filename-year match
        tif_members.sort(
            key=lambda x: x.file_size,
            reverse=True
        )

        return tif_members[0].filename


def extract_tiff(zip_path, year, temp_dir):
    """
    Extract only the selected GeoTIFF from the annual ZIP.
    """

    tiff_member = find_tiff_in_zip(zip_path, year)

    output_name = f"landscan_{year}.tif"
    output_path = temp_dir / output_name

    print(f"    TIFF inside ZIP: {tiff_member}")

    with ZipFile(zip_path, "r") as z:

        with z.open(tiff_member) as source, open(
            output_path, "wb"
        ) as destination:

            shutil.copyfileobj(source, destination)

    return output_path


# LOAD EXISTING 5 KM GRID
def load_grid(conn):
    """
    Load the existing processed.grid_5km table.

    This grid determines the exact:
      - spatial extent
      - CRS
      - 5 km alignment
      - grid IDs
    """

    sql = f"""
        SELECT
            {GRID_ID_COLUMN},
            {GEOMETRY_COLUMN}
        FROM {GRID_TABLE}
        ORDER BY {GRID_ID_COLUMN};
    """

    print("Loading existing 5 km grid...")

    gdf = gpd.read_postgis(
        sql,
        conn,
        geom_col=GEOMETRY_COLUMN
    )

    if gdf.empty:
        raise RuntimeError(
            f"{GRID_TABLE} is empty."
        )

    if gdf.crs is None:
        raise RuntimeError(
            f"{GRID_TABLE} has no CRS."
        )

    print(f"    Grid cells: {len(gdf):,}")
    print(f"    CRS: {gdf.crs}")

    # Check geometry dimensions
    bounds = gdf.total_bounds

    min_x, min_y, max_x, max_y = bounds

    width_float = (
        (max_x - min_x)
        / GRID_RESOLUTION_METERS
    )

    height_float = (
        (max_y - min_y)
        / GRID_RESOLUTION_METERS
    )

    width = int(round(width_float))
    height = int(round(height_float))

    if not np.isclose(
        width_float,
        width,
        atol=1e-6
    ):
        raise RuntimeError(
            f"Grid width is not aligned to 5 km: {width_float}"
        )

    if not np.isclose(
        height_float,
        height,
        atol=1e-6
    ):
        raise RuntimeError(
            f"Grid height is not aligned to 5 km: {height_float}"
        )

    print(f"    Raster width:  {width}")
    print(f"    Raster height: {height}")
    print(f"    Bounds: {bounds}")

    return (
        gdf,
        min_x,
        min_y,
        max_x,
        max_y,
        width,
        height,
    )



def create_grid_id_raster(
    grid_gdf,
    width,
    height,
    min_x,
    max_y,
):
    """
    Convert the existing polygon grid into a raster where
    each target 5 km cell contains its grid_id.

    Cells outside the existing grid remain 0.
    """

    print("Rasterizing grid IDs...")

    # Make sure grid IDs can safely fit inside int32
    max_grid_id = grid_gdf[GRID_ID_COLUMN].max()

    if max_grid_id > np.iinfo(np.int32).max:
        raise RuntimeError(
            "grid_id is too large for int32 rasterization."
        )

    target_transform = from_origin(
        min_x,
        max_y,
        GRID_RESOLUTION_METERS,
        GRID_RESOLUTION_METERS,
    )

    shapes = (
        (
            geom,
            int(grid_id)
        )
        for grid_id, geom
        in zip(
            grid_gdf[GRID_ID_COLUMN],
            grid_gdf[GEOMETRY_COLUMN]
        )
    )

    grid_id_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=target_transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    populated_cells = np.count_nonzero(
        grid_id_raster
    )

    print(
        f"    Rasterized cells: {populated_cells:,}"
    )

    expected_cells = len(grid_gdf)

    if populated_cells != expected_cells:
        raise RuntimeError(
            f"Rasterization mismatch.\n"
            f"Expected cells: {expected_cells:,}\n"
            f"Rasterized cells: {populated_cells:,}"
        )

    return grid_id_raster, target_transform



def create_output_table(conn):
    """
    Create output table if it does not exist.
    """

    sql = f"""
        CREATE SCHEMA IF NOT EXISTS processed;

        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            grid_id INTEGER NOT NULL,
            population_year INTEGER NOT NULL,
            population_count DOUBLE PRECISION NOT NULL,
            population_density_km2 DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (grid_id, population_year)
        );
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()



def process_year(
    conn,
    year,
    zip_path,
    grid_id_raster,
    target_transform,
    width,
    height,
    min_x,
    min_y,
    max_x,
    max_y,
    temp_dir,
):
    """
    Process one LandScan annual raster.

    Steps:
      1. Extract TIFF from ZIP
      2. Read only the source area covering the 5 km grid
      3. Reproject/aggregate using SUM
      4. Match values to grid_id raster
      5. Load into PostgreSQL
    """

    print()
    print("=" * 70)
    print(f"PROCESSING YEAR {year}")
    print("=" * 70)



    tiff_path = extract_tiff(
        zip_path,
        year,
        temp_dir
    )

    print(f"    Extracted TIFF: {tiff_path}")



    with rasterio.open(tiff_path) as src:

        print(f"    Source CRS: {src.crs}")
        print(
            f"    Source dimensions: "
            f"{src.width:,} x {src.height:,}"
        )
        print(f"    Source nodata: {src.nodata}")
        print(f"    Source dtype: {src.dtypes[0]}")
        print(f"    Source bounds: {src.bounds}")



        source_bounds = transform_bounds(
            grid_crs,
            src.crs,
            min_x,
            min_y,
            max_x,
            max_y,
            densify_pts=21,
        )

        (
            src_min_x,
            src_min_y,
            src_max_x,
            src_max_y,
        ) = source_bounds

        print(
            "    Required source bounds:"
        )
        print(
            f"      {source_bounds}"
        )


        from rasterio.windows import from_bounds

        source_window = from_bounds(
            src_min_x,
            src_min_y,
            src_max_x,
            src_max_y,
            src.transform,
        )

        source_window = source_window.round_offsets().round_lengths()

        # Prevent window from extending outside raster
        full_window = rasterio.windows.Window(
            0,
            0,
            src.width,
            src.height,
        )

        source_window = source_window.intersection(
            full_window
        )

        print(
            f"    Source window: "
            f"{source_window.width:.0f} x "
            f"{source_window.height:.0f}"
        )



        population = src.read(
            1,
            window=source_window,
            masked=True,
        )

        source_transform = src.window_transform(
            source_window
        )

        # Convert masked pixels to NaN
        population_array = population.astype(
            np.float64
	).filled(np.nan)


        target_population = np.zeros(
            (height, width),
            dtype=np.float64,
        )

        print(
            "    Reprojecting and aggregating "
            "population to 5 km grid..."
        )

        reproject(
            source=population_array,
            destination=target_population,
            src_transform=source_transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=grid_crs,
            src_nodata=np.nan,
            dst_nodata=0.0,
            resampling=Resampling.sum,
            num_threads=NUM_THREADS,
        )


    valid_mask = grid_id_raster != 0

    grid_ids = grid_id_raster[valid_mask]
    population_values = target_population[valid_mask]

    # Negative values should not occur
    population_values = np.nan_to_num(
        population_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    population_values = np.maximum(
        population_values,
        0.0
    )


    result = pd.DataFrame(
        {
            "grid_id": grid_ids.astype(np.int32),
            "population_year": year,
            "population_count": population_values,
        }
    )

    # Grid cells are 5 km x 5 km = 25 km²
    result["population_density_km2"] = (
        result["population_count"] / 25.0
    )

    print(
        f"    Result rows: {len(result):,}"
    )

    print(
        f"    Total estimated population in grid: "
        f"{result['population_count'].sum():,.2f}"
    )

    print(
        f"    Minimum population/cell: "
        f"{result['population_count'].min():,.4f}"
    )

    print(
        f"    Maximum population/cell: "
        f"{result['population_count'].max():,.4f}"
    )



    csv_path = temp_dir / f"population_{year}.csv"

    result.to_csv(
        csv_path,
        index=False,
        float_format="%.10f",
    )



    with conn.cursor() as cur:

        cur.execute(
            f"""
            DELETE FROM {OUTPUT_TABLE}
            WHERE population_year = %s;
            """,
            (year,),
        )

        with open(
            csv_path,
            "r",
            encoding="utf-8",
        ) as f:

            next(f)  # Skip header

            cur.copy_expert(
                f"""
                COPY {OUTPUT_TABLE}
                (
                    grid_id,
                    population_year,
                    population_count,
                    population_density_km2
                )
                FROM STDIN
                WITH (
                    FORMAT CSV
                );
                """,
                f,
            )

    conn.commit()



    csv_path.unlink(
        missing_ok=True
    )

    tiff_path.unlink(
        missing_ok=True
    )

    print(
        f"    Loaded {len(result):,} rows into "
        f"{OUTPUT_TABLE}"
    )




def create_indexes(conn):

    print()
    print("Creating indexes...")

    sql = f"""
        CREATE INDEX IF NOT EXISTS
        population_grid_year_idx
        ON {OUTPUT_TABLE}
        (grid_id, population_year);

        CREATE INDEX IF NOT EXISTS
        population_year_idx
        ON {OUTPUT_TABLE}
        (population_year);

        ANALYZE {OUTPUT_TABLE};
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()

    print("    Indexes created.")




def validate_results(conn):

    print()
    print("=" * 70)
    print("VALIDATING OUTPUT")
    print("=" * 70)

    with conn.cursor() as cur:

        # Row count
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {OUTPUT_TABLE};
            """
        )

        total_rows = cur.fetchone()[0]

        print(
            f"Total rows: {total_rows:,}"
        )

        # Distinct cells
        cur.execute(
            f"""
            SELECT COUNT(DISTINCT grid_id)
            FROM {OUTPUT_TABLE};
            """
        )

        distinct_cells = cur.fetchone()[0]

        print(
            f"Distinct grid cells: {distinct_cells:,}"
        )

        # Distinct years
        cur.execute(
            f"""
            SELECT
                MIN(population_year),
                MAX(population_year),
                COUNT(DISTINCT population_year)
            FROM {OUTPUT_TABLE};
            """
        )

        min_year, max_year, year_count = cur.fetchone()

        print(
            f"Years: {min_year}-{max_year}"
        )

        print(
            f"Number of years: {year_count}"
        )

        # Rows by year
        cur.execute(
            f"""
            SELECT
                population_year,
                COUNT(*) AS grid_count,
                SUM(population_count) AS total_population
            FROM {OUTPUT_TABLE}
            GROUP BY population_year
            ORDER BY population_year;
            """
        )

        rows = cur.fetchall()

    print()
    print(
        "Population summary by year:"
    )

    for (
        year,
        grid_count,
        total_population,
    ) in rows:

        print(
            f"  {year}: "
            f"{grid_count:,} cells, "
            f"{total_population:,.2f} population"
        )

    # Expected number of rows
    expected_rows = (
        distinct_cells * len(YEARS)
    )

    if total_rows != expected_rows:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected approximately "
            f"{expected_rows:,} rows."
        )

        print(
            f"Found {total_rows:,} rows."
        )

    else:

        print()
        print(
            "Row-count validation PASSED."
        )




if __name__ == "__main__":

    print("=" * 70)
    print("LANDSCAN GLOBAL -> EXISTING 5 KM GRID")
    print("=" * 70)



    if not DOWNLOAD_DIR.exists():
        raise RuntimeError(
            f"Download directory does not exist:\n"
            f"{DOWNLOAD_DIR}"
        )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    print()
    print("Searching for LandScan ZIP files...")

    zip_files = find_zip_files()

    for year in YEARS:
        print(
            f"  {year}: {zip_files[year].name}"
        )



    print()
    print("Connecting to PostgreSQL...")

    conn = get_connection()

    try:



        (
            grid_gdf,
            min_x,
            min_y,
            max_x,
            max_y,
            width,
            height,
        ) = load_grid(conn)


        global grid_crs
        grid_crs = grid_gdf.crs


        (
            grid_id_raster,
            target_transform,
        ) = create_grid_id_raster(
            grid_gdf,
            width,
            height,
            min_x,
            max_y,
        )

        print()
        print(
            f"Creating output table "
            f"{OUTPUT_TABLE}..."
        )

        create_output_table(conn)


        for year in YEARS:

            process_year(
                conn=conn,
                year=year,
                zip_path=zip_files[year],
                grid_id_raster=grid_id_raster,
                target_transform=target_transform,
                width=width,
                height=height,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                temp_dir=TEMP_DIR,
            )


        create_indexes(conn)


        validate_results(conn)

    finally:

        conn.close()

        # Remove temporary directory
        if TEMP_DIR.exists():
            shutil.rmtree(
                TEMP_DIR,
                ignore_errors=True
            )

    print()
    print("=" * 70)
    print("LANDSCAN PROCESSING COMPLETE")
    print("=" * 70)