"""
Process Historical Annual NLCD land-cover features for 
the wildfire-risk prediction training set.

Source:
    MRLC Annual NLCD CONUS Collection 1.2

Important:
    This script loops through all available historical years,
    processes the LndCov rasters, and appends them to a 
    historical table for ML feature joins.

Output:
    processed.historical_landcover

Required packages:
    pip install requests rasterio geopandas shapely psycopg2-binary pandas numpy tqdm
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import box
from tqdm import tqdm
import psycopg2


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "P1ttsburgh$",
}

GRID_TABLE = "processed.grid"

GRID_ID_COLUMN = "grid_id"

GRID_GEOMETRY_COLUMN = "centroid_5070"

OUTPUT_TABLE = "processed.historical_landcover"

# ------------------------------------------------------------
# HISTORICAL YEARS TO PROCESS
# ------------------------------------------------------------
# The Annual NLCD releases contain data spanning back to 1985.
# Adjust this list based on your wildfire target date range.
YEARS_TO_PROCESS = list(range(1992, 2026))

# The filename convention documented by MRLC for Annual NLCD:
# Annual_NLCD_LndCov_{year}_CU_C1V2.tif
NLCD_URL_TEMPLATE = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/"
    "data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
)

DATA_DIR = Path("nlcd_historical")


# ------------------------------------------------------------
# LANDCOVER GROUPS
# ------------------------------------------------------------
LANDCOVER_GROUPS = {
    "water_ice_pct": [11, 12],
    "developed_pct": [21, 22, 23, 24],
    "barren_pct": [31],
    "forest_pct": [41, 42, 43],
    "shrub_pct": [52],
    "grassland_pct": [71],
    "pasture_pct": [81],
    "cropland_pct": [82],
    "wetland_pct": [90, 95],
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# DOWNLOAD NLCD
# ============================================================

def download_nlcd(
    year: int,
) -> Path:
    """
    Download the Annual NLCD Land Cover bundle for a given year.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_filename = f"Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
    zip_path = DATA_DIR / zip_filename
    url = NLCD_URL_TEMPLATE.format(year=year)

    if zip_path.exists():

        logger.info(
            f"NLCD ZIP for {year} already exists: {zip_path}"
        )

        return zip_path

    logger.info(
        f"Downloading Annual NLCD Land Cover for {year}..."
    )

    response = requests.get(
        url,
        stream=True,
        timeout=300,
    )

    response.raise_for_status()

    total = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    downloaded = 0

    with open(
        zip_path,
        "wb",
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            f.write(chunk)

            downloaded += len(chunk)

            if total:

                logger.info(
                    f"[{year}] Downloaded "
                    f"{downloaded / 1024 / 1024:.1f} MB "
                    f"of "
                    f"{total / 1024 / 1024:.1f} MB"
                )

    logger.info(
        f"NLCD download complete for {year}: {zip_path}"
    )

    return zip_path


# ============================================================
# EXTRACT NLCD
# ============================================================

def extract_nlcd(
    zip_path: Path,
    year: int,
) -> Path:
    """
    Extract the NLCD bundle for the specific year.
    """
    
    extract_dir = DATA_DIR / f"extracted_{year}"
    
    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tif_files = list(
        extract_dir.rglob("*.tif")
    )

    lndcov_files = [
        p
        for p in tif_files
        if "LndCov" in p.name
    ]

    if lndcov_files:

        logger.info(
            f"Using existing NLCD raster for {year}: "
            f"{lndcov_files[0]}"
        )

        return lndcov_files[0]

    logger.info(
        f"Extracting NLCD ZIP for {year}..."
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as z:

        z.extractall(
            extract_dir
        )

    tif_files = list(
        extract_dir.rglob("*.tif")
    )

    lndcov_files = [
        p
        for p in tif_files
        if "LndCov" in p.name
    ]

    if not lndcov_files:

        raise RuntimeError(
            f"Could not find a Land Cover GeoTIFF for {year}."
        )

    if len(lndcov_files) > 1:

        logger.info(
            f"Found {len(lndcov_files)} tiles. Mosaicking..."
        )
        
        return create_mosaic(
            lndcov_files,
            extract_dir,
            year,
        )

    return lndcov_files[0]


# ============================================================
# MOSAIC TILED RASTERS
# ============================================================

def create_mosaic(
    raster_paths: list[Path],
    extract_dir: Path,
    year: int,
) -> Path:
    """
    Mosaic tiled LndCov rasters into one GeoTIFF.
    """

    from rasterio.merge import merge

    mosaic_path = (
        extract_dir /
        f"Annual_NLCD_LndCov_{year}_CONUS_mosaic.tif"
    )

    if mosaic_path.exists():
        return mosaic_path

    src_files = [
        rasterio.open(p)
        for p in raster_paths
    ]

    try:

        mosaic, transform = merge(
            src_files
        )

        profile = src_files[0].profile.copy()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw",
            }
        )

        with rasterio.open(
            mosaic_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                mosaic
            )

    finally:

        for src in src_files:
            src.close()

    return mosaic_path


# ============================================================
# LOAD GRID
# ============================================================

def load_grid() -> gpd.GeoDataFrame:
    """
    Load the grid centroids.
    """

    logger.info(
        "Loading prediction grid..."
    )

    sql = f"""
        SELECT
            {GRID_ID_COLUMN} AS grid_id,
            {GRID_GEOMETRY_COLUMN} AS geometry
        FROM {GRID_TABLE}
    """

    conn = get_connection()

    try:

        grid = gpd.read_postgis(
            sql,
            conn,
            geom_col="geometry",
        )

    finally:

        conn.close()

    grid = grid.set_crs(
        5070,
        allow_override=True,
    )

    return grid


# ============================================================
# CALCULATE 5 KM CELL BOUNDARIES
# ============================================================

def create_grid_polygons(
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Convert each grid centroid into a 5 km x 5 km square.
    """

    logger.info(
        "Creating 5 km grid-cell polygons..."
    )

    half_size = 2500.0
    polygons = []

    for geom in grid.geometry:

        x = geom.x
        y = geom.y

        polygons.append(
            box(
                x - half_size,
                y - half_size,
                x + half_size,
                y + half_size,
            )
        )

    polygons_gdf = gpd.GeoDataFrame(
        {
            "grid_id":
                grid["grid_id"].values
        },
        geometry=polygons,
        crs=5070,
    )

    return polygons_gdf


# ============================================================
# BUILD LAND-COVER FEATURES
# ============================================================

def build_landcover_features(
    grid_polygons: gpd.GeoDataFrame,
    raster_path: Path,
    year: int,
) -> pd.DataFrame:
    """
    Aggregate 30 m NLCD land cover to the 5 km prediction grid.
    """

    logger.info(
        f"Processing NLCD raster for {year}..."
    )

    results = []

    with rasterio.open(
        raster_path
    ) as src:

        if src.crs is None:
            raise RuntimeError("Raster has no CRS.")

        if grid_polygons.crs != src.crs:

            raster_grid = grid_polygons.to_crs(
                src.crs
            )

        else:

            raster_grid = grid_polygons

        for row in tqdm(
            raster_grid.itertuples(
                index=False
            ),
            total=len(raster_grid),
            desc=f"NLCD cells ({year})",
        ):

            grid_id = row.grid_id
            polygon = row.geometry

            window = from_bounds(
                polygon.bounds[0],
                polygon.bounds[1],
                polygon.bounds[2],
                polygon.bounds[3],
                transform=src.transform,
            )

            window = window.round_offsets().round_lengths()

            try:

                data = src.read(
                    1,
                    window=window,
                )

            except Exception:

                results.append(
                    {
                        "grid_id": grid_id,
                        **{name: np.nan for name in LANDCOVER_GROUPS},
                    }
                )
                continue

            nodata = src.nodata

            if nodata is not None:
                data = data[data != nodata]

            data = data[data > 0]

            if data.size == 0:

                results.append(
                    {
                        "grid_id": grid_id,
                        **{name: np.nan for name in LANDCOVER_GROUPS},
                    }
                )
                continue

            total_pixels = data.size
            result = {"grid_id": grid_id}

            for name, classes in LANDCOVER_GROUPS.items():

                count = np.isin(
                    data,
                    classes,
                ).sum()

                result[name] = (
                    count /
                    total_pixels *
                    100.0
                )

            results.append(result)

    features = pd.DataFrame(results)
    
    features["landcover_year"] = year

    return features


# ============================================================
# DATABASE SETUP & WRITE
# ============================================================

def initialize_database_table(
    df: pd.DataFrame,
    table_name: str,
):
    """
    Drop and create the historical table on the first iteration.
    """
    
    schema, table = table_name.split(".")
    
    logger.info(f"Initializing table {table_name}...")
    
    conn = get_connection()
    
    try:
        cur = conn.cursor()
        
        cur.execute(
            f'''
            DROP TABLE IF EXISTS
            "{schema}"."{table}";
            '''
        )

        columns = list(df.columns)
        definitions = []

        for column in columns:

            dtype = df[column].dtype

            if pd.api.types.is_integer_dtype(dtype):
                pg_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                pg_type = "DOUBLE PRECISION"
            elif pd.api.types.is_bool_dtype(dtype):
                pg_type = "BOOLEAN"
            else:
                pg_type = "TEXT"

            definitions.append(f'"{column}" {pg_type}')

        cur.execute(
            f'''
            CREATE TABLE
            "{schema}"."{table}" (
                {", ".join(definitions)}
            );
            '''
        )
        
        # Create an index for faster downstream joins
        cur.execute(
            f'''
            CREATE INDEX idx_{table}_grid_year 
            ON "{schema}"."{table}" (grid_id, landcover_year);
            '''
        )
        
        conn.commit()
        
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_to_database(
    df: pd.DataFrame,
    table_name: str,
):
    """
    Append dataframe rows to the PostgreSQL table.
    """

    schema, table = table_name.split(".")

    logger.info(
        f"Appending {len(df):,} rows to {table_name}..."
    )

    conn = get_connection()

    try:

        cur = conn.cursor()
        buffer = io.StringIO()

        export_df = df.copy()

        export_df = export_df.where(
            pd.notna(export_df),
            None,
        )

        export_df.to_csv(
            buffer,
            index=False,
            header=False,
            sep="\t",
            na_rep="\\N",
        )

        buffer.seek(0)
        columns = list(df.columns)

        column_sql = ", ".join(
            f'"{c}"'
            for c in columns
        )

        cur.copy_expert(
            f'''
            COPY "{schema}"."{table}"
            ({column_sql})
            FROM STDIN
            WITH (
                FORMAT CSV,
                DELIMITER E'\\t',
                NULL '\\N'
            );
            ''',
            buffer,
        )

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    logger.info(
        "============================================"
    )
    logger.info(
        "HISTORICAL NLCD LAND-COVER GENERATION"
    )
    logger.info(
        "============================================"
    )

    grid = load_grid()

    grid_polygons = create_grid_polygons(
        grid
    )
    
    is_first_year = True

    for year in YEARS_TO_PROCESS:
        
        logger.info(
            f"--- Starting processing for year: {year} ---"
        )
        
        try:
            zip_path = download_nlcd(year)
            
            raster_path = extract_nlcd(
                zip_path, 
                year
            )
            
            landcover = build_landcover_features(
                grid_polygons,
                raster_path,
                year,
            )
            
            if len(landcover) != len(grid):
                raise RuntimeError(
                    f"Mismatch in {year}. Expected {len(grid):,}; "
                    f"got {len(landcover):,}."
                )
            
            if is_first_year:
                initialize_database_table(
                    landcover,
                    OUTPUT_TABLE,
                )
                is_first_year = False
                
            append_to_database(
                landcover,
                OUTPUT_TABLE,
            )
            
            # --- CLEANUP LOGIC ---
            logger.info(f"Cleaning up files for year {year}...")
            if zip_path and zip_path.exists():
                zip_path.unlink()
                
            extract_dir = DATA_DIR / f"extracted_{year}"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
                
        except Exception as e:
            logger.error(
                f"Failed processing year {year}: {e}"
            )
            continue

    elapsed = (
        time.time() -
        start_time
    )

    logger.info(
        "============================================"
    )
    logger.info(
        "HISTORICAL NLCD UPDATE COMPLETE"
    )
    logger.info(
        f"Grid cells per year: {len(grid):,}"
    )
    logger.info(
        f"Output table: {OUTPUT_TABLE}"
    )
    logger.info(
        f"Execution time: {elapsed / 60:.1f} minutes"
    )
    logger.info(
        "============================================"
    )


if __name__ == "__main__":
    main()