import typer
import pickle
import numpy as np
import geopandas as gpd

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from shapely.geometry import Point, Polygon


from modeling_lanternfly_populations.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

MIN_LON, MIN_LAT = -124.848974, 24.396308   # Southwest corner
MAX_LON, MAX_LAT = -66.885444, 49.384358    # Northeast corner

def _load_data(
    input_path: Path 
):
    logger.info(f"Reading data from {input_path}...")
    try:
        gdf = gpd.read_parquet(input_path)
        logger.success("Loaded data succesfully!")
        return gdf
    except Exception as e:
        raise ValueError(f"Invalid parquet file found at {input_path}")
    
def _save_pickle(
    filepath: Path,
    object: object,
):
    logger.info(f"Saving pickle: {filepath}")
    with open(filepath, 'wb') as file:
        pickle.dump(object, file)

@app.command()
def grid_data(
    cell_size: float = 0.5, #degree grid,
    data_path: Path = RAW_DATA_DIR / "raw_lanternfly.parquet"
):  
    gdf = _load_data(input_path=data_path)
    lon_ticks = np.arange(MIN_LON, MAX_LON, cell_size)
    lat_ticks = np.arange(MIN_LAT, MAX_LAT, cell_size)

    polygons = []
    for x in lon_ticks:
        for y in lat_ticks:
            # Each grid cell is a small square/rectangle
            poly = Polygon([
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y + cell_size),
                (x, y + cell_size)
            ])
            polygons.append(poly)

    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')

    logger.info(f"Creating gridded data at {cell_size}-degree resolution...")
    joined = gpd.sjoin(gdf, grid_gdf, how='left', predicate='within')
    counts_per_cell = joined.groupby('index_right').size()
    grid_gdf['observation_count'] = counts_per_cell
    # grid_gdf['observation_count'] = grid_gdf['observation_count'].fillna(0)
    
    #Label centroids of polygons
    grid_gdf['lon'] = grid_gdf.geometry.centroid.x
    grid_gdf['lat'] = grid_gdf.geometry.centroid.y

    SAVE_PATH = INTERIM_DATA_DIR / f"{cell_size}_deg_lanternfly.parquet"
    grid_gdf.to_parquet(SAVE_PATH)
    logger.success(f"Data saved at {SAVE_PATH}")
    
@app.command()
def process_gp_data(
    input_path: Path = INTERIM_DATA_DIR / "0.5_deg_lanternfly.parquet",
    output_dir: Path = PROCESSED_DATA_DIR
):
    logger.info(f"Preprocessing data for GP to save in {PROCESSED_DATA_DIR}")
    from sklearn.model_selection import train_test_split

    gdf = _load_data(input_path=input_path)

    #Drop null observation counts in grid
    gdf = gdf.dropna(axis=0)

    X = gdf[['lon', 'lat']].values
    y = gdf['observation_count'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _save_pickle(output_dir / "gp_x_train.pkl", X_train)
    _save_pickle(output_dir / "gp_x_test.pkl", X_test)
    _save_pickle(output_dir / "gp_y_train.pkl", y_train)
    _save_pickle(output_dir / "gp_y_test.pkl", y_test)

    logger.info("Successfully saved GP data!")

if __name__ == "__main__":
    app()
