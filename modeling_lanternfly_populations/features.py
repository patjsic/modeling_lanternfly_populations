import os
import re
import typer
import pickle
import numpy as np
import geopandas as gpd

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from scipy.sparse import save_npz
from shapely.geometry import Point, Polygon

from modeling_lanternfly_populations.utils import _save_pickle
from modeling_lanternfly_populations.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# MIN_LON, MIN_LAT = -124.848974, 24.396308   # Southwest corner
# MAX_LON, MAX_LAT = -66.885444, 49.384358    # Northeast corner

# SELECT ONLY NORTHEAST REGION
MIN_LON, MIN_LAT = -90.0, 35.0
MAX_LON, MAX_LAT = -70.0, 45.0


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
    # grid_gdf['observation_count'] = grid_gdf['observation_count'].fillna(0) #Not sure if we want to fillna, since it makes observations sparse
    
    #Label centroids of polygons
    grid_gdf['lon'] = grid_gdf.geometry.centroid.x
    grid_gdf['lat'] = grid_gdf.geometry.centroid.y
    grid_gdf = grid_gdf.reset_index(drop=True)       # ensures 0..n-1
    grid_gdf['region_id'] = grid_gdf.index  

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

    logger.success("Successfully saved GP data!")

@app.command()
def build_adjacency(
    data_dir: Path = INTERIM_DATA_DIR,
    adj_type: str = "queen",
    grid_size: float = 0.5,
):
    """
    Generate adjacency matrix for existing gridded lanternfly data.
    Looks for file of the form <float>_deg_lanternfly.parquet and chooses the first one.

    'Queen' adjacency means polygons that share any boundary or corner are neighbors.
    'Rook' adjacency means polygons that share boundaries, but not corners, are neighbors.
    """
    import pysal
    import json
    from pysal.lib import weights
    logger.info(f"Finding neighbors based on {adj_type} adjacency.")
    #Search for matching gridded data and load into parquet
    # template = r'^\d+\.\d+_deg_lanternfly\.parquet$'
    # data_path = data_dir / next((s for s in os.listdir(data_dir) if re.match(template, s)), '')
    data_path = data_dir / f'{grid_size}_deg_lanternfly.parquet'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No gridded lanternfly data found in path: {data_path}")
    else:
        lf_gdf = gpd.read_parquet(data_path)

    # Build a spatial weights object (e.g., Queen contiguity)
    if adj_type == 'rook':
        w = weights.Rook.from_dataframe(lf_gdf, ids='region_id')
    else:
        w = weights.Queen.from_dataframe(lf_gdf, ids='region_id')

    # Convert to a PySAL W object
    w.transform = 'B'  # Make it binary (0/1 for adjacency)
    W_sparse = w.sparse.tocsr()  

    # Extract adjacency info as needed, e.g. neighbor list
    # neighbors_dict = w.neighbors  # {region_id: [neighbor_ids]}

    # with open(data_dir / 'neighbors.json', 'w', encoding='utf-8') as file:
    #     json.dump(W_sparse, file, ensure_ascii=False, indent=4)
    save_npz(data_dir / f'{grid_size}_neighbors.npz', W_sparse)

    logger.success(f"Successfully saved adjacency data to {data_dir / f'{grid_size}_neighbors.json'}")

if __name__ == "__main__":
    app()
