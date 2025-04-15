import typer
import pandas as pd
import geopandas as gpd

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from shapely.geometry import Point, Polygon

from modeling_lanternfly_populations.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from pyinaturalist import get_observations

app = typer.Typer()

@app.command()
def load_observations(
    output_path: Path = RAW_DATA_DIR / "raw_lanternfly.parquet",
):
    #Hardcoded variables to find lanternfly species
    MIN_LON, MIN_LAT = -124.848974, 24.396308   # Southwest corner
    MAX_LON, MAX_LAT = -66.885444, 49.384358    # Northeast corner
    taxon_name = 'Lycorma delicatula'
    per_page = 200
    n_pages=50

    all_results = []
    logger.info("Retrieving iNaturalist observations...")
    for page_num in tqdm(range(1, n_pages + 1), total=n_pages):
        resp = get_observations(
            taxon_name=taxon_name,
            per_page=per_page,
            page=page_num,
            swlat=MIN_LAT,   # 24.396308
            swlng=MIN_LON,   # -124.848974
            nelat=MAX_LAT,   # 49.384358
            nelng=MAX_LON,   # -66.885444
        )
        results = resp.get('results', [])
        if not results:
            logger.info(f"Reached end of page {page_num}")
            # If there are no results, we've likely reached the last page
            break
        all_results.extend(results)
    
    # Create lists of lat/lon coordinates
    lats = []
    lons = []
    obs_ids = []
    obs_dates = []

    logger.info("Creating geodataframe...")
    for obs in all_results:
        if 'geojson' in obs and obs['geojson']:
            coords = obs['geojson']['coordinates']  # [lon, lat]
            lons.append(coords[0])
            lats.append(coords[1])
            obs_ids.append(obs['id'])
            obs_dates.append(obs.get('observed_on', None))
        else:
            continue

    # Construct a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            'observation_id': obs_ids,
            'observed_date': obs_dates,
            'longitude': lons,
            'latitude': lats
        },
        geometry=[Point(xy) for xy in zip(lons, lats)],
        crs='EPSG:4326'  # WGS84 lat/lon
    )
    gdf['observed_date'] = pd.to_datetime(gdf['observed_date'], utc=True)
    print(gdf)
    gdf.to_parquet(output_path)
    logger.success(f"Loading dataset complete. Saved to {output_path}")

@app.command()
def test():
    logger.success("You ran the test!")

if __name__ == "__main__":
    app()
