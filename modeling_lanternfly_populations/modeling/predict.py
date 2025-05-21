from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from modeling_lanternfly_populations.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from modeling_lanternfly_populations.utils import loopy_belief_propagation, _load_pickle, _load_txt
app = typer.Typer()

def map_expected_bin_value():
    """Translates binned continuous values back to a continuous value 
    of the same scale as the input.
    
    Get predicted model value by calculating the expected value of the 
    belief w.r.t. the bin center. """
    return None

@app.command()
def get_inferred_bins(
    model_dir: Path = MODELS_DIR,
    grid_dir: Path = INTERIM_DATA_DIR,
    grid_size: float = 0.5,
    max_iterations: int = 1000,
    tolerance: float = 1e-5
):
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    #Load data grid dataframe

    logger.info(f"Performing model inference with loopy belief propataion, for grid size {grid_size}.")
    lf_gdf = gpd.read_parquet(grid_dir / f'{grid_size}_deg_lanternfly.parquet')

    #Load all parameters for lbp
    mrf_dir = Path(model_dir / f'{grid_size}_mrf/')

    domain_size = int(_load_txt(mrf_dir / f'{grid_size}_domain_size.txt'))
    n_bins = int(_load_txt(mrf_dir / f'{grid_size}_n_bin.txt'))
    neighbors = _load_pickle(mrf_dir / f'{grid_size}_neighbors.pkl')
    nodes = _load_pickle(mrf_dir / f'{grid_size}_nodes.pkl')
    pairwise_potentials = _load_pickle(mrf_dir / f'{grid_size}_potentials.pkl')
    unary_factors = _load_pickle(mrf_dir / f'{grid_size}_unary.pkl')

    #Generate beliefs with loopy bp with message passing
    beliefs = loopy_belief_propagation(
        nodes,
        neighbors,
        unary_factors,
        pairwise_potentials,
        domain_size,
        max_iters=max_iterations,
        tol=tolerance
    )

    #Generate inferred bins and append to original dataframe
    inferred_bins = {u: np.argmax(beliefs[u]) for u in nodes}
    lf_gdf["inferred_bin"] = lf_gdf["region_id"].astype(str).map(inferred_bins)

    #Map back to expected population value given bin centers
    counts = lf_gdf["observation_count"]
    non_na = counts.dropna()

    # pd.qcut returns both labels and the bin edges
    _, bin_edges = pd.qcut(non_na, q=n_bins, retbins=True, duplicates="drop")

    # make sure edges cover the full range
    bin_edges[0]   = non_na.min()
    bin_edges[-1]  = non_na.max()

    # compute bin‐centers for the smoothing potential
    centers     = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    expected_count = {
        u: np.dot(beliefs[u], centers)
        for u in beliefs
    }

    #append to bins gdf
    lf_gdf["expected_count"] = (
    lf_gdf["region_id"]
      .astype(str)
      .map(expected_count)
      .fillna(0)            # if some cells were unobserved/clamped
    )

    #save to models dir
    save_dir = mrf_dir / f'mrf_{grid_size}_predicted_counts.parquet'
    lf_gdf.to_parquet(save_dir)

    logger.success(f"Saved model outputs to {save_dir}")

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Performing inference for model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Inference complete.")
#     # -----------------------------------------


if __name__ == "__main__":
    app()
