import sys

sys.path.append('..')

import os
import json
import pickle
import numpy as np
import geopandas as gpd
# import pymc as pm

import pandas as pd
from pathlib import Path

from loguru import logger
# from tqdm import tqdm
# from utils import load_json_file
import typer

from modeling_lanternfly_populations.utils import _save_pickle, _save_txt, loopy_belief_propagation
from modeling_lanternfly_populations.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression
from UQpy.utilities.kernels.euclidean_kernels import RBF

app = typer.Typer()

def save_mn_lbp_data(nodes: object, neighbors: dict, unary_factors: dict, potential_factors: dict, domain_size: int, grid_size: float=0.5):
    """Save Markov Network metadata to load for loopy belief propagation.
    """
    try:
        mrf_dir = Path(MODELS_DIR / f'{grid_size}_mrf/')

        #Save nodes as pkl
        _save_pickle(mrf_dir / f'{grid_size}_nodes.pkl', nodes)

        #Save neighbors as pkl
        _save_pickle(mrf_dir / f'{grid_size}_neighbors.pkl', neighbors)

        #Save potentials as pkl
        _save_pickle(mrf_dir / f'{grid_size}_unary.pkl', unary_factors)
        _save_pickle(mrf_dir / f'{grid_size}_potentials.pkl', potential_factors)

        #Save Remaining metadata
        #TODO: Eventually make this a general **args situation with an input dictionary
        #where we can save any additional metadata as \n delimited list (saved to .txt of course)
        _save_txt(mrf_dir / f'{grid_size}_domain_size.txt', str(domain_size))
        return True
    except:
        return False

def load_json_file(file_path):
    """
    Loads JSON data from a file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {file_path}")
        return None

@app.command()
def train_gp(
    input_data: Path = PROCESSED_DATA_DIR
):
    from sklearn.metrics import r2_score
    GP_PREDS = MODELS_DIR / 'gp' / 'gp_preds.pkl'
    GP_STDS = MODELS_DIR / 'gp' / 'gp_stds.pkl'
    GP_MODEL = MODELS_DIR / 'gp' / 'gp_model.pkl'

    expected_filenames = ['gp_x_train.pkl', 'gp_y_train.pkl', 'gp_x_test.pkl', 'gp_y_test.pkl']
    data_dict = {}

    logger.info(f"Loading gp data from {input_data}...")
    for filename in expected_filenames:
        if os.path.exists(input_data / filename):
            var_name = os.path.splitext(filename)[0]
            file_path = input_data / filename

            with open(file_path, "rb") as f:
                #Create variable named var_name with pickle contents
                data_dict[var_name] = pickle.load(f)
        else:
            raise ValueError(f"Expected file {filename} not found at {file_path}")
    
    logger.success("Data loaded!")

    logger.info("Training model...")
    
    #Train a GP with RBF Kernel
    kernel = RBF(l_scale=1.0)
    gpr = GaussianProcessRegression(kernel=kernel, hyperparameters=[1, 1, 1])
    gpr.fit(data_dict['gp_x_train'], data_dict['gp_y_train'])

    logger.success("Model fit!")
    logger.info("Evaluating model...")

    y_pred, y_std = gpr.predict(data_dict['gp_x_test'], return_std=True)
    mse = np.mean((data_dict['gp_y_test'] - y_pred)**2)
    r2 = r2_score(data_dict['gp_y_test'], y_pred)

    logger.info(f"MSE: {mse}")
    logger.info(f"R2: {r2}")

    _save_pickle(GP_PREDS, y_pred)
    _save_pickle(GP_STDS, y_std)
    _save_pickle(GP_MODEL, gpr)

    logger.success(f"Model successfully saved at {GP_MODEL}")

@app.command()
def train_spatial_mrf(
    output_dir: Path = MODELS_DIR / 'mrf',
    data_dir: Path = INTERIM_DATA_DIR,
    grid_size: float = 0.5,
    n_bins: int = 15,
    alpha: float = 0.1,
):
    """
    Currently this function breaks the modular paradigm of the codebase. It performs training
    and eval in the same step, primarily since I'm not sure how I can separate the 
    """
    # import matplotlib.pyplot as plt
    # import arviz as az
    from pysal.lib import weights
    from pgmpy.models import MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor
    # from pgmpy.inference import BeliefPropagation, BeliefPropagationWithMessagePassing
    logger.info("Loading data...")
    # W = load_npz(data_dir / f'{grid_size}_neighbors.npz').toarray() #Convert to a dense matrix

    lf_gdf = gpd.read_parquet(data_dir / f'{grid_size}_deg_lanternfly.parquet')#.fillna(0)#.dropna()
    population_counts = lf_gdf['observation_count'].values
    W = weights.Rook.from_dataframe(lf_gdf, ids='region_id')
    W.transform='B'
    n_nodes = len(population_counts)
    # neighbors = [neighbors_dict[str(i)] for i in range(n_regions)]
    # num_neighbors = neighbors.shape[0] #np.array([len(nbr) for nbr in neighbors])
    logger.success("Data loaded successfully!")

    #Determine bin limits for population count data
    min_c = int(lf_gdf.observation_count.min())
    max_c = int(lf_gdf.observation_count.max())
    domain_size = max_c - min_c + 1

    counts = lf_gdf["observation_count"]
    non_na = counts.dropna()

    # pd.qcut returns both labels and the bin edges
    _, bin_edges = pd.qcut(non_na, q=n_bins, retbins=True, duplicates="drop")

    # make sure edges cover the full range
    bin_edges[0]   = non_na.min()
    bin_edges[-1]  = non_na.max()

    # compute bin‐centers for the smoothing potential
    centers     = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    domain_size = len(centers)  # this will be n_bins or fewer if duplicates dropped

    diff  = np.abs(centers[:, None] - centers[None, :])
    pot   = np.exp(-alpha * diff)
    flat  = pot.ravel(order="C").tolist()  # row-major flatten

    #Initialize Markov model
    mn = MarkovNetwork()

    #add all nodes (use strings for pgmpy)
    node_ids = lf_gdf["region_id"].astype(str).tolist()
    mn.add_nodes_from(node_ids)

    #add edges from your PySAL adjacency w.neighbors
    edges = [(str(rid), str(nbr))
            for rid, neighs in W.neighbors.items()
            for nbr in neighs if rid < nbr]
    mn.add_edges_from(edges)

    #add pairwise smoothing factors
    for u, v in edges:
        fac = DiscreteFactor(
            variables=[u, v],
            cardinality=[domain_size, domain_size],
            values=flat
        )
        mn.add_factors(fac)

    #add unary “clamping” factors for *observed* cells only
    unary_factors = {}
    for _, row in lf_gdf.iterrows():
        rid, cnt = str(row.region_id), row.observation_count
        if not pd.isna(cnt):
            # map raw count → bin index in [0..domain_size-1]
            # bin_idx = int(np.digitize(cnt, bin_edges) - 1)
            raw_idx = np.digitize(cnt, bin_edges) - 1
            bin_idx = int(np.clip(raw_idx, 0, domain_size - 1))
            values  = [1.0 if i == bin_idx else 0.0
                    for i in range(domain_size)]

            fac = DiscreteFactor(
                variables=[rid],
                cardinality=[domain_size],
                values=values
            )
            mn.add_factors(fac)

            #Save off clamped factors for lbp
            vec = np.zeros(domain_size)
            vec[bin_idx] = 1.0
            unary_factors[rid] = vec
    
    #save pairwise edges for smoothing
    pairwise_potentials = {}
    for u, v in edges:
        # ensure both directions
        pairwise_potentials[(u, v)] = pot
        pairwise_potentials[(v, u)] = pot.T

    #perform lbp
    nodes = mn.nodes()
    neighbors = {
        str(region_id): [str(nbr) for nbr in nbrs]
        for region_id, nbrs in W.neighbors.items()
    }

    #Save all parameters for performing LBP as defined in our custom utils function
    save_mn_lbp_data(nodes, neighbors, unary_factors, pairwise_potentials, domain_size)

    # beliefs = loopy_belief_propagation(
    #     nodes,
    #     neighbors,
    #     unary_factors,
    #     pairwise_potentials,
    #     domain_size,
    #     max_iters=1000,
    #     tol=1e-8
    # )

@app.command()
def test():
    logger.success("You ran the test! Congrats!")


if __name__ == "__main__":
    app()
