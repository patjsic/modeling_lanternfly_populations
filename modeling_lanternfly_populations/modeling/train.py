import sys

sys.path.append('..')

import os
import json
import pickle
import numpy as np
import pymc as pm
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

from loguru import logger
from tqdm import tqdm
# from utils import load_json_file
import typer

from modeling_lanternfly_populations.utils import _save_pickle
from modeling_lanternfly_populations.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression
from UQpy.utilities.kernels.euclidean_kernels import RBF

app = typer.Typer()

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
    output_dir: Path = MODELS_DIR,
    data_dir: Path = INTERIM_DATA_DIR,
    grid_size: float = 0.5,
):
    import matplotlib.pyplot as plt
    import arviz as az
    logger.info("Loading data...")
    W = load_npz(data_dir / f'{grid_size}_neighbors.npz').toarray() #Convert to a dense matrix
    N = W.shape[0]
    # print(neighbors.shape)
    grid_gdf = pd.read_parquet(data_dir / f'{grid_size}_deg_lanternfly.parquet')
    population_counts = grid_gdf['observation_count'].fillna(0).values
    # neighbors = [neighbors_dict[str(i)] for i in range(n_regions)]
    # num_neighbors = neighbors.shape[0] #np.array([len(nbr) for nbr in neighbors])
    logger.success("Data loaded successfully!")

    with pm.Model() as model:
        # Hyperpriors
        lambda_ = pm.Gamma('lambda', alpha=1., beta=1.)  # Regularization parameter
        sigma = pm.HalfNormal('sigma', sigma=1.)  # Standard deviation for observation noise

        # Latent variable for the population count at each grid cell
        # We model these as Gaussian variables
        # x = pm.Normal('x', mu=0, sigma=10, shape=len(population_counts))

        # Smoothness prior (penalizing large differences between neighbors)
        # smoothness_term = pm.math.dot(W, x) + lambda_#W.dot(x) * lambda_
 
        # print(population_counts)

        # Observation likelihood: the population counts at each grid cell
        # logger.info("Calculating likelihood...")
        for i in range(len(population_counts)):     
            pm.Normal(f'obs_{i}', mu=population_counts[i], sigma=sigma, observed=population_counts[i]) #+ smoothness_term[i], sigma=sigma, observed=population_counts[i])

        logger.success("Likelihood calculated successfully!")

        # Sampling the posterior
        logger.info("Sampling posterior...")
        # start = pm.find_MAP()
        try:
            i_data = pm.sample(200, tune=100, progressbar=True, exception_verbosity=True)
            # i_data = az.from_pymc(trace=trace, model=model)
            i_data.to_netcdf(f'./test.nc') #load it using az.from_netcdf("path.nc")
        except Exception as e:
            with open('./log.txt', 'a+') as file:
                file.write(f'ERROR: {e} \n')
            raise e
        logger.success("Data sampled successfully!")

        # phi_posterior = trace.posterior['phi']  # dims: [chain, draw, region]

        # # For example, compute posterior mean
        # phi_mean = phi_posterior.mean(dim=["chain", "draw"]).values

        # # Attach phi_mean back to your GeoDataFrame for mapping
        # grid_gdf["phi_mean"] = phi_mean
        # fig, ax = plt.subplots()
        # grid_gdf.plot(column="phi_mean", ax=ax, legend=True)
        # ax.set_title("Posterior Mean of Random Effect (phi)")
        # plt.savefig('./temp.png')


@app.command()
def test():
    logger.success("You ran the test! Congrats!")


if __name__ == "__main__":
    app()
