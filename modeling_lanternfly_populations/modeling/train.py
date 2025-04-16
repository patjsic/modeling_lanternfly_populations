import os
import pickle
import numpy as np
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from modeling_lanternfly_populations.config import MODELS_DIR, PROCESSED_DATA_DIR
from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression
from UQpy.utilities.kernels.euclidean_kernels import RBF

app = typer.Typer()

def _save_pickle(
    filepath: Path,
    object: object,
):
    logger.info(f"Saving pickle: {filepath}")
    save_dir = os.path.dirname(filepath)
    os.makedirs(save_dir, exist_ok = True) #Create the subdirectory if it doesn't already exist
    with open(filepath, 'wb') as file:
        pickle.dump(object, file)

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
    kernel = RBF()
    gpr = GaussianProcessRegression(kernel=kernel, hyperparameters=[1, 1, 0.1])
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
def test():
    logger.success("You ran the test! Congrats!")


if __name__ == "__main__":
    app()
