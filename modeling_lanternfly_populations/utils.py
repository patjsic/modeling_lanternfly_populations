import os
import pickle
from pathlib import Path
from loguru import logger

def _save_pickle(
    filepath: Path,
    object: object,
):
    logger.info(f"Saving pickle: {filepath}")
    save_dir = os.path.dirname(filepath)
    os.makedirs(save_dir, exist_ok = True) #Create the subdirectory if it doesn't already exist
    with open(filepath, 'wb') as file:
        pickle.dump(object, file)