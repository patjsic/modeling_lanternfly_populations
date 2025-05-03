import os
import json
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