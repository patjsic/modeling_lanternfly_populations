import os
import json
import pickle
import numpy as np
from pathlib import Path
from loguru import logger

def _save_pickle(
    filepath: Path,
    object: object,
):
    logger.info(f"Saving pickle: {filepath}")
    try:
        save_dir = os.path.dirname(filepath)
        os.makedirs(save_dir, exist_ok = True) #Create the subdirectory if it doesn't already exist
        with open(filepath, 'wb') as file:
            pickle.dump(object, file)
        logger.success(f"Successfully pickled file: {filepath}")
    except Exception as e:
        raise e

def _load_pickle(
    filepath: Path,
):
    logger.info(f"Loading pickle: {filepath}")
    with open(filepath, 'rb') as file:
        ret = pickle.load(file)
        logger.success(f"Successfully loaded pickle file: {filepath}")
        return ret

def _load_txt(
    filepath: Path,
):
    logger.info(f"Loading txt file: {filepath}")
    with open(filepath, "r") as file:
        ret = file.read()
        logger.success(f"Successfully loaded text file: {filepath}")
        return ret


def _save_txt(
    filepath: Path,
    save_text: str,
):
    logger.info(f"Saving txt file: {filepath}")
    try:
        save_dir = os.path.dirname(filepath)
        os.makedirs(save_dir, exist_ok = True)
        with open(filepath, 'w') as file:
            file.write(save_text)
        logger.success(f"Successfully saved text file: {filepath}")
    except Exception as e:
        raise e

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
    
def loopy_belief_propagation(nodes,
                             neighbors,
                             unary_factors,
                             pairwise_potentials,
                             domain_size,
                             max_iters=50,
                             tol=1e-3):
    """
    A simple loopy BP implementation on an MRF.

    nodes: list of node_ids (strings)
    neighbors: dict mapping node_id -> list of neighbor node_ids
    unary_factors: dict node_id → 1D array of length domain_size
    pairwise_potentials: dict (u,v) → 2D array [domain_size x domain_size]
                         must be symmetric, so (u,v) and (v,u) are transposes.
    domain_size: number of discrete states per node
    max_iters: maximum number of message‐passing rounds
    tol: convergence threshold on max message change
    """

    # 1) initialize all directed messages m[u→v] to uniform
    messages = {
        (u, v): np.ones(domain_size) / domain_size
        for u in nodes
        for v in neighbors[u]
    }

    for it in range(max_iters):
        delta = 0.0

        # 2) for each directed edge, update message
        for u in nodes:
            psi_u = unary_factors.get(u, np.ones(domain_size))
            for v in neighbors[u]:
                # collect incoming msgs into u from all w≠v
                incoming = [
                    messages[(w, u)]
                    for w in neighbors[u]
                    if w != v
                ]
                # start with φ_u
                tmp = psi_u.copy()
                for m in incoming:
                    tmp *= m
                # now compute new m_{u→v}(x_v)
                # m_uv[xv] = sum_{xu} φ_u[xu] · ψ_uv[xu,xv] · ∏_{w≠v} m_{w→u}[xu]
                phi = pairwise_potentials[(u, v)]
                # matrix-vector multiply: (domain_u × domain_v)
                new_msg = tmp @ phi
                # normalize
                new_msg = new_msg / new_msg.sum()
                # print(new_msg)
                # track largest change
                delta = max(delta, np.max(np.abs(new_msg - messages[(u, v)])))
                messages[(u, v)] = new_msg

        if delta < tol:
            # converged!
            break

    # 3) compute final beliefs
    beliefs = {}
    for u in nodes:
        psi_u = unary_factors.get(u, np.ones(domain_size))
        incoming = [messages[(w, u)] for w in neighbors[u]]
        b = psi_u.copy()
        for m in incoming:
            b *= m
        beliefs[u] = b / b.sum()

    return beliefs