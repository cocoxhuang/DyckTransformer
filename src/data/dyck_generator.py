import time
import torch
import os
import pickle
from sage.all import DyckWords  # Import your DyckWords implementation


def generate_dyck_data(n, data_path, force_regenerate=False, logger=None):
    """
    Generate Dyck words data for training with caching.
    
    Args:
        n (int): Semilength of Dyck words
        data_path (str): Full path to the cached data file. If None, uses default path with n.
        force_regenerate (bool): If True, regenerate data even if cache exists
        logger: Optional logger instance for logging messages
        
    Returns:
        dict: Dictionary containing 'inputs' and 'targets' tensors
    """
    def log_message(msg):
        """Helper function to log or print messages."""
        if logger:
            logger.info(msg)
        else:
            print(msg)

    assert data_path is not None, "data_path must be provided to save or load data."
    
    # Create directory for the data file if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    cache_file = data_path
    
    # Check if cached data exists and load it
    if not force_regenerate and os.path.exists(cache_file):
        log_message(f"Loading cached Dyck words data for n={n} from {cache_file}...")
        start_load = time.time()
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            log_message(f"Successfully loaded cached data with {data['inputs'].shape[0]} sequences.")
            log_message(f"Input shape: {data['inputs'].shape}")
            log_message(f"Target shape: {data['targets'].shape}")
            log_message(f"Time taken to load data: {time.time() - start_load:.4f} seconds.")
            return data
        except Exception as e:
            log_message(f"Error loading saved data: {e}. Regenerating...")
    
    # Generate new data
    start = time.time()
    log_message(f"Generating Dyck words of semilength {n}...")

    data = {}
    vecs = DyckWords(n)
    data['inputs'] = torch.tensor([list(entry) for entry in vecs], dtype=torch.long)
    data['targets'] = torch.tensor([entry.area_dinv_to_bounce_area_map() for entry in vecs], dtype=torch.long)
    log_message("Using actual DyckWords implementation.")

    log_message(f"Success. Generated {data['inputs'].shape[0]} sequences.")
    log_message(f"Input shape: {data['inputs'].shape}")
    log_message(f"Target shape: {data['targets'].shape}")
    log_message(f"Time taken to generate data: {time.time() - start:.4f} seconds.")

    # Save the generated data
    log_message(f"Saving data to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    log_message("Data saved successfully.")

    return data
