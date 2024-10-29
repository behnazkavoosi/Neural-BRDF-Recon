import torch
import torch.nn as nn
import numpy as np
import coords  # Assuming coords is a custom module you have
import pyexr
import json
import os

# Function to save model weights and structure
def save_model(model, h5, json=None):
    # Save the model weights to an .h5 file
    torch.save(model.state_dict(), h5)
    # Save model structure to JSON if specified
    if json is None:
        json = h5.replace('.h5', '.json')
    model_structure = {
        'model_class': type(model).__name__,
        'model_state_dict': model.state_dict()
    }
    with open(json, 'w') as f:
        json.dump(model_structure, f)

# Function to load a model's weights and structure
def load_model(h5, json=None, model_class=None):
    if json is None:
        json = h5.replace('.h5', '.json')

    # Read model structure from JSON file
    with open(json, 'r') as f:
        model_structure = json.load(f)

    # Create the model instance from the provided model_class
    if model_class is None:
        raise ValueError("A model_class is required to reconstruct the model in PyTorch.")
    
    model = model_class()
    model.load_state_dict(torch.load(h5))
    return model

# Normalize phi_d to the range [0, 2*pi]
def normalize_phid(orig_phid):
    phid = orig_phid.copy()
    phid = np.where(phid < 0, phid + 2 * np.pi, phid)
    phid = np.where(phid >= 2 * np.pi, phid - 2 * np.pi, phid)
    return phid

# Generate a mask based on the non-zero values in the array
def mask_from_array(arr):
    if len(arr.shape) > 1:
        mask = np.linalg.norm(arr, axis=1)
        mask[mask != 0] = 1
    else:
        mask = np.where(arr != 0, 1, 0)
    return mask

# Initialize PyTorch session for CUDA
def keras_init_session(allow_growth=True, logging=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(device)
        print("PyTorch initialized on CUDA (GPU).")
        if allow_growth:
            print("Note: PyTorch manages GPU memory dynamically; explicit allow_growth setting is not needed.")
    else:
        print("CUDA is not available. Running on CPU.")