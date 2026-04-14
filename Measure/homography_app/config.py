"""
Configuration for AI model device usage.
Set CPU_ONLY=True to force all models to use CPU for testing.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force CPU-only mode for testing
# Set environment variable: IMAGEMEASURE_CPU_ONLY=1
# Can be set in .env file or as environment variable
CPU_ONLY = os.getenv('IMAGEMEASURE_CPU_ONLY', '0') == '1'

# Enable/disable resource monitoring (CPU/GPU metrics logging)
# Set IMAGEMEASURE_RESOURCE_MONITOR=0 to disable (e.g. on production server)
RESOURCE_MONITOR = os.getenv('IMAGEMEASURE_RESOURCE_MONITOR', '1') == '1'

# Get device for model loading
def get_device():
    """
    Returns device string for model loading.
    Returns 'cpu' if CPU_ONLY is True, otherwise 'cuda' if available.
    """
    if CPU_ONLY:
        return 'cpu'
    
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def get_torch_device():
    """
    Returns torch device object for PyTorch models.
    """
    import torch
    if CPU_ONLY:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
