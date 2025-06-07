"""
Pytest configuration file for ember_ml tests.

This file contains common fixtures and configurations for pytest.
"""

import pytest
import os
import sys
import logging
from ember_ml.nn import tensor
from ember_ml.nn.tensor import set_seed

# Add parent directory to path to import ember_ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import backend utilities
from ember_ml.utils import backend_utils
from ember_ml.backend import get_backend, set_backend
from ember_ml import ops


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Set up logging for tests."""
    logger = logging.getLogger('ember_ml_tests')
    logger.setLevel(logging.INFO)
    return logger


@pytest.fixture(scope="session")
def available_backends():
    """Get list of available backends."""
    backends = ['numpy']
    
    try:
        import torch
        backends.append('torch')
    except ImportError:
        pass
    
    try:
        import mlx.core
        backends.append('mlx')
    except ImportError:
        pass
    
    return backends


# Removed the parameterized 'backend' fixture as tests are now separated by directory
# Individual backend fixtures (numpy_backend, torch_backend, mlx_backend) are kept

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    set_seed(seed)
    backend_utils.initialize_random_seed(seed)
    return seed


@pytest.fixture
def numpy_backend():
    """Set NumPy backend for tests."""
    original_backend = get_backend()
    ops.set_backend('numpy') # Use ops.set_backend to trigger alias updates
    yield
    ops.set_backend(original_backend or 'numpy')  # Use ops.set_backend


@pytest.fixture
def torch_backend():
    """Set PyTorch backend for tests if available."""
    try:
        import torch
        original_backend = get_backend()
        ops.set_backend('torch') # Use ops.set_backend
        yield
        ops.set_backend(original_backend or 'numpy')  # Use ops.set_backend
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def mlx_backend():
    """Set MLX backend for tests if available."""
    try:
        import mlx.core
        original_backend = get_backend()
        ops.set_backend('mlx') # Use ops.set_backend
        yield
        ops.set_backend(original_backend or 'numpy')  # Use ops.set_backend
    except ImportError:
        pytest.skip("MLX not available")


# Removed the parameterized 'any_backend' fixture as tests are now separated by directory

@pytest.fixture
def sample_tensor_1d():
    """Create a 1D sample tensor."""
    return tensor.convert_to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_tensor_2d():
    """Create a 2D sample tensor."""
    return tensor.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_tensor_3d():
    """Create a 3D sample tensor."""
    return tensor.convert_to_tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])