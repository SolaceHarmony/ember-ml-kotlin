"""
Pytest configuration and shared fixtures for ember_ml tests.
"""

import pytest
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def random_seed():
    """Fixture to ensure reproducible random numbers."""
    np.random.seed(42)
    return 42

@pytest.fixture
def sample_time_series():
    """Fixture providing sample time series data for testing."""
    t = np.linspace(0, 10, 1000)
    signal = np.sin(t) + 0.1 * np.random.randn(len(t))
    return t, signal

@pytest.fixture
def default_neuron_params():
    """Fixture providing default neuron parameters."""
    return {
        'tau': 1.0,
        'dt': 0.01,
        'gleak': 0.5,
        'cm': 1.0
    }

@pytest.fixture
def default_chain_params():
    """Fixture providing default chain parameters."""
    return {
        'num_neurons': 3,
        'base_tau': 1.0,
        'dt': 0.01,
        'gleak': 0.5,
        'cm': 1.0
    }

@pytest.fixture
def tolerance():
    """Fixture providing default numerical tolerance for comparisons."""
    return 1e-6

def pytest_configure(config):
    """Configure pytest for ember_ml tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration."""
    # Skip slow tests unless --runslow is specified
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests unless --gpu is specified
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU tests"
    )