import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.memory import metrics # Import the metrics module
from ember_ml.ops import set_backend
import time
# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for wave.memory.metrics components

def test_analysismetrics_dataclass():
    # Test AnalysisMetrics dataclass initialization and validation
    computation_time = 1.23
    interference_strength = 0.8
    energy_stability = 0.95
    phase_coherence = 0.7
    total_time = 1.5

    metrics = metrics.AnalysisMetrics(computation_time, interference_strength, energy_stability, phase_coherence, total_time)

    assert isinstance(metrics, metrics.AnalysisMetrics)
    assert metrics.computation_time == computation_time
    assert metrics.interference_strength == interference_strength
    assert metrics.energy_stability == energy_stability
    assert metrics.phase_coherence == phase_coherence
    assert metrics.total_time == total_time

    # Test validation (e.g., values should be non-negative)
    with pytest.raises(ValueError):
        metrics.AnalysisMetrics(-0.1, 0.8, 0.95, 0.7, 1.5) # Negative time


def test_metricscollector_initialization():
    # Test MetricsCollector initialization
    collector = metrics.MetricsCollector()

    assert isinstance(collector, metrics.MetricsCollector)
    assert collector.start_time is None
    assert collector.end_time is None
    assert len(collector.wave_history) == 0


def test_metricscollector_start_end_computation():
    # Test MetricsCollector start_computation and end_computation
    collector = metrics.MetricsCollector()
    collector.start_computation()
    assert collector.start_time is not None
    assert collector.end_time is None

    # Simulate some work
    time.sleep(0.001) # Use a smaller sleep for faster tests

    collector.end_computation()
    assert collector.end_time is not None
    assert collector.end_time > collector.start_time


def test_metricscollector_record_wave_states():
    # Test MetricsCollector record_wave_states
    collector = metrics.MetricsCollector()
    # Create dummy wave states (NumPy array)
    states1 = np.random.rand(2, 3).astype(tensor.float32) # (num_spheres, state_dim)
    states2 = np.random.rand(2, 3).astype(tensor.float32)

    collector.record_wave_states(states1)
    assert len(collector.wave_history) == 1
    assert tensor.convert_to_tensor_equal(collector.wave_history[0], states1)

    collector.record_wave_states(states2)
    assert len(collector.wave_history) == 2
    assert tensor.convert_to_tensor_equal(collector.wave_history[1], states2)


def test_metricscollector_get_wave_history():
    # Test MetricsCollector get_wave_history
    collector = metrics.MetricsCollector()
    states1 = np.random.rand(2, 3).astype(tensor.float32)
    states2 = np.random.rand(2, 3).astype(tensor.float32)

    collector.record_wave_states(states1)
    collector.record_wave_states(states2)

    history = collector.get_wave_history()

    assert isinstance(history, TensorLike)
    # Expected shape: (steps, num_spheres, state_dim)
    assert history.shape == (2, 2, 3)
    assert tensor.convert_to_tensor_equal(history[0], states1)
    assert tensor.convert_to_tensor_equal(history[1], states2)


def test_metricscollector_compute_metrics():
    # Test MetricsCollector compute_metrics
    # This test requires simulating a full run with recorded states and timings.
    # It also relies on the correctness of the underlying math_helpers functions.
    collector = metrics.MetricsCollector()

    # Simulate a run
    collector.start_computation()
    # Simulate recording states
    for i in range(10):
        states = np.random.rand(2, 3).astype(tensor.float32)
        collector.record_wave_states(states)
        time.sleep(0.001) # Simulate some time passing
    collector.end_computation()

    # Compute metrics
    analysis_metrics = collector.compute_metrics()

    assert isinstance(analysis_metrics, metrics.AnalysisMetrics)
    assert analysis_metrics.computation_time > 0
    assert analysis_metrics.total_time > 0
    # Checking the exact values of interference_strength, energy_stability,
    # and phase_coherence would require replicating the logic from math_helpers,
    # which is complex. We can check that they are within a reasonable range.
    assert 0.0 <= analysis_metrics.interference_strength <= 1.0 # Assuming normalized
    assert analysis_metrics.energy_stability >= 0.0
    assert 0.0 <= analysis_metrics.phase_coherence <= 1.0 # Assuming normalized


# Add more test functions for other metrics components if any exist and are testable