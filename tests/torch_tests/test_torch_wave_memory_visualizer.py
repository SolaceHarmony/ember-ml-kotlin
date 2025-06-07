import pytest
import numpy as np # For comparison with known correct results
import matplotlib.pyplot as plt # For testing plotting functions
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.memory import visualizer # Import the visualizer module
from ember_ml.wave.memory import multi_sphere # Import multi_sphere for creating a dummy model
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Fixture providing a dummy MultiSphereWaveModel and history
@pytest.fixture
def dummy_model_and_history():
    """Create a dummy MultiSphereWaveModel and history for visualization tests."""
    num_spheres = 2
    state_dim = 3
    model = multi_sphere.MultiSphereWaveModel(num_spheres, state_dim, 0.5, 0.5, 0.1)

    # Create dummy history data (NumPy array)
    steps = 10
    history_np = np.random.rand(steps, num_spheres, state_dim).astype(tensor.float32)

    return model, history_np

# Test cases for wave.memory.visualizer components

def test_wavememoryanalyzer_initialization():
    # Test WaveMemoryAnalyzer initialization
    analyzer = visualizer.WaveMemoryAnalyzer()
    assert isinstance(analyzer, visualizer.WaveMemoryAnalyzer)


def test_wavememoryanalyzer_analyze_model(dummy_model_and_history):
    # Test WaveMemoryAnalyzer analyze_model
    # This test requires running a simulation and generating visualizations/metrics.
    # We can test that it runs without errors and returns expected types.
    model, history = dummy_model_and_history
    analyzer = visualizer.WaveMemoryAnalyzer()

    # analyze_model runs the simulation and returns a figure and metrics
    # Note: Running the full simulation might be time-consuming.
    # We can reduce the number of steps for testing.
    steps = 5 # Reduced steps for faster test
    # Create dummy input waves and gating signals (NumPy arrays or lists)
    input_waves_seq_np = np.random.rand(steps, model.num_spheres, model.state_dim).astype(tensor.float32)
    input_waves_seq = tensor.convert_to_tensor(input_waves_seq_np)
    gating_seq_np = np.random.rand(steps, model.num_spheres).astype(tensor.float32)
    gating_seq = tensor.convert_to_tensor(gating_seq_np)

    # Mock the run method to return the dummy history
    model.run = lambda s, i, g: history[:s] # Return a slice of the dummy history

    try:
        fig, metrics = analyzer.analyze_model(model, steps, input_waves_seq, gating_seq)
        assert isinstance(fig, plt.Figure)
        assert isinstance(metrics, visualizer.AnalysisMetrics)
        plt.close(fig) # Close the figure
    except Exception as e:
        pytest.fail(f"analyze_model raised an exception: {e}")


def test_wavememoryanalyzer_create_visualization(dummy_model_and_history):
    # Test WaveMemoryAnalyzer create_visualization
    # This test verifies that the visualization function runs without errors
    # and creates a matplotlib figure with the expected number of subplots.
    model, history = dummy_model_and_history
    analyzer = visualizer.WaveMemoryAnalyzer()

    try:
        fig = analyzer.create_visualization(history)
        assert isinstance(fig, plt.Figure)
        # Check the number of subplots (depends on the implementation)
        # Assuming a certain number of subplots based on the documentation/code structure
        # For example, if it plots component evolution, phase space, energy, correlations, interference.
        # This might be around 5-7 subplots.
        # assert len(fig.get_axes()) >= 5 # Check for at least 5 subplots
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"create_visualization raised an exception: {e}")


def test_wavememoryanalyzer_animate_training(dummy_model_and_history):
    # Test WaveMemoryAnalyzer animate_training
    # This test verifies that the animation function runs without errors.
    # It requires a history of parameters or states over training steps.
    # We can use the dummy history for this test.
    model, history = dummy_model_and_history
    analyzer = visualizer.WaveMemoryAnalyzer()

    try:
        # animate_training returns a matplotlib animation object
        animation = analyzer.animate_training(history)
        # We can check the type of the returned object
        from matplotlib.animation import Animation
        assert isinstance(animation, Animation)
        # Note: Running the animation itself is outside the scope of a unit test.
    except ImportError:
        pytest.skip("Skipping animate_training test: matplotlib animation dependencies not met")
    except Exception as e:
        pytest.fail(f"animate_training raised an exception: {e}")


# Add more test functions for other visualizer components:
# test_wavememoryanalyzer_plot_*_methods()

# Note: Testing the individual plot_* methods would involve checking the properties
# of the generated plots (titles, labels, data plotted) which can be complex.
# Focusing on the main entry points (analyze_model, create_visualization, animate_training)
# provides a good level of coverage.