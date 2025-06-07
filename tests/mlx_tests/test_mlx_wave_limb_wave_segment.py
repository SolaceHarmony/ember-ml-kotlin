import pytest
import numpy as np # For comparison with known correct results
import array # For working with limb arrays

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.limb import wave_segment # Import the module
from ember_ml.wave.limb.hpc_limb_core import int_to_limbs, limbs_to_int # Import limb conversion helpers
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for wave.limb.wave_segment components

def test_wavesegment_initialization():
    # Test WaveSegment initialization
    initial_state_int = 100
    wave_max = 1000
    ca_threshold = 500
    k_threshold = 200

    initial_state_limbs = int_to_limbs(initial_state_int)

    wave_segment_instance = wave_segment.WaveSegment(initial_state_limbs, wave_max, ca_threshold, k_threshold)

    assert isinstance(wave_segment_instance, wave_segment.WaveSegment)
    assert limbs_to_int(wave_segment_instance.state) == initial_state_int
    assert wave_segment_instance.wave_max == wave_max
    assert wave_segment_instance.ca_threshold == ca_threshold
    assert wave_segment_instance.k_threshold == k_threshold
    assert hasattr(wave_segment_instance, 'propagation_history') # Should have history for delays


def test_wavesegment_get_normalized_state():
    # Test WaveSegment get_normalized_state
    initial_state_int = 500
    wave_max = 1000
    ca_threshold = 500
    k_threshold = 200

    initial_state_limbs = int_to_limbs(initial_state_int)
    wave_segment_instance = wave_segment.WaveSegment(initial_state_limbs, wave_max, ca_threshold, k_threshold)

    normalized_state = wave_segment_instance.get_normalized_state()

    assert isinstance(normalized_state, float) # Should return a float
    assert normalized_state == 0.5 # 500 / 1000


def test_wavesegmentarray_initialization():
    # Test WaveSegmentArray initialization
    num_segments = 5
    wave_segment_array = wave_segment.WaveSegmentArray(num_segments)

    assert isinstance(wave_segment_array, wave_segment.WaveSegmentArray)
    assert len(wave_segment_array.segments) == num_segments
    for segment in wave_segment_array.segments:
        assert isinstance(segment, wave_segment.WaveSegment)


def test_wavesegmentarray_get_wave_state():
    # Test WaveSegmentArray get_wave_state
    num_segments = 3
    wave_segment_array = wave_segment.WaveSegmentArray(num_segments)

    # Manually set states for segments (using limbs)
    wave_segment_array.segments[0].state = int_to_limbs(100)
    wave_segment_array.segments[0].wave_max = 1000
    wave_segment_array.segments[1].state = int_to_limbs(500)
    wave_segment_array.segments[1].wave_max = 1000
    wave_segment_array.segments[2].state = int_to_limbs(800)
    wave_segment_array.segments[2].wave_max = 1000

    wave_state = wave_segment_array.get_wave_state()

    assert wave_state.shape == (num_segments,)
    assert ops.allclose(wave_state, [0.1, 0.5, 0.8]) # Check normalized values


# Add more test functions for other wave_segment components:
# test_wavesegment_propagate(), test_wavesegment_update_ion_channels(),
# test_wavesegment_get_conduction_value(), test_wavesegmentarray_update(),
# test_wavesegmentarray_apply_boundary_reflection()

# Note: Testing the update and propagation logic will require simulating
# time steps and verifying state changes and conduction values.