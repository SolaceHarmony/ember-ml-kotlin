import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave import binary_pattern # Import the binary_pattern module
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

# Fixture providing sample binary wave pattern data
@pytest.fixture
def sample_binary_pattern_data():
    """Create sample binary wave pattern data."""
    # Use a consistent seed for reproducibility
    tensor.set_seed(42)
    # Create a simple binary wave pattern (e.g., a few pulses)
    pattern = tensor.zeros((10, 10), dtype=tensor.int32)
    pattern = tensor.tensor_scatter_nd_update(pattern, tensor.convert_to_tensor([[2, 3], [5, 6], [7, 8]]), tensor.ones(3, dtype=tensor.int32))
    return pattern

# Test cases for wave.binary_pattern components

def test_patternmatch_dataclass():
    # Test PatternMatch dataclass
    similarity = 0.8
    position = (5, 5)
    phase_shift = 2
    confidence = 0.9

    match = binary_pattern.PatternMatch(similarity, position, phase_shift, confidence)

    assert isinstance(match, binary_pattern.PatternMatch)
    assert match.similarity == similarity
    assert match.position == position
    assert match.phase_shift == phase_shift
    assert match.confidence == confidence


def test_interferencedetector_initialization():
    # Test InterferenceDetector initialization
    detector = binary_pattern.InterferenceDetector()
    assert isinstance(detector, binary_pattern.InterferenceDetector)


def test_interferencedetector_detect_interference(sample_binary_pattern_data):
    # Test InterferenceDetector detect_interference
    wave1 = sample_binary_pattern_data
    wave2 = tensor.random_uniform(tensor.shape(wave1), minval=0, maxval=2, dtype=tensor.int32) # Create another dummy wave

    # detect_interference returns a dictionary of results
    results = binary_pattern.InterferenceDetector.detect_interference(wave1, wave2)

    assert isinstance(results, dict)
    assert 'constructive_interference' in results
    assert 'destructive_interference' in results
    assert 'multiplicative_interference' in results
    assert 'constructive_strength' in results
    assert 'destructive_strength' in results
    assert 'multiplicative_strength' in results

    # Check that the results are tensors or scalar tensors
    for key in results:
        assert isinstance(results[key], (tensor.EmberTensor, float, int)) # Allow for scalar Python types


def test_patternmatcher_initialization():
    # Test PatternMatcher initialization
    matcher = binary_pattern.PatternMatcher()
    assert isinstance(matcher, binary_pattern.PatternMatcher)


def test_patternmatcher_match_pattern(sample_binary_pattern_data):
    # Test PatternMatcher match_pattern
    template = tensor.ones((2, 2), dtype=tensor.int32) # Simple 2x2 square template
    target = sample_binary_pattern_data # Use the sample pattern as target
    threshold = 0.8

    # match_pattern returns a list of PatternMatch objects
    matches = binary_pattern.PatternMatcher.match_pattern(template, target, threshold)

    assert isinstance(matches, list)
    # Check that each item in the list is a PatternMatch object
    for match in matches:
        assert isinstance(match, binary_pattern.PatternMatch)

    # More detailed tests would involve creating specific target patterns with known matches
    # and asserting the number and properties of the returned matches.


def test_binarypattern_initialization():
    # Test BinaryPattern initialization
    input_shape = (10, 10)
    binary_pattern_module = binary_pattern.BinaryPattern(input_shape)

    assert isinstance(binary_pattern_module, binary_pattern.BinaryPattern)
    assert binary_pattern_module.input_shape == input_shape
    assert hasattr(binary_pattern_module, 'encoder') # Should have an encoder
    assert hasattr(binary_pattern_module, 'pattern_matcher') # Should have a pattern matcher
    assert hasattr(binary_pattern_module, 'interference_detector') # Should have an interference detector


# Add more test functions for other binary_pattern components:
# test_interferencedetector_find_resonance(), test_binarypattern_extract_pattern(),
# test_binarypattern_match_pattern(), test_binarypattern_analyze_interference(),
# test_binarypattern_forward()

# Note: Some tests might require more complex setups or mock dependencies.
# Initial tests focus on instantiation and basic calls/shape checks.