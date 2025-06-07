import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
import ember_ml.wave.binary_memory as binary_memory # Import the module
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

# Fixture providing sample binary wave pattern data for memory tests
@pytest.fixture
def sample_memory_pattern_data():
    """Create sample binary wave pattern data for memory tests."""
    # Use a consistent seed for reproducibility
    tensor.set_seed(123)
    # Create a few distinct patterns
    pattern1 = tensor.zeros((5, 5), dtype=tensor.int32)
    pattern1 = tensor.tensor_scatter_nd_update(pattern1, tensor.convert_to_tensor([[1, 1], [3, 3]]), tensor.ones(2, dtype=tensor.int32))

    pattern2 = tensor.zeros((5, 5), dtype=tensor.int32)
    pattern2 = tensor.tensor_scatter_nd_update(pattern2, tensor.convert_to_tensor([[0, 4], [4, 0]]), tensor.ones(2, dtype=tensor.int32))

    pattern3 = tensor.zeros((5, 5), dtype=tensor.int32)
    pattern3 = tensor.tensor_scatter_nd_update(pattern3, tensor.convert_to_tensor([[2, 2]]), tensor.ones(1, dtype=tensor.int32))

    return pattern1, pattern2, pattern3

# Test cases for wave.memory.binary_memory components

def test_memorypattern_dataclass():
    # Test MemoryPattern dataclass
    pattern = tensor.ones((2, 2))
    timestamp = 12345
    metadata = {"source": "test"}

    memory_pattern = binary_memory.MemoryPattern(pattern, timestamp, metadata)

    assert isinstance(memory_pattern, binary_memory.MemoryPattern)
    assert isinstance(memory_pattern.pattern, tensor.EmberTensor)
    assert ops.allclose(memory_pattern.pattern, pattern).item()
    assert memory_pattern.timestamp == timestamp
    assert memory_pattern.metadata == metadata


def test_memorypattern_similarity(sample_memory_pattern_data):
    # Test MemoryPattern similarity method
    pattern1, pattern2, _ = sample_memory_pattern_data
    memory_pattern1 = binary_memory.MemoryPattern(pattern1, 1, {})
    memory_pattern2 = binary_memory.MemoryPattern(pattern2, 2, {})

    # Similarity calculation depends on the internal implementation (e.g., dot product, cosine similarity)
    # We can check that the similarity is a float or scalar tensor.
    similarity = memory_pattern1.similarity(memory_pattern2.pattern)
    assert isinstance(similarity, (float, tensor.floating, tensor.EmberTensor))
    # Check if similarity is within a reasonable range (e.g., 0 to 1 for normalized patterns)
    # For these dummy patterns, similarity should be low but non-zero due to random initialization.
    # A more precise test would require known patterns and expected similarity values.
    if isinstance(similarity, tensor.EmberTensor):
        similarity = tensor.item(similarity)
    assert 0.0 <= similarity <= 1.0 # Assuming normalized similarity


def test_wavestorage_initialization():
    # Test WaveStorage initialization
    capacity = 10
    storage = binary_memory.WaveStorage(capacity)

    assert isinstance(storage, binary_memory.WaveStorage)
    assert storage.capacity == capacity
    assert len(storage.patterns) == 0 # Should be empty initially


def test_wavestorage_store_and_retrieve(sample_memory_pattern_data):
    # Test WaveStorage store and retrieve
    pattern1, pattern2, pattern3 = sample_memory_pattern_data
    capacity = 2
    storage = binary_memory.WaveStorage(capacity)

    # Store patterns (exceeding capacity)
    storage.store(pattern1, 1, {"id": 1})
    assert len(storage.patterns) == 1

    storage.store(pattern2, 2, {"id": 2})
    assert len(storage.patterns) == 2

    storage.store(pattern3, 3, {"id": 3}) # This should cause pattern1 to be removed
    assert len(storage.patterns) == 2
    # Check that pattern1 is no longer in storage (based on metadata or pattern content)
    stored_ids = [p.metadata.get("id") for p in storage.patterns]
    assert 1 not in stored_ids
    assert 2 in stored_ids
    assert 3 in stored_ids

    # Retrieve patterns similar to pattern3
    threshold = 0.5 # Example threshold
    retrieved_matches = storage.retrieve(pattern3, threshold)

    assert isinstance(retrieved_matches, list)
    # Check that retrieved items are PatternMatch objects
    for match in retrieved_matches:
        assert isinstance(match, binary_memory.PatternMatch)

    # More detailed tests would involve setting specific patterns and thresholds
    # and asserting the number and content of retrieved matches.


def test_binarymemory_initialization():
    # Test BinaryMemory initialization
    grid_size = (10, 10)
    num_phases = 8
    capacity = 10
    binary_memory_module = binary_memory.BinaryMemory(grid_size, num_phases, capacity)

    assert isinstance(binary_memory_module, binary_memory.BinaryMemory)
    assert isinstance(binary_memory_module.encoder, binary_memory.BinaryWave) # Should have an encoder
    assert isinstance(binary_memory_module.storage, binary_memory.WaveStorage) # Should have storage
    assert hasattr(binary_memory_module, 'store_gate') # Should have learnable gates
    assert hasattr(binary_memory_module, 'retrieve_gate')


# Add more test functions for other binary_memory components:
# test_binarymemory_store_pattern(), test_binarymemory_retrieve_pattern(),
# test_binarymemory_clear_memory(), test_binarymemory_get_memory_state(),
# test_binarymemory_save_state(), test_binarymemory_load_state()

# Note: Testing the store/retrieve methods of BinaryMemory will involve
# using the encoder and gates, requiring more complex setups.