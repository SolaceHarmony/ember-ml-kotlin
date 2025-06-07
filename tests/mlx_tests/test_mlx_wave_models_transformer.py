import pytest

from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml.wave.models.wave_transformer import (
    WaveMultiHeadAttention,
    WaveTransformerEncoderLayer,
    WaveTransformerEncoder,
    WaveTransformer,
    create_wave_transformer,
)
from ember_ml.nn.modules import Module # Needed for isinstance checks

@pytest.fixture(params=['mlx'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Helper function to create dummy input data
def create_dummy_input_data(shape=(32, 10, 64)):
    """Creates a dummy input tensor for transformer models."""
    return tensor.random_normal(shape, dtype=tensor.float32)

# Test cases for initialization and forward pass shapes

def test_wavemultiheadattention_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveMultiHeadAttention initialization and forward pass shape."""
    embed_dim = 64
    num_heads = 8
    attention = WaveMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    assert isinstance(attention, WaveMultiHeadAttention)
    assert isinstance(attention, Module)
    query = create_dummy_input_data(shape=(32, 10, embed_dim))
    key = create_dummy_input_data(shape=(32, 10, embed_dim))
    value = create_dummy_input_data(shape=(32, 10, embed_dim))
    output = attention(query, key, value)
    assert tensor.shape(output) == (32, 10, embed_dim)

def test_wavetransformerencoderlayer_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveTransformerEncoderLayer initialization and forward pass shape."""
    embed_dim = 64
    num_heads = 8
    ff_hidden_dim = 128
    layer = WaveTransformerEncoderLayer(
        embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim
    )
    assert isinstance(layer, WaveTransformerEncoderLayer)
    assert isinstance(layer, Module)
    input_data = create_dummy_input_data(shape=(32, 10, embed_dim))
    output = layer(input_data)
    assert tensor.shape(output) == (32, 10, embed_dim)

def test_wavetransformerencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveTransformerEncoder initialization and forward pass shape."""
    embed_dim = 64
    num_heads = 8
    ff_hidden_dim = 128
    num_layers = 2
    encoder = WaveTransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
    )
    assert isinstance(encoder, WaveTransformerEncoder)
    assert isinstance(encoder, Module)
    input_data = create_dummy_input_data(shape=(32, 10, embed_dim))
    output = encoder(input_data)
    assert tensor.shape(output) == (32, 10, embed_dim)

def test_wavetransformer_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveTransformer initialization and forward pass shape."""
    seq_length = 10
    embed_dim = 64
    num_heads = 8
    ff_hidden_dim = 128
    num_layers = 2
    transformer = WaveTransformer(
        seq_length=seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
    )
    assert isinstance(transformer, WaveTransformer)
    assert isinstance(transformer, Module)
    input_data = create_dummy_input_data(shape=(32, seq_length, embed_dim))
    output = transformer(input_data)
    assert tensor.shape(output) == (32, seq_length, embed_dim)

def test_create_wave_transformer_factory(set_backend_fixture):
    """Test create_wave_transformer factory function."""
    seq_length = 10
    embed_dim = 64
    num_heads = 8
    ff_hidden_dim = 128
    num_layers = 2
    transformer = create_wave_transformer(
        seq_length=seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
    )
    assert isinstance(transformer, WaveTransformer)
    assert isinstance(transformer, Module)
    input_data = create_dummy_input_data(shape=(32, seq_length, embed_dim))
    output = transformer(input_data)
    assert tensor.shape(output) == (32, seq_length, embed_dim)

# TODO: Add tests for attention masks
# TODO: Add tests for parameter registration
# TODO: Add tests for different activation functions and dropout rates
# TODO: Add tests for edge cases and invalid inputs
# Note: Testing the correctness of attention weights and outputs might require
# complex matrix operations which may not be fully supported by all backends (e.g., NumPy).
# These tests are included but may need to be skipped or adapted for specific backends.