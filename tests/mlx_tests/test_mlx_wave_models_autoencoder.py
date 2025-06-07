import pytest
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.wave.models.wave_autoencoder import (
    WaveEncoder,
    WaveDecoder,
    WaveVariationalEncoder,
    WaveConvolutionalAutoencoder,
    WaveAutoencoder,
    create_wave_autoencoder,
    create_wave_conv_autoencoder,
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
def create_dummy_input_data(shape=(32, 100)):
    """Creates a dummy input tensor."""
    return tensor.random_normal(shape, dtype=tensor.float32)

# Test cases for initialization and forward pass shapes

def test_waveencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveEncoder initialization and forward pass shape."""
    input_size = 100
    latent_dim = 10
    encoder = WaveEncoder(input_size=input_size, latent_dim=latent_dim)
    assert isinstance(encoder, WaveEncoder)
    assert isinstance(encoder, Module)
    input_data = create_dummy_input_data(shape=(32, input_size))
    output = encoder(input_data)
    assert tensor.shape(output) == (32, latent_dim)

def test_wavedecoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveDecoder initialization and forward pass shape."""
    latent_dim = 10
    output_size = 100
    decoder = WaveDecoder(latent_dim=latent_dim, output_size=output_size)
    assert isinstance(decoder, WaveDecoder)
    assert isinstance(decoder, Module)
    input_data = create_dummy_input_data(shape=(32, latent_dim))
    output = decoder(input_data)
    assert tensor.shape(output) == (32, output_size)

def test_wavevariationalencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveVariationalEncoder initialization and forward pass shape."""
    input_size = 100
    latent_dim = 10
    encoder = WaveVariationalEncoder(input_size=input_size, latent_dim=latent_dim)
    assert isinstance(encoder, WaveVariationalEncoder)
    assert isinstance(encoder, Module)
    input_data = create_dummy_input_data(shape=(32, input_size))
    mean, log_var, z = encoder(input_data)
    assert tensor.shape(mean) == (32, latent_dim)
    assert tensor.shape(log_var) == (32, latent_dim)
    assert tensor.shape(z) == (32, latent_dim)

def test_waveautoencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveAutoencoder initialization and forward pass shape."""
    input_size = 100
    latent_dim = 10
    autoencoder = WaveAutoencoder(input_size=input_size, latent_dim=latent_dim)
    assert isinstance(autoencoder, WaveAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = create_dummy_input_data(shape=(32, input_size))
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == (32, input_size)
    assert tensor.shape(latent_z) == (32, latent_dim)
    # For non-variational autoencoder, mean and log_var should be None
    assert mean is None
    assert log_var is None

def test_waveautoencoder_variational_forward_shape(set_backend_fixture):
    """Test WaveAutoencoder forward pass shape with variational encoder."""
    input_size = 100
    latent_dim = 10
    # Create with variational=True
    autoencoder = WaveAutoencoder(input_size=input_size, latent_dim=latent_dim, variational=True)
    assert isinstance(autoencoder, WaveAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = create_dummy_input_data(shape=(32, input_size))
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == (32, input_size)
    assert tensor.shape(latent_z) == (32, latent_dim)
    assert tensor.shape(mean) == (32, latent_dim)
    assert tensor.shape(log_var) == (32, latent_dim)

def test_waveconvolutionalautoencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveConvolutionalAutoencoder initialization and forward pass shape."""
    input_shape = (32, 1, 100) # Batch, Channels, Sequence Length
    latent_dim = 10
    # Note: Convolutional autoencoder expects 3D input (batch, channels, length)
    autoencoder = WaveConvolutionalAutoencoder(input_shape=input_shape[1:], latent_dim=latent_dim)
    assert isinstance(autoencoder, WaveConvolutionalAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = tensor.random_normal(input_shape, dtype=tensor.float32)
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == input_shape
    assert tensor.shape(latent_z) == (input_shape[0], latent_dim)
    # For non-variational, mean and log_var should be None
    assert mean is None
    assert log_var is None

def test_waveconvolutionalautoencoder_variational_forward_shape(set_backend_fixture):
    """Test WaveConvolutionalAutoencoder forward pass shape with variational encoder."""
    input_shape = (32, 1, 100) # Batch, Channels, Sequence Length
    latent_dim = 10
    # Create with variational=True
    autoencoder = WaveConvolutionalAutoencoder(input_shape=input_shape[1:], latent_dim=latent_dim, variational=True)
    assert isinstance(autoencoder, WaveConvolutionalAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = tensor.random_normal(input_shape, dtype=tensor.float32)
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == input_shape
    assert tensor.shape(latent_z) == (input_shape[0], latent_dim)
    assert tensor.shape(mean) == (input_shape[0], latent_dim)
    assert tensor.shape(log_var) == (input_shape[0], latent_dim)


def test_create_wave_autoencoder_factory(set_backend_fixture):
    """Test create_wave_autoencoder factory function."""
    input_size = 100
    latent_dim = 10
    autoencoder = create_wave_autoencoder(input_size=input_size, latent_dim=latent_dim)
    assert isinstance(autoencoder, WaveAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = create_dummy_input_data(shape=(32, input_size))
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == (32, input_size)
    assert tensor.shape(latent_z) == (32, latent_dim)
    assert mean is None
    assert log_var is None

def test_create_wave_conv_autoencoder_factory(set_backend_fixture):
    """Test create_wave_conv_autoencoder factory function."""
    input_shape = (1, 100) # Channels, Sequence Length
    latent_dim = 10
    autoencoder = create_wave_conv_autoencoder(input_shape=input_shape, latent_dim=latent_dim)
    assert isinstance(autoencoder, WaveConvolutionalAutoencoder)
    assert isinstance(autoencoder, Module)
    input_data = tensor.random_normal((32,) + input_shape, dtype=tensor.float32) # Add batch dim
    reconstruction, latent_z, mean, log_var = autoencoder(input_data)
    assert tensor.shape(reconstruction) == tensor.shape(input_data)
    assert tensor.shape(latent_z) == (tensor.shape(input_data)[0], latent_dim)
    assert mean is None
    assert log_var is None

# TODO: Add tests for parameter registration
# TODO: Add tests for basic functionality (e.g., check if reconstruction is close to input for simple cases)
# TODO: Add tests for variational encoder's reparameterization trick (check distribution of z)
# TODO: Add tests for different activation functions and dropout rates
# TODO: Add tests for edge cases and invalid inputs