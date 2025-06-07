import pytest
import numpy as np
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.wave.models.multi_sphere import (
    SphereProjection,
    MultiSphereProjection,
    MultiSphereEncoder,
    MultiSphereDecoder,
    MultiSphereModel,
    SphericalHarmonics,
    MultiSphereWaveModel,
    create_multi_sphere_model,
    create_multi_sphere_wave_model,
)
from ember_ml.wave.models.multi_sphere import MultiSphereHarmonicModel
from ember_ml.wave.models.multi_sphere import create_multi_sphere_harmonic_model
from ember_ml.nn.modules import Module # Needed for isinstance checks
from ember_ml.wave.memory.sphere_overlap import OverlapNetwork # Needed for MultiSphereWaveModel

@pytest.fixture(params=['torch'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Helper function to create dummy input data
def create_dummy_input_data(shape=(32, 4)):
    """Creates a dummy input tensor for multi-sphere models (assuming 4D vectors)."""
    return tensor.random_normal(shape, dtype=tensor.float32)

# Test cases for initialization and forward pass shapes

def test_sphereprojection_initialization_and_forward_shape(set_backend_fixture):
    """Test SphereProjection initialization and forward pass shape."""
    projection = SphereProjection()
    assert isinstance(projection, SphereProjection)
    assert isinstance(projection, Module)
    input_data = create_dummy_input_data(shape=(32, 4)) # Batch, Vector Dim
    output = projection(input_data)
    assert tensor.shape(output) == (32, 4) # Shape should be preserved after projection
    # Check if projected onto unit sphere (norm should be ~1)
    norm = ops.sqrt(stats.sum(ops.square(output), axis=-1))
    assert ops.allclose(norm, tensor.ones(tensor.shape(norm)), atol=1e-6)

def test_multisphereprojection_initialization_and_forward_shape(set_backend_fixture):
    """Test MultiSphereProjection initialization and forward pass shape."""
    num_segments = 5
    projection = MultiSphereProjection(num_segments=num_segments)
    assert isinstance(projection, MultiSphereProjection)
    assert isinstance(projection, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 4)) # Batch, Segments, Vector Dim
    output = projection(input_data)
    assert tensor.shape(output) == (32, num_segments, 4) # Shape should be preserved
    # Check if each segment is projected onto a unit sphere
    norm = ops.sqrt(stats.sum(ops.square(output), axis=-1))
    assert ops.allclose(norm, tensor.ones(tensor.shape(norm)), atol=1e-6)

def test_multisphereencoder_initialization_and_forward_shape(set_backend_fixture):
    """Test MultiSphereEncoder initialization and forward pass shape."""
    num_segments = 5
    latent_dim = 10
    encoder = MultiSphereEncoder(num_segments=num_segments, latent_dim=latent_dim)
    assert isinstance(encoder, MultiSphereEncoder)
    assert isinstance(encoder, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 4))
    output = encoder(input_data)
    assert tensor.shape(output) == (32, latent_dim)

def test_multispheredecoder_initialization_and_forward_shape(set_backend_fixture):
    """Test MultiSphereDecoder initialization and forward pass shape."""
    num_segments = 5
    latent_dim = 10
    decoder = MultiSphereDecoder(num_segments=num_segments, latent_dim=latent_dim)
    assert isinstance(decoder, MultiSphereDecoder)
    assert isinstance(decoder, Module)
    input_data = create_dummy_input_data(shape=(32, latent_dim))
    output = decoder(input_data)
    assert tensor.shape(output) == (32, num_segments, 4)

def test_multispheremodel_initialization_and_forward_shape(set_backend_fixture):
    """Test MultiSphereModel initialization and forward pass shape."""
    num_segments = 5
    latent_dim = 10
    model = MultiSphereModel(num_segments=num_segments, latent_dim=latent_dim)
    assert isinstance(model, MultiSphereModel)
    assert isinstance(model, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 4))
    reconstruction, latent_z = model(input_data)
    assert tensor.shape(reconstruction) == (32, num_segments, 4)
    assert tensor.shape(latent_z) == (32, latent_dim)

def test_sphericalharmonics_initialization_and_forward_shape(set_backend_fixture):
    """Test SphericalHarmonics initialization and forward pass shape."""
    max_degree = 2 # Example degree
    harmonics = SphericalHarmonics(max_degree=max_degree)
    assert isinstance(harmonics, SphericalHarmonics)
    assert isinstance(harmonics, Module)
    # Input to spherical harmonics is typically 3D vectors (x, y, z)
    input_data = create_dummy_input_data(shape=(32, 3))
    output = harmonics(input_data)
    # The output size depends on the max_degree: (max_degree + 1)^2
    expected_output_size = (max_degree + 1)**2
    assert tensor.shape(output) == (32, expected_output_size)

def test_multisphereharmonicmodel_initialization_and_forward_shape(set_backend_fixture):
    """Test MultiSphereHarmonicModel initialization and forward pass shape."""
    num_segments = 5
    max_degree = 2
    output_size = 10
    model = MultiSphereHarmonicModel(num_segments=num_segments, max_degree=max_degree, output_size=output_size)
    assert isinstance(model, MultiSphereHarmonicModel)
    assert isinstance(model, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 3)) # Input is 3D vectors per segment
    output = model(input_data)
    assert tensor.shape(output) == (32, output_size)

def test_multispherewavemodel_initialization(set_backend_fixture):
    """Test MultiSphereWaveModel initialization."""
    num_spheres = 5
    # Create a dummy OverlapNetwork (assuming a simple one for testing init)
    # This requires creating dummy SphereOverlap objects
    from ember_ml.wave.memory.sphere_overlap import SphereOverlap
    overlaps = [SphereOverlap(sphere_a=0, sphere_b=1, reflection=0.5, transmission=0.5)] # Example overlap
    overlap_network = OverlapNetwork(overlaps=overlaps, num_spheres=num_spheres)

    model = MultiSphereWaveModel(
        num_spheres=num_spheres,
        overlap_network=overlap_network,
        reflection=0.5, # These might be redundant if using OverlapNetwork, check implementation
        transmission=0.5,
        noise_std=0.1,
        input_size=4, # Assuming input is 4D vectors
        output_size=10
    )
    assert isinstance(model, MultiSphereWaveModel)
    assert isinstance(model, Module)
    assert hasattr(model, 'spheres') # Check if spheres are initialized
    assert len(model.spheres) == num_spheres

# Note: Testing MultiSphereWaveModel.run requires simulating dynamics over time,
# which can be complex and potentially backend-dependent in terms of performance and exact numerical results.
# This test will focus on basic execution without checking for specific dynamic correctness.
def test_multispherewavemodel_run_basic_execution(set_backend_fixture):
    """Test MultiSphereWaveModel run method (basic execution)."""
    num_spheres = 2
    # Create a dummy OverlapNetwork
    from ember_ml.wave.memory.sphere_overlap import SphereOverlap
    overlaps = [SphereOverlap(sphere_a=0, sphere_b=1, reflection=0.5, transmission=0.5)]
    overlap_network = OverlapNetwork(overlaps=overlaps, num_spheres=num_spheres)

    model = MultiSphereWaveModel(
        num_spheres=num_spheres,
        overlap_network=overlap_network,
        reflection=0.5,
        transmission=0.5,
        noise_std=0.01,
        input_size=4,
        output_size=1 # Simple scalar output
    )

    steps = 10
    # Create dummy input waves sequence (Batch, Steps, Input Size)
    input_waves_seq = create_dummy_input_data(shape=(1, steps, 4))
    # Create dummy gating sequence (Batch, Steps, Num Spheres) - assuming scalar gating per sphere
    gating_seq = tensor.ones((1, steps, num_spheres), dtype=tensor.float32)

    try:
        history = model.run(steps=steps, input_waves_seq=input_waves_seq, gating_seq=gating_seq)
        # The history should contain the state of each sphere over time
        # Assuming history returns (Steps, Num Spheres, Vector Dim)
        assert tensor.shape(history) == (steps, num_spheres, 4)
    except Exception as e:
        pytest.fail(f"MultiSphereWaveModel.run failed: {e}")


def test_create_multi_sphere_model_factory(set_backend_fixture):
    """Test create_multi_sphere_model factory function."""
    num_segments = 5
    latent_dim = 10
    model = create_multi_sphere_model(num_segments=num_segments, latent_dim=latent_dim)
    assert isinstance(model, MultiSphereModel)
    assert isinstance(model, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 4))
    reconstruction, latent_z = model(input_data)
    assert tensor.shape(reconstruction) == (32, num_segments, 4)
    assert tensor.shape(latent_z) == (32, latent_dim)

def test_create_multi_sphere_harmonic_model_factory(set_backend_fixture):
    """Test create_multi_sphere_harmonic_model factory function."""
    num_segments = 5
    max_degree = 2
    output_size = 10
    model = create_multi_sphere_harmonic_model(num_segments=num_segments, max_degree=max_degree, output_size=output_size)
    assert isinstance(model, MultiSphereHarmonicModel)
    assert isinstance(model, Module)
    input_data = create_dummy_input_data(shape=(32, num_segments, 3)) # Input is 3D vectors per segment
    output = model(input_data)
    assert tensor.shape(output) == (32, output_size)

def test_create_multi_sphere_wave_model_factory(set_backend_fixture):
    """Test create_multi_sphere_wave_model factory function."""
    num_spheres = 5
    input_size = 4
    output_size = 10
    # Create a dummy OverlapNetwork for the factory
    from ember_ml.wave.memory.sphere_overlap import SphereOverlap
    overlaps = [SphereOverlap(sphere_a=0, sphere_b=1, reflection=0.5, transmission=0.5)]
    overlap_network = OverlapNetwork(overlaps=overlaps, num_spheres=num_spheres)

    model = create_multi_sphere_wave_model(
        num_spheres=num_spheres,
        overlap_network=overlap_network,
        input_size=input_size,
        output_size=output_size
    )
    assert isinstance(model, MultiSphereWaveModel)
    assert isinstance(model, Module)
    assert hasattr(model, 'spheres') # Check if spheres are initialized
    assert len(model.spheres) == num_spheres

# TODO: Add tests for parameter registration
# TODO: Add tests for different configurations (e.g., different overlap networks)
# TODO: Add tests for edge cases and invalid inputs
# TODO: Add more detailed tests for MultiSphereWaveModel.run, potentially checking
#       for expected state changes or properties over time, although exact numerical
#       results might vary between backends.