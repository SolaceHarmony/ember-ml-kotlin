import pytest
import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import EnhancedNCPMap
from ember_ml.nn.modules.rnn import seCfC

@pytest.fixture
def se_cfc_setup(numpy_backend):
    """Set up test fixtures for seCfC tests."""
    # Set random seed for reproducibility
    tensor.set_seed(42)

    # Create a small neuron map for testing
    neuron_map = EnhancedNCPMap(
        inter_neurons=4,
        command_neurons=2,
        motor_neurons=2,
        sensory_neurons=3,
        neuron_type="cfc",
        time_scale_factor=1.0,
        activation="tanh",
        recurrent_activation="sigmoid",
        sparsity_level=0.5,
        seed=42
    )

    # Create a seCfC model for testing
    model = seCfC(
        neuron_map=neuron_map,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
        regularization_strength=0.01
    )

    # Create test data
    batch_size = 2
    time_steps = 5
    input_features = 3
    output_features = 2

    inputs = tensor.random_normal(
        (batch_size, time_steps, input_features))

    return neuron_map, model, inputs, batch_size, time_steps, input_features, output_features

def test_initialization(se_cfc_setup, numpy_backend):
    """Test that the model initializes correctly."""
    neuron_map, model, _, _, _, _, _ = se_cfc_setup
    assert isinstance(model, seCfC)
    assert model.neuron_map.units == 11  # 3 sensory + 4 inter + 2 command + 2 motor
    assert model.neuron_map.output_dim == 2
    assert model.neuron_map.input_dim == 3

def test_build(se_cfc_setup, numpy_backend):
    """Test that the model builds correctly."""
    _, model, inputs, _, _, _, _ = se_cfc_setup

    # Build the model
    model.build(tensor.shape(inputs))

    # Check that the model is built
    assert model.built

    # Check that the parameters are initialized
    assert model.kernel is not None
    assert model.recurrent_kernel is not None
    assert model.bias is not None

    # Check parameter shapes
    # Note: units is now 11 (3+4+2+2)
    assert tensor.shape(model.kernel.data) == (3, 11 * 4)  # (input_dim, units * 4)
    assert tensor.shape(model.recurrent_kernel.data) == (11, 11 * 4)  # (units, units * 4)
    assert tensor.shape(model.bias.data) == (11 * 4,)  # (units * 4,)

def test_forward_pass(se_cfc_setup, numpy_backend):
    """Test the forward pass of the model."""
    _, model, inputs, batch_size, time_steps, _, output_features = se_cfc_setup

    # Build the model first
    model.build(tensor.shape(inputs))

    # Forward pass
    outputs = model(inputs)

    # Check output shape
    assert tensor.shape(outputs) == (batch_size, time_steps, output_features)

def test_return_state(se_cfc_setup, numpy_backend):
    """Test that the model can return state."""
    neuron_map, _, inputs, batch_size, time_steps, _, output_features = se_cfc_setup

    # Create a model that returns state
    model = seCfC(
        neuron_map=neuron_map,
        return_sequences=True,
        return_state=True,
        go_backwards=False,
        regularization_strength=0.01
    )

    # Build the model first
    model.build(tensor.shape(inputs))

    # Forward pass
    outputs, states = model(inputs)

    # Check output shape
    assert tensor.shape(outputs) == (batch_size, time_steps, output_features)

    # Check state shape
    assert len(states) == 2
    assert tensor.shape(states[0]) == (batch_size, 11)  # (batch_size, units)
    assert tensor.shape(states[1]) == (batch_size, 11)  # (batch_size, units)

def test_go_backwards(se_cfc_setup, numpy_backend):
    """Test that the model can process sequences backwards."""
    neuron_map, _, inputs, batch_size, time_steps, _, output_features = se_cfc_setup

    # Create a model that processes sequences backwards
    model = seCfC(
        neuron_map=neuron_map,
        return_sequences=True,
        return_state=False,
        go_backwards=True,
        regularization_strength=0.01
    )

    # Build the model first
    model.build(tensor.shape(inputs))

    # Forward pass
    outputs = model(inputs)

    # Check output shape
    assert tensor.shape(outputs) == (batch_size, time_steps, output_features)

def test_regularization_loss(se_cfc_setup, numpy_backend):
    """Test that the model computes regularization loss."""
    _, model, inputs, _, _, _, _ = se_cfc_setup

    # Build the model
    model.build(tensor.shape(inputs))

    # Compute regularization loss
    reg_loss = model.get_regularization_loss()

    # Check that the loss is a scalar
    assert tensor.shape(reg_loss) == ()

    # Check that the loss is non-negative
    assert tensor.to_numpy(reg_loss) >= 0.0

def test_reset_state(se_cfc_setup, numpy_backend):
    """Test that the model can reset state."""
    _, model, _, _, _, _, _ = se_cfc_setup

    # Reset state
    states = model.reset_state(batch_size=3)

    # Check state shape
    assert len(states) == 2
    assert tensor.shape(states[0]) == (3, 11)  # (batch_size, units)
    assert tensor.shape(states[1]) == (3, 11)  # (batch_size, units)

    # Check that the state is zeros
    assert ops.allclose(tensor.to_numpy(states[0]), 0.0)
    assert ops.allclose(tensor.to_numpy(states[1]), 0.0)

def test_get_config(se_cfc_setup, numpy_backend):
    """Test that the model can get its configuration."""
    _, model, _, _, _, _, _ = se_cfc_setup

    # Get config
    config = model.get_config()

    # Check that the config contains the expected keys
    assert "neuron_map" in config
    assert "return_sequences" in config
    assert "return_state" in config
    assert "go_backwards" in config
    assert "regularization_strength" in config

    # Check that the config values are correct
    assert config["return_sequences"] == True
    assert config["return_state"] == False
    assert config["go_backwards"] == False
    assert config["regularization_strength"] == 0.01

# Note: The training test involves gradient tape and optimizer, which are backend dependent.
# This test will be included in each backend-specific file.
def test_training(se_cfc_setup, numpy_backend):
    """Test that the model can be trained."""
    _, model, inputs, batch_size, time_steps, _, output_features = se_cfc_setup

    # Build the model
    model.build(tensor.shape(inputs))

    # Create target data
    targets = tensor.random_normal((batch_size, time_steps, output_features))

    # Create optimizer
    optimizer = ops.optimizers.Adam(learning_rate=0.01)

    # Initial forward pass
    with ops.GradientTape() as tape:
        outputs = model(inputs)
        loss = ops.mse(targets, outputs)

    # Initial loss - use tensor operations to get the value
    initial_loss = tensor.to_numpy(loss)

    # Train for a few steps
    for _ in range(5):
        with ops.GradientTape() as tape:
            outputs = model(inputs)
            loss = ops.mse(targets, outputs)

        # Compute gradients
        gradients = tape.gradient(loss, model.parameters())

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.parameters()))

    # Final forward pass
    with ops.GradientTape() as tape:
        outputs = model(inputs)
        loss = ops.mse(targets, outputs)

    # Final loss - use tensor operations to get the value
    final_loss = tensor.to_numpy(loss)

    # Check that the loss decreased
    assert final_loss < initial_loss