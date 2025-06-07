# tests/mlx_tests/test_nn_modules.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules
from ember_ml.ops import stats # Import stats needed for Dense activation check

# Note: Assumes conftest.py provides the mlx_backend fixture

# --- Helper Module ---
# Re-defined here for clarity
class SimpleModule(modules.Module):
    """A simple module for testing parameter registration."""
    def __init__(self, size):
        super().__init__()
        self.param1 = modules.Parameter(tensor.zeros(size))
        self.param2 = modules.Parameter(tensor.ones((size, size)))
        self.non_param = tensor.arange(size)
        units_val = tensor.item(ops.multiply(tensor.convert_to_tensor(size), tensor.convert_to_tensor(2)))
        self.nested = modules.Dense(input_dim=size, units=units_val)

def test_module_parameter_registration_mlx(mlx_backend): # Use fixture
    """Tests parameter registration with MLX backend."""
    module = SimpleModule(size=3)
    registered_params = list(module.parameters())
    assert len(registered_params) == 4, f"Incorrect number of parameters: {len(registered_params)}"
    param_shapes = [tensor.shape(p.data) for p in registered_params]
    expected_shapes = [(3,), (3, 3), (3, 6), (6,)]
    assert set(param_shapes) == set(expected_shapes), "Parameter shapes mismatch"
    is_non_param_registered = False
    for p in registered_params:
         try:
              if ops.allclose(p.data, module.non_param):
                   is_non_param_registered = True
                   break
         except: pass
    assert not is_non_param_registered, "Non-parameter tensor registered"

def test_parameter_properties_mlx(mlx_backend): # Use fixture
    """Tests Parameter properties with MLX backend."""
    data = tensor.convert_to_tensor([1.0, 2.0])
    param = modules.Parameter(data, requires_grad=True)
    # Import backend module locally for type check
    import mlx.core as mx
    assert isinstance(param.data, mx.array), "Data not mlx.array" # Direct backend type check
    assert param.requires_grad is True, "requires_grad not True"
    assert tensor.shape(param.data) == tensor.shape(data), "Shape mismatch"
    assert ops.allclose(param.data, data), "Data content mismatch"
    param_default = modules.Parameter(tensor.ones(3))
    assert param_default.requires_grad is True, "Default requires_grad not True"

def test_dense_forward_shape_mlx(mlx_backend): # Use fixture
    """Tests Dense forward pass shape with MLX backend."""
    in_features = 5
    out_features = 3
    layer = modules.Dense(input_dim=in_features, units=out_features)
    batch_size = 4
    input_tensor = tensor.random_normal((batch_size, in_features))
    output = layer(input_tensor)
    # Removed direct backend type check - rely on shape/content checks via ops/tensor API
    expected_shape = (batch_size, out_features)
    assert tensor.shape(output) == expected_shape, f"Shape mismatch: got {tensor.shape(output)}"

def test_dense_parameters_mlx(mlx_backend): # Use fixture
    """Tests Dense parameters with MLX backend."""
    in_features = 5
    out_features = 3
    layer = modules.Dense(input_dim=in_features, units=out_features)
    params = list(layer.parameters())
    assert len(params) == 2, "Should have weight and bias"
    weight_found = any(tensor.shape(p.data) == (in_features, out_features) for p in params)
    bias_found = any(tensor.shape(p.data) == (out_features,) for p in params)
    assert weight_found, "Weight not found/wrong shape"
    assert bias_found, "Bias not found/wrong shape"

def test_dense_no_bias_mlx(mlx_backend): # Use fixture
    """Tests Dense without bias with MLX backend."""
    layer = modules.Dense(input_dim=4, units=2, use_bias=False)
    params = list(layer.parameters())
    assert len(params) == 1, "Should only have weight"
    assert tensor.shape(params[0].data) == (4, 2), "Weight shape incorrect"
    input_tensor = tensor.random_normal((3, 4))
    output = layer(input_tensor)
    assert tensor.shape(output) == (3, 2), "Forward shape incorrect"

def test_dense_activation_mlx(mlx_backend): # Use fixture
    """Tests Dense with activation with MLX backend."""
    in_features = 4
    out_features = 3
    layer = modules.Dense(input_dim=in_features, units=out_features, activation='relu')
    batch_size = 2
    input_tensor = tensor.convert_to_tensor([[-1.0, -0.5, 0.5, 1.0], [0.1, -0.1, 2.0, -2.0]])
    output = layer(input_tensor)
    assert tensor.shape(output) == (batch_size, out_features), "Shape mismatch"
    # Use tensor operations to find the minimum value
    # First flatten the tensor to 1D
    flattened = tensor.reshape(output, (-1,))
    # Iterate through the flattened tensor to find the minimum value
    min_val = float('inf')
    for i in range(tensor.shape(flattened)[0]):
        val = tensor.item(flattened[i:i+1])
        min_val = min(min_val, val)
    threshold = ops.subtract(tensor.convert_to_tensor(0.0), tensor.convert_to_tensor(1e-7))
    assert min_val >= tensor.item(threshold), f"ReLU output negative: {min_val}"

def test_ncp_instantiation_shape_mlx(mlx_backend): # Use fixture
    """Tests NCP instantiation and shape with MLX backend."""
    # Test previously skipped due to initialization issues, now re-enabled.
    # The following code is skipped
    neuron_map = modules.wiring.NCPMap(inter_neurons=8, command_neurons=4, motor_neurons=3, sensory_neurons=5, seed=42)
    input_size = neuron_map.units
    neuron_map.build(input_size)
    ncp_module = modules.NCP(neuron_map=neuron_map)
    batch_size = 2
    input_tensor = tensor.random_normal((batch_size, input_size))
    output = ncp_module(input_tensor)
    # Removed direct backend type check - rely on shape/content checks via ops/tensor API
    expected_shape = (batch_size, neuron_map.output_dim)
    assert tensor.shape(output) == expected_shape, f"Shape mismatch: got {tensor.shape(output)}"
    assert len(list(ncp_module.parameters())) > 0, "No parameters found"

def test_autoncp_instantiation_shape_mlx(mlx_backend): # Use fixture
    """Tests AutoNCP instantiation and shape with MLX backend."""
    # Test previously skipped due to initialization issues, now re-enabled.
    
    # The following code should now work
    units = 15
    output_size = 4
    input_size = 6
    autoncp_module = modules.AutoNCP(units=units, output_size=output_size, sparsity_level=0.5, seed=43)
    autoncp_module.build((None, input_size))
    batch_size = 2
    input_tensor = tensor.random_normal((batch_size, input_size))
    output = autoncp_module(input_tensor)
    # Removed direct backend type check - rely on shape/content checks via ops/tensor API
    expected_shape = (batch_size, output_size)
    assert tensor.shape(output) == expected_shape, f"Shape mismatch: got {tensor.shape(output)}"
    assert len(list(autoncp_module.parameters())) > 0, "No parameters found"
    assert hasattr(autoncp_module, 'neuron_map'), "No neuron_map attribute"
    assert autoncp_module.neuron_map.units == units, "Units mismatch"
    assert autoncp_module.neuron_map.output_dim == output_size, "Output size mismatch"