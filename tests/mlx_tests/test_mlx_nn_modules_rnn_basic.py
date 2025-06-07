# tests/mlx_tests/test_nn_rnn.py
from ember_ml import ops
from ember_ml.nn import tensor
# from ember_ml.nn import modules # Removed old import
from ember_ml.nn.modules import rnn # Import rnn submodule
from ember_ml.nn.modules.wiring import NCPMap, FullyConnectedNCPMap

# Note: Assumes conftest.py provides the mlx_backend fixture

# Helper function to get RNN parameters
def _get_rnn_params():
    input_size = 5
    hidden_size = 10
    batch_size = 4
    seq_len = 3
    return input_size, hidden_size, batch_size, seq_len

def test_rnn_forward_mlx(mlx_backend): # Use fixture
    """Tests RNN layer single step forward pass with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    layer = rnn.RNN(input_size, hidden_size, return_sequences=False, return_state=True) # Use rnn.RNN
    x_t = tensor.random_normal((batch_size, 1, input_size))  # Single time step
    h_prev = tensor.zeros((1, batch_size, hidden_size))  # Initial state
    
    # Forward pass with initial state
    output, new_state = layer(x_t, h_prev)
    
    # Check shapes
    assert tensor.shape(output) == (batch_size, hidden_size), "Output shape mismatch"
    assert tensor.shape(new_state[0]) == (1, batch_size, hidden_size), "State shape mismatch"

def test_rnn_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests RNN layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = rnn.RNN(input_size, hidden_size) # Use rnn.RNN
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = rnn.RNN(input_size, hidden_size, return_state=True) # Use rnn.RNN
    outputs_state, final_state = layer_state(x)
    h_final = final_state[0] # Unpack final state list
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"

def test_lstm_forward_mlx(mlx_backend): # Use fixture
    """Tests LSTM layer single step forward pass with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    layer = rnn.LSTM(input_size, hidden_size, return_sequences=False, return_state=True) # Use rnn.LSTM
    x_t = tensor.random_normal((batch_size, 1, input_size))  # Single time step
    h_prev = tensor.zeros((1, batch_size, hidden_size))  # Initial hidden state
    c_prev = tensor.zeros((1, batch_size, hidden_size))  # Initial cell state
    
    # Forward pass with initial state
    output, new_state = layer(x_t, (h_prev, c_prev))
    h_next, c_next = new_state  # Unpack state tuple
    
    # Check shapes
    assert tensor.shape(output) == (batch_size, hidden_size), "Output shape mismatch"
    assert tensor.shape(h_next) == (1, batch_size, hidden_size), "h_next shape mismatch"
    assert tensor.shape(c_next) == (1, batch_size, hidden_size), "c_next shape mismatch"

def test_lstm_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests LSTM layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = rnn.LSTM(input_size, hidden_size) # Use rnn.LSTM
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = rnn.LSTM(input_size, hidden_size, return_state=True) # Use rnn.LSTM
    outputs_state, final_state = layer_state(x)
    h_final, c_final = final_state # Unpack final state tuple
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"
    assert tensor.shape(c_final) == (1, batch_size, hidden_size), "c_final shape mismatch"

def test_gru_forward_mlx(mlx_backend): # Use fixture
    """Tests GRU layer single step forward pass with MLX backend."""
    input_size, hidden_size, batch_size, _ = _get_rnn_params()
    layer = rnn.GRU(input_size, hidden_size, return_sequences=False, return_state=True) # Use rnn.GRU
    x_t = tensor.random_normal((batch_size, 1, input_size))  # Single time step
    h_prev = tensor.zeros((1, batch_size, hidden_size))  # Initial state
    
    # Forward pass with initial state
    output, new_state = layer(x_t, h_prev)
    
    # Check shapes
    assert tensor.shape(output) == (batch_size, hidden_size), "Output shape mismatch"
    assert tensor.shape(new_state[0]) == (1, batch_size, hidden_size), "State shape mismatch"

def test_gru_layer_forward_mlx(mlx_backend): # Use fixture
    """Tests GRU layer forward pass shape with MLX backend."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    layer = rnn.GRU(input_size, hidden_size) # Use rnn.GRU
    x = tensor.random_normal((batch_size, seq_len, input_size))
    # Layer returns outputs by default. Set return_state=True to get final state.
    outputs = layer(x) # Get only outputs
    y = outputs
    # Now test with state return
    layer_state = rnn.GRU(input_size, hidden_size, return_state=True) # Use rnn.GRU
    outputs_state, final_state = layer_state(x)
    h_final = final_state[0] # Unpack final state list
    assert tensor.shape(y) == (batch_size, seq_len, hidden_size), "y shape mismatch"
    # Shape is (num_layers * num_directions, batch_size, hidden_size)
    assert tensor.shape(h_final) == (1, batch_size, hidden_size), "h_final shape mismatch"

def test_enhanced_ncpmap_mlx(mlx_backend):
    """Tests enhanced NCPMap with cell-specific parameters."""
    input_size, hidden_size, _, _ = _get_rnn_params()
    
    # Create an enhanced NCPMap with cell-specific parameters
    neuron_map = NCPMap(
        inter_neurons=ops.floor_divide(hidden_size, 2),
        command_neurons=ops.floor_divide(hidden_size, 4),
        motor_neurons=ops.floor_divide(hidden_size, 4),
        sensory_neurons=input_size,
        time_scale_factor=1.5,
        activation="relu",
        recurrent_activation="sigmoid",
        mode="default",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        mixed_memory=False,
        ode_unfolds=8,
        epsilon=1e-9,
        implicit_param_constraints=True,
        input_mapping="affine",
        output_mapping="affine",
        sparsity_level=0.3
    )
    
    # Verify the parameters were stored correctly
    assert neuron_map.time_scale_factor == 1.5, "time_scale_factor not stored correctly"
    assert neuron_map.activation == "relu", "activation not stored correctly"
    assert neuron_map.recurrent_activation == "sigmoid", "recurrent_activation not stored correctly"
    assert neuron_map.mode == "default", "mode not stored correctly"
    assert neuron_map.use_bias == True, "use_bias not stored correctly"
    assert neuron_map.ode_unfolds == 8, "ode_unfolds not stored correctly"
    assert neuron_map.epsilon == 1e-9, "epsilon not stored correctly"
    assert neuron_map.input_mapping == "affine", "input_mapping not stored correctly"
    assert neuron_map.output_mapping == "affine", "output_mapping not stored correctly"
    
    # Test serialization and deserialization
    config = neuron_map.get_config()
    new_map = NCPMap.from_config(config)
    
    # Verify the parameters were restored correctly
    assert new_map.time_scale_factor == 1.5, "time_scale_factor not restored correctly"
    assert new_map.activation == "relu", "activation not restored correctly"
    assert new_map.recurrent_activation == "sigmoid", "recurrent_activation not restored correctly"
    assert new_map.mode == "default", "mode not restored correctly"
    assert new_map.use_bias == True, "use_bias not restored correctly"
    assert new_map.ode_unfolds == 8, "ode_unfolds not restored correctly"
    assert new_map.epsilon == 1e-9, "epsilon not restored correctly"
    assert new_map.input_mapping == "affine", "input_mapping not restored correctly"
    assert new_map.output_mapping == "affine", "output_mapping not restored correctly"

def test_fully_connected_ncpmap_mlx(mlx_backend):
    """Tests FullyConnectedNCPMap with cell-specific parameters."""
    input_size, hidden_size, _, _ = _get_rnn_params()
    
    # Create a FullyConnectedNCPMap with cell-specific parameters
    neuron_map = FullyConnectedNCPMap(
        units=hidden_size,
        input_dim=input_size,
        output_dim=hidden_size,
        time_scale_factor=1.5,
        activation="relu",
        recurrent_activation="sigmoid",
        mode="default",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        mixed_memory=False,
        ode_unfolds=8,
        epsilon=1e-9,
        implicit_param_constraints=True,
        input_mapping="affine",
        output_mapping="affine",
        sparsity_level=0.3
    )
    
    # Verify the parameters were stored correctly
    assert neuron_map.time_scale_factor == 1.5, "time_scale_factor not stored correctly"
    assert neuron_map.activation == "relu", "activation not stored correctly"
    assert neuron_map.recurrent_activation == "sigmoid", "recurrent_activation not stored correctly"
    assert neuron_map.mode == "default", "mode not stored correctly"
    assert neuron_map.use_bias == True, "use_bias not stored correctly"
    assert neuron_map.ode_unfolds == 8, "ode_unfolds not stored correctly"
    assert neuron_map.epsilon == 1e-9, "epsilon not stored correctly"
    assert neuron_map.input_mapping == "affine", "input_mapping not stored correctly"
    assert neuron_map.output_mapping == "affine", "output_mapping not stored correctly"
    
    # Test serialization and deserialization
    config = neuron_map.get_config()
    new_map = FullyConnectedNCPMap.from_config(config)
    
    # Verify the parameters were restored correctly
    assert new_map.time_scale_factor == 1.5, "time_scale_factor not restored correctly"
    assert new_map.activation == "relu", "activation not restored correctly"
    assert new_map.recurrent_activation == "sigmoid", "recurrent_activation not restored correctly"
    assert new_map.mode == "default", "mode not restored correctly"
    assert new_map.use_bias == True, "use_bias not restored correctly"
    assert new_map.ode_unfolds == 8, "ode_unfolds not restored correctly"
    assert new_map.epsilon == 1e-9, "epsilon not restored correctly"
    assert new_map.input_mapping == "affine", "input_mapping not restored correctly"
    assert new_map.output_mapping == "affine", "output_mapping not restored correctly"

def test_cfc_with_enhanced_ncpmap_mlx(mlx_backend):
    """Tests CfC layer with enhanced NCPMap."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    
    # Create an enhanced NCPMap with cell-specific parameters
    neuron_map = NCPMap(
        inter_neurons=ops.floor_divide(hidden_size, 2),
        command_neurons=ops.floor_divide(hidden_size, 4),
        motor_neurons=ops.floor_divide(hidden_size, 4),
        sensory_neurons=input_size,
        time_scale_factor=1.0,
        activation="tanh",
        recurrent_activation="sigmoid",
        mode="default",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        sparsity_level=0.5
    )
    
    # Create CfC layer with the enhanced NCPMap
    layer = rnn.CfC(neuron_map=neuron_map, return_sequences=True) # Use rnn.CfC

    # Test forward pass
    x = tensor.random_normal((batch_size, seq_len, input_size))
    outputs = layer(x)
    
    # Verify output shape - use the actual output shape
    output_dim = tensor.shape(outputs)[2]
    assert tensor.shape(outputs) == (batch_size, seq_len, output_dim), "Output shape mismatch"
    
    # Test with return_state=True
    layer_state = rnn.CfC(neuron_map=neuron_map, return_sequences=True, return_state=True) # Use rnn.CfC
    outputs_state, final_state = layer_state(x)

    # Verify output and state shapes
    output_dim = tensor.shape(outputs_state)[2]
    assert tensor.shape(outputs_state) == (batch_size, seq_len, output_dim), "Output shape mismatch"
    assert isinstance(final_state, list), "Final state should be a list"
    # Get the actual shapes of the final state elements
    h_final_shape = tensor.shape(final_state[0])
    assert len(h_final_shape) == 2 and h_final_shape[0] == batch_size, "h_final shape mismatch"
    if len(final_state) > 1:
        t_final_shape = tensor.shape(final_state[1])
        assert len(t_final_shape) == 2 and t_final_shape[0] == batch_size, "t_final shape mismatch"

def test_ltc_with_enhanced_ncpmap_mlx(mlx_backend):
    """Tests LTC layer with enhanced NCPMap."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    
    # Create an enhanced NCPMap with cell-specific parameters
    neuron_map = NCPMap(
        inter_neurons=ops.floor_divide(hidden_size, 2),
        command_neurons=ops.floor_divide(hidden_size, 4),
        motor_neurons=ops.floor_divide(hidden_size, 4),
        sensory_neurons=input_size,
        time_scale_factor=1.0,
        activation="tanh",
        recurrent_activation="sigmoid",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
        input_mapping="affine",
        output_mapping="affine",
        sparsity_level=0.5
    )
    
    # Create LTC layer with the enhanced NCPMap
    layer = rnn.LTC(neuron_map=neuron_map, return_sequences=True) # Use rnn.LTC

    # Test forward pass
    x = tensor.random_normal((batch_size, seq_len, input_size))
    outputs = layer(x)
    
    # Verify output shape - use neuron_map.motor_neurons for output dimension
    assert tensor.shape(outputs) == (batch_size, seq_len, neuron_map.motor_neurons), "Output shape mismatch"
    
    # Test with return_state=True
    layer_state = rnn.LTC(neuron_map=neuron_map, return_sequences=True, return_state=True) # Use rnn.LTC
    outputs_state, final_state = layer_state(x)

    # Verify output and state shapes
    assert tensor.shape(outputs_state) == (batch_size, seq_len, neuron_map.motor_neurons), "Output shape mismatch"
    assert tensor.shape(final_state) == (batch_size, neuron_map.units), "h_final shape mismatch"

def test_ltc_with_fully_connected_ncpmap_mlx(mlx_backend):
    """Tests LTC layer with FullyConnectedNCPMap."""
    input_size, hidden_size, batch_size, seq_len = _get_rnn_params()
    
    # Create a FullyConnectedNCPMap with cell-specific parameters
    neuron_map = FullyConnectedNCPMap(
        units=hidden_size,
        input_dim=input_size,
        output_dim=ops.floor_divide(hidden_size, 4),  # Explicitly set output_dim to match motor_neurons
        time_scale_factor=1.0,
        activation="tanh",
        recurrent_activation="sigmoid",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
        input_mapping="affine",
        output_mapping="affine",
        sparsity_level=0.0
    )
    
    # Create LTC layer with the FullyConnectedNCPMap
    layer = rnn.LTC(neuron_map=neuron_map, return_sequences=True) # Use rnn.LTC

    # Test forward pass
    x = tensor.random_normal((batch_size, seq_len, input_size))
    outputs = layer(x)
    
    # Verify output shape - use neuron_map.output_dim for output dimension
    assert tensor.shape(outputs) == (batch_size, seq_len, neuron_map.output_dim), "Output shape mismatch"
    
    # Test with return_state=True
    layer_state = rnn.LTC(neuron_map=neuron_map, return_sequences=True, return_state=True) # Use rnn.LTC
    outputs_state, final_state = layer_state(x)

    # Verify output and state shapes
    assert tensor.shape(outputs_state) == (batch_size, seq_len, neuron_map.output_dim), "Output shape mismatch"
    assert tensor.shape(final_state) == (batch_size, neuron_map.units), "h_final shape mismatch"