"""
Liquid Neural Network Implementations
-----------------------------------

This repository provides implementations of Liquid Neural Networks (LNNs) using:
1. TensorFlow + ncps library (in liquidanomaly.py, liquid_control_experiments.py, liquidtrainer.py)
2. Apple MLX framework (in this file)

Key Features:
- Multi-scale temporal processing with CfC (Closed-form Continuous-time) cells
- Mixed memory mechanism for adaptive time constants
- Gradient-stable training with proper initialization

Usage:
1. For TensorFlow-based implementation:
   ```python
   from liquidanomaly import LiquidAnomalyDetector
   detector = LiquidAnomalyDetector()
   ```

2. For MLX-based implementation (Apple Silicon):
   ```python
   from mlxncps import MultiScaleCfC
   model = MultiScaleCfC(input_dim=10)
   ```

The MLX implementation is optimized for Apple Silicon and provides similar 
functionality to the TensorFlow version, but with native Metal acceleration.
"""

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not installed. This module requires Apple MLX framework.")
    HAS_MLX = False

import numpy as np
import math

from ember_ml.nn import tensor

if HAS_MLX:
    class ConvLSTM2DCell(nn.Module):
        """Custom ConvLSTM2D cell for spatiotemporal feature extraction"""
        def __init__(self, input_dim, hidden_dim, kernel_size):
            super().__init__()
            pad = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad),
                nn.Sigmoid()
            )
            
        def __call__(self, x, hidden):
            h_prev, c_prev = hidden
            combined = mx.concatenate([x, h_prev], axis=1)
            gates = self.conv(combined)
            i, f, g, o = mx.split(gates, 4, axis=1)
            
            c_next = f * c_prev + i * mx.tanh(g)
            h_next = o * mx.tanh(c_next)
            return h_next, c_next

    class CfC(nn.Module):
        """Closed-form Continuous-time cell with NCP wiring"""
        def __init__(self, input_size, units, wiring):
            super().__init__()
            self.units = units
            self.wiring = wiring
            self.sensory = nn.Linear(input_size, wiring['sensory'])
            self.inter = nn.Linear(wiring['sensory'], wiring['inter'])
            self.command = nn.Linear(wiring['inter'], wiring['command'])
            self.motor = nn.Linear(wiring['command'], wiring['motor'])
            self._initialize_sparsity(wiring)
            
        def _initialize_sparsity(self, wiring):
            self.sensory.weight *= self._bernoulli_mask(wiring['p_sensory'])
            self.inter.weight *= self._bernoulli_mask(wiring['p_inter'])
            self.command.weight *= self._binomial_mask(wiring['p_command'])
            
        def _bernoulli_mask(self, p):
            return mx.random.bernoulli(p, self.sensory.weight.shape)
            
        def _binomial_mask(self, n):
            return mx.random.binomial(n, 0.5, self.inter.weight.shape)

        def __call__(self, x, h):
            s = mx.tanh(self.sensory(x))
            i = mx.sigmoid(self.inter(s) + h)
            c = mx.relu(self.command(i))
            m = self.motor(c)
            return m, c

    class ECGClassifier(nn.Module):
        """Complete model architecture combining ConvLSTM2D and NCP"""
        def __init__(self):
            super().__init__()
            self.conv_lstm = ConvLSTM2DCell(
                input_dim=12,  # 12-lead ECG
                hidden_dim=64,
                kernel_size=3
            )
            self.ncp = CfC(
                input_size=64,
                units=28,
                wiring={
                    'sensory': 75,
                    'inter': 14,
                    'command': 14,
                    'motor': 6,
                    'p_sensory': 0.6,
                    'p_inter': 0.4,
                    'p_command': 0.3
                }
            )
            self.classifier = nn.Linear(6, 6)

        def __call__(self, x):
            batch_size, time_steps = x.shape[:2]
            h = mx.zeros((batch_size, 64, x.shape[2], x.shape[3]))
            c = mx.zeros_like(h)
            for t in range(time_steps):
                h, c = self.conv_lstm(x[:, t], (h, c))
            features = h.mean(axis=(2,3))
            hidden = mx.zeros((batch_size, 14))
            output, _ = self.ncp(features, hidden)
            return self.classifier(output)

    log_entries = {
        "Log_ID": range(1, 101),
        "Timestamp": tensor.linspace(1000, 2000, 100),
        "Location": ops.random_choice(["Switch_A", "Switch_B", "Switch_C", "Switch_D", "Switch_E"], 100),
        "Message": ops.random_choice([
            "Link down", "Link up", "High latency detected", "Packet loss detected",
            "Authentication failure", "Configuration mismatch", "Power supply issue"
        ], 100),
        "Severity": ops.random_choice(["Low", "Medium", "High", "Critical"], 100)
    }

    log_ids = tensor.convert_to_tensor(log_entries["Log_ID"])
    timestamps = tensor.convert_to_tensor(log_entries["Timestamp"])
    locations = tensor.convert_to_tensor(log_entries["Location"])
    messages = tensor.convert_to_tensor(log_entries["Message"])
    severities = tensor.convert_to_tensor(log_entries["Severity"])

    unique_locations = np.unique(locations)
    location_mapping = {loc: i for i, loc in enumerate(unique_locations)}
    unique_messages = np.unique(messages)
    message_mapping = {msg: i for i, msg in enumerate(unique_messages)}
    unique_severities= np.unique(severities)
    severity_mapping = {sev: i for i, sev in enumerate(unique_severities)}

    locations_encoded = tensor.convert_to_tensor([location_mapping[loc] for loc in locations])
    messages_encoded = tensor.convert_to_tensor([message_mapping[msg] for msg in messages])
    severities_encoded = tensor.convert_to_tensor([severity_mapping[sev] for sev in severities])

    log_ids_mx = tensor.convert_to_tensor(log_ids)
    timestamps_mx = tensor.convert_to_tensor(timestamps)
    locations_mx = tensor.convert_to_tensor(locations_encoded)
    messages_mx = tensor.convert_to_tensor(messages_encoded)
    severities_mx = tensor.convert_to_tensor(severities_encoded)

    embedding_dim_location = 10
    embedding_dim_message = 20
    embedding_dim_severity = 5

    location_embedding = nn.Embedding(len(unique_locations), embedding_dim_location)
    message_embedding = nn.Embedding(len(unique_messages), embedding_dim_message)
    severity_embedding = nn.Embedding(len(unique_severities), embedding_dim_severity)

    test_location_input = tensor.convert_to_tensor([1, 2])
    embedded_locations = location_embedding(test_location_input)
    print(f"Shape of embedded locations: {embedded_locations.shape}")

    test_message_input = tensor.convert_to_tensor([1-3])
    embedded_messages = message_embedding(test_message_input)
    print(f"Shape of embedded messages: {embedded_messages.shape}")

    test_severity_input = tensor.convert_to_tensor([1])
    embedded_severities = severity_embedding(test_severity_input)
    print(f"Shape of embedded severities: {embedded_severities.shape}")

    class ConvLSTM2D(nn.Module):
        """Spatiotemporal feature extractor for ECG signals"""
        def __init__(self, input_channels=12, hidden_dim=64):
            super().__init__()
            self.conv_lstm_cell = nn.LSTMCell(
                input_size=input_channels,
                hidden_size=hidden_dim,
                num_layers=2
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def __call__(self, x):
            batch, time = x.shape[:2]
            h = mx.zeros((batch, 64))
            c = mx.zeros((batch, 64))
            for t in range(time):
                h, c = self.conv_lstm_cell(x[:, t], (h, c))
                h = self.layer_norm(h)
            return h

    class ConvLSTM2DCell(nn.Module):
        """Convolutional LSTM Cell with PyTorch-compatible initialization"""
        def __init__(self, input_dim, hidden_dim, kernel_size=3):
            super().__init__()
            self.hidden_dim = hidden_dim
            pad = kernel_size // 2
            
            # Convolutional gates (input, forget, cell, output)
            self.conv = nn.Conv2d(
                input_dim + hidden_dim,
                4 * hidden_dim,
                kernel_size=kernel_size,
                padding=pad
            )
            
            # Initialize weights using PyTorch-style initialization
            k = 1 / math.sqrt(hidden_dim)
            self.conv.weight = mx.random.uniform(-k, k, self.conv.weight.shape)
            self.conv.bias = mx.zeros(4 * hidden_dim)

        def __call__(self, x, hidden):
            h_prev, c_prev = hidden
            combined = mx.concatenate([x, h_prev], axis=1)
            gates = self.conv(combined)
            
            # Split into input/forget/cell/output gates
            i, f, g, o = mx.split(gates, 4, axis=1)
            
            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            c_next = f * c_prev + i * g
            h_next = o * mx.tanh(c_next)
            
            return h_next, c_next

    class LTCell(nn.Module):
        """Liquid Time-Constant Cell with adaptive time dynamics"""
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.input_gate = nn.Linear(input_dim, hidden_dim)
            self.forget_gate = nn.Linear(input_dim, hidden_dim)
            self.time_constant = nn.Linear(input_dim, hidden_dim)
            self.output_gate = nn.Linear(input_dim, hidden_dim)
            
            # Initialize parameters using kaiming uniform
            nn.init.kaiming_uniform(self.input_gate.weight)
            nn.init.kaiming_uniform(self.forget_gate.weight)
            nn.init.kaiming_uniform(self.time_constant.weight)
            nn.init.kaiming_uniform(self.output_gate.weight)

        def __call__(self, x, h_prev):
            i = mx.sigmoid(self.input_gate(x))
            f = mx.sigmoid(self.forget_gate(x))
            tc = mx.sigmoid(self.time_constant(x))
            o = mx.sigmoid(self.output_gate(x))
            
            # Liquid time-constant dynamics
            h = tc * mx.tanh(f * h_prev + i) + (1 - tc) * h_prev
            return o * h

    class CLTC(nn.Module):
        """Complete CLTC Architecture for Spatiotemporal Data"""
        def __init__(self, input_channels=12, hidden_dim=64, num_classes=6):
            super().__init__()
            self.conv_lstm = ConvLSTM2DCell(input_channels, hidden_dim)
            self.ltc = LTCell(hidden_dim, 128)
            self.classifier = nn.Linear(128, num_classes)
            
            # Spatial pooling parameters
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def __call__(self, x):
            """
            Args:
                x: Input tensor of shape [batch, time, height, width, channels]
                
            Returns:
                Class probabilities [batch, num_classes]
            """
            batch_size, time_steps = x.shape[:2]
            
            # Initialize ConvLSTM states
            h_conv = mx.zeros((batch_size, self.conv_lstm.hidden_dim, *x.shape[2:4]))
            c_conv = mx.zeros_like(h_conv)
            
            # Initialize LTC state
            h_ltc = mx.zeros((batch_size, 128))
            
            for t in range(time_steps):
                # Process each timestep through ConvLSTM
                h_conv, c_conv = self.conv_lstm(x[:, t], (h_conv, c_conv))
                
                # Spatial pooling and squeeze dimensions
                features = self.pool(h_conv).squeeze(axis=(-2, -1))
                
                # Update LTC state
                h_ltc = self.ltc(features, h_ltc)
            
            return self.classifier(h_ltc)


    class LSTMCell(nn.Module):
        """MLX implementation of LSTM cell with PyTorch-compatible initialization"""
        
        def __init__(self, input_size, hidden_size, bias=True):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.bias = bias

            # Weight initialization parameters
            k = 1 / math.sqrt(hidden_size)
            self._init_range = (-math.sqrt(k), math.sqrt(k))
            
            # Input-hidden weights [4*hidden_size, input_size]
            self.weight_ih = nn.Linear(input_size, 4*hidden_size, bias=False)
            # Hidden-hidden weights [4*hidden_size, hidden_size]
            self.weight_hh = nn.Linear(hidden_size, 4*hidden_size, bias=False)
            
            if bias:
                self.bias_ih = mx.zeros(4*hidden_size)
                self.bias_hh = mx.zeros(4*hidden_size)
            else:
                self.bias_ih = None
                self.bias_hh = None
                
            self._init_weights()

        def _init_weights(self):
            """PyTorch-style weight initialization"""
            # Initialize input-hidden weights
            self.weight_ih.weight = mx.random.uniform(
                low=self._init_range[0],
                high=self._init_range[1],
                shape=self.weight_ih.weight.shape
            )
            
            # Initialize hidden-hidden weights
            self.weight_hh.weight = mx.random.uniform(
                low=self._init_range[0],
                high=self._init_range[1],
                shape=self.weight_hh.weight.shape
            )

        def __call__(self, x, hidden):
            """
            Args:
                x: Input tensor of shape (batch, input_size)
                hidden: Tuple of (h_prev, c_prev) both shape (batch, hidden_size)
                
            Returns:
                h_next: Next hidden state (batch, hidden_size)
                c_next: Next cell state (batch, hidden_size)
            """
            h_prev, c_prev = hidden
            
            # Combine input and hidden transformations
            gates = (self.weight_ih(x) + 
                    self.weight_hh(h_prev))
            
            # Add biases if enabled
            if self.bias:
                gates += self.bias_ih + self.bias_hh

            # Split combined gates matrix
            i, f, g, o = mx.split(gates, 4, axis=-1)
            
            # Apply activation functions
            i = mx.sigmoid(i)  # Input gate
            f = mx.sigmoid(f)  # Forget gate
            g = mx.tanh(g)     # Cell gate
            o = mx.sigmoid(o)  # Output gate

            # Update cell state
            c_next = f * c_prev + i * g
            
            # Compute new hidden state
            h_next = o * mx.tanh(c_next)
            
            return h_next, c_next

        @property
        def state_size(self):
            return self.hidden_size

        @property
        def parameters(self):
            """Return all trainable parameters with Michael Khany's ordering"""
            params = [
                self.weight_ih.weight,
                self.weight_hh.weight
            ]
            if self.bias:
                params.extend([self.bias_ih, self.bias_hh])
            return params
    class LiquidTimeConstants(nn.Module):
        def __init__(self, d_model, d_state):
            super().__init__()
            self.time_constant = nn.Sequential(
                nn.Linear(d_model, d_state),
                nn.Sigmoid()
            )
            self.input_gate = nn.Sequential(
                nn.Linear(d_model, d_state),
                nn.Sigmoid()
            )

    class LiquidS4Cell(nn.Module):
        """Liquid Structural State-Space Model Cell"""
        def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.dt_min = dt_min
            self.dt_max = dt_max

            # HiPPO-LegS initialization for state matrix
            self.A = self._hippo_legs(d_state)
            self.B = nn.Linear(d_model, d_state, bias=False)
            self.C = nn.Linear(d_state, d_model, bias=False)
            
            # Liquid time-constant parameters
            self.W = nn.Linear(d_model, d_state*d_model)
            self.P = nn.Linear(d_model, d_state)
            
            # DPLR parameterization
            self.Lambda, self.P, self.Q = self._dplr_decomp()

        def _hippo_legs(self, n):
            """HiPPO-LegS matrix initialization"""
            A = mx.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    A[i,j] = math.sqrt(2*i+1) * (1 if i == j else 0)
            return A

        def _dplr_decomp(self):
            """Diagonal Plus Low-Rank decomposition"""
            Lambda = mx.diag(self.A)
            P = mx.random.normal((self.d_state, 1))
            Q = mx.random.normal((1, self.d_state))
            return Lambda, P, Q

        def _compute_kernel(self, L):
            """Liquid-S4 convolutional kernel"""
            # Compute Cauchy kernel (simplified)
            gamma = mx.logspace(math.log10(self.dt_min), 
                            math.log10(self.dt_max), 
                            self.d_state)
            
            # Combine DPLR parameters
            A = self.Lambda + self.P @ self.Q
            K = mx.exp(-gamma[:, None] * mx.arange(L)[None, :])
            return mx.fft.ifft(K.real)

        def __call__(self, x, hidden=None):
            """
            Args:
                x: Input tensor [batch, seq_len, d_model]
                hidden: Previous hidden state [batch, d_state]
                
            Returns:
                output: [batch, seq_len, d_model]
                hidden: Final hidden state
            """
            batch, seq_len, _ = x.shape
            L = seq_len
            
            # Compute liquid time constants
            W = mx.tanh(self.W(x)).reshape(batch, seq_len, self.d_state, self.d_model)
            P = mx.sigmoid(self.P(x))  # [batch, seq_len, d_state]
            
            # Compute input-dependent dynamics
            B_eff = self.B(x) * P[..., None]  # [batch, seq_len, d_state]
            
            # Compute SSM kernel
            K = self._compute_kernel(L)  # [d_state, L]
            
            # Liquid-S4 convolution
            x_fft = mx.fft.fft(x, axis=1)
            K_fft = mx.fft.fft(K, axis=1)
            y = mx.fft.ifft(x_fft * K_fft[None, ...], axis=1).real
            
            # Add input correlation terms
            corr_term = mx.einsum('bti,btij,btj->bti', x, W, x)
            y += corr_term
            
            # Final projection
            output = self.C(y)
            return output, y[:,-1,:]

    class LiquidS4(nn.Module):
        """Full Liquid-S4 Architecture"""
        def __init__(self, d_input, d_model=256, d_state=64, n_layers=6):
            super().__init__()
            self.embed = nn.Linear(d_input, d_model)
            self.layers = [LiquidS4Cell(d_model, d_state) for _ in range(n_layers)]
            self.classifier = nn.Linear(d_model, d_input)
            
        def __call__(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x, _ = layer(x)
            return self.classifier(x)

    class CCfC(nn.Module):
        """ConvLSTM2D + CfC architecture"""
        def __init__(self, num_classes=6):
            super().__init__()
            self.feature_extractor = ConvLSTM2D()
            self.cfc = CfC(64, 128)
            self.classifier = nn.Linear(128, num_classes)
        def __call__(self, x):
            features = self.feature_extractor(x)
            hidden = mx.zeros_like(features)
            output = self.cfc(features, hidden)
            return self.classifier(output)

    # class CustomLTC_NCP(nn.Module):
    #     def __init__(self, input_size, wiring):
    #         super().__init__()
    #         self.input_size = input_size
    #         self.wiring = wiring
    #         self.ltc = Your_LTC_Implementation(...)
    #     def __call__(self, x):
    #         output = self.ltc(x)
    #         return output

    class CombinedModel(nn.Module):
        def __init__(self, input_size, wiring):
            super().__init__()
            self.ltc_ncp = CustomLTC_NCP(input_size, wiring)
            self.linear = nn.Linear(motor_neurons, output_size)
        def __call__(self, x):
            output = self.ltc_ncp(x)
            output = self.linear(output)
            return output

    ecg_data = mx.random.uniform(shape=(32, 256, 12, 1))
    cltc_model = CLTC()
    ccfc_model = CCfC()
    cltc_output = cltc_model(ecg_data)
    ccfc_output = ccfc_model(ecg_data)
    quantized_model = nn.QuantizedLinear.from_linear(cltc_model.classifier)
    mlx.coreml.convert(cltc_model, input_shape=(1, 256, 12, 1))

    class CfCCell(nn.Module):
        """MLX implementation of Closed-form Continuous-time (CfC) cell"""
        
        def __init__(self, input_dim, hidden_dim, mixed_memory=True):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.mixed_memory = mixed_memory
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Memory gates
            if mixed_memory:
                self.memory_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
            
            # State update components
            self.state_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            
            # Initialize using MLX's kaiming uniform
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights using Kaiming initialization"""
            k = 1 / math.sqrt(self.hidden_dim)
            for layer in [self.input_proj, self.state_update]:
                layer.weight = mx.random.uniform(-k, k, layer.weight.shape)
                if layer.bias is not None:
                    layer.bias = mx.zeros_like(layer.bias)
            
            if self.mixed_memory:
                self.memory_gate.weight = mx.random.uniform(-k, k, self.memory_gate.weight.shape)
                if self.memory_gate.bias is not None:
                    self.memory_gate.bias = mx.zeros_like(self.memory_gate.bias)
        
        def __call__(self, x, state):
            """
            Args:
                x: Input tensor [batch, input_dim]
                state: Previous state [batch, hidden_dim]
            Returns:
                new_state: Updated state [batch, hidden_dim]
            """
            # Project input
            x_proj = self.input_proj(x)
            
            # Combine input and state for processing
            combined = mx.concatenate([x_proj, state], axis=-1)
            
            if self.mixed_memory:
                # Compute adaptive time constants through memory gating
                memory_gate = mx.sigmoid(self.memory_gate(combined))
                
                # Update state with mixed memory
                state_update = mx.tanh(self.state_update(combined))
                new_state = memory_gate * state + (1 - memory_gate) * state_update
            else:
                # Simple continuous-time update
                state_update = mx.tanh(self.state_update(combined))
                new_state = 0.1 * state + 0.9 * state_update
            
            return new_state

    class MultiScaleCfC(nn.Module):
        """Multi-scale CfC network implementation in MLX"""
        
        def __init__(self, input_dim, hidden_dims=[64, 32, 16], mixed_memory=True):
            super().__init__()
            self.layers = []
            
            # Create multi-scale CfC layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(CfCCell(prev_dim, hidden_dim, mixed_memory))
                prev_dim = hidden_dim
            
            self.output_proj = nn.Linear(hidden_dims[-1], input_dim)
        
        def __call__(self, x, states=None):
            """
            Args:
                x: Input sequence [batch, time, input_dim]
                states: Optional initial states for each layer
            Returns:
                outputs: Sequence of outputs [batch, time, input_dim]
                final_states: Final states of each layer
            """
            batch_size, seq_len, _ = x.shape
            
            # Initialize states if not provided
            if states is None:
                states = [mx.zeros((batch_size, layer.hidden_dim)) for layer in self.layers]
            
            outputs = []
            current_states = list(states)
            
            # Process sequence
            for t in range(seq_len):
                x_t = x[:, t]
                
                # Pass through each layer
                for i, layer in enumerate(self.layers):
                    current_states[i] = layer(x_t, current_states[i])
                    x_t = current_states[i]
                
                # Project to output space
                output = self.output_proj(x_t)
                outputs.append(output)
            
            # Stack outputs along time dimension
            outputs = mx.stack(outputs, axis=1)
            
            return outputs, current_states

    def example_usage():
        """Example of how to use the MLX CfC implementation"""
        # Create sample data
        batch_size = 32
        seq_len = 100
        input_dim = 10
        
        # Random input data
        x = mx.random.normal((batch_size, seq_len, input_dim))
        
        # Create model
        model = MultiScaleCfC(
            input_dim=input_dim,
            hidden_dims=[64, 32, 16],
            mixed_memory=True
        )
        
        # Forward pass
        outputs, final_states = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Number of hidden states: {len(final_states)}")
        for i, state in enumerate(final_states):
            print(f"Hidden state {i} shape: {state.shape}")
        
        return outputs, final_states

else:
    def example_usage():
        print("MLX framework not available. Please install MLX to use this module.")
