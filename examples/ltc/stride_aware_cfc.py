import os
from ncps import wirings


#Import Keras after setting backend
import ember_ml
import ember_ml.nn.initializers
import ember_ml.nn.modules.activations

# LeCun improved tanh activation
def lecun_tanh(x):
    return 1.7159 * ember_ml.nn.modules.activations.tanh(0.66666667 * x)

# Binomial Initializer
class BinomialInitializer(ember_ml.nn.initializers.BinomialInitializer):
    def __init__(self, probability=0.5, seed=None):
        super().__init__()
        self.probability = probability
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = backend.floatx()
        return tensor.cast(
            keras.random.uniform(shape, minval=0.0, maxval=1.0, seed=self.seed) < self.probability,
            dtype=dtype
        )

    def get_config(self):
        return {"probability": self.probability, "seed": self.seed}


@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareWiredCfCCell")
class StrideAwareWiredCfCCell(keras.layers.Layer):

    def __init__(
            self,
            wiring: wirings.Wiring,
            stride_length: int = 1,
            time_scale_factor: float = 1.0,
            fully_recurrent: bool = True,
            mode: str = "default",
            activation = lecun_tanh,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.wiring = wiring
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        self.mode = mode
        self.activation = ember_ml.nn.modules.activations.get_activation(activation)

        self.units = wiring.units
        self.input_dim = wiring.input_dim
        self.output_dim = wiring.output_dim
        self.recurrent_activation = ember_ml.nn.modules.activations.get_activation('sigmoid')


    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units), # Use self.units here
            initializer='glorot_uniform',
            name='kernel',
            regularizer=ember_ml.nn.modules.regularizers.L2(0.01),
            constraint=ember_ml.nn.modules.constraints.MaxNorm(3)
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units), # Use self.units here
            initializer='orthogonal',
            name='recurrent_kernel',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
        )

        self.backbone_out = self.add_weight( # Added backbone
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            name='backbone_out',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
        )

        self.time_kernel = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            name='time_kernel',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
        )

        self.bias = self.add_weight(
            shape=(self.units,),  # Use self.units here
            initializer='zeros',
            name='bias'
        )

        self.recurrent_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='recurrent_bias'
        )

        if self.mode != "no_gate":
            self.gate_kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer='glorot_uniform',
                name='gate_kernel',
                regularizer=keras.regularizers.L2(0.01),
                constraint=keras.constraints.MaxNorm(3)
            )

            self.gate_recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='orthogonal',
                name='gate_recurrent_kernel',
                regularizer=keras.regularizers.L2(0.01),
                constraint=keras.constraints.MaxNorm(3)
            )

            self.gate_bias = self.add_weight(
                shape=(self.units,),
                initializer='ones',
                name='gate_bias'
            )

        sparsity = self.wiring.get_config()["sparsity_level"]
        self.input_mask = self.add_weight(
            shape=(input_dim,),
            initializer=BinomialInitializer(probability=sparsity, seed=42),
            name='input_mask',
            trainable=False
        )
        self.recurrent_mask = self.add_weight(
            shape=(self.units, self.units),
            initializer=BinomialInitializer(probability=sparsity, seed=43),
            name='recurrent_mask',
            trainable=False
        )
        self.output_mask = self.add_weight(
            shape=(self.units,),
            initializer=BinomialInitializer(probability=sparsity, seed=44),
            name='output_mask',
            trainable=False
        )

        self.built = True

    def _compute_time_scaling(self, inputs, kwargs):
        """Helper function to compute time scaling."""
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
        else:
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            t = keras.tensor.cast(t, dtype=keras.backend.floatx())
        return inputs, t


    def call(self, inputs, states, **kwargs):
        h_prev = states[0]
        inputs, t = self._compute_time_scaling(inputs, kwargs)

        masked_inputs = inputs * self.input_mask
        masked_h_prev = keras.ops.matmul(h_prev, self.recurrent_mask)

        x = keras.ops.matmul(masked_inputs, self.kernel) + self.bias
        x = self.activation(x + keras.ops.matmul(masked_h_prev, self.recurrent_kernel))

        # Removed backbone layers/dropout.  Handled in build
        h_candidate = keras.ops.matmul(x, self.backbone_out) + self.recurrent_bias  # Added backbone
        time_gate = keras.ops.exp(-keras.ops.abs(t) * keras.ops.exp(self.time_kernel))

        if self.mode == "no_gate":
            h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
            output = h_new * self.output_mask
        else:
            gate_in = keras.ops.matmul(inputs, self.gate_kernel)
            gate_rec = keras.ops.matmul(h_prev, self.gate_recurrent_kernel)
            gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)

            if self.mode == "pure":
                h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
            else:
                h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)
            output = h_new * self.output_mask

        return output, [h_new]

    def get_config(self):
        # Custom serialization to avoid TensorFlow dependencies
        try:
            # Try standard serialization first
            activation_config = keras.activations.serialize(self.activation)
        except (ImportError, TypeError, AttributeError) as e:
            # Fallback: use function name if it's a standard activation
            if self.activation == keras.activations.tanh:
                activation_config = "tanh"
            elif self.activation == keras.activations.relu:
                activation_config = "relu"
            elif self.activation == keras.activations.sigmoid:
                activation_config = "sigmoid"
            elif self.activation == lecun_tanh:
                activation_config = "lecun_tanh"
            else:
                # Default to the custom activation's name
                activation_config = getattr(self.activation, "__name__", "unknown")
        
        # Get wiring config and add class name information
        wiring_config = self.wiring.get_config()
        wiring_class_name = self.wiring.__class__.__name__
        
        config = {
            "wiring": {
                "class_name": wiring_class_name,
                "config": wiring_config
            },
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
            "activation": activation_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        wiring_config = config.pop("wiring")
        from ncps import wirings
        wiring_class = getattr(wirings, wiring_config["class_name"])
        wiring = wiring_class.from_config(wiring_config["config"])
        
        # Handle activation deserialization with fallback
        activation_config = config.pop('activation')
        try:
            # Try standard deserialization
            activation = keras.activations.deserialize(activation_config)
        except (ImportError, TypeError, AttributeError) as e:
            # Handle string activation names
            if activation_config == "tanh":
                activation = keras.activations.tanh
            elif activation_config == "relu":
                activation = keras.activations.relu
            elif activation_config == "sigmoid":
                activation = keras.activations.sigmoid
            elif activation_config == "lecun_tanh":
                activation = lecun_tanh
            else:
                # Default to lecun_tanh as fallback
                activation = lecun_tanh
                
        return cls(wiring=wiring, activation=activation, **config)

    @property
    def state_size(self):
        return self.units

    @property
    def input_size(self):
        return self.input_dim

    @property
    def output_size(self):
        return self.output_dim


@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareCfCCell")
class StrideAwareCfCCell(keras.layers.Layer):
    def __init__(self, units, stride_length=1, time_scale_factor=1.0,
                    mode="default", activation=lecun_tanh,
                    backbone_units=128, backbone_layers=1, backbone_dropout=0.0,
                    **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.mode = mode
        self.activation = keras.activations.get(activation)
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.recurrent_activation = keras.activations.get('sigmoid')

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.backbone_units),
            initializer='glorot_uniform',
            name='kernel'
        )

        self.backbone_kernels = []
        self.backbone_biases = []
        for i in range(self.backbone_layers):
            self.backbone_kernels.append(
                self.add_weight(
                    shape=(self.backbone_units, self.backbone_units),
                    initializer='glorot_uniform',
                    name=f'backbone_kernel_{i}'
                )
            )
            self.backbone_biases.append(
                self.add_weight(
                    shape=(self.backbone_units,),
                    initializer='zeros',
                    name=f'backbone_bias_{i}'
                )
            )

        self.backbone_out = self.add_weight(
            shape=(self.backbone_units, self.units),
            initializer='glorot_uniform',
            name='backbone_out'
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.backbone_units),
            initializer='orthogonal',
            name='recurrent_kernel'
        )

        self.time_kernel = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            name='time_kernel'
        )

        self.bias = self.add_weight(
            shape=(self.backbone_units,),
            initializer='zeros',
            name='bias'
        )

        self.recurrent_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='recurrent_bias'
        )
        if self.mode != "no_gate":
            self.gate_kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer='glorot_uniform',
                name='gate_kernel'
            )
            self.gate_recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='orthogonal',
                name='gate_recurrent_kernel'
            )
            self.gate_bias = self.add_weight(
                shape=(self.units,),
                initializer='ones',
                name='gate_bias'
            )
        self.built = True

    @property
    def state_size(self):
        return self.units
    
    def _compute_time_scaling(self, inputs, kwargs):
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
        else:
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            t = keras.tensor.cast(t, dtype=keras.backend.floatx())
        return inputs, t

    def call(self, inputs, states, **kwargs):
        h_prev = states[0]
        inputs, t = self._compute_time_scaling(inputs, kwargs)

        x = keras.ops.matmul(inputs, self.kernel) + self.bias
        x = self.activation(x + keras.ops.matmul(h_prev, self.recurrent_kernel))

        for i in range(self.backbone_layers):
            x = keras.ops.matmul(x, self.backbone_kernels[i]) + self.backbone_biases[i]
            x = self.activation(x)
            if self.backbone_dropout > 0:
                x = keras.ops.dropout(x, rate=self.backbone_dropout)

        h_candidate = keras.ops.matmul(x, self.backbone_out) + self.recurrent_bias
        time_gate = keras.ops.exp(-keras.ops.abs(t) * keras.ops.exp(self.time_kernel))

        if self.mode == "no_gate":
            h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
        else:
            gate_in = keras.ops.matmul(inputs, self.gate_kernel)
            gate_rec = keras.ops.matmul(h_prev, self.gate_recurrent_kernel)
            gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)
            if self.mode == "pure":
                h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
            else:
                h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)

        return h_new, [h_new]

    def get_config(self):
        # Custom serialization to avoid TensorFlow dependencies
        try:
            # Try standard serialization first
            activation_config = keras.activations.serialize(self.activation)
        except (ImportError, TypeError, AttributeError) as e:
            # Fallback: use function name if it's a standard activation
            if self.activation == keras.activations.tanh:
                activation_config = "tanh"
            elif self.activation == keras.activations.relu:
                activation_config = "relu"
            elif self.activation == keras.activations.sigmoid:
                activation_config = "sigmoid"
            elif self.activation == lecun_tanh:
                activation_config = "lecun_tanh"
            else:
                # Default to the custom activation's name
                activation_config = getattr(self.activation, "__name__", "unknown")
        
        config = {
            "units": self.units,
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "mode": self.mode,
            "activation": activation_config,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Handle activation deserialization with fallback
        activation_config = config.pop('activation')
        try:
            # Try standard deserialization
            activation = keras.activations.deserialize(activation_config)
        except (ImportError, TypeError, AttributeError) as e:
            # Handle string activation names
            if activation_config == "tanh":
                activation = keras.activations.tanh
            elif activation_config == "relu":
                activation = keras.activations.relu
            elif activation_config == "sigmoid":
                activation = keras.activations.sigmoid
            elif activation_config == "lecun_tanh":
                activation = lecun_tanh
            else:
                # Default to lecun_tanh as fallback
                activation = lecun_tanh
                
        return cls(activation=activation, **config)


class StrideAwareCfC(ember_ml.nn.modules.RNN):
    def __init__(
        self,
        cell,  # Now takes a cell *instance*
        mixed_memory: bool = False,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        zero_output_for_mask: bool = False,
        **kwargs
    ):
        if mixed_memory:
            class MixedMemoryRNN(ember_ml.nn.modules.RNN):
                def __init__(self, cell, **kwargs):
                    super().__init__(**kwargs)
                    self.rnn_cell = cell
                    self.units = cell.units
                    self.state_size = [cell.state_size, cell.units]

                def build(self, input_shape):
                    self.rnn_cell.build(input_shape)
                    input_dim = input_shape[-1]

                    self.memory_kernel = self.add_weight(
                        shape=(input_dim, self.units),
                        initializer='glorot_uniform',
                        name='memory_kernel'
                    )

                    self.memory_recurrent_kernel = self.add_weight(
                        shape=(self.units, self.units),
                        initializer='orthogonal',
                        name='memory_recurrent_kernel'
                    )

                    self.memory_bias = self.add_weight(
                        shape=(self.units,),
                        initializer='zeros',
                        name='memory_bias'
                    )
                    self.built = True

                def call(self, inputs, states, **kwargs):
                    rnn_state = states[0]
                    memory_state = states[1]

                    output, new_rnn_state = self.rnn_cell(inputs, [rnn_state], **kwargs)

                    memory_gate = keras.ops.sigmoid(
                        keras.ops.matmul(inputs, self.memory_kernel) +
                        keras.ops.matmul(memory_state, self.memory_recurrent_kernel) +
                        self.memory_bias
                    )

                    new_memory_state = memory_state * memory_gate + output * (1 - memory_gate)

                    return output, [new_rnn_state[0], new_memory_state]

                def get_config(self):
                    config = {
                        "cell": keras.layers.serialize(self.rnn_cell),
                    }
                    base_config = super().get_config()
                    return {**base_config, **config}
                
                @classmethod
                def from_config(cls, config):
                    cell_config = config.pop("cell")
                    cell = keras.layers.deserialize(cell_config)
                    return cls(cell = cell, **config)

            # Apply mixed memory wrapping *before* passing to super().__init__
            cell = MixedMemoryRNN(cell)

        # Now, always pass the cell instance directly to the superclass
        super(StrideAwareCfC, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            zero_output_for_mask=zero_output_for_mask,
            **kwargs,
        )
        # Store properties from the *cell*.  No more redundant attributes.
        if isinstance(cell, keras.layers.Layer) and hasattr(cell, 'rnn_cell'):
            # For MixedMemoryRNN case, access properties from the inner cell
            self.stride_length = cell.rnn_cell.stride_length
            self.time_scale_factor = cell.rnn_cell.time_scale_factor
        else:
            # For regular cell case
            self.stride_length = cell.stride_length
            self.time_scale_factor = cell.time_scale_factor


    def get_config(self):
        # Serialize the *cell* correctly.
        # Use try-except to handle TensorFlow serialization issues when using numpy backend
        try:
            # Try standard keras serialization first
            cell_config = keras.layers.serialize(self.cell)
        except (ImportError, TypeError, AttributeError) as e:
            # Fallback to manual serialization if TensorFlow components cause issues
            if isinstance(self.cell, keras.layers.Layer) and hasattr(self.cell, 'rnn_cell'):
                # For MixedMemoryRNN case
                inner_cell_config = {
                    "class_name": self.cell.rnn_cell.__class__.__name__,
                    "config": self.cell.rnn_cell.get_config()
                }
                cell_config = {
                    "class_name": "MixedMemoryRNN",
                    "config": {
                        "cell": inner_cell_config
                    }
                }
            else:
                # For regular cell case
                cell_config = {
                    "class_name": self.cell.__class__.__name__,
                    "config": self.cell.get_config()
                }
        
        config = {
                "cell": cell_config,
                "mixed_memory": isinstance(self.cell, keras.layers.Layer) and hasattr(self.cell, 'rnn_cell'), #check for mixed memory
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "go_backwards": self.go_backwards,
                "stateful":self.stateful,
                "unroll": self.unroll,
                "zero_output_for_mask": self.zero_output_for_mask,
        }
        return config # Don't call super().get_config()

    @classmethod
    def from_config(cls, config):
        cell_config = config.pop("cell")
        # Try standard deserialization first, but be prepared to handle exceptions
        try:
            cell = keras.layers.deserialize(cell_config)
        except (ImportError, TypeError, AttributeError) as e:
            # Fallback to manual deserialization
            class_name = cell_config.get("class_name")
            cell_class_config = cell_config.get("config", {})
            
            if class_name == "MixedMemoryRNN":
                # Handle MixedMemoryRNN case
                inner_cell_config = cell_class_config.get("cell", {})
                inner_class_name = inner_cell_config.get("class_name")
                
                # Import the correct cell class
                if inner_class_name == "StrideAwareWiredCfCCell":
                    from ncps import wirings
                    inner_cell_config_data = inner_cell_config.get("config", {})
                    wiring_config = inner_cell_config_data.pop("wiring", None)
                    if wiring_config:
                        wiring_class = getattr(wirings, wiring_config["class_name"])
                        wiring = wiring_class.from_config(wiring_config["config"])
                        inner_cell = StrideAwareWiredCfCCell(wiring=wiring, **inner_cell_config_data)
                    else:
                        inner_cell = StrideAwareWiredCfCCell(**inner_cell_config_data)
                elif inner_class_name == "StrideAwareCfCCell":
                    inner_cell = StrideAwareCfCCell(**inner_cell_config.get("config", {}))
                else:
                    raise ValueError(f"Unsupported cell type: {inner_class_name}")
                
                # Create MixedMemoryRNN wrapper manually
                @keras.utils.register_keras_serializable(package="ncps", name="MixedMemoryRNN")
                class MixedMemoryRNN(keras.layers.Layer):
                    def __init__(self, cell, **kwargs):
                        super().__init__(**kwargs)
                        self.rnn_cell = cell
                        self.units = cell.units
                        self.state_size = [cell.state_size, cell.units]

                    def build(self, input_shape):
                        self.rnn_cell.build(input_shape)
                        input_dim = input_shape[-1]

                        self.memory_kernel = self.add_weight(
                            shape=(input_dim, self.units),
                            initializer='glorot_uniform',
                            name='memory_kernel'
                        )

                        self.memory_recurrent_kernel = self.add_weight(
                            shape=(self.units, self.units),
                            initializer='orthogonal',
                            name='memory_recurrent_kernel'
                        )

                        self.memory_bias = self.add_weight(
                            shape=(self.units,),
                            initializer='zeros',
                            name='memory_bias'
                        )
                        self.built = True

                    def call(self, inputs, states, **kwargs):
                        rnn_state = states[0]
                        memory_state = states[1]

                        output, new_rnn_state = self.rnn_cell(inputs, [rnn_state], **kwargs)

                        memory_gate = keras.ops.sigmoid(
                            keras.ops.matmul(inputs, self.memory_kernel) +
                            keras.ops.matmul(memory_state, self.memory_recurrent_kernel) +
                            self.memory_bias
                        )

                        new_memory_state = memory_state * memory_gate + output * (1 - memory_gate)

                        return output, [new_rnn_state[0], new_memory_state]

                cell = MixedMemoryRNN(inner_cell)
            elif class_name == "StrideAwareWiredCfCCell":
                from ncps import wirings
                wiring_config = cell_class_config.pop("wiring", None)
                if wiring_config:
                    wiring_class = getattr(wirings, wiring_config["class_name"])
                    wiring = wiring_class.from_config(wiring_config["config"])
                    cell = StrideAwareWiredCfCCell(wiring=wiring, **cell_class_config)
                else:
                    cell = StrideAwareWiredCfCCell(**cell_class_config)
            elif class_name == "StrideAwareCfCCell":
                cell = StrideAwareCfCCell(**cell_class_config)
            else:
                raise ValueError(f"Unsupported cell type: {class_name}")
                
        # Make sure mixed memory is properly handled
        mixed_memory = config.pop("mixed_memory")
        if mixed_memory and not (isinstance(cell, keras.layers.Layer) and hasattr(cell, 'rnn_cell')):
            # If mixed_memory is True but cell is not already a MixedMemoryRNN
            return cls(cell, mixed_memory=True, **config)  # Apply MixedMemoryRNN wrapper
        elif mixed_memory:
            # Already a MixedMemoryRNN
            return cls(cell.rnn_cell, mixed_memory=True, **config)  # Unwrap and rewrap to ensure consistency
        return cls(cell, **config)  # Pass the deserialized cell