import enum
from typing import List, Optional, Tuple

# Ember ML imports
from ember_ml import ops
from ember_ml.nn import modules as nn_modules
from ember_ml.nn import tensor
from ember_ml import initializers
from ember_ml.nn.modules import activations # For functional activations

# Enums (remain unchanged)
class MappingType(enum.Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(enum.Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

# LTCCell using Ember ML abstractions
class LTCCell(nn_modules.Module):
    def __init__(self, num_units, input_mapping=MappingType.Affine,
                 solver=ODESolver.SemiImplicit, ode_solver_unfolds=6,
                 activation_fn_name="tanh", **kwargs): # Use activation name string
        super().__init__(**kwargs)
        self._num_units = num_units
        self._input_mapping = input_mapping # Note: input_mapping logic not fully implemented here
        self._solver = solver
        self._ode_solver_unfolds = ode_solver_unfolds
        self._activation_fn_name = activation_fn_name # Store name
        self.kernel: Optional[nn_modules.Parameter] = None
        self.recurrent_kernel: Optional[nn_modules.Parameter] = None
        self.bias: Optional[nn_modules.Parameter] = None
        # self.built is handled by base Module class

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        # input_shape: (..., input_dim)
        input_dim = input_shape[-1]
        # Use Ember ML Parameter and initializers
        self.kernel = nn_modules.Parameter(initializers.glorot_uniform()((input_dim, self._num_units)))
        self.recurrent_kernel = nn_modules.Parameter(initializers.glorot_uniform()((self._num_units, self._num_units)))
        self.bias = nn_modules.Parameter(tensor.zeros((self._num_units,)))
        # self.built = True # Handled by base class

    def __call__(self, inputs, states):
        # build is called automatically by base Module if not built
        prev_output = states[0]
        # Use ops functions
        net_input = ops.matmul(inputs, self.kernel)
        net_input = ops.add(net_input, ops.matmul(prev_output, self.recurrent_kernel))
        net_input = ops.add(net_input, self.bias)

        # Get activation function dynamically
        activation_fn = activations.get_activation(self._activation_fn_name)

        if self._solver == ODESolver.SemiImplicit:
            output = self._semi_implicit_solver(prev_output, net_input, activation_fn)
        elif self._solver == ODESolver.Explicit:
            output = self._explicit_solver(prev_output, net_input, activation_fn)
        elif self._solver == ODESolver.RungeKutta:
            output = self._runge_kutta_solver(prev_output, net_input, activation_fn)
        else:
            raise ValueError("Unsupported ODE Solver type.")

        return output, [output]

    def _semi_implicit_solver(self, prev_output, net_input, activation_fn):
        # Euler-style update toward activation result using ops
        activated_input = activation_fn(net_input)
        diff = ops.subtract(activated_input, prev_output)
        update = ops.multiply(float(self._ode_solver_unfolds), diff) # Cast unfolds to float
        return ops.add(prev_output, update)

    def _explicit_solver(self, prev_output, net_input, activation_fn):
         # Use ops functions
        activated_input = activation_fn(net_input)
        update = ops.multiply(float(self._ode_solver_unfolds), activated_input) # Cast unfolds to float
        return ops.add(prev_output, update)

    def _runge_kutta_solver(self, prev_output, net_input, activation_fn):
        # Use ops functions
        dt = ops.divide(1.0, float(self._ode_solver_unfolds)) # Cast unfolds to float
        k1 = activation_fn(net_input)
        k2 = activation_fn(ops.add(net_input, ops.multiply(0.5 * dt, k1)))
        k3 = activation_fn(ops.add(net_input, ops.multiply(0.5 * dt, k2)))
        k4 = activation_fn(ops.add(net_input, ops.multiply(dt, k3)))
        term1 = ops.add(k1, ops.multiply(2.0, k2))
        term2 = ops.add(ops.multiply(2.0, k3), k4)
        update = ops.multiply(ops.divide(dt, 6.0), ops.add(term1, term2))
        return ops.add(prev_output, update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_units": self._num_units,
            "solver": self._solver.name, # Store enum name
            "ode_solver_unfolds": self._ode_solver_unfolds,
            "activation_fn_name": self._activation_fn_name
        })
        return config

# CTRNN using Ember ML abstractions
class CTRNN(nn_modules.Module):
    def __init__(self, units, global_feedback=False, activation_fn_name="tanh", cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.global_feedback = global_feedback # Note: global_feedback logic not implemented here
        self.activation_fn_name = activation_fn_name
        self.cell_clip = cell_clip
        self.kernel: Optional[nn_modules.Parameter] = None
        self.recurrent_kernel: Optional[nn_modules.Parameter] = None
        self.bias: Optional[nn_modules.Parameter] = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = nn_modules.Parameter(initializers.glorot_uniform()((input_dim, self.units)))
        self.recurrent_kernel = nn_modules.Parameter(initializers.glorot_uniform()((self.units, self.units)))
        self.bias = nn_modules.Parameter(tensor.zeros((self.units,)))

    def __call__(self, inputs, states):
        prev_output = states[0]
        # Use ops functions
        net_input = ops.matmul(inputs, self.kernel)
        net_input = ops.add(net_input, ops.matmul(prev_output, self.recurrent_kernel))
        net_input = ops.add(net_input, self.bias)
        # Get activation function dynamically
        activation_fn = activations.get_activation(self.activation_fn_name)
        output = activation_fn(net_input)
        if self.cell_clip is not None:
            output = ops.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]

# NODE using Ember ML abstractions (Similar to CTRNN with tanh)
class NODE(nn_modules.Module):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip
        self.kernel: Optional[nn_modules.Parameter] = None
        self.recurrent_kernel: Optional[nn_modules.Parameter] = None
        self.bias: Optional[nn_modules.Parameter] = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = nn_modules.Parameter(initializers.glorot_uniform()((input_dim, self.units)))
        self.recurrent_kernel = nn_modules.Parameter(initializers.glorot_uniform()((self.units, self.units)))
        self.bias = nn_modules.Parameter(tensor.zeros((self.units,)))

    def __call__(self, inputs, states):
        prev_output = states[0]
        # Use ops functions
        net_input = ops.matmul(inputs, self.kernel)
        net_input = ops.add(net_input, ops.matmul(prev_output, self.recurrent_kernel))
        net_input = ops.add(net_input, self.bias)
        output = ops.tanh(net_input) # Directly use ops.tanh
        if self.cell_clip is not None:
            output = ops.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]

# CTGRU using Ember ML abstractions
class CTGRU(nn_modules.Module):
    def __init__(self, units, cell_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell_clip = cell_clip
        self.kernel: Optional[nn_modules.Parameter] = None
        self.recurrent_kernel: Optional[nn_modules.Parameter] = None
        self.bias: Optional[nn_modules.Parameter] = None
        self.kernel_c: Optional[nn_modules.Parameter] = None
        self.recurrent_kernel_c: Optional[nn_modules.Parameter] = None
        self.bias_c: Optional[nn_modules.Parameter] = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Combined kernel for z and r gates.
        self.kernel = nn_modules.Parameter(initializers.glorot_uniform()((input_dim, 2 * self.units)))
        self.recurrent_kernel = nn_modules.Parameter(initializers.glorot_uniform()((self.units, 2 * self.units)))
        self.bias = nn_modules.Parameter(tensor.zeros((2 * self.units,)))
        # Parameters for candidate c.
        self.kernel_c = nn_modules.Parameter(initializers.glorot_uniform()((input_dim, self.units)))
        self.recurrent_kernel_c = nn_modules.Parameter(initializers.glorot_uniform()((self.units, self.units)))
        self.bias_c = nn_modules.Parameter(tensor.zeros((self.units,)))

    def __call__(self, inputs, states):
        prev_output = states[0]
        # Use ops functions
        zr = ops.matmul(inputs, self.kernel)
        zr = ops.add(zr, ops.matmul(prev_output, self.recurrent_kernel))
        zr = ops.add(zr, self.bias)
        # Assuming ops.split works like mx.split
        # Check ops.split documentation for exact signature if needed
        z, r = ops.split(zr, 2, axis=-1)
        z = ops.sigmoid(z)
        r = ops.sigmoid(r)
        c_input = ops.matmul(inputs, self.kernel_c)
        c_rec = ops.multiply(r, ops.matmul(prev_output, self.recurrent_kernel_c)) # Use ops.multiply
        c = ops.add(c_input, c_rec)
        c = ops.add(c, self.bias_c)
        c = ops.tanh(c)
        # Use ops functions for final output calculation
        output = ops.add(ops.multiply(ops.subtract(1.0, z), prev_output), ops.multiply(z, c))
        if self.cell_clip is not None:
            output = ops.clip(output, -self.cell_clip, self.cell_clip)
        return output, [output]