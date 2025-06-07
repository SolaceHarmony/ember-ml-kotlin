"""
Ray implementation of the MetalKernelActor, integrated into the ember_ml.actors package,
utilizing asynchronous neural network modules.
"""

import ray
import mlx.core as mx
import time
import asyncio
from typing import Dict, List, Optional, Any

# Import EmberTensor and async liquid_cfc_xlstm
from ember_ml.nn.tensor import EmberTensor
from ember_ml.asyncml.nn.modules.rnn.liquid_cfc_xlstm import async_liquid_cfc_xlstm
from ember_ml.ops import stats
from ember_ml.nn.tensor.types import TensorLike
import ray
import time
import asyncio
from typing import Dict, List, Optional, Any
from ember_ml import ops
@ray.remote
class MetalKernelActor:
    """
    Actor that processes data through the Metal kernel.
    """

    def __init__(self, hidden_dim: int, input_dim: int = 2):
        """
        Initialize the Metal kernel actor.

        Args:
            hidden_dim: Dimension of hidden state
            input_dim: Dimension of input
        """
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.model_params = self._initialize_model_params()

        # Initialize states
        self.reset_states()

        # History tracking (can be kept here or moved to a separate actor/component)
        self.history = {
            'h_liquid': [],
            'c_t': [],
            'n_t': [],
            'inputs': [],
            'gate_i': [], # Note: Gate history might need to be handled differently if not returned by the async module
            'gate_f': [],
            'gate_o': [],
            'gate_g': [],
        }

        # Performance metrics
        self.processing_times = []

        # Neural clock
        self.last_update_time = time.time()
        self.gamma_phase = 0.0
        self.theta_phase = 0.0

        print(f"[MetalKernelActor] Initialized with hidden_dim={hidden_dim}, input_dim={input_dim}")

    def reset_states(self):
        """Reset the kernel state using EmberTensor."""
        # Initialize states as EmberTensor
        self.h_liquid = EmberTensor.zeros((self.hidden_dim,), dtype='float32')
        self.c_t = EmberTensor.zeros((self.hidden_dim,), dtype='float32')
        self.n_t = EmberTensor.zeros((self.hidden_dim,), dtype='float32')
        # W_recurrent is initialized in _initialize_model_params as EmberTensor
        return {"status": "reset"}

    def _initialize_model_params(self):
        """Initialize model parameters as EmberTensor."""
        # Simplified implementation for testing
        return {
            'W_recurrent': EmberTensor.random_normal((self.hidden_dim, self.hidden_dim)),
            'W_i': EmberTensor.random_normal((self.hidden_dim,)),
            'U_i': EmberTensor.random_normal((self.hidden_dim,)),
            'b_i': EmberTensor.zeros((self.hidden_dim,)),
            'W_f': EmberTensor.random_normal((self.hidden_dim,)),
            'U_f': EmberTensor.random_normal((self.hidden_dim,)),
            'b_f': EmberTensor.zeros((self.hidden_dim,)),
            'W_o': EmberTensor.random_normal((self.hidden_dim,)),
            'U_o': EmberTensor.random_normal((self.hidden_dim,)),
            'b_o': EmberTensor.zeros((self.hidden_dim,)),
            'W_g': EmberTensor.random_normal((self.hidden_dim,)),
            'U_g': EmberTensor.random_normal((self.hidden_dim,)),
            'b_g': EmberTensor.zeros((self.hidden_dim,)),
            'lambda_vals': EmberTensor.random_uniform((self.hidden_dim,), 0.1, 1.0),
            'gate_mask': EmberTensor.ones((self.hidden_dim,), dtype='int32'), # Assuming int32 for mask
            'lambda_mask': EmberTensor.ones((self.hidden_dim,), dtype='int32'), # Assuming int32 for mask
            'dt': 0.01,
            'alpha': 0.01,
            'target_sum': 3.0,
            'neural_clock': 0.0, # This is a scalar, can remain float
            'use_hebbian': False,
            'eta': 0.0001,
            'decay_rate': 0.0001
        }

    def _update_neural_clock(self):
        """Update the neural clock based on oscillatory dynamics."""
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update oscillation phases
        gamma_frequency = 40.0
        theta_frequency = 8.0
        self.gamma_phase += 2.0 * 3.14159 * gamma_frequency * delta_time
        self.theta_phase += 2.0 * 3.14159 * theta_frequency * delta_time

        # Keep phases in [0, 2π] range
        self.gamma_phase = self.gamma_phase % (2.0 * 3.14159)
        self.theta_phase = self.theta_phase % (2.0 * 3.14159)

        # Theta-modulated gamma oscillation (using EmberTensor for sin and multiplication)
        # Assuming EmberTensor supports scalar operations and math functions
        theta_component = (EmberTensor(1.0) + EmberTensor.sin(EmberTensor(self.theta_phase))) / 2.0
        gamma_component = (EmberTensor(1.0) + EmberTensor.sin(EmberTensor(self.gamma_phase))) / 2.0

        # Final neural clock is a combination of both rhythms
        # Convert back to scalar if needed, or keep as scalar EmberTensor
        return (theta_component * gamma_component).item() # Convert to scalar float

    async def process_data(self, request):
        """
        Process data through the Metal kernel using the asynchronous RNN module.

        Args:
            request: Process request with input_data, sequence_id, and timestep

        Returns:
            Process result
        """
        # Log processing start
        sequence_id = request.get('sequence_id', 'unknown')
        timestep = request.get('timestep', -1)
        print(f"[MetalKernelActor] Processing data for sequence {sequence_id}, timestep {timestep}")

        # Extract input
        input_x = request['input_data']

        # Update neural clock
        neural_clock = self._update_neural_clock()
        self.model_params['neural_clock'] = neural_clock

        # Store input and current states in history
        self.history['inputs'].append(input_x.tolist())
        self.history['h_liquid'].append(self.h_liquid.tolist())
        self.history['c_t'].append(self.c_t.tolist())
        self.history['n_t'].append(self.n_t.tolist())

        # Process through the kernel using the asynchronous RNN function
        start_time = time.time()

        try:
            # Call the asynchronous liquid_cfc_xlstm function and await its result
            h_liquid_next, c_t_next, n_t_next, W_recurrent_next = await async_liquid_cfc_xlstm(
                input_x,
                self.model_params,
                self.h_liquid,
                self.c_t,
                self.n_t,
                self.W_recurrent
            )

            # Note: The async_liquid_cfc_xlstm function includes simulated computation time.

            end_time = time.time()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)

            # Update states
            self.h_liquid = h_liquid_next
            self.c_t = c_t_next
            self.n_t = n_t_next
            self.W_recurrent = W_recurrent_next

            print(f"[MetalKernelActor] Completed kernel computation for sequence {sequence_id}, timestep {timestep} in {processing_time:.4f}s")

            # Return result
            return {
                'output': h_liquid_next, # Assuming h_liquid_next is the primary output
                'sequence_id': sequence_id,
                'timestep': timestep,
                'processing_time': processing_time
            }
        except Exception as e:
            print(f"[MetalKernelActor] Error in kernel computation: {e}")
            return {
                'error': str(e),
                'sequence_id': sequence_id,
                'timestep': timestep
            }

    def get_history(self):
        """Get the history of states and gates."""
        # Note: Gate history might not be available if not returned by the async module.
        # This method might need adjustment based on what the async module returns or if history is handled elsewhere.
        return self.history

    def get_performance_stats(self):
        """Get performance statistics."""
        if not self.processing_times:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'total': 0
            }
        from ember_ml.nn import tensor
        times = tensor.convert_to_tensor(self.processing_times)
        return {
            'count': len(times),
            'mean': stats.mean(times),
            'min': stats.min(times),
            'max': stats.max(times),
            'total': stats.sum(times)
        }

    def get_state(self):
        """Get the current state of the actor."""
        return {
            'h_liquid': self.h_liquid,
            'c_t': self.c_t,
            'n_t': self.n_t,
            'gamma_phase': self.gamma_phase,
            'theta_phase': self.theta_phase,
            'neural_clock': self.model_params.get('neural_clock', 0.0)
        }