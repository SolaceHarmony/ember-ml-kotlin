"""
Blocky road neuron implementation with frequency-aware wave generation
and precise fraction-based computations.
"""

from typing import Optional, Tuple, Dict, Any
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.neurons import BaseNeuron
class BlockySigmoid:
    """Quantized sigmoid implementation with discrete steps."""
    
    _steps = [-12, -10.5, -9, -7.5, -6, -4.5, -3, -1.5,
              0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12]
    _values = [ops.divide(i, 16) for i in range(17)]
    
    @classmethod
    def compute(cls, x: float, mu: float = 0.5, sigma: float = 1.0) -> float:
        """
        Compute quantized sigmoid value.

        Args:
            x: Input value
            mu: Center point
            sigma: Steepness

        Returns:
            Quantized sigmoid value
        """
        threshold = ops.multiply(ops.multiply(ops.subtract(x, mu), sigma), 10)
        
        # Find the index of the first step that is greater than threshold
        idx = ops.subtract(len(cls._steps), 1)  # Default to last index
        for i, step in enumerate(cls._steps):
            if ops.less_equal(threshold, step):
                idx = i
                break
                
        return cls._values[idx]

def blocky_sin(freq: float = 1.0,
               pattern_time: float = 5.0,
               num_steps: int = 5000) -> Tuple:
    """
    Generate frequency-aware blocky sine wave.

    Args:
        freq: Wave frequency
        pattern_time: Duration of pattern
        num_steps: Number of time steps

    Returns:
        Tuple of (time points, wave values)
    """
    x_stair, y_stair = [], []
    dt_step = ops.divide(pattern_time, num_steps)
    
    for i in range(ops.add(num_steps, 1)):
        x_val = ops.multiply(i, dt_step)
        y_val = ops.sin(ops.multiply(ops.multiply(2, ops.pi), ops.multiply(freq, x_val)))
        
        if i == 0:
            x_stair.append(x_val)
            y_stair.append(y_val)
        else:
            x_stair.append(x_val)
            y_stair.append(y_stair[-1])
            x_stair.append(x_val)
            y_stair.append(y_val)
            
    return tensor.convert_to_tensor(x_stair), tensor.convert_to_tensor(y_stair)

class BlockyRoadNeuron(BaseNeuron):
    """
    LTC neuron with blocky wave processing and quantized activation.
    """
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 cm: float = 1.0):
        """
        Initialize blocky road neuron.

        Args:
            neuron_id: Unique identifier
            tau: Time constant
            dt: Time step
            gleak: Leak conductance
            cm: Membrane capacitance
        """
        super().__init__(neuron_id, tau, dt)
        self.gleak = gleak
        self.cm = cm
        
    def _initialize_state(self) -> float:
        """Initialize neuron state."""
        return 0.0
        
    def update(self,
               input_signal: float,
               **kwargs) -> float:
        """
        Update neuron state.

        Args:
            input_signal: Input signal
            **kwargs: Additional parameters

        Returns:
            Updated state
        """
        # Apply blocky sigmoid to input
        u = BlockySigmoid.compute(input_signal)
        
        # LTC update with leak using ops functions
        dh_input = ops.multiply(
            ops.divide(1.0, self.tau),
            ops.subtract(u, self.state)
        )
        dh_leak = ops.multiply(self.gleak, self.state)
        dh = ops.subtract(dh_input, dh_leak)
        
        # Update state
        state_change = ops.divide(
            ops.multiply(self.dt, dh),
            self.cm
        )
        self.state = ops.add(self.state, state_change)
        
        # Store history
        self.history.append(self.state)
        
        return self.state

class BlockyRoadChain(BaseChain):
    """Chain of blocky road neurons with progressive time constants."""
    
    def __init__(self,
                 num_neurons: int,
                 base_tau: float = 1.0,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 cm: float = 1.0):
        """
        Initialize blocky road chain.

        Args:
            num_neurons: Number of neurons
            base_tau: Base time constant
            dt: Time step
            gleak: Leak conductance
            cm: Membrane capacitance
        """
        # Create a factory class that will be used to create neurons
        class BlockyRoadNeuronFactory(BlockyRoadNeuron):
            def __new__(cls, neuron_id, tau, dt):
                return BlockyRoadNeuron(
                    neuron_id=neuron_id,
                    tau=tau,
                    dt=dt,
                    gleak=gleak,
                    cm=cm
                )
        
        super().__init__(
            num_neurons=num_neurons,
            neuron_class=BlockyRoadNeuronFactory,
            base_tau=base_tau,
            dt=dt
        )
        # Initialize weights using ops random functions
        self.weights = tensor.random_uniform(
            shape=(num_neurons,),
            minval=0.5,
            maxval=1.5
        )
        
    def update(self, input_signals):
        """
        Update chain state.

        Args:
            input_signals: Input array [batch_size, input_size]

        Returns:
            Updated states [batch_size, num_neurons]
        """
        # Create zeros tensor using ops
        states = tensor.zeros(self.num_neurons)
        
        # Update first neuron
        states_0 = self.neurons[0].update(input_signals[0])
        states = tensor.tensor_scatter_update(states, [0], [states_0])
        
        # Update chain
        for i in range(1, self.num_neurons):
            prev_idx = ops.subtract(i, 1)
            chain_input = ops.multiply(self.weights[prev_idx], states[prev_idx])
            states_i = self.neurons[i].update(chain_input)
            states = tensor.tensor_scatter_update(states, [i], [states_i])
            
        # Store history
        self.state_history.append(tensor.copy(states))
        
        return states
    
    def analyze_memory(self,
                      threshold: float = 0.05,
                      pattern_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze memory retention in the chain.

        Args:
            threshold: Forgetting threshold
            pattern_time: Optional pattern duration for timing analysis

        Returns:
            Dictionary of memory statistics
        """
        stats = {}
        
        # Convert history list to tensor
        history = tensor.stack(self.state_history)
        
        # Compute forgetting times if pattern_time provided
        if pattern_time is not None:
            pattern_idx = tensor.cast(ops.divide(pattern_time, self.dt), tensor.int32)
            forget_times = []
            
            for i in range(self.num_neurons):
                forget_time = None
                history_i = history[:, i]
                
                for idx in range(tensor.to_numpy(pattern_idx), tensor.shape(history)[0]):
                    val = history[idx, i]
                    if ops.less(ops.abs(val), threshold):
                        forget_time = ops.multiply(
                            ops.subtract(idx, pattern_idx),
                            self.dt
                        )
                        break
                        
                forget_times.append(forget_time)
                
            stats['forget_times'] = forget_times
            
        # Compute final states
        final_states = history[-1]
        stats['final_states'] = tensor.to_numpy(final_states)
        
        # Create forgot mask
        forgot_mask = ops.less(ops.abs(final_states), threshold)
        stats['forgot_mask'] = tensor.to_numpy(forgot_mask)
        
        return stats

def create_blocky_chain(num_neurons: int,
                       base_tau: float = 1.0,
                       dt: float = 0.01,
                       gleak: float = 0.5,
                       cm: float = 1.0) -> BlockyRoadChain:
    """
    Factory function to create blocky road chain.

    Args:
        num_neurons: Number of neurons
        base_tau: Base time constant
        dt: Time step
        gleak: Leak conductance
        cm: Membrane capacitance

    Returns:
        Configured blocky road chain
    """
    return BlockyRoadChain(
        num_neurons=num_neurons,
        base_tau=base_tau,
        dt=dt,
        gleak=gleak,
        cm=cm
    )