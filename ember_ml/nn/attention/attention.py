"""
Attention-enhanced neuron implementation.

This module provides the LTCNeuronWithAttention class that combines
LTC dynamics with an attention mechanism for adaptive behavior.
"""

from typing import Optional

# Use absolute import instead of relative import
from ember_ml.nn.attention.mechanisms.mechanism import CausalAttention


class LTCNeuronWithAttention:
    """
    A neuron that combines LTC dynamics with an attention mechanism.
    
    This neuron type uses attention to modulate its time constant and
    input processing based on prediction accuracy and novelty detection.
    
    Attributes:
        id (int): Unique identifier for the neuron
        tau (float): Base time constant
        state (float): Current activation state
        attention (CausalAttention): Attention mechanism
        last_prediction (float): Previous state prediction
    """
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 attention_params: Optional[dict] = None):
        """
        Initialize an attention-enhanced LTC neuron.
        
        Args:
            neuron_id: Unique identifier for the neuron
            tau: Base time constant for LTC dynamics
            attention_params: Optional parameters for attention mechanism
                            (decay_rate, novelty_threshold, memory_length)
        """
        self.id = neuron_id
        self.tau = tau
        self.state = 0.0
        self.last_prediction = 0.0
        
        # Initialize attention mechanism with optional custom parameters
        if attention_params is None:
            attention_params = {}
        self.attention = CausalAttention(**attention_params)
    
    def update(self, input_signal: float, dt: float) -> float:
        """
        Update the neuron's state using attention-modulated dynamics.
        
        Args:
            input_signal: Current input value
            dt: Time step size for integration
            
        Returns:
            float: Updated neuron state
            
        Note:
            The update process:
            1. Calculate prediction error
            2. Update attention based on error and current state
            3. Modulate time constant based on attention
            4. Update state using attention-weighted input
        """
        # Import ops and tensor here to avoid circular imports
        from ember_ml import ops
        from ember_ml.nn import tensor
        
        # Calculate prediction error
        input_tensor = tensor.convert_to_tensor(input_signal, tensor.float32)
        last_pred_tensor = tensor.convert_to_tensor(self.last_prediction, tensor.float32)
        prediction_error = ops.subtract(input_tensor, last_pred_tensor)
        
        # Update attention
        attention_value = self.attention.update(
            neuron_id=self.id,
            prediction_error=prediction_error,
            current_state=self.state,
            target_state=input_signal
        )
        
        # Modulate time constant based on attention
        # Higher attention -> faster response (smaller effective tau)
        attention_factor = ops.multiply(tensor.convert_to_tensor(0.3), attention_value)
        one_minus_factor = ops.subtract(tensor.convert_to_tensor(1.0), attention_factor)
        effective_tau = ops.multiply(self.tau, one_minus_factor)
        
        # Update LTC dynamics with attention-modulated input
        inv_tau = ops.divide(tensor.convert_to_tensor(1.0), effective_tau)
        attention_plus_one = ops.add(tensor.convert_to_tensor(1.0), attention_value)
        weighted_input = ops.multiply(input_signal, attention_plus_one)
        state_tensor = tensor.convert_to_tensor(self.state, tensor.float32)
        input_minus_state = ops.subtract(weighted_input, state_tensor)
        d_state_factor = ops.multiply(inv_tau, input_minus_state)
        d_state = ops.multiply(d_state_factor, dt)
        
        # Update state
        self.state += d_state
        
        # Store prediction for next update
        self.last_prediction = self.state
        
        return self.state
    
    def reset(self) -> None:
        """Reset the neuron's state and prediction history."""
        self.state = 0.0
        self.last_prediction = 0.0
    
    def get_attention_value(self) -> float:
        """
        Get the current total attention value for this neuron.
        
        Returns:
            float: Current attention value
        """
        # Get the attention state for this neuron, or return 0.0 if not found
        attention_state = self.attention.states.get(self.id)
        if attention_state is None:
            return 0.0
        return attention_state.compute_total()
    
    def __repr__(self) -> str:
        """
        Get a string representation of the neuron.
        
        Returns:
            str: Neuron description including ID and tau
        """
        return (f"LTCNeuronWithAttention(id={self.id}, "
                f"tau={self.tau:.2f}, "
                f"attention={self.get_attention_value():.3f})")