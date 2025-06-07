import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Import from ember_ml.wave.limb
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.wave.limb.hpc_limb_core import (
    int_to_limbs,
    limbs_to_int,
    hpc_add,
    hpc_sub,
    hpc_shr,
    hpc_compare,
    HPCWaveSegment
)

class BinaryWaveNeuron:
    """Binary wave neuron using HPC limb arithmetic"""
    
    def __init__(self, wave_max_val: int = 1_000_000):
        self.segment = HPCWaveSegment(
            wave_max_val=wave_max_val,
            start_val=wave_max_val // 2  # Start at equilibrium
        )
        # Initialize thresholds based on wave_max
        self.upper_threshold = int_to_limbs(int(wave_max_val * 0.75))  # 75% of max
        self.lower_threshold = int_to_limbs(int(wave_max_val * 0.25))  # 25% of max
        
    def process_input(self, input_wave: 'HPCWaveSegment') -> 'HPCWaveSegment':
        """Process input through wave dynamics"""
        # Add input to state
        self.segment.wave_state = hpc_add(
            self.segment.wave_state,
            input_wave.wave_state
        )
        
        # Wave dynamics
        if hpc_compare(self.segment.wave_state, self.upper_threshold) > 0:
            # Above equilibrium - decrease
            decrement = hpc_shr(self.segment.wave_state, 3)  # wave//8
            self.segment.wave_state = hpc_sub(self.segment.wave_state, decrement)
            
        if hpc_compare(self.lower_threshold, self.segment.wave_state) > 0:
            # Below equilibrium - increase
            increment = hpc_shr(self.segment.wave_state, 3)  # wave//8
            self.segment.wave_state = hpc_add(self.segment.wave_state, increment)
        
        # Generate output
        output = HPCWaveSegment(wave_max_val=limbs_to_int(self.segment.wave_max))
        output.wave_state = hpc_shr(self.segment.wave_state, 4)  # wave//16
        
        return output

class BinaryWaveNetwork:
    """Network of binary wave neurons"""
    
    def __init__(self, num_neurons: int = 8, wave_max: int = 1_000_000):
        self.neurons = [BinaryWaveNeuron(wave_max) for _ in range(num_neurons)]
        self.equilibrium = wave_max // 2
        self.pcm_scale = wave_max // (4 * 32768)  # Scale factor for PCM conversion
        
    def process_pcm(self, pcm_data: TensorLike) -> TensorLike:
        """Process PCM audio through the network"""
        output_samples = []
        
        for sample in pcm_data:
            # Convert PCM to wave amplitude around equilibrium
            # Scale PCM value to wave space
            wave_offset = abs(sample) * self.pcm_scale
            if sample >= 0:
                wave_val = self.equilibrium + wave_offset
            else:
                wave_val = self.equilibrium - wave_offset
            wave_val = min(999_999, max(0, wave_val))
            
            # Create input wave segment
            input_wave = HPCWaveSegment(
                wave_max_val=1_000_000,
                start_val=wave_val
            )
            
            # Process through network
            network_output = HPCWaveSegment(wave_max_val=1_000_000)
            network_output.wave_state = int_to_limbs(self.equilibrium)  # Start at equilibrium
            
            # Forward pass
            for i, neuron in enumerate(self.neurons):
                # Process through neuron
                neuron_output = neuron.process_input(input_wave)
                network_output.wave_state = hpc_add(
                    network_output.wave_state,
                    neuron_output.wave_state
                )
                
                # Connect to neighbors
                if i > 0:
                    # Send 1/4 to previous
                    prev_input = HPCWaveSegment(wave_max_val=1_000_000)
                    prev_input.wave_state = hpc_shr(neuron_output.wave_state, 2)
                    self.neurons[i-1].segment.wave_state = hpc_add(
                        self.neurons[i-1].segment.wave_state,
                        prev_input.wave_state
                    )
                    
                if i < len(self.neurons) - 1:
                    # Send 1/4 to next
                    next_input = HPCWaveSegment(wave_max_val=1_000_000)
                    next_input.wave_state = hpc_shr(neuron_output.wave_state, 2)
                    self.neurons[i+1].segment.wave_state = hpc_add(
                        self.neurons[i+1].segment.wave_state,
                        next_input.wave_state
                    )
            
            # Convert output back to PCM
            output_val = limbs_to_int(network_output.wave_state)
            # Convert from wave space back to PCM
            wave_offset = output_val - self.equilibrium
            output_pcm = wave_offset // self.pcm_scale
            output_pcm = max(-32768, min(32767, output_pcm))
            
            output_samples.append(output_pcm)
            
        return tensor.convert_to_tensor(output_samples, dtype=tensor.int16)

def create_test_signal(duration_sec: float, sample_rate: int) -> TensorLike:
    """Create test signal with multiple frequencies"""
    t = tensor.linspace(0, duration_sec, int(duration_sec * sample_rate))
    signal = (
        0.5 * ops.sin(2 * ops.pi * 440 * t) +  # A4
        0.3 * ops.sin(2 * ops.pi * 880 * t) +  # A5
        0.2 * ops.sin(2 * ops.pi * 1760 * t)   # A6
    )
    return (signal * 32767).astype(tensor.int16)