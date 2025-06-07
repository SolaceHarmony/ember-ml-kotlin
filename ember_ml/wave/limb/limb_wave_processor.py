import numpy as np
from typing import List, Tuple, Optional
from array import array
from dataclasses import dataclass

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike

# HPC-limb constants
CHUNK_BITS = 64
CHUNK_BASE = 1 << CHUNK_BITS
CHUNK_MASK = CHUNK_BASE - 1

def int_to_limbs(value: int) -> array:
    """Convert integer to HPC-limb array"""
    if value < 0:
        raise ValueError("Negative values not supported in limb representation")
    limbs = array('Q')
    while value > 0:
        limbs.append(value & CHUNK_MASK)
        value >>= CHUNK_BITS
    if not limbs:
        limbs.append(0)
    return limbs

def limbs_to_int(limbs: array) -> int:
    """Convert HPC-limb array to integer"""
    val = 0
    shift = 0
    for limb in limbs:
        val += limb << shift
        shift += CHUNK_BITS
    return val

def hpc_add(A: array, B: array) -> array:
    """Add two HPC-limb arrays"""
    out_len = max(len(A), len(B))
    out = array('Q', [0] * (out_len + 1))
    carry = 0
    for i in range(out_len):
        av = A[i] if i < len(A) else 0
        bv = B[i] if i < len(B) else 0
        s_val = av + bv + carry
        out[i] = s_val & CHUNK_MASK
        carry = s_val >> CHUNK_BITS
    if carry:
        out[out_len] = carry
    else:
        out.pop()
    return out

def hpc_sub(A: array, B: array) -> array:
    """Subtract two HPC-limb arrays (A - B)"""
    out_len = max(len(A), len(B))
    out = array('Q', [0] * out_len)
    carry = 0
    for i in range(out_len):
        av = A[i] if i < len(A) else 0
        bv = B[i] if i < len(B) else 0
        diff = av - bv - carry
        if diff < 0:
            diff += CHUNK_BASE
            carry = 1
        else:
            carry = 0
        out[i] = diff & CHUNK_MASK
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out

def hpc_shr(A: array, shift_bits: int) -> array:
    """Right shift HPC-limb array"""
    if shift_bits <= 0:
        return array('Q', A)
    out = array('Q', A)
    limb_shifts = shift_bits // CHUNK_BITS
    bit_shifts = shift_bits % CHUNK_BITS
    if limb_shifts >= len(out):
        return array('Q', [0])
    out = out[limb_shifts:]
    if bit_shifts == 0:
        if not out:
            out.append(0)
        return out
    carry = 0
    for i in reversed(range(len(out))):
        cur = out[i] | (carry << CHUNK_BITS)
        out[i] = (cur >> bit_shifts) & CHUNK_MASK
        carry = cur & ((1 << bit_shifts) - 1)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    if not out:
        out.append(0)
    return out

class LimbWaveNeuron:
    """Binary wave neuron using HPC-limb representation"""
    
    def __init__(self, wave_max: int = 1_000_000):
        self.wave_max = int_to_limbs(wave_max)
        self.wave_state = int_to_limbs(0)
        # Lower thresholds for more gradual response
        self.ca_threshold = int_to_limbs(int(wave_max * 0.01))  # 1%
        self.k_threshold = int_to_limbs(int(wave_max * 0.015))  # 1.5%
        self.conduction_div = 16  # Increased division for less gain
        self.leak_div = 8  # Increased leak for better stability
        
        # Automatic gain control
        self.output_history = []
        self.agc_window = 1000
        self.target_output = wave_max // 4
        
    def _apply_agc(self, output: array) -> array:
        """Apply automatic gain control"""
        output_val = limbs_to_int(output)
        self.output_history.append(output_val)
        if len(self.output_history) > self.agc_window:
            self.output_history.pop(0)
        
        if len(self.output_history) > 0:
            avg_output = sum(self.output_history) / len(self.output_history)
            if avg_output > 0:
                gain = self.target_output / avg_output
                gain = min(max(gain, 0.5), 2.0)  # Limit gain range
                output_val = int(output_val * gain)
                return int_to_limbs(output_val)
        
        return output
        
    def process_input(self, input_wave: array) -> array:
        """Process input wave through ion channels and leak currents"""
        # Add attenuated input to state
        input_attenuated = hpc_shr(input_wave, 2)  # Reduce input gain
        self.wave_state = hpc_add(self.wave_state, input_attenuated)
        
        # Ion channels with smoother response
        wave_val = limbs_to_int(self.wave_state)
        ca_val = limbs_to_int(self.ca_threshold)
        k_val = limbs_to_int(self.k_threshold)
        
        if wave_val > ca_val:
            # Gradual Ca2+ activation
            ratio = min((wave_val - ca_val) / ca_val, 1.0)
            increment = hpc_shr(self.wave_state, int(4 + (1 - ratio) * 4))
            self.wave_state = hpc_add(self.wave_state, increment)
            
        if wave_val > k_val:
            # Gradual K+ activation
            ratio = min((wave_val - k_val) / k_val, 1.0)
            decrement = hpc_shr(self.wave_state, int(3 + (1 - ratio) * 4))
            self.wave_state = hpc_sub(self.wave_state, decrement)
        
        # Adaptive leak current
        leak_shift = max(3, min(6, int(np.log2(wave_val / 1000)) if wave_val > 0 else 3))
        leak = hpc_shr(self.wave_state, leak_shift)
        self.wave_state = hpc_sub(self.wave_state, leak)
        
        # Conduction output with AGC
        output = hpc_shr(self.wave_state, int(np.log2(self.conduction_div)))
        output = self._apply_agc(output)
        
        return output

class LimbWaveNetwork:
    """Network of binary wave neurons using HPC-limb representation"""
    
    def __init__(self, num_neurons: int, wave_max: int = 1_000_000):
        self.neurons = [LimbWaveNeuron(wave_max) for _ in range(num_neurons)]
        self.wave_max = wave_max
        
        # Output smoothing
        self.output_buffer = []
        self.smooth_window = 4
        
    def _smooth_output(self, output_val: int) -> int:
        """Apply output smoothing"""
        self.output_buffer.append(output_val)
        if len(self.output_buffer) > self.smooth_window:
            self.output_buffer.pop(0)
        return int(sum(self.output_buffer) / len(self.output_buffer))
        
    def process_pcm(self, pcm_data: TensorLike) -> TensorLike:
        """Process PCM audio through the network"""
        # Convert PCM to limb representation
        pcm_max = 32767  # Max value for 16-bit audio
        scale = self.wave_max / pcm_max  # Scale to wave_max range
        input_waves = []
        
        # Handle both positive and negative PCM values
        for sample in pcm_data:
            # Scale absolute value, keeping sign for later
            sign = 1 if sample >= 0 else -1
            scaled = int(abs(sample) * scale)
            input_waves.append((sign, int_to_limbs(scaled)))
        
        # Process through network
        outputs = []
        for sign, input_wave in input_waves:
            network_output = int_to_limbs(0)
            
            # Forward pass with reduced neighbor coupling
            for i, neuron in enumerate(self.neurons):
                neuron_output = neuron.process_input(input_wave)
                network_output = hpc_add(network_output, neuron_output)
                
                # Connect to neighbors with reduced coupling
                if i > 0:
                    neighbor_input = hpc_shr(neuron_output, 3)  # 1/8 to previous
                    self.neurons[i-1].wave_state = hpc_add(
                        self.neurons[i-1].wave_state,
                        neighbor_input
                    )
                if i < len(self.neurons) - 1:
                    neighbor_input = hpc_shr(neuron_output, 3)  # 1/8 to next
                    self.neurons[i+1].wave_state = hpc_add(
                        self.neurons[i+1].wave_state,
                        neighbor_input
                    )
            
            # Convert output back to PCM scale with smoothing
            output_int = limbs_to_int(network_output)
            output_pcm = int(output_int / scale)
            output_pcm = self._smooth_output(output_pcm)
            
            # Restore sign and clamp to int16 range
            output_pcm *= sign
            output_pcm = max(-32768, min(32767, output_pcm))
            
            outputs.append(output_pcm)
        
        return tensor.convert_to_tensor(outputs, dtype=tensor.int16)

def create_test_signal(duration_sec: float, sample_rate: int) -> TensorLike:
    """Create test signal with multiple frequencies"""
    t = tensor.linspace(0, duration_sec, int(duration_sec * sample_rate))
    signal = (
        0.5 * ops.sin(2 * ops.pi * 440 * t) +  # A4
        0.3 * ops.sin(2 * ops.pi * 880 * t) +  # A5
        0.2 * ops.sin(2 * ops.pi * 1760 * t)   # A6
    )
    return (signal * 32767).astype(tensor.int16)

if __name__ == "__main__":
    # Create test signal
    sample_rate = 44100
    duration = 0.1
    test_signal = create_test_signal(duration, sample_rate)
    
    # Create network and process
    network = LimbWaveNetwork(num_neurons=8)
    output = network.process_pcm(test_signal)
    
    # Print stats
    print(f"Input shape: {test_signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: [{stats.min(test_signal)}, {stats.max(test_signal)}]")
    print(f"Output range: [{stats.min(output)}, {stats.max(output)}]")