import numpy as np
from typing import List, Tuple, Optional
from array import array
import math

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

LEAK_SHIFT = 16
def make_exponential_factor(tau_ms: float = 100.0, dt_ms: float = 1.0) -> int:
    """Create exponential decay factor for leak current"""
    alpha = math.exp(-dt_ms / tau_ms)
    alpha_scaled = int(alpha * (1 << LEAK_SHIFT))
    return alpha_scaled

def exponential_leak(wave_state: array, alpha_scaled: int) -> array:
    """Apply exponential leak to wave state"""
    wave_int = limbs_to_int(wave_state)
    wave_after = (wave_int * alpha_scaled) >> LEAK_SHIFT
    return int_to_limbs(wave_after)

class WaveInterferenceNeuron:
    """Binary wave neuron using interference patterns"""
    
    def __init__(self, wave_max: int = 1_000_000):
        self.wave_max = int_to_limbs(wave_max)
        self.wave_state = int_to_limbs(0)
        self.phase_state = 0.0
        self.leak_factor = make_exponential_factor()
        
        # Wave interference parameters
        self.interference_window = []
        self.max_window = 16
        self.interference_threshold = wave_max // 64  # Reduced threshold
        
    def _compute_interference(self, input_wave: array) -> array:
        """Compute wave interference pattern"""
        # Store in interference window
        self.interference_window.append(input_wave)
        if len(self.interference_window) > self.max_window:
            self.interference_window.pop(0)
            
        # Compute interference
        interference = array('Q', [0])
        for i, wave in enumerate(self.interference_window):
            # Phase-shifted addition with attenuation
            shift = (i % 8) + 2  # Additional shift for more attenuation
            shifted = hpc_shr(wave, shift)
            interference = hpc_add(interference, shifted)
            
        return interference
    
    def process_input(self, input_wave: array) -> array:
        """Process input through interference and leak"""
        # Compute interference pattern
        interference = self._compute_interference(input_wave)
        
        # Add attenuated interference to wave state
        interference_attenuated = hpc_shr(interference, 2)
        self.wave_state = hpc_add(self.wave_state, interference_attenuated)
        
        # Apply exponential leak
        self.wave_state = exponential_leak(self.wave_state, self.leak_factor)
        
        # Generate output based on threshold with attenuation
        wave_val = limbs_to_int(self.wave_state)
        if wave_val > self.interference_threshold:
            # Progressive output based on how far above threshold
            ratio = min((wave_val - self.interference_threshold) / self.interference_threshold, 2.0)
            shift_amount = max(4, min(6, int(6 - ratio * 2)))  # Dynamic shift based on ratio
            output = hpc_shr(self.wave_state, shift_amount)
            self.wave_state = hpc_sub(self.wave_state, output)
            return output
        
        return array('Q', [0])

class WaveInterferenceNetwork:
    """Network of wave interference neurons"""
    
    def __init__(self, num_neurons: int, wave_max: int = 1_000_000):
        self.neurons = [WaveInterferenceNeuron(wave_max) for _ in range(num_neurons)]
        self.wave_max = wave_max
        self.output_buffer = []
        self.buffer_size = 4
        
    def _smooth_output(self, value: float) -> float:
        """Apply output smoothing"""
        self.output_buffer.append(value)
        if len(self.output_buffer) > self.buffer_size:
            self.output_buffer.pop(0)
        return sum(self.output_buffer) / len(self.output_buffer)
    
    def process_pcm(self, pcm_data: TensorLike) -> TensorLike:
        """Process PCM audio through wave interference network"""
        # Convert PCM to wave representation
        pcm_max = 32767
        scale = self.wave_max / (pcm_max * 2)  # Reduced scale factor
        outputs = []
        
        # Process each sample
        for sample in pcm_data:
            # Get sign and magnitude
            sign = np.sign(sample)
            magnitude = abs(sample)
            
            # Scale to wave representation
            wave_val = int(magnitude * scale)
            input_wave = int_to_limbs(wave_val)
            
            # Process through network
            network_output = array('Q', [0])
            
            # Forward pass with interference
            for i, neuron in enumerate(self.neurons):
                neuron_output = neuron.process_input(input_wave)
                network_output = hpc_add(network_output, neuron_output)
                
                # Propagate to neighbors through interference
                if i > 0:
                    self.neurons[i-1].interference_window.append(
                        hpc_shr(neuron_output, 3)
                    )
                if i < len(self.neurons) - 1:
                    self.neurons[i+1].interference_window.append(
                        hpc_shr(neuron_output, 3)
                    )
            
            # Convert back to PCM with smoothing
            output_val = limbs_to_int(network_output)
            output_pcm = int(output_val / scale)
            output_pcm = self._smooth_output(output_pcm)
            
            # Apply sign and clamp
            output_pcm = int(output_pcm * sign)
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

class WaveInterferenceProcessor:
    """
    Processor for wave interference computations.
    
    This class provides a high-level interface for processing audio data
    using wave interference patterns for complex wave transformations.
    """
    
    def __init__(self, num_neurons: int = 8, wave_max: int = 1_000_000):
        """
        Initialize the wave interference processor.
        
        Args:
            num_neurons: Number of neurons in the network
            wave_max: Maximum wave amplitude
        """
        self.network = WaveInterferenceNetwork(num_neurons, wave_max)
        
    def process(self, input_data: TensorLike) -> TensorLike:
        """
        Process input data through the wave interference network.
        
        Args:
            input_data: Input audio data as numpy array
            
        Returns:
            Processed output data
        """
        return self.network.process_pcm(input_data)
        
    def analyze_frequency_response(self, sample_rate: int = 44100) -> dict:
        """
        Analyze the frequency response of the processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Dictionary with frequency response analysis
        """
        # Generate sweep signal
        duration = 1.0
        t = tensor.linspace(0, duration, int(duration * sample_rate))
        freqs = np.logspace(np.log10(20), np.log10(20000), 10)  # 10 frequencies from 20Hz to 20kHz
        
        responses = {}
        for freq in freqs:
            # Generate sine wave at this frequency
            sine = ops.sin(2 * ops.pi * freq * t)
            sine_pcm = (sine * 32767).astype(tensor.int16)
            
            # Process through network
            output = self.process(sine_pcm)
            
            # Calculate response
            input_power = stats.mean(ops.abs(sine_pcm))
            output_power = stats.mean(ops.abs(output))
            
            # Store response ratio
            responses[freq] = output_power / input_power if input_power > 0 else 0
            
        return {
            'frequencies': list(freqs),
            'responses': list(responses.values())
        }

if __name__ == "__main__":
    # Create test signal
    sample_rate = 44100
    duration = 0.1
    test_signal = create_test_signal(duration, sample_rate)
    
    # Create processor and process
    processor = WaveInterferenceProcessor(num_neurons=8)
    output = processor.process(test_signal)
    
    # Analyze frequency response
    freq_response = processor.analyze_frequency_response()
    
    # Print stats
    print(f"Input shape: {test_signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: [{stats.min(test_signal)}, {stats.max(test_signal)}]")
    print(f"Output range: [{stats.min(output)}, {stats.max(output)}]")
    print(f"Frequency response: {freq_response}")