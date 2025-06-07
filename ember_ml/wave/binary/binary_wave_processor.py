import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import struct

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike

@dataclass
class WaveConfig:
    sample_rate: int
    bit_depth: int
    buffer_size: int
    num_freq_bands: int
    phase_resolution: int

class BinaryWaveProcessor:
    """Handles conversion between PCM audio and binary wave representations"""
    
    def __init__(self, config: WaveConfig):
        self.config = config
        self.phase_accumulator = 0
        self.previous_sample = 0
        self._initialize_frequency_bands()
    
    def _initialize_frequency_bands(self):
        """Initialize frequency band filters"""
        nyquist = self.config.sample_rate // 2
        self.band_frequencies = np.logspace(
            np.log10(20),  # Start at 20 Hz
            np.log10(nyquist),
            self.config.num_freq_bands
        )
        # Create band filters with sharper cutoffs
        self.band_filters = []
        for freq in self.band_frequencies:
            width = freq * 0.3  # Narrower bands
            low = max(0, freq - width)
            high = min(nyquist, freq + width)
            self.band_filters.append((low, high))
    
    def pcm_to_binary(self, pcm_data: TensorLike) -> TensorLike:
        """Convert PCM audio data to binary representation using improved delta-sigma"""
        # Normalize PCM data to [-1, 1]
        if pcm_data.dtype == tensor.int16:
            normalized = pcm_data / 32768.0
        elif pcm_data.dtype == tensor.int32:
            normalized = pcm_data / 2147483648.0
        else:
            normalized = pcm_data.astype(tensor.float32)
        
        # Second-order delta-sigma modulation
        binary = tensor.zeros(len(normalized), dtype=tensor.uint8)
        error1 = 0  # First integrator
        error2 = 0  # Second integrator
        
        for i in range(len(normalized)):
            # Add errors with different weights
            input_value = normalized[i] + error1 * 1.5 - error2 * 0.5
            
            # Quantize
            if input_value > 0:
                binary[i] = 1
                quantization_error = input_value - 1
            else:
                binary[i] = 0
                quantization_error = input_value
            
            # Update error integrators
            error2 = error1 + quantization_error * 0.5
            error1 = quantization_error
                
        return binary
    
    def extract_frequency_bands(self, binary_data: TensorLike) -> List[TensorLike]:
        """Extract frequency band information with improved filtering"""
        # Convert binary to float for FFT
        float_data = binary_data.astype(tensor.float32) * 2 - 1
        
        # Apply Hanning window
        window = np.hanning(len(float_data))
        windowed_data = float_data * window
        
        # Compute FFT
        spectrum = linearalg.rfft(windowed_data)
        freqs = linearalg.rfftfreq(len(float_data), 1/self.config.sample_rate)
        
        # Extract bands with smoother transitions
        band_data = []
        for low, high in self.band_filters:
            # Create band mask with smooth transitions
            center_freq = (low + high) / 2
            bandwidth = high - low
            freq_response = 1 / (1 + ((freqs - center_freq)/(bandwidth/2))**4)
            
            # Apply mask and inverse FFT
            band_spectrum = spectrum * freq_response
            band_signal = np.fft.irfft(band_spectrum)
            
            # Convert back to binary with hysteresis
            threshold = 0.0
            hysteresis = 0.1
            binary = tensor.zeros_like(band_signal, dtype=tensor.uint8)
            state = 0
            
            for i in range(len(band_signal)):
                if state == 0 and band_signal[i] > threshold + hysteresis:
                    state = 1
                elif state == 1 and band_signal[i] < threshold - hysteresis:
                    state = 0
                binary[i] = state
            
            band_data.append(binary)
            
        return band_data
    
    def encode_phase(self, binary_data: TensorLike) -> Tuple[TensorLike, float]:
        """Encode phase information with improved stability"""
        # Use Hilbert transform for phase calculation
        analytic_signal = self._hilbert_transform(binary_data)
        phase = np.angle(analytic_signal)
        
        # Calculate average phase
        mean_phase = stats.mean(phase) % (2 * ops.pi)
        
        return binary_data, mean_phase
    
    def _hilbert_transform(self, binary_data: TensorLike) -> TensorLike:
        """Compute Hilbert transform of binary signal"""
        float_data = binary_data.astype(tensor.float32) * 2 - 1
        spectrum = np.fft.fft(float_data)
        n = len(spectrum)
        h = tensor.zeros(n)
        
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
            
        return np.fft.ifft(spectrum * h)
    
    def process_frame(self, pcm_data: TensorLike) -> Tuple[List[TensorLike], List[float]]:
        """Process a frame of PCM data into binary waves with phase information"""
        # Convert to binary
        binary_data = self.pcm_to_binary(pcm_data)
        
        # Extract frequency bands
        band_data = self.extract_frequency_bands(binary_data)
        
        # Encode phase for each band
        phases = []
        encoded_bands = []
        for band in band_data:
            encoded, phase = self.encode_phase(band)
            encoded_bands.append(encoded)
            phases.append(phase)
            
        return encoded_bands, phases
    
    def binary_to_pcm(self, binary_data: TensorLike) -> TensorLike:
        """Convert binary representation back to PCM audio with improved filtering"""
        # Convert to float
        float_data = binary_data.astype(tensor.float32) * 2 - 1
        
        # Apply sinc reconstruction filter
        filter_length = 31
        t = tensor.arange(-filter_length//2, filter_length//2 + 1)
        sinc = np.sinc(t/2)  # Nyquist frequency
        window = np.hamming(len(sinc))
        filter_kernel = sinc * window
        filter_kernel = filter_kernel / stats.sum(filter_kernel)
        
        # Apply filter
        filtered = np.convolve(float_data, filter_kernel, mode='same')
        
        # Convert to 16-bit PCM
        pcm_int16 = ops.clip(filtered * 32767, -32768, 32767).astype(tensor.int16)
        return pcm_int16

class BinaryWaveNeuron:
    """Binary wave neuron with phase sensitivity and STDP learning"""
    
    def __init__(self, num_freq_bands: int, phase_sensitivity: float = 0.5):
        self.num_freq_bands = num_freq_bands
        self.phase_sensitivity = phase_sensitivity
        self.state = tensor.zeros(num_freq_bands, dtype=tensor.uint8)
        self.phase = tensor.zeros(num_freq_bands)
        self.threshold = 0.3  # Lower threshold for more activation
        self.weights = np.random.random(num_freq_bands) * 0.5 + 0.25  # Better initialization
        
        # STDP parameters
        self.learning_rate = 0.01
        self.stdp_window = 20  # samples
        self.state_history = []
        self.phase_history = []
    
    def compute_interference(self, input_waves: List[TensorLike], input_phases: List[float]) -> TensorLike:
        """Compute wave interference patterns with improved phase sensitivity"""
        interference = tensor.zeros_like(self.state, dtype=tensor.float32)
        
        for i, (wave, phase) in enumerate(zip(input_waves, input_phases)):
            # Compute phase difference with wrapping
            phase_diff = ops.abs(((self.phase[i] - phase + ops.pi) % (2 * ops.pi)) - ops.pi)
            
            # Gaussian phase sensitivity
            phase_factor = ops.exp(-phase_diff**2 / (2 * self.phase_sensitivity**2))
            
            # Compute weighted contribution with temporal integration
            wave_energy = stats.sum(wave) / len(wave)  # Energy in the wave
            interference[i] = wave_energy * self.weights[i] * phase_factor
            
        return interference
    
    def update_phase(self, interference: TensorLike, input_phases: List[float]):
        """Update neuron phase state with momentum"""
        phase_momentum = 0.8
        learn_rate = 0.2
        
        for i, phase in enumerate(input_phases):
            if interference[i] > self.threshold:
                # Update phase with momentum
                phase_diff = ((phase - self.phase[i] + ops.pi) % (2 * ops.pi)) - ops.pi
                self.phase[i] = (self.phase[i] + 
                               phase_momentum * self.phase[i] +
                               learn_rate * phase_diff) % (2 * ops.pi)
    
    def apply_stdp(self, input_waves: List[TensorLike], output: TensorLike):
        """Apply STDP learning rule"""
        # Store state history
        self.state_history.append(output)
        if len(self.state_history) > self.stdp_window:
            self.state_history.pop(0)
        
        # Skip if not enough history
        if len(self.state_history) < 2:
            return
        
        # Compute STDP updates
        for i in range(self.num_freq_bands):
            # Compute correlation between input and output
            input_energy = stats.mean([wave[i] if i < len(wave) else 0 for wave in input_waves])
            output_energy = stats.mean(output)
            
            # Compute weight update
            if output_energy > 0:
                # Hebbian update
                delta_w = self.learning_rate * input_energy * (1 - self.weights[i])
            else:
                # Anti-Hebbian update
                delta_w = -self.learning_rate * input_energy * self.weights[i]
            
            # Apply weight update with bounds
            self.weights[i] = ops.clip(self.weights[i] + delta_w, 0.1, 0.9)
    
    def generate_output(self, interference: TensorLike) -> TensorLike:
        """Generate binary output with hysteresis"""
        output = tensor.zeros_like(interference, dtype=tensor.uint8)
        
        # Add hysteresis to prevent rapid switching
        hysteresis = 0.05
        for i in range(len(interference)):
            if self.state[i] == 0 and interference[i] > self.threshold + hysteresis:
                output[i] = 1
            elif self.state[i] == 1 and interference[i] < self.threshold - hysteresis:
                output[i] = 0
            else:
                output[i] = self.state[i]
        
        return output
    
    def process(self, input_waves: List[TensorLike], input_phases: List[float]) -> TensorLike:
        """Process input waves and generate output"""
        # Compute interference pattern
        interference = self.compute_interference(input_waves, input_phases)
        
        # Update phase
        self.update_phase(interference, input_phases)
        
        # Generate output
        output = self.generate_output(interference)
        
        # Apply STDP learning
        self.apply_stdp(input_waves, output)
        
        # Update state
        self.state = output
        
        return output

def create_test_signal(duration: float, sample_rate: int) -> TensorLike:
    """Create a test signal with improved harmonics"""
    t = tensor.linspace(0, duration, int(duration * sample_rate))
    
    # Create a signal with multiple frequencies and amplitude modulation
    am = 0.5 * (1 + 0.3 * ops.sin(2 * ops.pi * 5 * t))  # 5 Hz amplitude modulation
    
    signal = am * (
        0.5 * ops.sin(2 * ops.pi * 440 * t) +  # A4 note
        0.3 * ops.sin(2 * ops.pi * 880 * t) +  # A5 note
        0.2 * ops.sin(2 * ops.pi * 1760 * t)   # A6 note
    )
    
    # Add some noise
    noise = tensor.random_normal(0, 0.05, len(t))
    signal = signal + noise
    
    # Normalize and convert to int16
    signal = signal / stats.max(ops.abs(signal))
    return (signal * 32767).astype(tensor.int16)