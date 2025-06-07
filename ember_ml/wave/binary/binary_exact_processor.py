import numpy as np
from typing import List, Tuple, Optional

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike

class BinaryWaveState:
    """Exact binary wave state using Python's arbitrary precision integers"""
    
    def __init__(self, initial_value: int = 0):
        # Convert to positive value and store as Python integer
        self.value = abs(int(initial_value))
        self.phase = 0  # Phase accumulator
        
    def add_wave(self, other: 'BinaryWaveState'):
        """Add another wave state with overflow protection"""
        try:
            self.value = (self.value + other.value) & ((1 << 32) - 1)  # 32-bit max
        except OverflowError:
            self.value = (1 << 32) - 1
        
    def sub_wave(self, other: 'BinaryWaveState'):
        """Subtract another wave state with underflow protection"""
        try:
            self.value = max(0, self.value - other.value)
        except OverflowError:
            self.value = 0
        
    def shift_right(self, bits: int):
        """Arithmetic right shift"""
        if bits <= 0:
            return
        self.value >>= bits

class ExactBinaryNeuron:
    """Binary neuron using exact arithmetic"""
    
    def __init__(self, threshold: int = 1000):
        self.state = BinaryWaveState()
        self.threshold = threshold
        self.phase_acc = 0
        self.output_scale = 1.0  # Adaptive output scaling
        
    def process(self, input_wave: BinaryWaveState) -> BinaryWaveState:
        """Process input wave with exact arithmetic"""
        # Add input to state
        self.state.add_wave(input_wave)
        
        # Update phase with smoother progression
        self.phase_acc = (self.phase_acc + 7) & 0xFFFF  # Prime step for better distribution
        phase_factor = max(1, self.phase_acc >> 12)  # 0-15 range, minimum 1
        
        # Generate output based on threshold with phase
        output = BinaryWaveState()
        
        if self.state.value > self.threshold:
            # Compute output with better scaling
            base_output = (self.state.value - self.threshold) >> 3  # Take excess above threshold
            output_val = int(base_output * self.output_scale * phase_factor)
            output.value = min((1 << 32) - 1, output_val)  # Clamp to 32 bits
            
            # Subtract from state with refractory period
            self.state.sub_wave(output)
            
            # Adjust output scale based on activity
            if output.value > 0:
                target = self.threshold >> 1  # Target half of threshold
                ratio = target / output.value
                self.output_scale = max(0.1, min(2.0, self.output_scale * (0.9 + 0.1 * ratio)))
        
        return output

class ExactBinaryNetwork:
    """Network of binary neurons using exact arithmetic"""
    
    def __init__(self, num_neurons: int = 8, threshold: int = 1000):
        self.neurons = [
            ExactBinaryNeuron(threshold * (i + 1) // num_neurons)  # Progressive thresholds
            for i in range(num_neurons)
        ]
        
    def process_pcm(self, pcm_data: TensorLike) -> TensorLike:
        """Process PCM audio through exact binary network"""
        output_samples = []
        output_scale = 1.0  # Adaptive output scaling
        
        # Create output smoothing buffer
        smooth_buffer = [0] * 4
        
        for sample in pcm_data:
            # Convert to unsigned magnitude
            sign = 1 if sample >= 0 else -1
            magnitude = abs(sample)
            
            # Scale to larger range for better precision
            scaled_mag = magnitude << 12  # Scale up by 4096
            
            # Create input wave state
            input_wave = BinaryWaveState(scaled_mag)
            
            # Process through network
            network_output = BinaryWaveState()
            
            # Forward pass with wave interference
            for i, neuron in enumerate(self.neurons):
                # Process with phase relationship
                neuron_output = neuron.process(input_wave)
                
                # Add to network output with phase-based scaling
                phase_scale = (neuron.phase_acc & 0xFF) / 256.0  # 0-1 range
                scaled_output = BinaryWaveState(int(neuron_output.value * phase_scale))
                network_output.add_wave(scaled_output)
                
                # Connect to neighbors with interference
                if i > 0:
                    # Send portion to previous based on phase difference
                    phase_diff = abs(neuron.phase_acc - self.neurons[i-1].phase_acc) / 65536.0
                    prev_amount = int(neuron_output.value * (1.0 - phase_diff) / 4)
                    prev_input = BinaryWaveState(prev_amount)
                    self.neurons[i-1].state.add_wave(prev_input)
                    
                if i < len(self.neurons) - 1:
                    # Send portion to next based on phase difference
                    phase_diff = abs(neuron.phase_acc - self.neurons[i+1].phase_acc) / 65536.0
                    next_amount = int(neuron_output.value * (1.0 - phase_diff) / 4)
                    next_input = BinaryWaveState(next_amount)
                    self.neurons[i+1].state.add_wave(next_input)
            
            # Convert output back to signed PCM with smoothing
            output_val = network_output.value >> 12  # Scale back down
            
            # Update smoothing buffer
            smooth_buffer.pop(0)
            smooth_buffer.append(output_val)
            smooth_val = sum(smooth_buffer) // len(smooth_buffer)
            
            # Apply adaptive scaling
            scaled_val = int(smooth_val * output_scale)
            scaled_val = min(32767, scaled_val)  # Clamp to int16 range
            scaled_val *= sign
            
            # Adjust output scaling
            if abs(scaled_val) > 16384:  # If output is too high
                output_scale *= 0.95
            elif abs(scaled_val) < 8192:  # If output is too low
                output_scale *= 1.05
            output_scale = max(0.1, min(10.0, output_scale))
            
            output_samples.append(scaled_val)
            
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

class BinaryExactProcessor:
    """
    Processor for exact binary wave computations.
    
    This class provides a high-level interface for processing audio data
    using exact binary arithmetic for high precision wave computations.
    """
    
    def __init__(self, num_neurons: int = 8, threshold: int = 1000):
        """
        Initialize the binary exact processor.
        
        Args:
            num_neurons: Number of neurons in the network
            threshold: Base threshold for neurons
        """
        self.network = ExactBinaryNetwork(num_neurons, threshold)
        
    def process(self, input_data: TensorLike) -> TensorLike:
        """
        Process input data through the binary exact network.
        
        Args:
            input_data: Input audio data as numpy array
            
        Returns:
            Processed output data
        """
        return self.network.process_pcm(input_data)
        
    def analyze(self, input_data: TensorLike, output_data: TensorLike) -> dict:
        """
        Analyze the processing results.
        
        Args:
            input_data: Original input data
            output_data: Processed output data
            
        Returns:
            Dictionary with analysis metrics
        """
        return {
            'input_range': (stats.min(input_data), stats.max(input_data)),
            'output_range': (stats.min(output_data), stats.max(output_data)),
            'input_mean': stats.mean(ops.abs(input_data)),
            'output_mean': stats.mean(ops.abs(output_data)),
            'input_std': stats.std(input_data),
            'output_std': stats.std(output_data),
        }

if __name__ == "__main__":
    # Create test signal
    sample_rate = 44100
    duration = 0.1
    test_signal = create_test_signal(duration, sample_rate)
    
    # Create processor and process
    processor = BinaryExactProcessor(num_neurons=8)
    output = processor.process(test_signal)
    
    # Analyze results
    analysis = processor.analyze(test_signal, output)
    
    # Print stats
    print(f"Input shape: {test_signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: {analysis['input_range']}")
    print(f"Output range: {analysis['output_range']}")