"""Binary wave neural processing components."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.container.linear import Linear

def _roll(x: tensor.EmberTensor, shifts: int, axis: int = 0) -> tensor.EmberTensor:
    """Roll tensor along a given axis using NumPy as a fallback."""
    x_np = tensor.to_numpy(x)
    rolled = np.roll(x_np, shifts, axis)
    return tensor.convert_to_tensor(rolled, dtype=tensor.dtype(x), device=ops.get_device(x))


def _flip(x: tensor.EmberTensor, axis: int) -> tensor.EmberTensor:
    """Flip tensor along a given axis using NumPy as a fallback."""
    x_np = tensor.to_numpy(x)
    flipped = np.flip(x_np, axis)
    return tensor.convert_to_tensor(flipped, dtype=tensor.dtype(x), device=ops.get_device(x))


@dataclass
class WaveConfig:
    """Configuration for binary wave processing."""
    grid_size: int = 4
    num_phases: int = 8
    fade_rate: float = 0.1
    threshold: float = 0.5

class BinaryWave(Module):
    """Base class for binary wave processing."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize binary wave processor.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config

        # Learnable parameters
        self.phase_shift = Parameter(tensor.zeros(config.num_phases))
        self.amplitude_scale = Parameter(tensor.ones(config.num_phases))
        
    def encode(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Encode input into wave pattern.

        Args:
            x: Input tensor

        Returns:
            Wave pattern
        """
        # Project to phase space
        phases = tensor.arange(self.config.num_phases, dtype=tensor.float32)
        phases = ops.add(phases, self.phase_shift)
        
        # Generate wave pattern
        t = tensor.linspace(0.0, 1.0, self.config.grid_size * self.config.grid_size)
        phase_term = ops.multiply(
            ops.multiply(ops.multiply(2.0, ops.pi), tensor.expand_dims(phases, -1)),
            tensor.expand_dims(t, 0),
        )
        wave = ops.sin(phase_term)

        wave = ops.multiply(wave, tensor.expand_dims(self.amplitude_scale, -1))
        
        # Apply input modulation
        x_flat = tensor.reshape(x, (-1,))
        wave = ops.multiply(wave, tensor.expand_dims(x_flat, 0))
        
        # Reshape to grid
        wave = tensor.reshape(
            wave,
            (
                self.config.num_phases,
                self.config.grid_size,
                self.config.grid_size,
            ),
        )
        
        return wave
    
    def decode(self, wave: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Decode wave pattern to output.

        Args:
            wave: Wave pattern

        Returns:
            Decoded output
        """
        # Apply inverse wave transform
        wave_flat = tensor.reshape(wave, (self.config.num_phases, -1))
        phases = tensor.arange(self.config.num_phases, dtype=tensor.float32)
        phases = ops.add(phases, self.phase_shift)

        t = tensor.linspace(0.0, 1.0, tensor.shape(wave_flat)[1])
        phase_term = ops.multiply(
            ops.multiply(ops.multiply(2.0, ops.pi), tensor.expand_dims(phases, -1)),
            tensor.expand_dims(t, 0),
        )
        basis = ops.sin(phase_term)
        basis = ops.multiply(basis, tensor.expand_dims(self.amplitude_scale, -1))

        # Solve for output using NumPy pseudoinverse as fallback
        pinv_basis_np = np.linalg.pinv(tensor.to_numpy(basis))
        pinv_basis = tensor.convert_to_tensor(
            pinv_basis_np,
            dtype=tensor.dtype(basis),
            device=ops.get_device(basis),
        )
        output = ops.matmul(pinv_basis, wave_flat)

        
        # Reshape to grid
        output = tensor.reshape(
            output,
            (
                self.config.grid_size,
                self.config.grid_size,
            ),
        )
        
        return output
    
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Process input through wave transform.

        Args:
            x: Input tensor

        Returns:
            Processed output
        """
        wave = self.encode(x)
        return self.decode(wave)

class BinaryWaveProcessor(Module):
    """Processes binary wave patterns."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize processor.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
    def wave_interference(self,
                         wave1: tensor.convert_to_tensor,
                         wave2: tensor.convert_to_tensor,
                         mode: str = 'XOR') -> tensor.convert_to_tensor:
        """
        Apply wave interference between two patterns.
        
        Args:
            wave1, wave2: Binary wave patterns
            mode: Interference type ('XOR', 'AND', or 'OR')
            
        Returns:
            Interference pattern
        """
        # Threshold to binary
        binary1 = ops.greater(wave1, self.config.threshold)
        binary2 = ops.greater(wave2, self.config.threshold)
        
        if mode == 'XOR':
            result = ops.logical_xor(binary1, binary2)
        elif mode == 'AND':
            result = ops.logical_and(binary1, binary2)
        else:  # OR
            result = ops.logical_or(binary1, binary2)

        return tensor.cast(result, tensor.float32)
    
    def phase_similarity(self,
                        wave1: tensor.convert_to_tensor,
                        wave2: tensor.convert_to_tensor,
                        max_shift: Optional[int] = None) -> Dict[str, tensor.convert_to_tensor]:
        """
        Calculate similarity allowing for phase shifts.
        
        Args:
            wave1, wave2: Binary wave patterns
            max_shift: Maximum phase shift to try
            
        Returns:
            Dict containing similarity metrics
        """
        if max_shift is None:
            max_shift = self.config.num_phases // 4
            
        best_similarity = tensor.convert_to_tensor(0.0, device=ops.get_device(wave1))
        best_shift = tensor.convert_to_tensor(0, device=ops.get_device(wave1))
        
        for shift in range(max_shift):
            shifted = _roll(wave2, shifts=shift, axis=0)
            similarity = ops.subtract(1.0, ops.mean(ops.abs(ops.subtract(wave1, shifted))))

            if tensor.item(similarity) > tensor.item(best_similarity):

                best_similarity = similarity
                best_shift = tensor.convert_to_tensor(shift, device=ops.get_device(wave1))
                
        return {
            'similarity': best_similarity,
            'shift': best_shift
        }
    
    def extract_features(self,
                        wave: tensor.convert_to_tensor) -> Dict[str, tensor.convert_to_tensor]:
        """
        Extract characteristic features from wave pattern.
        
        Args:
            wave: Binary wave pattern
            
        Returns:
            Dict of features
        """
        binary = ops.greater(wave, self.config.threshold)
        binary_float = tensor.cast(binary, tensor.float32)
        
        # Basic features
        density = ops.mean(binary_float)
        
        # Transitions (changes between 0 and 1)
        transitions = ops.sum(
            ops.abs(
                ops.subtract(
                    binary_float[..., 1:],
                    binary_float[..., :-1]
                )
            )
        )
        
        # Symmetry measure
        flipped = _flip(binary_float, axis=-1)
        flipped = _flip(flipped, axis=-2)
        symmetry = ops.subtract(1.0, ops.mean(ops.abs(ops.subtract(binary_float, flipped))))

        
        return {
            'density': density,
            'transitions': transitions,
            'symmetry': symmetry
        }

class BinaryWaveEncoder(Module):
    """Encodes data into binary wave patterns."""
    
    def __init__(self, config: WaveConfig = WaveConfig()):
        """
        Initialize encoder.

        Args:
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
    def encode_char(self, char: str) -> tensor.convert_to_tensor:
        """
        Encode a character into a binary wave pattern.
        
        Args:
            char: Single character to encode
            
        Returns:
            4D tensor of shape (num_phases, grid_size, grid_size, 1)
            representing the binary wave pattern
        """
        # Convert to binary
        code_point = ord(char)
        bin_repr = f"{code_point:016b}"
        
        # Create 2D grid
        bit_matrix = tensor.convert_to_tensor(
            [int(b) for b in bin_repr], dtype=tensor.float32
        )
        bit_matrix = tensor.reshape(
            bit_matrix,
            (self.config.grid_size, self.config.grid_size),
        )

        
        # Generate phase shifts
        time_slices = []
        for t in range(self.config.num_phases):
            # Roll the matrix for phase shift
            shifted = _roll(bit_matrix, shifts=t, axis=1)

            
            # Apply fade factor
            fade_factor = max(0.0, 1.0 - t * self.config.fade_rate)
            time_slices.append(ops.multiply(shifted, fade_factor))
            
        # Stack into 4D tensor
        wave_pattern = tensor.stack(time_slices)
        return tensor.expand_dims(wave_pattern, -1)

    
    def encode_sequence(self, sequence: str) -> tensor.convert_to_tensor:
        """
        Encode a sequence of characters into wave patterns.
        
        Args:
            sequence: String to encode
            
        Returns:
            5D tensor of shape (seq_len, num_phases, grid_size, grid_size, 1)
        """
        patterns = [self.encode_char(c) for c in sequence]
        return tensor.stack(patterns)

class BinaryWaveNetwork(Module):
    """Neural network using binary wave processing."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 config: WaveConfig = WaveConfig()):
        """
        Initialize network.

        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
            config: Wave configuration
        """
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = BinaryWaveEncoder(config)
        self.processor = BinaryWaveProcessor(config)
        
        # Learnable parameters
        self.input_proj = Linear(input_size, hidden_size)
        self.wave_proj = Linear(

            hidden_size,
            config.grid_size * config.grid_size,
        )
        self.output_proj = Linear(hidden_size, output_size)

        
        # Wave memory
        self.register_buffer(
            'memory_gate',
            tensor.random_normal(
                (config.grid_size, config.grid_size)
            ),
        )
        self.register_buffer(
            'update_gate',
            tensor.random_normal(
                (config.grid_size, config.grid_size)
            ),

        )
        
    def forward(self,
                x: tensor.convert_to_tensor,
                memory: Optional[tensor.convert_to_tensor] = None) -> Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Process input through binary wave network.
        
        Args:
            x: Input tensor
            memory: Optional previous memory state
            
        Returns:
            (output, new_memory) tuple
        """
        # Project to hidden
        hidden = self.input_proj(x)
        
        # Generate wave pattern
        wave = self.wave_proj(hidden)
        wave = tensor.reshape(
            wave,
            (-1, self.config.grid_size, self.config.grid_size),
        )
        
        # Apply memory gate
        if memory is not None:
            gate = self.processor.wave_interference(
                wave,
                self.memory_gate,
                mode='AND'
            )
            wave = self.processor.wave_interference(
                wave,
                gate,
                mode='AND'
            )
            
            # Update memory
            update = self.processor.wave_interference(
                wave,
                self.update_gate,
                mode='AND'
            )
            memory = self.processor.wave_interference(
                memory,
                update,
                mode='OR'
            )
        else:
            memory = wave
            
        # Project to output
        output = self.output_proj(hidden)
        
        return output, memory