"""
Vision Wiring Pattern for Computer Vision Tasks

This module provides a specialized wiring pattern for visual processing tasks,
implementing local receptive fields and feature hierarchies.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn import tensor

class VisionWiring(NeuronMap):
    """
    Wiring pattern for visual processing.
    
    Architecture:
    - Local receptive fields
    - Feature hierarchies
    - Skip connections
    
    This wiring pattern is designed for computer vision tasks, with a structure
    similar to convolutional neural networks but implemented using recurrent
    neural networks with specific connectivity patterns.
    """
    
    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: List[int],
        kernel_size: int = 3,
        stride: int = 2,
        **kwargs
    ):
        """
        Initialize the vision wiring pattern.
        
        Args:
            input_height: Height of the input image
            input_width: Width of the input image
            channels: List of channel sizes for each layer
            kernel_size: Size of the local receptive field
            stride: Stride for downsampling between layers
            **kwargs: Additional keyword arguments
        """
        # Calculate feature map sizes
        self.feature_maps = self._get_feature_maps(
            input_height,
            input_width,
            channels,
            stride
        )
        
        # Calculate total units
        total_units = sum(h * w * c for h, w, c in self.feature_maps)
        
        # Initialize base class
        super().__init__(units=total_units, output_dim=channels[-1], **kwargs)
        
        # Store configuration
        self.input_height = input_height
        self.input_width = input_width
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Mark as not built yet
        self.built = False
    
    def _get_feature_maps(self, h: int, w: int, channels: List[int], stride: int) -> List[Tuple[int, int, int]]:
        """
        Calculate feature map sizes for each layer.
        
        Args:
            h: Input height
            w: Input width
            channels: List of channel sizes
            stride: Stride for downsampling
            
        Returns:
            List of (height, width, channels) tuples for each layer
        """
        maps = []
        for c in channels:
            maps.append((h, w, c))
            h = (h - 1) // stride + 1
            w = (w - 1) // stride + 1
        return maps
    
    def _get_receptive_field(self, h: int, w: int, layer: int) -> List[Tuple[int, int]]:
        """
        Get neurons in local receptive field.
        
        Args:
            h: Height position
            w: Width position
            layer: Layer index
            
        Returns:
            List of (height, width) positions in the receptive field
        """
        k = self.kernel_size
        h_start = max(0, h - k//2)
        h_end = min(self.feature_maps[layer][0], h + k//2 + 1)
        w_start = max(0, w - k//2)
        w_end = min(self.feature_maps[layer][1], w + k//2 + 1)
        
        field = []
        for i in range(h_start, h_end):
            for j in range(w_start, w_end):
                field.append((i, j))
        return field
    
    def build(self, input_dim: Optional[int] = None):
        """
        Build the wiring pattern.
        
        Args:
            input_dim: Input dimension (optional)
        """
        if self.built:
            return
        
        # Set input dimension if provided
        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = self.input_height * self.input_width * 3  # Assuming RGB input
        
        # Initialize adjacency matrix if not already done
        if self.adjacency_matrix is None:
            self.adjacency_matrix = tensor.zeros((self.units, self.units))
        
        # Build connectivity
        self._build_local_connections()
        self._build_skip_connections()
        
        # Mark as built
        self.built = True
    
    def _build_local_connections(self):
        """Build connections with local receptive fields."""
        offset = 0
        for layer in range(len(self.channels) - 1):
            h, w, c = self.feature_maps[layer]
            next_h, next_w, next_c = self.feature_maps[layer + 1]
            
            # Connect each neuron to its local receptive field
            for i in range(0, h, self.stride):
                for j in range(0, w, self.stride):
                    # Calculate source index
                    for c_idx in range(c):
                        src_idx = offset + (i * w + j) * c + c_idx
                        
                        # Get receptive field in next layer
                        field = self._get_receptive_field(i//self.stride, j//self.stride, layer + 1)
                        
                        # Connect to each position in the receptive field
                        for ni, nj in field:
                            # Connect to all channels in the next layer
                            for nc_idx in range(next_c):
                                dest_idx = (offset + h * w * c) + (ni * next_w + nj) * next_c + nc_idx
                                self.adjacency_matrix = tensor.with_value(
                                    self.adjacency_matrix, src_idx, dest_idx, 1.0
                                )
            
            # Update offset for next layer
            offset += h * w * c
    
    def _build_skip_connections(self):
        """Build skip connections between layers."""
        offset = 0
        for layer in range(len(self.channels) - 2):
            h, w, c = self.feature_maps[layer]
            
            # Connect to layer + 2 (skip one layer)
            skip_h, skip_w, skip_c = self.feature_maps[layer + 2]
            skip_offset = offset + h * w * c + self.feature_maps[layer + 1][0] * self.feature_maps[layer + 1][1] * self.feature_maps[layer + 1][2]
            
            # Sparse skip connections
            for i in range(0, h, self.stride * 2):
                for j in range(0, w, self.stride * 2):
                    # Calculate source index
                    for c_idx in range(c):
                        src_idx = offset + (i * w + j) * c + c_idx
                        
                        # Calculate destination index
                        for sc_idx in range(skip_c):
                            dest_idx = skip_offset + ((i//(self.stride * 2)) * skip_w + j//(self.stride * 2)) * skip_c + sc_idx
                            self.adjacency_matrix = tensor.with_value(
                                self.adjacency_matrix, src_idx, dest_idx, 1.0
                            )
            
            # Update offset for next layer
            offset += h * w * c
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the wiring."""
        config = super().get_config()
        config.update({
            "input_height": self.input_height,
            "input_width": self.input_width,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VisionWiring':
        """Creates a wiring from its configuration."""
        return cls(**config)