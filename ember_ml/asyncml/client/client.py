"""
Client interface for the Ray-based neuromorphic computing system,
integrated with Ember ML's asynchronous capabilities.
"""

import ray
import asyncio
import time
from typing import Dict, List, Any, Optional
import mlx.core as mx # Keep mlx import for now, as EmberTensor might wrap it
import numpy as np
 
# Import EmberTensor
from ember_ml.nn.tensor import EmberTensor
 
# Assuming actors will be moved/adapted to this new structure
# from .actors.ray_metal_kernel import MetalKernelActor
# from .actors.ray_sequence_processor import SequenceProcessorActor
 
# Placeholder imports for now, actual imports will depend on actor organization
MetalKernelActor = Any
SequenceProcessorActor = Any
 
 
class NeuromorphicClient:
    """
    Client interface for the neuromorphic computing system.
    """
 
    def __init__(self, hidden_dim=16, input_dim=2):
        """
        Initialize the client.
 
        Args:
            hidden_dim: Dimension of hidden state
            input_dim: Dimension of input
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
 
        # Create actors (assuming they are defined and accessible in the Ray environment)
        # In a real scenario, these would be imported and then .remote() called
        try:
            # Attempt to get actor classes from the Ray registry if they are registered
            # This is a placeholder and might need adjustment based on actual actor registration
            self.metal_kernel = ray.get_actor("MetalKernelActor").remote(hidden_dim=hidden_dim, input_dim=input_dim)
            self.sequence_processor = ray.get_actor("SequenceProcessorActor").remote(self.metal_kernel)
        except ValueError:
             # If not registered, assume they are directly importable (as in the original code)
             # This part will need to be updated based on the final actor organization
             from ember_ml.actors.task.metal_kernel_actor import MetalKernelActor as ImportedMetalKernelActor
             from ember_ml.actors.supervisor.sequence_processor_actor import SequenceProcessorActor as ImportedSequenceProcessorActor
             self.metal_kernel = ImportedMetalKernelActor.remote(hidden_dim=hidden_dim, input_dim=input_dim)
             self.sequence_processor = ImportedSequenceProcessorActor.remote(self.metal_kernel)
 
 
        # Store dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
 
        print(f"[NeuromorphicClient] Initialized with hidden_dim={hidden_dim}, input_dim={input_dim}")
 
    async def process_sequence(self, sequence_data: EmberTensor):
        """
        Process a sequence through the system.
 
        Args:
            sequence_data: Input sequence data (as EmberTensor)
 
        Returns:
            Processing result
        """
        # Input is expected to be EmberTensor, no conversion from TensorLike needed here.
        # If conversion is needed, it should happen before calling this method.
 
        # Check dimensions using .shape on EmberTensor
        if len(sequence_data.shape) != 2 or sequence_data.shape[1] != self.input_dim:
            raise ValueError(f"Expected sequence_data with shape (seq_len, {self.input_dim}), got {sequence_data.shape}")
 
        # Record start time
        start_time = time.time()
 
        # Process the sequence
        print(f"[NeuromorphicClient] Processing sequence with shape {sequence_data.shape}")
        # Pass EmberTensor to the sequence processor
        result = await self.sequence_processor.process_sequence.remote(sequence_data)
 
        # Record end time and performance
        end_time = time.time()
        processing_time = end_time - start_time
 
        print(f"[NeuromorphicClient] Sequence processing completed in {processing_time:.4f}s")
 
        # The result['outputs'] is expected to be an EmberTensor from the SequenceProcessorActor
        return result
 
    async def get_metal_kernel_state(self):
        """Get the current state of the Metal kernel."""
        return await self.metal_kernel.get_state.remote()
 
    async def get_metal_kernel_history(self):
        """Get the history of the Metal kernel."""
        return await self.metal_kernel.get_history.remote()
 
    async def get_metal_kernel_performance(self):
        """Get performance statistics for the Metal kernel."""
        return await self.metal_kernel.get_performance_stats.remote()
 
    async def get_sequence_status(self, sequence_id):
        """Get the status of a sequence."""
        return await self.sequence_processor.get_sequence_status.remote(sequence_id)
 
    async def get_active_sequences(self):
        """Get a list of active sequences."""
        return await self.sequence_processor.get_active_sequences.remote()
 
    async def get_completed_sequences(self):
        """Get a list of completed sequences."""
        return await self.sequence_processor.get_completed_sequences.remote()
 
    async def clear_completed_sequences(self):
        """Clear completed sequences from memory."""
        return await self.sequence_processor.clear_completed_sequences.remote()
 
    async def shutdown(self):
        """Shutdown the system."""
        # Ray will handle actor termination
        ray.shutdown()
        print("[NeuromorphicClient] Shutdown complete")

__all__ = [
    "NeuromorphicClient"
]