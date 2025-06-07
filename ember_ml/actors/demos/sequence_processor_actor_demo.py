"""
Demonstrates using the SequenceProcessorActor from the ember_ml.actors package.
"""

import asyncio
import ray
import asyncio
import ray
# import mlx.core as mx # Removed as EmberTensor is used
 
# Import EmberTensor
from ember_ml.nn.tensor import EmberTensor
 
# Import the actors from their new locations
from ember_ml.actors.task.metal_kernel_actor import MetalKernelActor
from ember_ml.actors.supervisor.sequence_processor_actor import SequenceProcessorActor
 
async def test_sequence_processor_demo():
    """
    Demonstrates creating and interacting with the SequenceProcessorActor.
    """
    # Initialize Ray
    ray.init()
 
    # Create the actors
    metal_kernel = MetalKernelActor.remote(hidden_dim=16, input_dim=2)
    sequence_processor = SequenceProcessorActor.remote(metal_kernel)
 
    # Process a sequence
    # Create sequence data as EmberTensor
    sequence_data = EmberTensor.random_normal((10, 2))
    result = await sequence_processor.process_sequence.remote(sequence_data)
    # Result['outputs'] is expected to be an EmberTensor
    print(f"Sequence processing result: {result}")
 
    # Get active sequences (should be empty now)
    active_sequences = await sequence_processor.get_active_sequences.remote()
    print(f"Active sequences: {active_sequences}")
 
    # Get completed sequences
    completed_sequences = await sequence_processor.get_completed_sequences.remote()
    print(f"Completed sequences: {completed_sequences}")
 
    # Clear completed sequences
    cleared = await sequence_processor.clear_completed_sequences.remote()
    print(f"Cleared sequences: {cleared}")
 
    # Shut down Ray
    ray.shutdown()
 
if __name__ == "__main__":
    # Run the asynchronous demo
    asyncio.run(test_sequence_processor_demo())