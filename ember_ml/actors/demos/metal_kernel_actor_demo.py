"""
Demonstrates using the MetalKernelActor from the ember_ml.actors package.
"""

import asyncio
import ray
import asyncio
import ray
# import mlx.core as mx # Removed as EmberTensor is used

from ember_ml.nn import tensor 
# Import the MetalKernelActor from its new location
from ember_ml.actors.task.metal_kernel_actor import MetalKernelActor
 
async def test_metal_kernel_demo():
    """
    Demonstrates creating and interacting with the MetalKernelActor.
    """
    # Initialize Ray
    ray.init()
 
    # Create the actor
    metal_kernel = MetalKernelActor.remote(hidden_dim=16, input_dim=2)
 
    # Reset states
    reset_result = await metal_kernel.reset_states.remote()
    print(f"Reset result: {reset_result}")
 
    # Process some data
    for i in range(5):
        request = {
            # Create input data as EmberTensor
            'input_data': tensor.random_normal((2,)),
            'sequence_id': 'test-sequence',
            'timestep': i
        }
        result = await metal_kernel.process_data.remote(request)
        print(f"Process result for timestep {i}: {result}")
 
    # Get performance stats
    stats = await metal_kernel.get_performance_stats.remote()
    print(f"Performance stats: {stats}")
 
    state = await metal_kernel.get_state.remote()
    print(f"Current state: {state}")
 
    # Shut down Ray
    ray.shutdown()
 
if __name__ == "__main__":
    # Run the asynchronous demo
    asyncio.run(test_metal_kernel_demo())