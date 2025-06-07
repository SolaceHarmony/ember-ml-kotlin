"""
Demonstrates using the asynchronous Ray client for the neuromorphic computing system.
"""

import asyncio
import matplotlib.pyplot as plt
# Demonstrates using the asynchronous Ray client for the neuromorphic computing system.
 
 
# Import EmberTensor
from ember_ml.nn.tensor import EmberTensor
 
# Import the asynchronous client from the new structure
from ember_ml.asyncml.client.client import NeuromorphicClient
from ember_ml import ops
from ember_ml.nn import tensor
async def test_client_demo():
    """
    Demonstrates creating a client, processing a sequence, and visualizing results.
    """
    # Create client
    client = NeuromorphicClient(hidden_dim=16, input_dim=2)
 
    # Generate test data (sine wave) using NumPy
    # In a scenario with async tensor ops, this could use async_ops or async EmberTensor methods
    seq_len = 50
    t = tensor.linspace(0, 4 * ops.pi, seq_len)
    sin_data = ops.sin(t)
    cos_data = ops.cos(t)
    sequence_data_np = np.column_stack((sin_data, cos_data))
 
    # Convert NumPy data to EmberTensor
    # If EmberTensor had async methods, this might be:
    # sequence_data = await EmberTensor.async_convert_to_tensor(sequence_data_np)
    sequence_data = EmberTensor(sequence_data_np)
 
 
    # Process the sequence asynchronously
    print("Processing sequence using the asynchronous client...")
    result = await client.process_sequence(sequence_data)
 
    # Check if there was an error
    if 'error' in result and result['error']:
        print(f"Error processing sequence: {result['error']}")
    else:
        # Get the outputs (expected to be EmberTensor)
        outputs = result['outputs']
        print(f"Output shape: {outputs.shape}")
 
        # Plot the results using Matplotlib
        plt.figure(figsize=(12, 8))
 
        # Plot input (using original NumPy data for plotting simplicity)
        plt.subplot(2, 1, 1)
        plt.plot(t, sin_data, label='Sin Input')
        plt.plot(t, cos_data, label='Cos Input')
        plt.title('Input Sequence')
        plt.legend()
 
        # Plot output
        plt.subplot(2, 1, 2)
        # Convert EmberTensor output to list for plotting
        output_data = outputs.tolist()
        plt.plot(t, [x[0] for x in output_data], label='Output 0')
        plt.plot(t, [x[1] for x in output_data], label='Output 1')
        plt.title('Output Sequence')
        plt.legend()
 
        plt.tight_layout()
        plt.savefig('ray_sequence_results_demo.png')
        plt.show()
 
    # Get performance stats asynchronously
    performance = await client.get_metal_kernel_performance()
    print(f"Performance stats: {performance}")
 
    # Shutdown the client asynchronously
    await client.shutdown()
 
if __name__ == "__main__":
    # Run the asynchronous demo
    asyncio.run(test_client_demo())