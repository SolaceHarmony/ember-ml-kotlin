"""
Ray implementation of the SequenceProcessorActor, integrated into the ember_ml.actors package.
"""

import ray
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
import mlx.core as mx # Keep mlx import for now, as EmberTensor might wrap it

# Import EmberTensor and ops
from ember_ml.nn.tensor import EmberTensor
from ember_ml import ops

# Import the MetalKernelActor from its new location
from ember_ml.actors.task.metal_kernel_actor import MetalKernelActor


@ray.remote
class SequenceProcessorActor:
    """
    Actor that processes sequences through the Metal kernel.
    """

    def __init__(self, metal_kernel_actor: MetalKernelActor):
        """
        Initialize a sequence processor actor.

        Args:
            metal_kernel_actor: Reference to the Metal kernel actor
        """
        self.metal_kernel_actor = metal_kernel_actor
        self.sequences = {}
        print(f"[SequenceProcessorActor] Initialized")

    async def process_sequence(self, sequence_data: EmberTensor):
        """
        Process a sequence through the Metal kernel.

        Args:
            sequence_data: Input sequence data (as EmberTensor)

        Returns:
            Processing result
        """
        sequence_id = str(uuid.uuid4())
        # Use .shape on EmberTensor
        print(f"[SequenceProcessorActor] Processing sequence {sequence_id} with {sequence_data.shape[0]} timesteps")

        # Initialize sequence tracking
        self.sequences[sequence_id] = {
            'input_seq': sequence_data,
            'outputs': [],
            # Use sequence_data.shape[0] for number of timesteps
            'pending_timesteps': set(range(sequence_data.shape[0])),
            'completed': False,
            'error': None,
            'start_time': time.time()
        }

        # Reset the Metal kernel
        print(f"[SequenceProcessorActor] Resetting Metal kernel for sequence {sequence_id}")
        await self.metal_kernel_actor.reset_states.remote()

        # Process each timestep
        tasks = []
        # Iterate based on the number of timesteps from EmberTensor shape
        for t in range(sequence_data.shape[0]):
            # Create a request for this timestep
            request = {
                # Extract timestep data as EmberTensor
                'input_data': sequence_data[t],
                'sequence_id': sequence_id,
                'timestep': t
            }

            # Send the request to the Metal kernel
            task = self.metal_kernel_actor.process_data.remote(request)
            tasks.append(task)

            # Log every 10 timesteps or the last timestep
            if t % 10 == 0 or t == sequence_data.shape[0] - 1:
                print(f"[SequenceProcessorActor] Sent timestep {t}/{sequence_data.shape[0]-1} for sequence {sequence_id}")

        # Wait for all tasks to complete
        try:
            # Use await ray.get for asyncio compatibility
            results = await ray.get(tasks)

            # Process results
            for result in results:
                if 'error' in result and result['error']:
                    self.sequences[sequence_id]['error'] = result['error']
                    self.sequences[sequence_id]['completed'] = True
                    return {
                        'sequence_id': sequence_id,
                        'error': result['error']
                    }

                timestep = result['timestep']
                # Append the EmberTensor output directly
                self.sequences[sequence_id]['outputs'].append((timestep, result['output']))
                self.sequences[sequence_id]['pending_timesteps'].remove(timestep)

            # All timesteps processed
            print(f"[SequenceProcessorActor] All timesteps processed for sequence {sequence_id}, assembling final result")

            # Sort outputs by timestep
            self.sequences[sequence_id]['outputs'].sort(key=lambda x: x[0])

            # Extract outputs (list of EmberTensor)
            outputs = [output for _, output in self.sequences[sequence_id]['outputs']]

            # Stack outputs into a single EmberTensor using ember_ml.ops
            stacked_outputs = ops.stack(outputs)

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - self.sequences[sequence_id]['start_time']

            print(f"[SequenceProcessorActor] Sequence {sequence_id} completed in {processing_time:.4f} seconds")

            # Mark as completed
            self.sequences[sequence_id]['completed'] = True

            # Return result (outputs is an EmberTensor)
            return {
                'sequence_id': sequence_id,
                'outputs': stacked_outputs,
                'processing_time': processing_time
            }

        except Exception as e:
            print(f"[SequenceProcessorActor] Error processing sequence: {e}")
            self.sequences[sequence_id]['error'] = str(e)
            self.sequences[sequence_id]['completed'] = True
            return {
                'sequence_id': sequence_id,
                'error': str(e)
            }

    def get_sequence_status(self, sequence_id):
        """Get the status of a sequence."""
        if sequence_id not in self.sequences:
            return {"error": f"Sequence {sequence_id} not found"}

        sequence = self.sequences[sequence_id]
        return {
            'completed': sequence['completed'],
            'error': sequence['error'],
            'pending_timesteps': len(sequence['pending_timesteps']),
            'processed_timesteps': len(sequence['outputs']),
            'total_timesteps': len(sequence['input_seq']),
            'processing_time': time.time() - sequence['start_time'] if not sequence['completed'] else None
        }

    def get_active_sequences(self):
        """Get a list of active sequences."""
        active_sequences = []
        for sequence_id, sequence in self.sequences.items():
            if not sequence['completed']:
                active_sequences.append({
                    'sequence_id': sequence_id,
                    'pending_timesteps': len(sequence['pending_timesteps']),
                    'processed_timesteps': len(sequence['outputs']),
                    'total_timesteps': len(sequence['input_seq']),
                    'processing_time': time.time() - sequence['start_time']
                })
        return active_sequences

    def get_completed_sequences(self):
        """Get a list of completed sequences."""
        completed_sequences = []
        for sequence_id, sequence in self.sequences.items():
            if sequence['completed']:
                completed_sequences.append({
                    'sequence_id': sequence_id,
                    'error': sequence['error'],
                    'total_timesteps': len(sequence['input_seq']),
                    'processing_time': time.time() - sequence['start_time']
                })
        return completed_sequences

    def clear_completed_sequences(self):
        """Clear completed sequences from memory."""
        sequence_ids = list(self.sequences.keys())
        cleared_count = 0

        for sequence_id in sequence_ids:
            if self.sequences[sequence_id]['completed']:
                del self.sequences[sequence_id]
                cleared_count += 1

        return {"cleared_count": cleared_count}