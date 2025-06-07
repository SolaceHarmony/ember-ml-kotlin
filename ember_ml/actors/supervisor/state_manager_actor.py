"""
Implementation of the StateManagerActor for the Ray-based neuromorphic computing system,
integrated into the ember_ml.actors package.

This actor is responsible for managing neural state, including persistence,
loading, and analysis, using EmberTensor for state representation.
"""

import ray
import time
import json
import pickle
import os
import math
from typing import Dict, List, Any, Optional
from ember_ml.nn.tensor.types import TensorLike

# Import EmberTensor
from ember_ml.nn import tensor


@ray.remote
class StateManagerActor:
    """
    Actor responsible for managing neural state, including persistence,
    loading, and analysis, using EmberTensor for state representation.
    """

    def __init__(self, model_params: Dict, persistence_dir: str = None):
        """
        Initialize the state manager actor.

        Args:
            model_params: Model parameters (expected to contain EmberTensor where applicable)
            persistence_dir: Directory for state persistence
        """
        self.model_params = model_params
        self.current_state: Dict[str, Any] = {} # Use type hint for clarity
        self.step_counter = 0
        self.recent_activity = []
        self.activity_history_size = 100

        # Set up persistence directory
        if persistence_dir is None:
            persistence_dir = os.path.expanduser("~/neuromorphic_states")
        self.persistence_dir = persistence_dir
        os.makedirs(self.persistence_dir, exist_ok=True)

        print(f"[StateManagerActor] Initialized with persistence directory: {self.persistence_dir}")

    async def update_state(self, state_data: Dict[str, Any]):
        """
        Update the current state with new data.

        Args:
            state_data: New state data (expected to contain EmberTensor where applicable)

        Returns:
            Update status
        """
        self.current_state = state_data
        self.step_counter += 1

        # Track activity levels
        h_liquid = self.current_state.get('h_liquid', [])
        # Check if h_liquid is EmberTensor before converting to list
        if isinstance(h_liquid, TensorLike):
             h_liquid_list = h_liquid.tolist()
        else:
             h_liquid_list = h_liquid # Assume it's already a list or empty

        if h_liquid_list:
            activity = sum(1 for v in h_liquid_list
                               if v > self.model_params.get('activity_threshold', 0.5)) / len(h_liquid_list)

            if len(self.recent_activity) >= self.activity_history_size:
                self.recent_activity.pop(0)
            self.recent_activity.append(activity)

        return {"status": "updated", "step": self.step_counter}

    def get_step_counter(self):
        """Get the current step counter."""
        return self.step_counter

    def get_activity_history(self):
        """Get the activity history."""
        return self.recent_activity

    async def get_activity_stats(self):
        """Get activity statistics."""
        if not self.recent_activity:
            return {"current": 0, "average": 0, "trend": 0}

        current = self.recent_activity[-1]
        average = sum(self.recent_activity) / len(self.recent_activity)

        # Calculate trend
        trend = 0
        if len(self.recent_activity) > 10:
            recent_avg = sum(self.recent_activity[-5:]) / 5
            older_avg = sum(self.recent_activity[:5]) / 5
            trend = recent_avg - older_avg

        return {
            "current": current,
            "average": average,
            "trend": trend
        }

    async def save_state(self, filename=None):
        """
        Save the current state to disk.

        Args:
            filename: Optional filename override
        """
        if not self.current_state:
            return {"error": "No state to save"}

        if filename is None:
            filename = f"neural_state_{self.step_counter}.pkl"

        state_file = os.path.join(self.persistence_dir, filename)

        try:
            # Convert EmberTensor (or other tensor types) to lists for serialization
            serializable_state = {}
            for key, value in self.current_state.items():
                if isinstance(value, TensorLike): # Handle other tensor types
                    serializable_state[key] = value.tolist()
                else:
                    serializable_state[key] = value

            with open(state_file, 'wb') as f:
                pickle.dump(serializable_state, f)

            # Save metadata
            metadata = {
                "step": self.step_counter,
                # Get length from EmberTensor shape if h_liquid is EmberTensor
                "neurons": len(self.current_state.get('h_liquid', []).tolist()) if isinstance(self.current_state.get('h_liquid'), TensorLike) else len(self.current_state.get('h_liquid', [])),
                "timestamp": time.time(),
                "use_hebbian": self.model_params.get('use_hebbian', False),
                "eta": self.model_params.get('eta', 0.0),
                "activity": self.recent_activity[-1] if self.recent_activity else 0
            }

            metadata_file = os.path.join(self.persistence_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {"status": "saved", "file": state_file}
        except Exception as e:
            return {"error": f"Error saving state: {str(e)}"}

    async def load_state(self, step=None, filename=None):
        """
        Load state from disk.

        Args:
            step: Step number to load
            filename: Specific filename to load

        Returns:
            Load status
        """
        try:
            # Determine which file to load
            if filename is not None:
                state_file = os.path.join(self.persistence_dir, filename)
            else:
                metadata_file = os.path.join(self.persistence_dir, "metadata.json")
                if not os.path.exists(metadata_file):
                    return {"error": "No saved state found"}

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Determine which step to load
                if step is None:
                    step = metadata.get("step", 0)

                state_file = os.path.join(self.persistence_dir, f"neural_state_{step}.pkl")

            # Check if the file exists
            if not os.path.exists(state_file):
                return {"error": f"State file not found: {state_file}"}

            # Load the state
            with open(state_file, 'rb') as f:
                loaded_state = pickle.load(f)

            # Convert lists back to EmberTensor
            for key, value in loaded_state.items():
                if isinstance(value, list):
                    # Convert list to EmberTensor
                    loaded_state[key] = tensor.convert_to_tensor(value)
                # If it's already a tensor type (e.g., from a previous save/load cycle), keep it
                elif not isinstance(value, TensorLike):
                     # If it's not a list or EmberTensor, try to convert to EmberTensor
                     try:
                         loaded_state[key] = tensor.convert_to_tensor(value)
                     except Exception:
                         # If conversion fails, keep the original value
                         pass


            self.current_state = loaded_state

            # Update step counter if loading by step
            if step is not None:
                self.step_counter = step

            return {"status": "loaded", "step": self.step_counter}
        except Exception as e:
            return {"error": f"Error loading state: {str(e)}"}

    async def analyze_network_dynamics(self):
        """
        Analyze network dynamics.

        Returns:
            Analysis results
        """
        h_liquid = self.current_state.get('h_liquid', [])
        # Check if h_liquid is EmberTensor before converting to list for analysis
        if isinstance(h_liquid, TensorLike):
            h_liquid_list = h_liquid.tolist()
        elif isinstance(h_liquid, TensorLike): # Handle other tensor types
            h_liquid_list = h_liquid.tolist()
        else:
            h_liquid_list = h_liquid # Assume it's already a list or empty


        if not h_liquid_list or all(v == 0 for v in h_liquid_list):
            return {"error": "Network state is empty"}

        # Activity analysis
        active_neurons = sum(1 for v in h_liquid_list
                            if v > self.model_params.get('activity_threshold', 0.5))
        active_ratio = active_neurons / len(h_liquid_list)

        # Basic statistics
        mean = sum(h_liquid_list) / len(h_liquid_list)
        min_val = min(h_liquid_list)
        max_val = max(h_liquid_list)

        # Standard deviation
        variance = sum((v - mean) ** 2 for v in h_liquid_list) / len(h_liquid_list)
        std_dev = math.sqrt(variance)

        return {
            "activeNeurons": active_neurons,
            "activeRatio": active_ratio,
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "stdDev": std_dev,
            "stepCount": self.step_counter,
            "oscillationPhase": self._get_oscillation_phase()
        }

    def _get_oscillation_phase(self):
        """
        Get oscillation phase information.

        Returns:
            Phase information
        """
        # Simplified implementation
        return {
            "theta": (self.step_counter % 8) / 8.0 * 2.0 * math.pi,
            "gamma": (self.step_counter % 40) / 40.0 * 2.0 * math.pi
        }

    def list_saved_states(self):
        """
        List all saved states.

        Returns:
            List of saved states
        """
        states = []
        for filename in os.listdir(self.persistence_dir):
            if filename.startswith("neural_state_") and filename.endswith(".pkl"):
                try:
                    step = int(filename.replace("neural_state_", "").replace(".pkl", ""))
                    states.append({"step": step, "filename": filename})
                except ValueError:
                    continue

        # Sort by step
        states.sort(key=lambda x: x["step"])
        return states

    def delete_state(self, step=None, filename=None):
        """
        Delete a saved state.

        Args:
            step: Step number to delete
            filename: Specific filename to delete
        """
        try:
            # Determine which file to delete
            if filename is not None:
                state_file = os.path.join(self.persistence_dir, filename)
            else:
                if step is None:
                    return {"error": "Must specify either step or filename"}
                state_file = os.path.join(self.persistence_dir, f"neural_state_{step}.pkl")

            # Check if the file exists
            if not os.path.exists(state_file):
                return {"error": f"State file not found: {state_file}"}

            # Delete the file
            os.remove(state_file)
            return {"status": "deleted", "file": state_file}
        except Exception as e:
            return {"error": f"Error deleting state: {str(e)}"}