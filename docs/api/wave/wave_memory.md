# Wave Memory Systems

This section describes components related to storing and retrieving information using wave-based memory systems, including multi-sphere models and binary wave storage.

## Core Concepts

Wave memory systems utilize the properties of wave interference and propagation to store and recall patterns. This can involve storing binary wave patterns directly or using more complex models like interacting spherical domains.

## Components

### `ember_ml.wave.binary_memory`

*   **`MemoryPattern`**: Dataclass storing a wave pattern, timestamp, and metadata. Includes a `similarity` method.
*   **`WaveStorage`**: Manages a list of `MemoryPattern` objects with a fixed capacity.
    *   `store(pattern, timestamp, metadata)`: Stores a new pattern, removing the oldest if capacity is exceeded.
    *   `retrieve(query_pattern, threshold)`: Retrieves patterns similar to the query based on a threshold.
    *   `clear()`: Clears all stored patterns.
*   **`BinaryMemory(nn.Module)`**: A memory system using binary waves. It combines `BinaryWave` processing, `WaveStorage`, and learnable gates (`store_gate`, `retrieve_gate`) for storing and retrieving patterns.
    *   `store_pattern(pattern, metadata)`: Encodes the input pattern, applies gating, and stores it using `WaveStorage`.
    *   `retrieve_pattern(query, threshold)`: Encodes the query, applies gating, retrieves similar patterns from `WaveStorage`, decodes them, and returns patterns and metadata.
    *   `clear_memory()`: Clears the underlying `WaveStorage`.
    *   `get_memory_state()`: Returns information about the current memory state (number of patterns, capacity).
    *   `save_state()` / `load_state()`: Methods for saving and loading the memory system's state, including configuration, gates, and stored patterns.

### `ember_ml.wave.memory.sphere_overlap`

*   **`SphereOverlap`**: Dataclass defining interaction parameters (reflection, transmission coefficients) between two adjacent spheres. Includes validation logic.
*   **`SphereState`**: Dataclass representing the state of a single sphere, including fast and slow wave state vectors (4D numpy arrays) and noise standard deviation. Includes validation logic.
*   **`OverlapNetwork`**: Dataclass representing the network of overlaps between spheres.
    *   `__init__(overlaps, num_spheres)`: Initializes with a list of `SphereOverlap` objects and the total number of spheres. Includes validation.
    *   `get_neighbors(sphere_idx)`: Returns indices of spheres overlapping with the given sphere.
    *   `get_overlap(idx_A, idx_B)`: Returns the `SphereOverlap` object between two specified spheres, if it exists.

### `ember_ml.wave.memory.multi_sphere`

*   **`MultiSphereWaveModel`**: Models wave interactions across multiple spherical domains defined by an `OverlapNetwork`.
    *   `__init__(M, reflection, transmission, noise_std)`: Initializes `M` spheres and the overlap network connecting adjacent spheres.
    *   `set_initial_state(idx, fast_vec, slow_vec)`: Sets the initial state for a specific sphere.
    *   `run(steps, input_waves_seq, gating_seq)`: Runs the simulation for a number of steps, applying input waves and gating signals. Returns the history of fast states.
    *   `_update_sphere_state(idx, input_wave)`: Updates a single sphere's state based on input, incorporating noise. Uses `partial_interference`.
    *   `_process_overlaps()`: Processes wave interactions (reflection, transmission) between overlapping spheres based on the `OverlapNetwork`.
    *   `get_sphere_states()`: Returns the current fast and slow states of all spheres.

### `ember_ml.wave.memory.visualizer`

*   **`WaveMemoryAnalyzer`**: Provides comprehensive visualization and analysis tools for wave memory systems, particularly `MultiSphereWaveModel`.
    *   `__init__()`: Initializes matplotlib settings.
    *   `analyze_model(model, steps)`: Runs a simulation and generates a comprehensive visualization figure and analysis metrics. Uses `MetricsCollector`.
    *   `create_visualization(history)`: Creates a multi-panel matplotlib figure visualizing various aspects of the wave dynamics (component evolution, phase space, energy, correlations, interference).
    *   `_plot_*` methods: Helper methods for generating individual subplots within the main visualization.
    *   `animate_model(history)`: Creates a matplotlib animation of the wave evolution (specifically the x-component).

### `ember_ml.wave.memory.metrics`

*   **`AnalysisMetrics`**: Dataclass storing analysis metrics (computation time, interference strength, energy stability, phase coherence, total time). Includes validation and formatting methods.
*   **`MetricsCollector`**: Collects and computes `AnalysisMetrics` during a simulation.
    *   `__init__()`: Initializes timers and history storage.
    *   `start_computation()` / `end_computation()`: Used to time the core computation phase.
    *   `record_wave_states(states)`: Records the wave states at a time step.
    *   `compute_metrics()`: Computes the final `AnalysisMetrics` based on recorded history and timings. Uses helper functions from `math_helpers`.
    *   `get_wave_history()`: Returns the recorded wave history as a numpy array.

### `ember_ml.wave.memory.math_helpers` (Relevant Functions)

*   **`normalize_vector(vec, epsilon)`**: Normalizes a vector to unit length.
*   **`compute_phase_angle(vec)`**: Computes phase angle (arctan2) based on vector components.
*   **`compute_energy(vec)`**: Computes energy (squared L2 norm).
*   **`partial_interference(base, new, alpha, epsilon)`**: Computes partial interference between two vectors based on their angle and an interference strength `alpha`.
*   **`compute_phase_coherence(vectors)`**: Computes average phase coherence (mean cosine of phase differences) between multiple vectors.
*   **`compute_interference_strength(vectors)`**: Computes average interference strength (based on dot products) between multiple vectors.
*   **`compute_energy_stability(energy_history)`**: Computes energy stability based on the standard deviation of energy over time.
*   **`create_rotation_matrix(angle, axis)`**: Creates a 4D rotation matrix.