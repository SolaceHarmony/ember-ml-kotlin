# Wave Limb Arithmetic (High-Precision Computing)

This section describes components related to high-precision computing (HPC) using 64-bit "limbs" for exact integer arithmetic, primarily intended for wave-based computations where precision is critical.

## Core Concepts

Limb arithmetic breaks down large integers into arrays of 64-bit unsigned integers (limbs). Arithmetic operations (addition, subtraction, shifts) are performed on these arrays, allowing for calculations that exceed the precision limits of standard floating-point types. This is particularly relevant for certain wave simulation models that rely on exact integer state representations.

## Components

### `ember_ml.wave.limb.hpc_limb_core` & `ember_ml.wave.limb.hpc_limb`

These files contain the core functions and a class wrapper for HPC limb arithmetic. Note that there is significant duplication between these two files.

*   **Constants (`CHUNK_BITS`, `CHUNK_BASE`, `CHUNK_MASK`)**: Define the parameters for 64-bit limb arithmetic.
*   **`int_to_limbs(value)`**: Converts a non-negative Python integer into an `array.array('Q')` of 64-bit limbs.
*   **`limbs_to_int(limbs)`**: Converts an array of 64-bit limbs back into a Python integer.
*   **`hpc_add(A, B)`**: Adds two HPC limb arrays (`A` and `B`), handling carries.
*   **`hpc_sub(A, B)`**: Subtracts HPC limb array `B` from `A` (assuming `A >= B`), handling borrows.
*   **`hpc_shr(A, shift_bits)`**: Right-shifts an HPC limb array `A` by `shift_bits`, handling shifts across limb boundaries.
*   **`hpc_compare(A, B)`**: Compares two HPC limb arrays (`hpc_limb_core.py` only). Returns -1 if A < B, 0 if A == B, 1 if A > B.
*   **`HPCLimb` (Class in `hpc_limb.py`)**: A wrapper class around the limb array (`array.array('Q')`) providing methods like `copy()`, `to_int()`, and `__repr__()`. It uses the core functions internally.
*   **`HPCWaveSegment` (Class in `hpc_limb_core.py`)**: Represents a wave segment using HPC limbs. Includes `update`, `get_state`, and `get_normalized_state` methods.

### `ember_ml.wave.limb.limb_wave_processor`

*   **HPC Limb Functions**: Duplicates of `int_to_limbs`, `limbs_to_int`, `hpc_add`, `hpc_sub`, `hpc_shr`.
*   **`LimbWaveNeuron`**: Implements a binary wave neuron using the HPC limb representation for its internal state and calculations (thresholds, updates, conduction). Includes automatic gain control (AGC).
    *   `__init__(wave_max)`: Initializes neuron with thresholds based on `wave_max`.
    *   `_apply_agc(output)`: Applies automatic gain control to the output.
    *   `process_input(input_wave)`: Updates the neuron's state based on input, ion channel dynamics (Ca2+, K+), leak current, and AGC. Returns the conduction output.
*   **`LimbWaveNetwork`**: A network composed of `LimbWaveNeuron` instances.
    *   `__init__(num_neurons, wave_max)`: Initializes the network.
    *   `_smooth_output(output_val)`: Applies smoothing to the network output.
    *   `process_pcm(pcm_data)`: Processes an array of PCM audio samples through the network, converting to/from limb representation and handling neighbor coupling. Returns the processed PCM data.
*   **`create_test_signal(...)`**: Generates a test audio signal.

### `ember_ml.wave.limb.wave_segment`

*   **`WaveSegment`**: Represents a segment of a wave using `HPCLimb` objects for its state. Manages propagation delays and ion channel dynamics.
    *   `__init__(initial_state, wave_max, ca_threshold, k_threshold)`: Initializes the segment with parameters converted to `HPCLimb`.
    *   `propagate(conduction_val)`: Updates the segment state based on conduction from neighbors, managing history for delays.
    *   `update_ion_channels()`: Updates state based on Ca2+ and K+ channel thresholds.
    *   `get_conduction_value()`: Calculates the conduction value to propagate.
    *   `get_normalized_state()`: Returns the current state normalized to [0, 1].
*   **`WaveSegmentArray`**: An array of `WaveSegment` objects that interact.
    *   `__init__(num_segments)`: Initializes the array.
    *   `update()`: Updates all segments for one time step, handling propagation and boundary reflections.
    *   `_apply_boundary_reflection(segment)`: Applies partial reflection at boundaries.
    *   `get_wave_state()`: Returns an array of normalized states for all segments.

### `ember_ml.wave.limb.pwm_processor`

*   **`PWMProcessor`**: Handles conversion between PCM audio samples and Pulse Width Modulation (PWM) signals. While placed in the `limb` directory, it currently uses standard numpy operations, not HPC limb arithmetic.
    *   `__init__(bits_per_block, carrier_freq, sample_rate)`: Initializes PWM parameters.
    *   `pcm_to_pwm(pcm_data)`: Converts 16-bit PCM data to a binary PWM signal based on duty cycle quantization.
    *   `pwm_to_pcm(pwm_signal)`: Converts a PWM signal back to PCM by calculating duty cycle per block.
    *   `analyze_pwm_signal(pwm_signal)`: Calculates statistics about the duty cycles in a PWM signal.
    *   `get_pwm_parameters()`: Returns the current PWM parameters.