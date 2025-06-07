# Binary Wave Processing

This section describes components related to representing and processing information using binary wave patterns and interference phenomena.

## Core Concepts

Binary waves represent information through patterns of binary states (0s and 1s) evolving over time and space. Key operations involve interference, phase shifts, and pattern matching.

## Components

### `ember_ml.wave.binary_wave`

*   **`WaveConfig`**: Dataclass holding configuration for binary wave processing (grid size, number of phases, fade rate, threshold).
*   **`BinaryWave(nn.Module)`**: Base class for binary wave processing. Includes learnable parameters for phase shift and amplitude scaling.
    *   `encode(x)`: Encodes input `x` into a wave pattern based on phase shifts and amplitude scaling.
    *   `decode(wave)`: Decodes a wave pattern back into its original representation using an inverse transform.
    *   `forward(x)`: Encodes and then decodes the input.
*   **`BinaryWaveProcessor(nn.Module)`**: Processes binary wave patterns.
    *   `wave_interference(wave1, wave2, mode)`: Applies interference ('XOR', 'AND', 'OR') between two binary wave patterns.
    *   `phase_similarity(wave1, wave2, max_shift)`: Calculates similarity between two waves, allowing for phase shifts. Returns similarity score and best shift.
    *   `extract_features(wave)`: Extracts features like density, transitions, and symmetry from a wave pattern.
*   **`BinaryWaveEncoder(nn.Module)`**: Encodes data (specifically characters and sequences) into binary wave patterns.
    *   `encode_char(char)`: Encodes a single character into a 4D wave pattern tensor.
    *   `encode_sequence(sequence)`: Encodes a string sequence into a 5D tensor of wave patterns.
*   **`BinaryWaveNetwork(nn.Module)`**: A neural network architecture using binary wave processing components (encoder, processor) and learnable projections. Includes a simple memory mechanism using gating.
    *   `forward(x, memory)`: Processes input `x` and optional previous memory state, returning output and new memory state.

### `ember_ml.wave.binary_pattern`

*   **`PatternMatch`**: Dataclass storing results of pattern matching (similarity, position, phase shift, confidence).
*   **`InterferenceDetector`**: Detects and analyzes interference patterns between waves.
    *   `detect_interference(wave1, wave2)`: Computes constructive, destructive, and multiplicative interference and their strengths.
    *   `find_resonance(wave, num_shifts)`: Finds resonance patterns by checking interference with phase-shifted versions of the same wave.
*   **`PatternMatcher`**: Matches and aligns binary wave patterns using a sliding window approach with phase shift tolerance.
    *   `match_pattern(template, target, threshold)`: Finds occurrences of a template pattern within a target pattern.
*   **`BinaryPattern(nn.Module)`**: A module for pattern recognition using binary wave interference. It combines feature extraction (using Conv2D layers) with the `PatternMatcher` and `InterferenceDetector`.
    *   `extract_pattern(wave)`: Extracts features from a wave pattern using internal Conv2D layers.
    *   `match_pattern(template, target, threshold)`: Encodes template and target, extracts features, and uses `PatternMatcher`.
    *   `analyze_interference(wave1, wave2)`: Extracts features and analyzes interference and resonance using `InterferenceDetector`.
    *   `forward(input_wave, template)`: Processes an input wave, extracts features, finds resonance, and optionally performs pattern matching against a template.

### `ember_ml.wave.binary.binary_exact_processor`

*   **`BinaryWaveState`**: Represents an exact binary wave state using Python's arbitrary precision integers.
*   **`ExactBinaryNeuron`**: Implements a binary neuron using exact integer arithmetic.
*   **`ExactBinaryNetwork`**: A network of `ExactBinaryNeuron` instances.
*   **`create_test_signal(...)`**: Generates a test signal.
*   **`BinaryExactProcessor`**: Handles conversion between PCM audio and exact binary wave representations using arbitrary precision integers.

### `ember_ml.wave.binary.binary_wave_neuron`

*   **`BinaryWaveNeuron`**: Implements a binary wave neuron using HPC limb arithmetic (likely intended, but uses numpy/torch in this specific file). Includes phase sensitivity and STDP learning concepts (though STDP not fully implemented here).
*   **`BinaryWaveNetwork`**: A network of `BinaryWaveNeuron` instances.
*   **`create_test_signal(...)`**: Generates a test signal.

### `ember_ml.wave.binary.binary_wave_processor`

*   **`WaveConfig`**: Dataclass for configuration (duplicate of the one in `binary_wave.py`).
*   **`BinaryWaveProcessor`**: Handles conversion between PCM audio and binary wave representations. Includes phase sensitivity and STDP learning concepts.
*   **`BinaryWaveNeuron`**: Neuron implementation within the processor (similar to the standalone one).
*   **`create_test_signal(...)`**: Generates a test signal.

### `ember_ml.wave.binary.wave_interference_processor`

*   **HPC Limb Functions (`int_to_limbs`, `limbs_to_int`, `hpc_add`, `hpc_sub`, `hpc_shr`)**: Core functions for high-precision arithmetic using 64-bit limbs (duplicates from `hpc_limb_core.py`).
*   **`make_exponential_factor(...)`**: Creates an exponential decay factor.
*   **`exponential_leak(...)`**: Applies exponential leak to a wave state.
*   **`WaveInterferenceNeuron`**: Implements a binary wave neuron using interference patterns and HPC limb arithmetic.
*   **`WaveInterferenceNetwork`**: A network of `WaveInterferenceNeuron` instances.
*   **`create_test_signal(...)`**: Generates a test signal.
*   **`WaveInterferenceProcessor`**: Handles conversion and processing using wave interference neurons and HPC limb arithmetic.