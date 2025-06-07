# Harmonic Wave Processing

This section describes components related to representing information, particularly embeddings, using combinations of harmonic functions (sine waves).

## Core Concepts

Harmonic wave processing aims to represent complex data, such as text embeddings, as a superposition of sine waves with varying amplitudes, frequencies, and phases. This allows for potentially efficient storage and manipulation of high-dimensional data.

## Components

### `ember_ml.wave.harmonic`

*   **`FrequencyAnalyzer`**: Analyzes frequency components of signals.
    *   `__init__(sampling_rate, window_size, overlap)`: Initializes the analyzer.
    *   `compute_spectrum(signal)`: Computes the frequency spectrum using FFT.
    *   `find_peaks(signal, threshold, tolerance)`: Finds peak frequencies in the spectrum.
    *   `harmonic_ratio(signal)`: Computes the harmonic-to-noise ratio.
*   **`WaveSynthesizer`**: Synthesizes various waveforms.
    *   `__init__(sampling_rate)`: Initializes the synthesizer.
    *   `sine_wave(frequency, duration, amplitude, phase)`: Generates a sine wave.
    *   `harmonic_wave(frequencies, amplitudes, duration)`: Generates a wave with specified harmonics.
    *   `apply_envelope(wave, envelope)`: Applies an amplitude envelope to a wave.
*   **`HarmonicProcessor`**: Combines analysis and synthesis for harmonic signals.
    *   `__init__(sampling_rate)`: Initializes the processor, creating `FrequencyAnalyzer` and `WaveSynthesizer` instances.
    *   `decompose(signal)`: Decomposes a signal into its harmonic components (frequencies and amplitudes).
    *   `reconstruct(frequencies, amplitudes, duration)`: Reconstructs a signal from harmonic components.
    *   `filter_harmonics(signal, keep_frequencies, tolerance)`: Filters a signal to keep only specified harmonics.

### `ember_ml.wave.harmonic.embedding_utils`

*   **`EmbeddingGenerator`**: Handles text embedding generation using transformer models (e.g., BERT).
    *   `__init__(model_name)`: Initializes with a transformer model name.
    *   `generate_embeddings(texts)`: Generates embeddings for a list of texts.

### `ember_ml.wave.harmonic.training`

*   **`HarmonicTrainer`**: Handles training of harmonic wave parameters to match target embeddings.
    *   `__init__(embedding_dim, learning_rate)`: Initializes the trainer.
    *   `loss_function(params, t, target_embedding)`: Computes MSE loss between target embedding and generated harmonic wave.
    *   `compute_gradients(params, t, target_embedding, epsilon)`: Computes numerical gradients using finite differences.
    *   `train(embeddings, t, epochs)`: Trains harmonic parameters using gradient descent.

### `ember_ml.wave.harmonic.visualization`

*   **`HarmonicVisualizer`**: Handles visualization of embeddings and harmonic waves.
    *   `__init__(sampling_rate)`: Initializes the visualizer.
    *   `plot_embeddings(target, learned)`: Plots target vs. learned harmonic embeddings.
    *   `plot_wave_comparison(target_wave, learned_wave, t)`: Compares target and learned waveforms.
    *   `animate_training(params_history, t, target_embedding)`: Creates an animation showing the evolution of the learned wave during training.

### `ember_ml.wave.harmonic.wave_generator`

*   **`harmonic_wave(params, t, batch_size)`**: Generates a harmonic wave based on parameters (amplitudes, frequencies, phases) for a batch of embeddings.
*   **`map_embeddings_to_harmonics(embeddings)`**: Initializes harmonic parameters (randomly) for a batch of embeddings.

### `ember_ml.models.HarmonicWaveDemo` (Relevant Functions)

*   **`harmonic_wave(params, t, batch_size)`**: (Duplicate of function in `wave_generator.py`) Generates harmonic waves from parameters.
*   **`generate_embeddings(texts)`**: Generates embeddings using a specified transformer model (BERT in the demo).
*   **`map_embeddings_to_harmonics(embeddings)`**: (Duplicate of function in `wave_generator.py`) Initializes harmonic parameters randomly.
*   **`loss_function(params, t, target_embedding)`**: (Duplicate of function in `training.py`) Computes MSE loss.
*   **`compute_gradients(params, t, target_embedding, epsilon)`**: (Duplicate of function in `training.py`) Computes numerical gradients.
*   **`train_harmonic_embeddings(embeddings, t, batch_size, learning_rate, epochs)`**: (Duplicate of function in `training.py`) Trains harmonic parameters.
*   **`visualize_embeddings(target, learned)`**: (Duplicate of function in `visualization.py`) Visualizes target vs. learned embeddings.