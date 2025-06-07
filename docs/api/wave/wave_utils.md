# Wave Utilities

This section describes various utility functions supporting wave-based processing and analysis within Ember ML.

## Core Concepts

These utilities provide foundational mathematical operations, signal analysis tools, data conversion functions, and visualization capabilities specifically tailored for working with wave data and models. Many functions leverage the `ops` abstraction layer for backend-agnostic computation, while others rely on standard libraries like NumPy, SciPy, or Librosa (where available).

## Components

### Mathematical Helpers (`ember_ml.wave.utils.math_helpers`)

Provides backend-agnostic math functions using the `ops` layer.

*   **Activation Functions**: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `softmax`.
*   **Normalization/Standardization**: `normalize` (to unit norm), `standardize` (to zero mean, unit variance).
*   **Distance/Similarity**: `euclidean_distance`, `cosine_similarity`.
*   **Basic Functions**: `exponential_decay`, `gaussian`.
*   **Vector/Wave Specific**:
    *   `normalize_vector`: Normalizes a vector (duplicate of `normalize`?).
    *   `compute_energy_stability(wave, window_size)`: Computes stability based on energy variance over windows.
    *   `compute_interference_strength(wave1, wave2)`: Computes interference based on correlation and simplified phase difference.
    *   `compute_phase_coherence(wave1, wave2, freq_range)`: Computes a simplified phase coherence metric (placeholder, needs FFT).
    *   `partial_interference(wave1, wave2, window_size)`: Computes interference over sliding windows.

### Wave Analysis (`ember_ml.wave.utils.wave_analysis`)

Provides functions for analyzing wave signals, often relying on NumPy/SciPy or Librosa.

*   **`compute_fft(wave, sample_rate)`**: Computes Fast Fourier Transform using NumPy. Returns frequencies and magnitudes.
*   **`compute_stft(wave, sample_rate, window_size, hop_length)`**: Computes Short-Time Fourier Transform using SciPy. Returns times, frequencies, and spectrogram magnitudes.
*   **Librosa-Based Functions (if available)**:
    *   `compute_mfcc(wave, sample_rate, n_mfcc)`: Computes Mel-Frequency Cepstral Coefficients.
    *   `compute_spectral_centroid(wave, sample_rate)`: Computes the center of mass of the spectrum. (Includes NumPy fallback).
    *   `compute_spectral_bandwidth(wave, sample_rate)`: Computes the bandwidth of the spectrum. (Includes NumPy fallback).
    *   `compute_spectral_contrast(wave, sample_rate)`: Computes spectral contrast.
    *   `compute_spectral_rolloff(wave, sample_rate)`: Computes the frequency below which a specified percentage of the total spectral energy lies. (Includes NumPy fallback).
    *   `compute_zero_crossing_rate(wave)`: Computes the rate at which the signal crosses zero. (Includes NumPy fallback).
    *   `compute_harmonic_ratio(wave, sample_rate)`: Computes the ratio of harmonic to percussive components.
*   **NumPy-Based Functions**:
    *   `compute_rms(wave)`: Computes Root Mean Square energy.
    *   `compute_peak_amplitude(wave)`: Computes the maximum absolute amplitude.
    *   `compute_crest_factor(wave)`: Computes the ratio of peak amplitude to RMS energy.
    *   `compute_dominant_frequency(wave, sample_rate)`: Finds the frequency with the highest magnitude in the FFT.
*   **`compute_wave_features(wave, sample_rate)`**: Computes a dictionary of various features (RMS, peak, crest, dominant frequency, and Librosa features if available).

### Wave Conversion (`ember_ml.wave.utils.wave_conversion`)

Provides utilities for converting between different audio/wave data representations using NumPy.

*   **`pcm_to_float(pcm_data, dtype)`**: Converts integer PCM data to floating-point [-1.0, 1.0].
*   **`float_to_pcm(float_data, dtype)`**: Converts floating-point data to integer PCM.
*   **`pcm_to_db(pcm_data, ref, min_db)`**: Converts PCM data to decibels.
*   **`db_to_amplitude(db)`**: Converts decibels to amplitude.
*   **`amplitude_to_db(amplitude, min_db)`**: Converts amplitude to decibels.
*   **`pcm_to_binary(pcm_data, threshold)`**: Converts PCM data to binary (0 or 1) based on a threshold.
*   **`binary_to_pcm(binary_data, amplitude, dtype)`**: Converts binary data back to PCM.
*   **`pcm_to_phase(pcm_data)`**: Computes the phase spectrum of PCM data using FFT.
*   **`phase_to_pcm(phase_data, magnitude, dtype)`**: Reconstructs PCM data from phase (and optional magnitude) using inverse FFT.

### Wave Visualization (`ember_ml.wave.utils.wave_visualization`)

Provides functions for plotting wave data using Matplotlib (and Librosa where available).

*   **`plot_waveform(wave, sample_rate, title)`**: Plots the wave amplitude over time.
*   **`plot_spectrum(wave, sample_rate, title)`**: Plots the frequency spectrum (magnitude vs. frequency) using FFT.
*   **`plot_spectrogram(wave, sample_rate, window_size, hop_length, title)`**: Plots the spectrogram (frequency vs. time vs. magnitude) using STFT.
*   **Librosa-Based Plots (if available)**:
    *   `plot_mel_spectrogram(...)`: Plots the mel spectrogram.
    *   `plot_chromagram(...)`: Plots the chromagram (distribution of energy across pitch classes).
    *   `plot_mfcc(...)`: Plots the Mel-Frequency Cepstral Coefficients over time.
*   **`plot_wave_features(wave, sample_rate)`**: Generates a dictionary of plots for various features (waveform, spectrum, spectrogram, and Librosa plots if available).
*   **`plot_wave_comparison(waves, labels, sample_rate, title)`**: Plots multiple waveforms on separate subplots for comparison.
*   **`fig_to_image(fig)`**: Converts a Matplotlib figure to a PIL Image object.
*   **`plot_to_numpy(fig)`**: Converts a Matplotlib figure to a NumPy array representation.