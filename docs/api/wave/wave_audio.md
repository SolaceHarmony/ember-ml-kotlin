# Wave Audio Processing

This section describes components specifically designed for processing audio data within the wave framework.

## Core Concepts

These components handle tasks like loading audio files, converting between audio formats (PCM, float), and potentially applying audio-specific transformations or feature extractions relevant to wave-based models.

## Components

### `ember_ml.wave.audio.audio_processor`

*   **`AudioProcessor`**: Handles loading, processing, and saving audio files.
    *   `__init__(sampling_rate)`: Initializes with a target sampling rate.
    *   `load_audio(file_path, target_sr)`: Loads an audio file, resamples it to the target rate, and converts it to mono float32 format. Requires `librosa`.
    *   `save_audio(file_path, audio_data, source_sr)`: Saves audio data to a WAV file. Requires `soundfile`.
    *   `normalize_audio(audio_data)`: Normalizes audio data to have a peak amplitude of 1.0.
    *   `segment_audio(audio_data, segment_length, hop_length)`: Splits audio data into overlapping segments.

*(Note: This file relies on external libraries like `librosa` and `soundfile` which are not part of the core Ember ML backend system and may introduce dependencies.)*