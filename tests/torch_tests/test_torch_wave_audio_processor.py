import pytest
import numpy as np # For comparison with known correct results
import os
import wave # For creating dummy wave files
import tempfile # For creating temporary files
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Fixture to create a dummy wave file
@pytest.fixture
def dummy_wave_file():
    """Creates a dummy wave file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        file_path = fp.name
        # Create a simple sine wave audio
        sample_rate = 44100
        duration = 1.0
        frequency = 440
        nframes = int(sample_rate * duration)
        amplitude = 32760 # Max amplitude for 16-bit audio

        t = tensor.linspace(0, duration, nframes)
        wave_data = amplitude * ops.sin(2 * ops.pi * frequency * t)
        wave_data_int16 = wave_data.astype(tensor.int16)

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1) # Mono
            wf.setsampwidth(2) # 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(wave_data_int16.tobytes())

    yield file_path
    os.remove(file_path) # Clean up the dummy file

# Test cases for wave.audio.audio_processor functions

def test_audioprocessor_initialization():
    # Test AudioProcessor initialization
    sample_rate = 16000
    processor = audio_processor.AudioProcessor(sample_rate)

    assert isinstance(processor, audio_processor.AudioProcessor)
    assert processor.sampling_rate == sample_rate


def test_audioprocessor_load_audio(dummy_wave_file):
    # Test AudioProcessor load_audio
    # This test requires librosa to be installed.
    try:
        import librosa
    except ImportError:
        pytest.skip("Skipping load_audio test: librosa not installed")

    target_sample_rate = 8000
    processor = audio_processor.AudioProcessor(target_sample_rate)

    audio_data = processor.load_audio(dummy_wave_file, target_sample_rate)

    assert isinstance(audio_data, TensorLike) # load_audio returns numpy array
    assert len(audio_data.shape) == 1 # Should be mono
    # Check sample rate (should be close to target)
    # The loaded audio should have the target sample rate, so its length should be duration * target_sample_rate
    expected_length = int(1.0 * target_sample_rate) # Duration of dummy file is 1.0s
    assert abs(audio_data.shape[0] - expected_length) <= 1 # Allow for minor off-by-one
    assert audio_data.dtype == tensor.float32 # Should be float32


def test_audioprocessor_normalize_audio():
    # Test AudioProcessor normalize_audio
    audio_data = tensor.convert_to_tensor([0.5, -0.5, 1.0, -1.0, 0.0], dtype=tensor.float32)
    processor = audio_processor.AudioProcessor(1000) # Sample rate doesn't matter for this test

    normalized_data = processor.normalize_audio(audio_data)

    assert isinstance(normalized_data, TensorLike)
    assert ops.allclose(stats.max(ops.abs(normalized_data)), 1.0) # Peak amplitude should be 1.0


def test_audioprocessor_segment_audio():
    # Test AudioProcessor segment_audio
    audio_data = tensor.arange(20, dtype=tensor.float32) # [0, 1, ..., 19]
    segment_length = 5
    hop_length = 2
    processor = audio_processor.AudioProcessor(1000) # Sample rate doesn't matter

    segments = processor.segment_audio(audio_data, segment_length, hop_length)

    assert isinstance(segments, TensorLike)
    assert segments.shape[1] == segment_length # Second dimension is segment length
    # Calculate expected number of segments: (total_length - segment_length) // hop_length + 1
    expected_num_segments = (20 - 5) // 2 + 1 # 15 // 2 + 1 = 7 + 1 = 8
    assert segments.shape[0] == expected_num_segments

    # Check content of a few segments
    assert tensor.convert_to_tensor_equal(segments[0], [0, 1, 2, 3, 4])
    assert tensor.convert_to_tensor_equal(segments[1], [2, 3, 4, 5, 6])
    assert tensor.convert_to_tensor_equal(segments[expected_num_segments - 1], [14, 15, 16, 17, 18])


# Add more test functions for other audio_processor methods:
# test_audioprocessor_save_audio() - requires soundfile
# test_audioprocessor_load_audio_resampling()
# test_audioprocessor_load_audio_stereo_to_mono()