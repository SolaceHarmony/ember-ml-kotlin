import pytest
import numpy as np
from regex import T # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.limb import pwm_processor # Import the module
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for wave.limb.pwm_processor components

def test_pwmprocessor_initialization():
    # Test PWMProcessor initialization
    bits_per_block = 8
    carrier_freq = 1000
    sample_rate = 8000
    processor = pwm_processor.PWMProcessor(bits_per_block, carrier_freq, sample_rate)

    assert isinstance(processor, pwm_processor.PWMProcessor)
    assert processor.bits_per_block == bits_per_block
    assert processor.carrier_freq == carrier_freq
    assert processor.sample_rate == sample_rate


def test_pwmprocessor_pcm_to_pwm():
    # Test PWMProcessor pcm_to_pwm
    bits_per_block = 8
    carrier_freq = 1000
    sample_rate = 8000
    processor = pwm_processor.PWMProcessor(bits_per_block, carrier_freq, sample_rate)

    # Create dummy PCM data (NumPy array, float32)
    # Values should be in the range [-1.0, 1.0]
    pcm_data = tensor.convert_to_tensor([0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25], dtype=tensor.float32)

    # pcm_to_pwm returns a binary NumPy array
    pwm_signal = processor.pcm_to_pwm(pcm_data)

    assert ops.all(ops.logical_or(pwm_signal == 0, pwm_signal == 1)) # Should contain only 0s and 1s
    # Check the length of the PWM signal
    # Expected length = len(pcm_data) * bits_per_block * (sample_rate / carrier_freq)
    # For this example: 7 * 8 * (8000 / 1000) = 7 * 8 * 8 = 448
    expected_length = len(pcm_data) * bits_per_block * int(sample_rate / carrier_freq)
    assert pwm_signal.shape[0] == expected_length

    # More detailed tests would involve checking the duty cycle of the generated PWM signal
    # for specific PCM input values.


def test_pwmprocessor_pwm_to_pcm():
    # Test PWMProcessor pwm_to_pcm
    bits_per_block = 8
    carrier_freq = 1000
    sample_rate = 8000
    processor = pwm_processor.PWMProcessor(bits_per_block, carrier_freq, sample_rate)

    # Create a dummy PWM signal (NumPy array, binary)
    # Simulate a signal with varying duty cycles
    pwm_signal = np.concatenate([
        tensor.zeros(bits_per_block * int(sample_rate / carrier_freq)), # 0% duty cycle
        tensor.ones(bits_per_block * int(sample_rate / carrier_freq) // 2), tensor.zeros(bits_per_block * int(sample_rate / carrier_freq) // 2), # 50% duty cycle
        tensor.ones(bits_per_block * int(sample_rate / carrier_freq)), # 100% duty cycle
    ]).astype(tensor.int32)

    # pwm_to_pcm returns a NumPy array (float32)
    reconstructed_pcm = processor.pwm_to_pcm(pwm_signal)

    # Expected length = len(pwm_signal) / (bits_per_block * (sample_rate / carrier_freq))
    expected_length = len(pwm_signal) // (bits_per_block * int(sample_rate / carrier_freq))
    assert reconstructed_pcm.shape[0] == expected_length
    assert reconstructed_pcm.dtype == tensor.float32

    # More detailed tests would involve checking the reconstructed PCM values
    # against the expected values based on the PWM duty cycles.


# Add more test functions for other pwm_processor methods:
# test_pwmprocessor_analyze_pwm_signal(), test_pwmprocessor_get_pwm_parameters()