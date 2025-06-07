import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
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

# Test cases for ops.fft functions

def test_fft_ifft_1d():
    # Test 1D FFT and IFFT
    x = tensor.convert_to_tensor([1.0, 2.0, 1.0, 0.0]) # Simple signal
    fft_result = ops.fft(x)
    ifft_result = ops.ifft(fft_result)

    # Convert to numpy for assertion
    x_np = tensor.to_numpy(x)
    fft_result_np = tensor.to_numpy(fft_result)
    ifft_result_np = tensor.to_numpy(ifft_result)

    # Assert correctness (IFFT of FFT should be original signal)
    assert ops.allclose(ifft_result_np, x_np)

    # Compare FFT result with numpy's FFT
    expected_fft_np = np.fft.fft(x_np)
    assert ops.allclose(fft_result_np, expected_fft_np)

def test_fft2_ifft2():
    # Test 2D FFT and IFFT
    x = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) # Simple 2D signal
    fft2_result = ops.fft2(x)
    ifft2_result = ops.ifft2(fft2_result)

    # Convert to numpy for assertion
    x_np = tensor.to_numpy(x)
    fft2_result_np = tensor.to_numpy(fft2_result)
    ifft2_result_np = tensor.to_numpy(ifft2_result)

    # Assert correctness (IFFT2 of FFT2 should be original signal)
    assert ops.allclose(ifft2_result_np, x_np)

    # Compare FFT2 result with numpy's FFT2
    expected_fft2_np = np.fft.fft2(x_np)
    assert ops.allclose(fft2_result_np, expected_fft2_np)

# Add more test functions for other ops.fft functions:
# test_fftn_ifftn(), test_rfft_irfft(), test_rfft2_irfft2(), test_rfftn_irfftn()

# Example structure for test_rfft_irfft
# def test_rfft_irfft():
#     # Test 1D Real FFT and Inverse Real FFT
#     x = tensor.convert_to_tensor([1.0, 2.0, 1.0, 0.0]) # Real signal
#     rfft_result = ops.rfft(x)
#     irfft_result = ops.irfft(rfft_result)
#
#     # Convert to numpy for assertion
#     x_np = tensor.to_numpy(x)
#     rfft_result_np = tensor.to_numpy(rfft_result)
#     irfft_result_np = tensor.to_numpy(irfft_result)
#
#     # Assert correctness (IRFFT of RFFT should be original signal)
#     assert ops.allclose(irfft_result_np, x_np)
#
#     # Compare RFFT result with numpy's RFFT
#     expected_rfft_np = linearalg.rfft(x_np)
#     assert ops.allclose(rfft_result_np, expected_rfft_np)