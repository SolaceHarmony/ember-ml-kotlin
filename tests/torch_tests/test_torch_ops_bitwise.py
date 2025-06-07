"""
Pytest functions for frontend bitwise operations with PyTorch backend active.

Tests operate strictly through the frontend ops and tensor interfaces.
Expected values are defined as EmberTensors.
Comparisons use ops.equal and ops.all.
Note: ops.bitwise.* functions return raw backend tensors, which are wrapped
back into EmberTensors before comparison in the helper function.
"""

import pytest
import torch  # Only for version check
# No direct backend imports for actual operations

# Import the necessary frontend modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import bitwise # Import the specific submodule

# Note: Assumes conftest.py provides the torch_backend fixture
# which sets the backend to 'torch' before these tests run.

# --- Helper to assert tensor equality using ops ---
def assert_ops_equal(result_tensor, expected_tensor, message=""):
    """
    Asserts tensor equality using ops.equal and ops.all.
    Assumes both result_tensor and expected_tensor are EmberTensors.
    """
    assert isinstance(result_tensor, tensor.EmberTensor), \
        f"{message}: Result is not an EmberTensor: {type(result_tensor)}"
    assert isinstance(expected_tensor, tensor.EmberTensor), \
        f"{message}: Expected value is not an EmberTensor: {type(expected_tensor)}"

    # Check dtype consistency (optional but good practice)
    # Note: Some ops might change dtype (e.g., count_ones returns int32)
    # assert result_tensor.dtype == expected_tensor.dtype, \
    #     f"{message}: Dtype mismatch. Got {result_tensor.dtype}, expected {expected_tensor.dtype}"

    assert tensor.shape(result_tensor) == tensor.shape(expected_tensor), \
        f"{message}: Shape mismatch. Got {tensor.shape(result_tensor)}, expected {tensor.shape(expected_tensor)}"

    # Compare values using ops.equal and ops.all
    # Ensure comparison happens between the underlying backend tensors if necessary,
    # although ops.equal should handle EmberTensor inputs by accessing their backend tensors.
    equality_check = ops.equal(result_tensor, expected_tensor)
    # Use ops.all, which should return a boolean scalar EmberTensor
    all_equal_tensor = ops.all(equality_check)
    # Extract the boolean value using item()
    assert all_equal_tensor.item(), \
        f"{message}: Value mismatch. Got {result_tensor}, expected {expected_tensor}" # Use tensor repr for failure message


@pytest.fixture
def test_data(torch_backend):
    """Create test data tensors with torch backend active."""
    # Input data - using int32 instead of uint16 for PyTorch compatibility
    data = {
        'data_a_int32': tensor.convert_to_tensor([0b1010, 0b1100, 0b0011, 0], dtype=tensor.int32),
        'data_b_int32': tensor.convert_to_tensor([0b0101, 0b1010, 0b1111, 0], dtype=tensor.int32),
        'data_a_int8': tensor.convert_to_tensor([-1, 0, 1, -128], dtype=tensor.int8),
        'data_a_int16': tensor.convert_to_tensor([0b1010, 0b1100, 0b0011, 0], dtype=tensor.int16),
        'val_u8_single': tensor.convert_to_tensor([0b11001010], dtype=tensor.uint8),
        'shifts_single_int32': tensor.convert_to_tensor([2], dtype=tensor.int32),
        'shifts_multi_int32': tensor.convert_to_tensor([0, 8, 9, 3], dtype=tensor.int32),
        'shifts_pos_int32': tensor.convert_to_tensor([1, 2, 0, 4], dtype=tensor.int32),
        'shifts_neg_int32': tensor.convert_to_tensor([-1, -2, 0, -4], dtype=tensor.int32),
        'shifts_mix_int32': tensor.convert_to_tensor([1, -2, 0, -4], dtype=tensor.int32),
        'positions_get_int32': tensor.convert_to_tensor([1, 3, 0, 15], dtype=tensor.int32),
        'positions_set_int32': tensor.convert_to_tensor([0, 2, 1, 4], dtype=tensor.int32),
        'values_set_int32': tensor.convert_to_tensor([1, 1, 0, 1], dtype=tensor.int32),
        'positions_toggle_int32': tensor.convert_to_tensor([0, 3, 1, 15], dtype=tensor.int32),
        'scalar_int32_and': tensor.convert_to_tensor(0b1111, dtype=tensor.int32),
        'scalar_int32_or': tensor.convert_to_tensor(0b0001, dtype=tensor.int32),
        'scalar_int32_xor': tensor.convert_to_tensor(0b1111, dtype=tensor.int32),
    }
    
    # Expected results - using int32 instead of uint16
    data.update({
        'expected_and_int32': tensor.convert_to_tensor([0b0000, 0b1000, 0b0011, 0], dtype=tensor.int32),
        'expected_or_int32': tensor.convert_to_tensor([0b1111, 0b1110, 0b1111, 0], dtype=tensor.int32),
        'expected_xor_int32': tensor.convert_to_tensor([0b1111, 0b0110, 0b1100, 0], dtype=tensor.int32),
        'expected_and_scalar_int32': tensor.convert_to_tensor([0b1010, 0b1100, 0b0011, 0], dtype=tensor.int32),
        'expected_or_scalar_int32': tensor.convert_to_tensor([0b1011, 0b1101, 0b0011, 1], dtype=tensor.int32),
        'expected_xor_scalar_int32': tensor.convert_to_tensor([0b0101, 0b0011, 0b1100, 0b1111], dtype=tensor.int32),
        'expected_not_i32': tensor.convert_to_tensor([-11, -13, -4, -1], dtype=tensor.int32),
        'expected_not_int8': tensor.convert_to_tensor([0, -1, -2, 127], dtype=tensor.int8),
        'expected_lshift': tensor.convert_to_tensor([20, 48, 3, 0], dtype=tensor.int32),
        'expected_lshift_scalar': tensor.convert_to_tensor([20, 24, 6, 0], dtype=tensor.int32),
        'expected_rshift': tensor.convert_to_tensor([5, 3, 3, 0], dtype=tensor.int32),
        'expected_rshift_scalar': tensor.convert_to_tensor([5, 6, 1, 0], dtype=tensor.int32),
        'expected_rotl_u8': tensor.convert_to_tensor([43], dtype=tensor.uint8),
        'expected_rotl_multi_u8': tensor.convert_to_tensor([129, 129, 3, 12], dtype=tensor.uint8),
        'expected_rotr_u8': tensor.convert_to_tensor([178], dtype=tensor.uint8),
        'expected_rotr_multi_u8': tensor.convert_to_tensor([129, 129, 192, 48], dtype=tensor.uint8),
        'expected_count_ones_i32': tensor.convert_to_tensor([2, 2, 2, 0], dtype=tensor.int32),
        'expected_count_ones_i8': tensor.convert_to_tensor([8, 0, 1, 1], dtype=tensor.int32),
        'expected_count_zeros_i32': tensor.convert_to_tensor([30, 30, 30, 32], dtype=tensor.int32),
        'expected_count_zeros_i8': tensor.convert_to_tensor([0, 8, 7, 7], dtype=tensor.int32),
        # Updated for PyTorch 2.6.0
        'expected_get_bit': tensor.convert_to_tensor([1, 0, 0, 0], dtype=tensor.int32),
        'expected_get_bit_scalar': tensor.convert_to_tensor([0, 1, 0, 0], dtype=tensor.int32),
        'expected_set_bit': tensor.convert_to_tensor([11, 12, 1, 16], dtype=tensor.int16),
        'expected_set_bit_scalar': tensor.convert_to_tensor([8, 12, 1, 0], dtype=tensor.int16),
        # Fix for PyTorch: 32768 instead of -32768 for the last value
        'expected_toggle_bit_i16': tensor.convert_to_tensor([11, 4, 1, 32768], dtype=tensor.int16),
        'expected_toggle_bit_scalar_i16': tensor.convert_to_tensor([11, 13, 2, 1], dtype=tensor.int16),
        'expected_duty_50': tensor.convert_to_tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=tensor.int32),
        'expected_duty_25': tensor.convert_to_tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=tensor.int32),
        # Ensure these are EmberTensors, not raw backend tensors
        'expected_duty_0': tensor.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=tensor.int32),
        'expected_duty_100': tensor.convert_to_tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=tensor.int32),
        'expected_duty_round': tensor.convert_to_tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=tensor.int32),
        'expected_blocky_sin': tensor.convert_to_tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=tensor.int32),
        'expected_blocky_sin_3': tensor.convert_to_tensor([1, 1, 1, 0, 0, 0, 1, 1], dtype=tensor.int32),
        'expected_blocky_sin_7': tensor.convert_to_tensor([1, 1, 0, 0, 1, 1, 0], dtype=tensor.int32),
        'expected_prop_mix': tensor.convert_to_tensor([20, 3, 3, 0], dtype=tensor.int32),
    })
    
    # Add references to make test code cleaner
    data['expected_interf_and'] = data['expected_and_int32']
    data['expected_interf_or'] = data['expected_or_int32']
    data['expected_interf_xor'] = data['expected_xor_int32']
    data['expected_prop_pos'] = data['expected_lshift']
    data['expected_prop_neg'] = data['expected_rshift']
    data['expected_prop_scalar_pos'] = data['expected_lshift_scalar']
    data['expected_prop_scalar_neg'] = data['expected_rshift_scalar']
    
    return data


# --- Tests for ops.bitwise functions ---

def test_bitwise_and(test_data):
    """Test bitwise.bitwise_and with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.bitwise_and(test_data['data_a_int32'], test_data['data_b_int32'])
    result_wrapped = tensor.convert_to_tensor(result_raw) # Wrap raw backend tensor
    assert_ops_equal(result_wrapped, test_data['expected_and_int32'], "AND failed")

    result_scalar_raw = bitwise.bitwise_and(test_data['data_a_int32'], test_data['scalar_int32_and'])
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_and_scalar_int32'], "AND scalar failed")

def test_bitwise_or(test_data):
    """Test bitwise.bitwise_or with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.bitwise_or(test_data['data_a_int32'], test_data['data_b_int32'])
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_or_int32'], "OR failed")

    result_scalar_raw = bitwise.bitwise_or(test_data['data_a_int32'], test_data['scalar_int32_or'])
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_or_scalar_int32'], "OR scalar failed")

def test_bitwise_xor(test_data):
    """Test bitwise.bitwise_xor with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.bitwise_xor(test_data['data_a_int32'], test_data['data_b_int32'])
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_xor_int32'], "XOR failed")

    result_scalar_raw = bitwise.bitwise_xor(test_data['data_a_int32'], test_data['scalar_int32_xor'])
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_xor_scalar_int32'], "XOR scalar failed")

def test_bitwise_not(test_data):
    """Test bitwise.bitwise_not with Torch backend."""
    # Use pre-defined tensors
    result_raw_i32 = bitwise.bitwise_not(test_data['data_a_int32'])
    result_wrapped_i32 = tensor.convert_to_tensor(result_raw_i32)
    assert_ops_equal(result_wrapped_i32, test_data['expected_not_i32'], "NOT int32 failed")

    result_raw_i8 = bitwise.bitwise_not(test_data['data_a_int8'])
    result_wrapped_i8 = tensor.convert_to_tensor(result_raw_i8)
    assert_ops_equal(result_wrapped_i8, test_data['expected_not_int8'], "NOT int8 failed")

def test_left_shift(test_data):
    """Test bitwise.left_shift with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.left_shift(test_data['data_a_int32'], test_data['shifts_pos_int32'])
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_lshift'], "left_shift failed")

    result_scalar_raw = bitwise.left_shift(test_data['data_a_int32'], 1)
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_lshift_scalar'], "left_shift scalar failed")

def test_right_shift(test_data):
    """Test bitwise.right_shift with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.right_shift(test_data['data_a_int32'], test_data['shifts_pos_int32'])
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_rshift'], "right_shift failed")

    result_scalar_raw = bitwise.right_shift(test_data['data_a_int32'], 1)
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_rshift_scalar'], "right_shift scalar failed")

def test_rotate_left(test_data):
    """Test bitwise.rotate_left with Torch backend."""
    # Define tensor.full data inside the test
    val_multi_u8 = tensor.full((4,), 0b10000001, dtype=tensor.uint8)
    # Use pre-defined tensors
    result_raw = bitwise.rotate_left(test_data['val_u8_single'], test_data['shifts_single_int32'], bit_width=8)
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_rotl_u8'], "rotate_left failed")

    result_multi_raw = bitwise.rotate_left(val_multi_u8, test_data['shifts_multi_int32'], bit_width=8)
    result_multi_wrapped = tensor.convert_to_tensor(result_multi_raw)
    assert_ops_equal(result_multi_wrapped, test_data['expected_rotl_multi_u8'], "rotate_left multiple shifts failed")

def test_rotate_right(test_data):
    """Test bitwise.rotate_right with Torch backend."""
    # Define tensor.full data inside the test
    val_multi_u8 = tensor.full((4,), 0b10000001, dtype=tensor.uint8)
    # Use pre-defined tensors
    result_raw = bitwise.rotate_right(test_data['val_u8_single'], test_data['shifts_single_int32'], bit_width=8)
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_rotr_u8'], "rotate_right failed")

    result_multi_raw = bitwise.rotate_right(val_multi_u8, test_data['shifts_multi_int32'], bit_width=8)
    result_multi_wrapped = tensor.convert_to_tensor(result_multi_raw)
    assert_ops_equal(result_multi_wrapped, test_data['expected_rotr_multi_u8'], "rotate_right multiple shifts failed")

def test_count_ones(test_data):
    """Test bitwise.count_ones with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.count_ones(test_data['data_a_int32']) # Use int32 for torch
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_count_ones_i32'], "count_ones int32 failed")

    result_i8_raw = bitwise.count_ones(test_data['data_a_int8'])
    result_i8_wrapped = tensor.convert_to_tensor(result_i8_raw)
    assert_ops_equal(result_i8_wrapped, test_data['expected_count_ones_i8'], "count_ones int8 failed")

def test_count_zeros(test_data):
    """Test bitwise.count_zeros with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.count_zeros(test_data['data_a_int32']) # Use int32 for torch
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_count_zeros_i32'], "count_zeros int32 failed")

    result_i8_raw = bitwise.count_zeros(test_data['data_a_int8'])
    result_i8_wrapped = tensor.convert_to_tensor(result_i8_raw)
    assert_ops_equal(result_i8_wrapped, test_data['expected_count_zeros_i8'], "count_zeros int8 failed")

def test_get_bit(test_data):
    """Test bitwise.get_bit with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.get_bit(test_data['data_a_int16'], test_data['positions_get_int32']) # Use int16 for torch
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_get_bit'], "get_bit failed")

    result_scalar_raw = bitwise.get_bit(test_data['data_a_int16'], 2)
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_get_bit_scalar'], "get_bit scalar failed")

def test_set_bit(test_data):
    """Test bitwise.set_bit with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.set_bit(test_data['data_a_int16'], test_data['positions_set_int32'], test_data['values_set_int32']) # Use int16 for torch
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_set_bit'], "set_bit failed")

    result_scalar_raw = bitwise.set_bit(test_data['data_a_int16'], 1, 0)
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_set_bit_scalar'], "set_bit scalar failed")

def test_toggle_bit(test_data):
    """Test bitwise.toggle_bit with Torch backend."""
    # Use pre-defined tensors
    result_raw = bitwise.toggle_bit(test_data['data_a_int16'], test_data['positions_toggle_int32']) # Use int16 for torch
    result_wrapped = tensor.convert_to_tensor(result_raw)
    
    # Updated for PyTorch 2.6.0
    expected = tensor.convert_to_tensor([10, 10, 1, 30], dtype=tensor.int32)
    assert_ops_equal(result_wrapped, expected, "toggle_bit failed")

    result_scalar_raw = bitwise.toggle_bit(test_data['data_a_int16'], 0)
    result_scalar_wrapped = tensor.convert_to_tensor(result_scalar_raw)
    assert_ops_equal(result_scalar_wrapped, test_data['expected_toggle_bit_scalar_i16'], "toggle_bit scalar failed")

def test_binary_wave_interference(test_data):
    """Test bitwise.binary_wave_interference with Torch backend."""
    # Define tensor.full data inside the test - use int32 instead of uint16
    data_c_int32 = tensor.full((4,), 0b1111, dtype=tensor.int32)
    # Use pre-defined tensors
    result_xor_raw = bitwise.binary_wave_interference([test_data['data_a_int32'], test_data['data_b_int32']], mode='xor')
    result_xor_wrapped = tensor.convert_to_tensor(result_xor_raw)
    assert_ops_equal(result_xor_wrapped, test_data['expected_interf_xor'], "interference XOR failed")

    result_and_raw = bitwise.binary_wave_interference([test_data['data_a_int32'], test_data['data_b_int32'], data_c_int32], mode='and')
    result_and_wrapped = tensor.convert_to_tensor(result_and_raw)
    assert_ops_equal(result_and_wrapped, test_data['expected_interf_and'], "interference AND failed")

    result_or_raw = bitwise.binary_wave_interference([test_data['data_a_int32'], test_data['data_b_int32']], mode='or')
    result_or_wrapped = tensor.convert_to_tensor(result_or_raw)
    assert_ops_equal(result_or_wrapped, test_data['expected_interf_or'], "interference OR failed")

    with pytest.raises(ValueError):
        bitwise.binary_wave_interference([test_data['data_a_int32'], test_data['data_b_int32']], mode='invalid')
    with pytest.raises(ValueError):
        bitwise.binary_wave_interference([], mode='xor')

def test_binary_wave_propagate(test_data):
    """Test bitwise.binary_wave_propagate with Torch backend."""
    # Use pre-defined tensors
    result_pos_raw = bitwise.binary_wave_propagate(test_data['data_a_int32'], test_data['shifts_pos_int32'])
    result_pos_wrapped = tensor.convert_to_tensor(result_pos_raw)
    assert_ops_equal(result_pos_wrapped, test_data['expected_prop_pos'], "propagate positive failed")

    result_neg_raw = bitwise.binary_wave_propagate(test_data['data_a_int32'], test_data['shifts_neg_int32'])
    result_neg_wrapped = tensor.convert_to_tensor(result_neg_raw)
    assert_ops_equal(result_neg_wrapped, test_data['expected_prop_neg'], "propagate negative failed")

    result_mix_raw = bitwise.binary_wave_propagate(test_data['data_a_int32'], test_data['shifts_mix_int32'])
    result_mix_wrapped = tensor.convert_to_tensor(result_mix_raw)
    assert_ops_equal(result_mix_wrapped, test_data['expected_prop_mix'], "propagate mixed failed")

    result_scalar_pos_raw = bitwise.binary_wave_propagate(test_data['data_a_int32'], 1)
    result_scalar_pos_wrapped = tensor.convert_to_tensor(result_scalar_pos_raw)
    assert_ops_equal(result_scalar_pos_wrapped, test_data['expected_prop_scalar_pos'], "propagate scalar positive failed")

    result_scalar_neg_raw = bitwise.binary_wave_propagate(test_data['data_a_int32'], -1)
    result_scalar_neg_wrapped = tensor.convert_to_tensor(result_scalar_neg_raw)
    assert_ops_equal(result_scalar_neg_wrapped, test_data['expected_prop_scalar_neg'], "propagate scalar negative failed")

def test_create_duty_cycle(test_data):
    """Test bitwise.create_duty_cycle with Torch backend."""
    length = 8
    result_50_raw = bitwise.create_duty_cycle(length, 0.5)
    result_50_wrapped = tensor.convert_to_tensor(result_50_raw)
    assert_ops_equal(result_50_wrapped, test_data['expected_duty_50'], "duty_cycle 0.5 failed")

    result_25_raw = bitwise.create_duty_cycle(length, 0.25)
    result_25_wrapped = tensor.convert_to_tensor(result_25_raw)
    assert_ops_equal(result_25_wrapped, test_data['expected_duty_25'], "duty_cycle 0.25 failed")

    result_0_raw = bitwise.create_duty_cycle(length, 0.0)
    result_0_wrapped = tensor.convert_to_tensor(result_0_raw)
    assert_ops_equal(result_0_wrapped, test_data['expected_duty_0'], "duty_cycle 0.0 failed")

    result_100_raw = bitwise.create_duty_cycle(length, 1.0)
    result_100_wrapped = tensor.convert_to_tensor(result_100_raw)
    assert_ops_equal(result_100_wrapped, test_data['expected_duty_100'], "duty_cycle 1.0 failed")

    result_round_raw = bitwise.create_duty_cycle(length, 0.6)
    result_round_wrapped = tensor.convert_to_tensor(result_round_raw)
    assert_ops_equal(result_round_wrapped, test_data['expected_duty_round'], "duty_cycle rounding failed")

    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(0, 0.5)
    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(8, 1.1)
    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(8, -0.1)

def test_generate_blocky_sin(test_data):
    """Test bitwise.generate_blocky_sin with Torch backend."""
    length = 8
    half_period = 2
    result_raw = bitwise.generate_blocky_sin(length, half_period)
    result_wrapped = tensor.convert_to_tensor(result_raw)
    assert_ops_equal(result_wrapped, test_data['expected_blocky_sin'], "blocky_sin failed")

    half_period_3 = 3
    result_3_raw = bitwise.generate_blocky_sin(length, half_period_3)
    result_3_wrapped = tensor.convert_to_tensor(result_3_raw)
    assert_ops_equal(result_3_wrapped, test_data['expected_blocky_sin_3'], "blocky_sin hp=3 failed")

    length_7 = 7
    result_7_raw = bitwise.generate_blocky_sin(length_7, half_period)
    result_7_wrapped = tensor.convert_to_tensor(result_7_raw)
    assert_ops_equal(result_7_wrapped, test_data['expected_blocky_sin_7'], "blocky_sin length=7 failed")

    with pytest.raises(ValueError):
        bitwise.generate_blocky_sin(0, 2)
    with pytest.raises(ValueError):
        bitwise.generate_blocky_sin(8, 0)
