"""
Pytest functions for frontend bitwise operations with MLX backend active.

Tests operate strictly through the frontend ops and tensor interfaces.
"""

import pytest
from typing import List, Tuple, Any

# Import the necessary frontend modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import bitwise # Import the specific submodule

# Note: Assumes conftest.py provides the mlx_backend fixture
# which sets the backend to 'mlx' before these tests run.

# --- Test Data (Python lists/values) ---
data_a_uint16 = [0b1010, 0b1100, 0b0011, 0]
data_b_uint16 = [0b0101, 0b1010, 0b1111, 0]
data_a_int8 = [-1, 0, 1, -128]
data_c_uint16 = [0b1111]*4 # For interference test

# Expected results (Python lists/values)
expected_and_uint16 = [0b0000, 0b1000, 0b0011, 0]
expected_or_uint16 = [0b1111, 0b1110, 0b1111, 0]
expected_xor_uint16 = [0b1111, 0b0110, 0b1100, 0]
expected_not_uint16 = [65525, 65523, 65532, 65535] # ~10, ~12, ~3, ~0 for uint16
expected_not_int8 = [0, -1, -2, 127] # ~(-1), ~(0), ~(1), ~(-128) for int8
expected_lshift = [20, 48, 3, 0] # 10<<1, 12<<2, 3<<0, 0<<4
expected_lshift_scalar = [20, 24, 6, 0] # << 1
expected_rshift = [5, 3, 3, 0] # 10>>1, 12>>2, 3>>0, 0>>4
expected_rshift_scalar = [5, 6, 1, 0] # >> 1
expected_rotl_u8 = [43] # 0b11001010 rotl 2 = 0b00101011
expected_rotl_multi_u8 = [129, 129, 3, 12] # 0b10000001 rotl 0, 8, 9, 3
expected_rotr_u8 = [178] # 0b11001010 rotr 2 = 0b10110010
expected_rotr_multi_u8 = [129, 129, 192, 48] # 0b10000001 rotr 0, 8, 9(1), 3 -> Corrected expected value for shift 9(1)
expected_count_ones_u16 = [2, 2, 2, 0]
expected_count_ones_i8 = [8, 0, 1, 1]
expected_count_zeros_u16 = [14, 14, 14, 16]
expected_count_zeros_i8 = [0, 8, 7, 7]
expected_get_bit = [1, 1, 1, 0] # bit 1 of 10, bit 3 of 12, bit 0 of 3, bit 15 of 0
expected_get_bit_scalar = [0, 1, 0, 0] # bit 2
expected_set_bit = [11, 12, 1, 16] # set bits [0, 2, 1, 4] to [1, 1, 0, 1]
expected_set_bit_scalar = [8, 12, 1, 0] # set bit 1 to 0
expected_toggle_bit = [11, 4, 1, 32768] # toggle bits [0, 3, 1, 15]
expected_toggle_bit_scalar = [11, 13, 2, 1] # toggle bit 0
expected_interf_and = expected_and_uint16 # a&b&c where c is all 1s
expected_interf_or = expected_or_uint16
expected_interf_xor = expected_xor_uint16
expected_prop_pos = expected_lshift
expected_prop_neg = expected_rshift
expected_prop_mix = [20, 3, 3, 0] # 10<<1, 12>>2, 3<<0, 0>>4
expected_prop_scalar_pos = expected_lshift_scalar
expected_prop_scalar_neg = expected_rshift_scalar
expected_duty_50 = [1, 1, 1, 1, 0, 0, 0, 0]
expected_duty_25 = [1, 1, 0, 0, 0, 0, 0, 0]
expected_duty_0 = [0] * 8
expected_duty_100 = [1] * 8
expected_duty_round = [1, 1, 1, 1, 1, 0, 0, 0] # 8 * 0.6 = 4.8 -> 5 ones
expected_blocky_sin = [1, 1, 0, 0, 1, 1, 0, 0] # len=8, hp=2
expected_blocky_sin_3 = [1, 1, 1, 0, 0, 0, 1, 1] # len=8, hp=3
expected_blocky_sin_7 = [1, 1, 0, 0, 1, 1, 0] # len=7, hp=2


# --- Tests for ops.bitwise functions ---

def test_bitwise_and(mlx_backend):
    """Test bitwise.bitwise_and with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    b = tensor.convert_to_tensor(data_b_uint16, dtype='uint16')
    expected = tensor.convert_to_tensor(expected_and_uint16, dtype='uint16')
    result = bitwise.bitwise_and(a, b) # Use bitwise submodule
    assert tensor.shape(result) == tensor.shape(expected)
    assert tensor.dtype(result) == tensor.dtype(expected)
    assert ops.all(ops.equal(result, expected)), f"AND failed"

    scalar = tensor.convert_to_tensor(0b1111, dtype='uint16')
    result_scalar = bitwise.bitwise_and(a, scalar)
    expected_scalar = tensor.convert_to_tensor([0b1010, 0b1100, 0b0011, 0], dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"AND scalar failed"

def test_bitwise_or(mlx_backend):
    """Test bitwise.bitwise_or with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    b = tensor.convert_to_tensor(data_b_uint16, dtype='uint16')
    expected = tensor.convert_to_tensor(expected_or_uint16, dtype='uint16')
    result = bitwise.bitwise_or(a, b)
    assert tensor.shape(result) == tensor.shape(expected)
    assert tensor.dtype(result) == tensor.dtype(expected)
    assert ops.all(ops.equal(result, expected)), f"OR failed"

    scalar = tensor.convert_to_tensor(0b0001, dtype='uint16')
    result_scalar = bitwise.bitwise_or(a, scalar)
    expected_scalar = tensor.convert_to_tensor([0b1011, 0b1101, 0b0011, 1], dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"OR scalar failed"

def test_bitwise_xor(mlx_backend):
    """Test bitwise.bitwise_xor with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    b = tensor.convert_to_tensor(data_b_uint16, dtype='uint16')
    expected = tensor.convert_to_tensor(expected_xor_uint16, dtype='uint16')
    result = bitwise.bitwise_xor(a, b)
    assert tensor.shape(result) == tensor.shape(expected)
    assert tensor.dtype(result) == tensor.dtype(expected)
    assert ops.all(ops.equal(result, expected)), f"XOR failed"

    scalar = tensor.convert_to_tensor(0b1111, dtype='uint16')
    result_scalar = bitwise.bitwise_xor(a, scalar)
    expected_scalar = tensor.convert_to_tensor([0b0101, 0b0011, 0b1100, 0b1111], dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"XOR scalar failed"

def test_bitwise_not(mlx_backend):
    """Test bitwise.bitwise_not with MLX backend."""
    a_u16 = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    result_uint = bitwise.bitwise_not(a_u16)
    expected_uint = tensor.convert_to_tensor(expected_not_uint16, dtype='uint16')
    assert tensor.shape(result_uint) == tensor.shape(expected_uint) # Use tensor.shape
    assert tensor.dtype(result_uint) == tensor.dtype(expected_uint) # Use tensor.dtype
    assert ops.all(ops.equal(result_uint, expected_uint)), f"NOT uint failed"

    a_i8 = tensor.convert_to_tensor(data_a_int8, dtype='int8')
    result_sint = bitwise.bitwise_not(a_i8)
    expected_sint = tensor.convert_to_tensor(expected_not_int8, dtype='int8')
    assert tensor.shape(result_sint) == tensor.shape(expected_sint)
    assert tensor.dtype(result_sint) == tensor.dtype(expected_sint) # Use tensor.dtype
    assert ops.all(ops.equal(result_sint, expected_sint)), f"NOT sint failed"

def test_left_shift(mlx_backend):
    """Test bitwise.left_shift with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    shifts = tensor.convert_to_tensor([1, 2, 0, 4], dtype='int32')
    result = bitwise.left_shift(a, shifts)
    expected = tensor.convert_to_tensor(expected_lshift, dtype='uint16')
    assert ops.all(ops.equal(result, expected)), f"left_shift failed"

    result_scalar = bitwise.left_shift(a, 1)
    expected_scalar = tensor.convert_to_tensor(expected_lshift_scalar, dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"left_shift scalar failed"

def test_right_shift(mlx_backend):
    """Test bitwise.right_shift with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    shifts = tensor.convert_to_tensor([1, 2, 0, 4], dtype='int32')
    result = bitwise.right_shift(a, shifts)
    expected = tensor.convert_to_tensor(expected_rshift, dtype='uint16')
    assert ops.all(ops.equal(result, expected)), f"right_shift failed"

    result_scalar = bitwise.right_shift(a, 1)
    expected_scalar = tensor.convert_to_tensor(expected_rshift_scalar, dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"right_shift scalar failed"

def test_rotate_left(mlx_backend):
    """Test bitwise.rotate_left with MLX backend."""
    val_u8 = tensor.convert_to_tensor([0b11001010], dtype='uint8')
    shifts = tensor.convert_to_tensor([2], dtype='int32')
    result = bitwise.rotate_left(val_u8, shifts, bit_width=8)
    expected = tensor.convert_to_tensor(expected_rotl_u8, dtype='uint8')
    assert ops.all(ops.equal(result, expected)), f"rotate_left failed"

    shifts_multi = tensor.convert_to_tensor([0, 8, 9, 3], dtype='int32')
    val_multi_u8 = tensor.convert_to_tensor([0b10000001] * 4, dtype='uint8')
    result_multi = bitwise.rotate_left(val_multi_u8, shifts_multi, bit_width=8)
    expected_multi = tensor.convert_to_tensor(expected_rotl_multi_u8, dtype='uint8')
    assert ops.all(ops.equal(result_multi, expected_multi)), f"rotate_left multiple shifts failed"

def test_rotate_right(mlx_backend):
    """Test bitwise.rotate_right with MLX backend."""
    val_u8 = tensor.convert_to_tensor([0b11001010], dtype='uint8')
    shifts = tensor.convert_to_tensor([2], dtype='int32')
    result = bitwise.rotate_right(val_u8, shifts, bit_width=8)
    expected = tensor.convert_to_tensor(expected_rotr_u8, dtype='uint8')
    assert ops.all(ops.equal(result, expected)), f"rotate_right failed"

    shifts_multi = tensor.convert_to_tensor([0, 8, 9, 3], dtype='int32')
    val_multi_u8 = tensor.convert_to_tensor([0b10000001] * 4, dtype='uint8')
    result_multi = bitwise.rotate_right(val_multi_u8, shifts_multi, bit_width=8)
    expected_multi = tensor.convert_to_tensor(expected_rotr_multi_u8, dtype='uint8')
    assert ops.all(ops.equal(result_multi, expected_multi)), f"rotate_right multiple shifts failed"

def test_count_ones(mlx_backend):
    """Test bitwise.count_ones with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    result = bitwise.count_ones(a)
    expected = tensor.convert_to_tensor(expected_count_ones_u16, dtype='int32')
    assert ops.all(ops.equal(result, expected)), f"count_ones u16 failed"

    a_i8 = tensor.convert_to_tensor(data_a_int8, dtype='int8')
    result_i8 = bitwise.count_ones(a_i8)
    expected_i8 = tensor.convert_to_tensor(expected_count_ones_i8, dtype='int32')
    assert ops.all(ops.equal(result_i8, expected_i8)), f"count_ones int8 failed"

def test_count_zeros(mlx_backend):
    """Test bitwise.count_zeros with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    result = bitwise.count_zeros(a)
    expected = tensor.convert_to_tensor(expected_count_zeros_u16, dtype='int32')
    assert ops.all(ops.equal(result, expected)), f"count_zeros u16 failed"

    a_i8 = tensor.convert_to_tensor(data_a_int8, dtype='int8')
    result_i8 = bitwise.count_zeros(a_i8)
    expected_i8 = tensor.convert_to_tensor(expected_count_zeros_i8, dtype='int32')
    assert ops.all(ops.equal(result_i8, expected_i8)), f"count_zeros int8 failed"

def test_get_bit(mlx_backend):
    """Test bitwise.get_bit with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    positions = tensor.convert_to_tensor([1, 3, 0, 15], dtype='int32')
    result = bitwise.get_bit(a, positions)
    expected = tensor.convert_to_tensor(expected_get_bit, dtype='int32')
    assert ops.all(ops.equal(result, expected)), f"get_bit failed"

    result_scalar = bitwise.get_bit(a, 2)
    expected_scalar = tensor.convert_to_tensor(expected_get_bit_scalar, dtype='int32')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"get_bit scalar failed"

def test_set_bit(mlx_backend):
    """Test bitwise.set_bit with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    positions = tensor.convert_to_tensor([0, 2, 1, 4], dtype='int32')
    values = tensor.convert_to_tensor([1, 1, 0, 1], dtype='int32')
    result = bitwise.set_bit(a, positions, values)
    expected = tensor.convert_to_tensor(expected_set_bit, dtype='uint16')
    assert ops.all(ops.equal(result, expected)), f"set_bit failed"

    result_scalar = bitwise.set_bit(a, 1, 0)
    expected_scalar = tensor.convert_to_tensor(expected_set_bit_scalar, dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"set_bit scalar failed"

def test_toggle_bit(mlx_backend):
    """Test bitwise.toggle_bit with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    positions = tensor.convert_to_tensor([0, 3, 1, 15], dtype='int32')
    result = bitwise.toggle_bit(a, positions)
    expected = tensor.convert_to_tensor(expected_toggle_bit, dtype='uint16')
    assert ops.all(ops.equal(result, expected)), f"toggle_bit failed"

    result_scalar = bitwise.toggle_bit(a, 0)
    expected_scalar = tensor.convert_to_tensor(expected_toggle_bit_scalar, dtype='uint16')
    assert ops.all(ops.equal(result_scalar, expected_scalar)), f"toggle_bit scalar failed"

def test_binary_wave_interference(mlx_backend):
    """Test bitwise.binary_wave_interference with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    b = tensor.convert_to_tensor(data_b_uint16, dtype='uint16')
    c = tensor.convert_to_tensor(data_c_uint16, dtype='uint16')

    result_xor = bitwise.binary_wave_interference([a, b], mode='xor')
    expected_xor = tensor.convert_to_tensor(expected_interf_xor, dtype='uint16')
    assert ops.all(ops.equal(result_xor, expected_xor)), f"interference XOR failed"

    result_and = bitwise.binary_wave_interference([a, b, c], mode='and')
    expected_and = tensor.convert_to_tensor(expected_interf_and, dtype='uint16')
    assert ops.all(ops.equal(result_and, expected_and)), f"interference AND failed"

    result_or = bitwise.binary_wave_interference([a, b], mode='or')
    expected_or = tensor.convert_to_tensor(expected_interf_or, dtype='uint16')
    assert ops.all(ops.equal(result_or, expected_or)), f"interference OR failed"

    with pytest.raises(ValueError):
        bitwise.binary_wave_interference([a, b], mode='invalid')
    with pytest.raises(ValueError):
        bitwise.binary_wave_interference([], mode='xor')

def test_binary_wave_propagate(mlx_backend):
    """Test bitwise.binary_wave_propagate with MLX backend."""
    a = tensor.convert_to_tensor(data_a_uint16, dtype='uint16')
    shifts_pos = tensor.convert_to_tensor([1, 2, 0, 4], dtype='int32')
    shifts_neg = tensor.convert_to_tensor([-1, -2, 0, -4], dtype='int32')
    shifts_mix = tensor.convert_to_tensor([1, -2, 0, -4], dtype='int32')

    result_pos = bitwise.binary_wave_propagate(a, shifts_pos)
    expected_pos = tensor.convert_to_tensor(expected_prop_pos, dtype='uint16')
    assert ops.all(ops.equal(result_pos, expected_pos)), f"propagate positive failed"

    result_neg = bitwise.binary_wave_propagate(a, shifts_neg)
    expected_neg = tensor.convert_to_tensor(expected_prop_neg, dtype='uint16')
    assert ops.all(ops.equal(result_neg, expected_neg)), f"propagate negative failed"

    result_mix = bitwise.binary_wave_propagate(a, shifts_mix)
    expected_mix = tensor.convert_to_tensor(expected_prop_mix, dtype='uint16')
    assert ops.all(ops.equal(result_mix, expected_mix)), f"propagate mixed failed"

    result_scalar_pos = bitwise.binary_wave_propagate(a, 1)
    expected_scalar_pos = tensor.convert_to_tensor(expected_prop_scalar_pos, dtype='uint16')
    assert ops.all(ops.equal(result_scalar_pos, expected_scalar_pos)), f"propagate scalar positive failed"

    result_scalar_neg = bitwise.binary_wave_propagate(a, -1)
    expected_scalar_neg = tensor.convert_to_tensor(expected_prop_scalar_neg, dtype='uint16')
    assert ops.all(ops.equal(result_scalar_neg, expected_scalar_neg)), f"propagate scalar negative failed"

def test_create_duty_cycle(mlx_backend):
    """Test bitwise.create_duty_cycle with MLX backend."""
    length = 8
    result_50 = bitwise.create_duty_cycle(length, 0.5)
    expected_50 = tensor.convert_to_tensor(expected_duty_50, dtype='int32')
    assert ops.all(ops.equal(result_50, expected_50)), f"duty_cycle 0.5 failed"

    result_25 = bitwise.create_duty_cycle(length, 0.25)
    expected_25 = tensor.convert_to_tensor(expected_duty_25, dtype='int32')
    assert ops.all(ops.equal(result_25, expected_25)), f"duty_cycle 0.25 failed"

    result_0 = bitwise.create_duty_cycle(length, 0.0)
    expected_0 = tensor.convert_to_tensor(expected_duty_0, dtype='int32')
    assert ops.all(ops.equal(result_0, expected_0)), f"duty_cycle 0.0 failed"

    result_100 = bitwise.create_duty_cycle(length, 1.0)
    expected_100 = tensor.convert_to_tensor(expected_duty_100, dtype='int32')
    assert ops.all(ops.equal(result_100, expected_100)), f"duty_cycle 1.0 failed"

    result_round = bitwise.create_duty_cycle(length, 0.6)
    expected_round = tensor.convert_to_tensor(expected_duty_round, dtype='int32')
    assert ops.all(ops.equal(result_round, expected_round)), f"duty_cycle rounding failed"

    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(0, 0.5)
    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(8, 1.1)
    with pytest.raises(ValueError):
        bitwise.create_duty_cycle(8, -0.1)

def test_generate_blocky_sin(mlx_backend):
    """Test bitwise.generate_blocky_sin with MLX backend."""
    length = 8
    half_period = 2
    result = bitwise.generate_blocky_sin(length, half_period)
    expected = tensor.convert_to_tensor(expected_blocky_sin, dtype='int32')
    assert ops.all(ops.equal(result, expected)), f"blocky_sin failed"

    half_period_3 = 3
    result_3 = bitwise.generate_blocky_sin(length, half_period_3)
    expected_3 = tensor.convert_to_tensor(expected_blocky_sin_3, dtype='int32')
    assert ops.all(ops.equal(result_3, expected_3)), f"blocky_sin hp=3 failed"

    length_7 = 7
    result_7 = bitwise.generate_blocky_sin(length_7, half_period)
    expected_7 = tensor.convert_to_tensor(expected_blocky_sin_7, dtype='int32')
    assert ops.all(ops.equal(result_7, expected_7)), f"blocky_sin length=7 failed"

    with pytest.raises(ValueError):
        bitwise.generate_blocky_sin(0, 2)
    with pytest.raises(ValueError):
        bitwise.generate_blocky_sin(8, 0)