"""MLX bitwise operations for ember_ml."""

# Import functions from submodules using absolute paths
from ember_ml.backend.mlx.bitwise.basic_ops import bitwise_and, bitwise_or, bitwise_xor, bitwise_not
from ember_ml.backend.mlx.bitwise.shift_ops import left_shift, right_shift, rotate_left, rotate_right
from ember_ml.backend.mlx.bitwise.bit_ops import count_ones, count_zeros, get_bit, set_bit, toggle_bit
from ember_ml.backend.mlx.bitwise.wave_ops import (
    binary_wave_interference,
    binary_wave_propagate,
    create_duty_cycle,
    generate_blocky_sin,
)

__all__ = [
    # Basic Ops
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    # Shift Ops
    "left_shift",
    "right_shift",
    "rotate_left",
    "rotate_right",
    # Bit Ops
    "count_ones",
    "count_zeros",
    "get_bit",
    "set_bit",
    "toggle_bit",
    # Wave Ops
    "binary_wave_interference",
    "binary_wave_propagate",
    "create_duty_cycle",
    "generate_blocky_sin",
]