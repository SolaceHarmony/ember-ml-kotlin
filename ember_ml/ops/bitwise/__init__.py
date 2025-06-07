"""
Bitwise operations module for the Ember ML frontend.

This module provides a unified interface to bitwise operations from the current backend
(NumPy, PyTorch, MLX) using the proxy module pattern. It dynamically forwards
attribute access to the appropriate backend module.
"""

# Import the bitwise proxy from the ops proxy module
from ember_ml.ops.proxy import bitwise as bitwise_proxy

# Import all operations from the bitwise proxy
bitwise_and = bitwise_proxy.bitwise_and
bitwise_or = bitwise_proxy.bitwise_or
bitwise_xor = bitwise_proxy.bitwise_xor
bitwise_not = bitwise_proxy.bitwise_not
left_shift = bitwise_proxy.left_shift
right_shift = bitwise_proxy.right_shift
rotate_left = bitwise_proxy.rotate_left
rotate_right = bitwise_proxy.rotate_right
count_ones = bitwise_proxy.count_ones
count_zeros = bitwise_proxy.count_zeros
get_bit = bitwise_proxy.get_bit
set_bit = bitwise_proxy.set_bit
toggle_bit = bitwise_proxy.toggle_bit
binary_wave_interference = bitwise_proxy.binary_wave_interference
binary_wave_propagate = bitwise_proxy.binary_wave_propagate
create_duty_cycle = bitwise_proxy.create_duty_cycle
generate_blocky_sin = bitwise_proxy.generate_blocky_sin

# Define __all__ to include all operations
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
