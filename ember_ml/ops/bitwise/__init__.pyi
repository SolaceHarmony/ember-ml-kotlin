"""
Type stub file for ember_ml.ops.bitwise module.

This provides explicit type hints for the dynamically aliased bitwise operations,
allowing type checkers to recognize them properly based on the expected backend signatures.
Follows the pattern in ember_ml/ops/__init__.pyi using TensorLike for inputs
and a generic Tensor (Any) for the backend-specific return type.
"""

from typing import List, Optional, Any, Union, Tuple, Literal

# Import common types used across ops
# Assuming TensorLike is defined to accept various inputs and Tensor is Any for backend type
from ember_ml.nn.tensor.types import TensorLike
type Tensor = Any # Represents the backend-specific tensor type (e.g., mx.array)

# Basic bitwise operations
def bitwise_and(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_or(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_xor(x: TensorLike, y: TensorLike) -> Tensor: ...
def bitwise_not(x: TensorLike) -> Tensor: ...

# Shift operations
def left_shift(x: TensorLike, shifts: TensorLike) -> Tensor: ...
def right_shift(x: TensorLike, shifts: TensorLike) -> Tensor: ...
def rotate_left(x: TensorLike, shifts: TensorLike, bit_width: int = ...) -> Tensor: ...
def rotate_right(x: TensorLike, shifts: TensorLike, bit_width: int = ...) -> Tensor: ...

# Bit counting operations
def count_ones(x: TensorLike) -> Tensor: ...
def count_zeros(x: TensorLike) -> Tensor: ...

# Bit manipulation operations
def get_bit(x: TensorLike, position: TensorLike) -> Tensor: ...
def set_bit(x: TensorLike, position: TensorLike, value: TensorLike) -> Tensor: ...
def toggle_bit(x: TensorLike, position: TensorLike) -> Tensor: ...

# Binary wave operations
def binary_wave_interference(waves: List[TensorLike], mode: str = ...) -> Tensor: ...
def binary_wave_propagate(wave: TensorLike, shift: TensorLike) -> Tensor: ...
# Use specific types if known, otherwise TensorLike
def create_duty_cycle(length: Union[int, TensorLike], duty_cycle: Union[float, TensorLike]) -> Tensor: ...
def generate_blocky_sin(length: Union[int, TensorLike], half_period: Union[int, TensorLike]) -> Tensor: ...

# Define __all__ to match the implementation module
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