"""
NumPy implementation of MegaBinary from BizarroMath.

This module provides the NumpyMegaBinary class, inheriting from NumpyMegaNumber,
using numpy.ndarray with dtype=uint16 as the underlying representation for
binary wave and bitwise operations.
"""

import numpy as np
from typing import Tuple, Union, List, Optional, Any
from enum import Enum

# Import backend types
from ember_ml.backend.numpy.types import TensorLike

# Import the base class from the same directory
from ember_ml.backend.numpy.bizarromath.mega_number import NumpyMegaNumber

class InterferenceMode(Enum):
    """Interference modes for binary wave operations."""
    XOR = "xor"
    AND = "and"
    OR  = "or"

class NumpyMegaBinary(NumpyMegaNumber):
    """
    NumPy-based binary data class, storing bits in NumPy arrays with uint16 dtype.
    Includes wave generation, duty-cycle patterns, interference, and
    optional leading-zero preservation. Inherits from NumpyMegaNumber.
    """

    def __init__(self, value: Union[str, bytes, bytearray] = "0",
                 keep_leading_zeros: bool = True,
                 **kwargs):
        """
        Initialize a NumpyMegaBinary object.

        Args:
            value: Initial value, can be:
                - String of binary digits (e.g., "1010" or "0b1010")
                - bytes/bytearray (will parse each byte => 8 bits => NumPy array)
                - Default "0" => NumPy array of just [0]
            keep_leading_zeros: Whether to keep leading zeros (default: True)
            **kwargs: Additional arguments for NumpyMegaNumber
        """
        # Initialize the base NumpyMegaNumber first, setting is_float=False
        super().__init__(
            mantissa=None, # Will be set by _parse_binary_string
            exponent=None, # Not used for binary
            negative=False, # Binary is unsigned
            is_float=False,
            exponent_negative=False, # Not used for binary
            keep_leading_zeros=keep_leading_zeros,
            **kwargs
        )

        # Step 1) Auto-detect and convert input
        if isinstance(value, (bytes, bytearray)):
            # Store original bytes if needed
            self.byte_data = bytearray(value)
            # Convert them to a binary string => MLX array
            bin_str = "".join(format(b, "08b") for b in self.byte_data)
        elif isinstance(value, str):
            # Assume it's a string of bits (e.g., "1010" or "0b1010")
            # or possibly an empty string => "0"
            bin_str = value
            if bin_str.startswith("0b"):
                bin_str = bin_str[2:]
            if not bin_str:
                bin_str = "0"

            # Also build self.byte_data from this binary string
            # so we have a consistent stored representation if needed.
            # We'll chunk every 8 bits => int => byte
            self.byte_data = bytearray()
            # Pad with leading zeros if necessary to make length a multiple of 8
            padded_bin_str = bin_str.zfill((len(bin_str) + 7) // 8 * 8)
            for i in range(0, len(padded_bin_str), 8):
                chunk = padded_bin_str[i:i+8]
                self.byte_data.append(int(chunk, 2))
        elif isinstance(value, NumpyMegaBinary):
             # Handle copy case
             bin_str = value.to_string()
             self.byte_data = value.byte_data
        else:
             raise TypeError(f"Unsupported initial value type for NumpyMegaBinary: {type(value)}")


        # Step 2) Parse bin_str into MLX array mantissa
        self._parse_binary_string(bin_str)

        # Step 3) Normalize with respect to keep_leading_zeros
        # Base class normalize handles sign, but we might need binary specific?
        # For now, rely on base class _normalize and _parse_binary_string logic
        self._normalize() # Call base class normalize

        # Store bit length based on the initial string potentially before normalization
        # This might need adjustment depending on desired behavior with keep_leading_zeros
        self._bit_length = len(bin_str)

    def _parse_binary_string(self, bin_str: str) -> None:
        """
        Convert binary string => MLX array in little-endian chunk form.
        Overrides base class method if necessary, or called by base init.

        Args:
            bin_str: Binary string (e.g., "1010") - assumes no "0b" prefix here.
        """
        if not bin_str:
            bin_str = "0"

        # Store bit length based on this potentially normalized string
        self._bit_length = len(bin_str)

        # Convert to integer (use Python's arbitrary precision int)
        val = int(bin_str, 2)

        # Convert to NumPy array (use int64 for potentially large integers)
        val_np = np.array(val, dtype=np.int64)

        # Convert to limbs (int16) using the base class static method
        self.mantissa = NumpyMegaNumber._int_to_chunklist(val_np, self._global_chunk_size)
        # Ensure exponent is zero for binary
        self.exponent = np.array([0], dtype=np.int16)
        self.is_float = False
        self.negative = False
        self.exponent_negative = False


    def bitwise_and(self, other: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Perform bitwise AND operation.

        Args:
            other: Another NumpyMegaBinary object

        Returns:
            Result of bitwise AND operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))

        # Pad arrays to the same length
        self_arr = np.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = np.pad(other.mantissa, [(0, max_len - len(other.mantissa))])

        # Perform bitwise AND
        result_arr = np.bitwise_and(self_arr, other_arr)

        # Create result using mantissa directly
        result = NumpyMegaBinary(mantissa=result_arr, keep_leading_zeros=self._keep_leading_zeros)
        result._normalize() # Normalize the result

        return result

    def bitwise_or(self, other: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Perform bitwise OR operation.

        Args:
            other: Another NumpyMegaBinary object

        Returns:
            Result of bitwise OR operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))

        # Pad arrays to the same length
        self_arr = np.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = np.pad(other.mantissa, [(0, max_len - len(other.mantissa))])

        # Perform bitwise OR
        result_arr = np.bitwise_or(self_arr, other_arr)

        # Create result
        result = NumpyMegaBinary(mantissa=result_arr, keep_leading_zeros=self._keep_leading_zeros)
        result._normalize()

        return result

    def bitwise_xor(self, other: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Perform bitwise XOR operation.

        Args:
            other: Another NumpyMegaBinary object

        Returns:
            Result of bitwise XOR operation
        """
        # Get maximum length
        max_len = max(len(self.mantissa), len(other.mantissa))

        # Pad arrays to the same length
        self_arr = np.pad(self.mantissa, [(0, max_len - len(self.mantissa))])
        other_arr = np.pad(other.mantissa, [(0, max_len - len(other.mantissa))])

        # Perform bitwise XOR
        result_arr = np.bitwise_xor(self_arr, other_arr)

        # Create result
        result = NumpyMegaBinary(mantissa=result_arr, keep_leading_zeros=self._keep_leading_zeros)
        result._normalize()

        return result

    def bitwise_not(self) -> "NumpyMegaBinary":
        """
        Perform bitwise NOT operation.
        Note: This performs a bitwise inversion based on the current limb representation.
              The effective bit length might influence the result if interpreted as
              a fixed-width integer's NOT operation.

        Returns:
            Result of bitwise NOT operation
        """
        # Perform bitwise NOT on existing limbs
        # NumPy invert acts like bitwise_not, which is suitable here.
        result_arr = np.invert(self.mantissa)

        # Create result
        result = NumpyMegaBinary(mantissa=result_arr, keep_leading_zeros=self._keep_leading_zeros)
        # Normalization might trim leading ones if keep_leading_zeros is False,
        # which might not be the desired behavior for NOT. Consider implications.
        result._normalize()

        return result

    # --- Arithmetic Operations (Inherited, but ensure they return MLXMegaBinary) ---

    def add(self, other: "NumpyMegaNumber") -> "NumpyMegaBinary":
        """
        Add two NumpyMegaBinary objects (treating them as unsigned integers).

        Args:
            other: Another NumpyMegaNumber (should ideally be NumpyMegaBinary)

        Returns:
            Sum as NumpyMegaBinary
        """
        if not isinstance(other, NumpyMegaBinary):
            # Handle addition with non-binary MegaNumber if necessary,
            # potentially by converting other to binary first or raising error.
            # For now, assume other is NumpyMegaBinary.
             raise TypeError("Addition is only defined between NumpyMegaBinary instances.")

        # Use base class integer addition logic
        base_result = super().add(other)

        # Create new NumpyMegaBinary from the result mantissa
        # The base add method already returns NumpyMegaNumber, just cast it
        # Need to ensure the result is treated as binary
        result_bin = NumpyMegaBinary(mantissa=base_result.mantissa, keep_leading_zeros=self._keep_leading_zeros)
        result_bin._normalize()
        return result_bin


    def sub(self, other: "NumpyMegaNumber") -> "NumpyMegaBinary":
        """
        Subtract other from self (treating them as unsigned integers).
        Result is undefined if other > self.

        Args:
            other: Another NumpyMegaNumber (should ideally be NumpyMegaBinary)

        Returns:
            Difference as NumpyMegaBinary
        """
        if not isinstance(other, NumpyMegaBinary):
             raise TypeError("Subtraction is only defined between NumpyMegaBinary instances.")

        # Use base class integer subtraction logic
        base_result = super().sub(other)

        # Check for negative result which is invalid for binary representation
        if base_result.negative:
            raise ValueError("Subtraction resulted in a negative value, invalid for NumpyMegaBinary")

        # Create new NumpyMegaBinary from the result mantissa
        result_bin = NumpyMegaBinary(mantissa=base_result.mantissa, keep_leading_zeros=self._keep_leading_zeros)
        result_bin._normalize()
        return result_bin

    def mul(self, other: "NumpyMegaNumber") -> "NumpyMegaBinary":
        """
        Multiply two NumpyMegaBinary objects (treating them as unsigned integers).

        Args:
            other: Another NumpyMegaNumber (should ideally be NumpyMegaBinary)

        Returns:
            Product as NumpyMegaBinary
        """
        if not isinstance(other, NumpyMegaBinary):
             raise TypeError("Multiplication is only defined between NumpyMegaBinary instances.")

        # Use base class integer multiplication logic
        base_result = super().mul(other)

        # Create new NumpyMegaBinary from the result mantissa
        result_bin = NumpyMegaBinary(mantissa=base_result.mantissa, keep_leading_zeros=self._keep_leading_zeros)
        result_bin._normalize()
        return result_bin

    def div(self, other: "NumpyMegaNumber") -> "NumpyMegaBinary":
        """
        Divide self by other (integer division).

        Args:
            other: Another NumpyMegaNumber (should ideally be NumpyMegaBinary)

        Returns:
            Quotient as NumpyMegaBinary
        """
        if not isinstance(other, NumpyMegaBinary):
             raise TypeError("Division is only defined between NumpyMegaBinary instances.")

        # Use base class integer division logic
        base_result = super().div(other) # This performs integer division

        # Create new NumpyMegaBinary from the result mantissa
        result_bin = NumpyMegaBinary(mantissa=base_result.mantissa, keep_leading_zeros=self._keep_leading_zeros)
        result_bin._normalize()
        return result_bin

    # --- Shift Operations ---

    def shift_left(self, bits: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Shift left by bits.

        Args:
            bits: Number of bits to shift (as NumpyMegaBinary)

        Returns:
            Shifted NumpyMegaBinary
        """
        # Convert bits to integer (use int64 for potentially large shifts)
        shift_val = self._chunklist_to_int(bits.mantissa)

        # Convert self to integer
        self_val = self._chunklist_to_int(self.mantissa)

        # Perform left shift
        shifted_val = np.left_shift(self_val, shift_val)

        # Convert back to limbs
        result_limbs = self._int_to_chunklist(shifted_val, self._global_chunk_size)

        # Create result
        result = NumpyMegaBinary(mantissa=result_limbs, keep_leading_zeros=self._keep_leading_zeros)
        result._normalize()
        return result

    def shift_right(self, bits: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Shift right by bits.

        Args:
            bits: Number of bits to shift (as NumpyMegaBinary)

        Returns:
            Shifted NumpyMegaBinary
        """
        # Convert bits to integer
        shift_val = self._chunklist_to_int(bits.mantissa)

        # Convert self to integer
        self_val = self._chunklist_to_int(self.mantissa)

        # Perform right shift
        shifted_val = np.right_shift(self_val, shift_val)

        # Convert back to limbs
        result_limbs = self._int_to_chunklist(shifted_val, self._global_chunk_size)

        # Create result
        result = NumpyMegaBinary(mantissa=result_limbs, keep_leading_zeros=self._keep_leading_zeros)
        result._normalize()
        return result

    # --- Bit Manipulation ---

    def get_bit(self, position: "NumpyMegaBinary") -> bool:
        """
        Get the bit at the specified position.

        Args:
            position: Bit position (0-based, from least significant bit)

        Returns:
            Bit value (True or False)
        """
        # Convert position to integer
        pos_val = self._chunklist_to_int(position.mantissa)

        # Convert self to integer
        self_val = self._chunklist_to_int(self.mantissa)

        # Create mask
        one = np.array(1, dtype=self_val.dtype)
        mask = np.left_shift(one, pos_val)

        # Check the bit
        is_set = np.bitwise_and(self_val, mask)
        return np.any(np.not_equal(is_set, np.array(0, dtype=is_set.dtype))).item()


    def set_bit(self, position: "NumpyMegaBinary", value: bool) -> None:
        """
        Set the bit at the specified position. Modifies the object in place.

        Args:
            position: Bit position (0-based, from least significant bit)
            value: Bit value (True or False)
        """
        # Convert position to integer
        pos_val = self._chunklist_to_int(position.mantissa)

        # Convert self to integer
        self_val = self._chunklist_to_int(self.mantissa)

        # Create mask
        one = np.array(1, dtype=self_val.dtype)
        mask = np.left_shift(one, pos_val)

        if value:
            # Set bit using OR
            new_val = np.bitwise_or(self_val, mask)
        else:
            # Clear bit using AND with NOT mask
            new_val = np.bitwise_and(self_val, np.invert(mask))

        # Convert back to limbs and update mantissa
        self.mantissa = self._int_to_chunklist(new_val, self._global_chunk_size)
        self._normalize() # Re-normalize after modification


    # --- Wave Operations ---

    @classmethod
    def interfere(cls, waves: List["NumpyMegaBinary"], mode: InterferenceMode) -> "NumpyMegaBinary":
        """
        Combine multiple waves bitwise (XOR, AND, OR).

        Args:
            waves: List of NumpyMegaBinary objects
            mode: Interference mode (XOR, AND, OR)

        Returns:
            Interference pattern
        """
        if not waves:
            raise ValueError("Need at least one wave for interference")

        # Find max length among all wave mantissas
        max_len = 0
        for wave in waves:
            max_len = max(max_len, len(wave.mantissa))

        # Pad all mantissas to max_len and perform operation
        result_arr = np.pad(waves[0].mantissa, [(0, max_len - len(waves[0].mantissa))])

        for wave in waves[1:]:
            padded_wave_arr = np.pad(wave.mantissa, [(0, max_len - len(wave.mantissa))])
            if mode == InterferenceMode.XOR:
                result_arr = np.bitwise_xor(result_arr, padded_wave_arr)
            elif mode == InterferenceMode.AND:
                result_arr = np.bitwise_and(result_arr, padded_wave_arr)
            elif mode == InterferenceMode.OR:
                result_arr = np.bitwise_or(result_arr, padded_wave_arr)
            else:
                 raise ValueError(f"Unsupported interference mode: {mode}")

        # Create result
        # Determine keep_leading_zeros based on the first wave? Or default?
        keep_zeros = waves[0]._keep_leading_zeros if waves else True
        result = cls(mantissa=result_arr, keep_leading_zeros=keep_zeros)
        result._normalize()
        return result


    @classmethod
    def generate_blocky_sin(cls, length: "NumpyMegaBinary", half_period: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Create a blocky sine wave pattern.

        Args:
            length: Length of the pattern in bits (as NumpyMegaBinary)
            half_period: Half the period of the wave in bits (as NumpyMegaBinary)

        Returns:
            Blocky sine wave pattern
        """
        # Convert inputs to integers
        len_int = int(cls._chunklist_to_int(length.mantissa).item())
        hp_int = int(cls._chunklist_to_int(half_period.mantissa).item())

        if hp_int <= 0:
            raise ValueError("Half period must be positive")
        if len_int <= 0:
            return cls("0") # Return zero for zero length

        # Generate pattern using standard Python integers for loop control
        bin_str = ""
        for i in range(len_int):
            if (i // hp_int) % 2 == 0:
                bin_str += "1"
            else:
                bin_str += "0"

        return cls(bin_str, keep_leading_zeros=length._keep_leading_zeros)


    @classmethod
    def create_duty_cycle(cls, length: "NumpyMegaBinary", duty_cycle_val: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Create a binary pattern with the specified duty cycle.
        Assumes duty_cycle_val represents the number of '1's directly.

        Args:
            length: Length of the pattern in bits (as NumpyMegaBinary)
            duty_cycle_val: Number of '1' bits (as NumpyMegaBinary)

        Returns:
            Binary pattern with the specified duty cycle
        """
        len_int = int(cls._chunklist_to_int(length.mantissa).item())
        num_ones = int(cls._chunklist_to_int(duty_cycle_val.mantissa).item())

        if num_ones < 0 or num_ones > len_int:
             raise ValueError("Number of ones must be between 0 and length")
        if len_int <= 0:
             return cls("0")

        # Create pattern string
        bin_str = "1" * num_ones + "0" * (len_int - num_ones)

        return cls(bin_str, keep_leading_zeros=length._keep_leading_zeros)


    def propagate(self, shift: "NumpyMegaBinary") -> "NumpyMegaBinary":
        """
        Propagate the wave by shifting it left.

        Args:
            shift: Number of bits to shift (as NumpyMegaBinary)

        Returns:
            Propagated wave
        """
        # Propagation is typically a left shift in wave contexts
        return self.shift_left(shift)

    # --- Conversion and Utility Methods ---

    def to_bits(self) -> List[int]:
        """
        Convert to list of bits (LSB first).

        Returns:
            List of bits (0 or 1)
        """
        bin_str = self.to_string()
        # Pad with leading zeros if keep_leading_zeros is true and _bit_length is set
        if self._keep_leading_zeros and hasattr(self, '_bit_length'):
             bin_str = bin_str.zfill(self._bit_length)
        return [int(bit) for bit in reversed(bin_str)]

    def to_bits_bigendian(self) -> List[int]:
        """
        Convert to list of bits (MSB first).

        Returns:
            List of bits (0 or 1)
        """
        bin_str = self.to_string()
        if self._keep_leading_zeros and hasattr(self, '_bit_length'):
             bin_str = bin_str.zfill(self._bit_length)
        return [int(bit) for bit in bin_str]

    def to_string(self) -> str:
        """
        Convert to binary string (MSB first).

        Returns:
            Binary string representation
        """
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"

        # Convert limbs to integer (use int64)
        val = self._chunklist_to_int(self.mantissa)

        # Convert to binary string, remove "0b" prefix
        bin_str = bin(int(val.item()))[2:] # Convert NumPy scalar to Python int

        # Handle potential padding if keep_leading_zeros is True
        if self._keep_leading_zeros and hasattr(self, '_bit_length'):
             bin_str = bin_str.zfill(self._bit_length)

        return bin_str if bin_str else "0"


    def to_string_bigendian(self) -> str:
        """
        Convert to binary string (MSB first). Alias for to_string.

        Returns:
            Binary string representation (MSB first)
        """
        return self.to_string()

    def is_zero(self) -> bool:
        """
        Check if the value is zero.

        Returns:
            True if the value is zero, False otherwise
        """
        # Check if mantissa represents zero after normalization
        self._normalize() # Ensure it's normalized
        return len(self.mantissa) == 1 and self.mantissa[0] == 0

    def to_bytes(self) -> bytearray:
        """
        Convert to bytes (big-endian).

        Returns:
            Byte representation
        """
        bin_str = self.to_string()
        # Pad with leading zeros to make length a multiple of 8
        padded_bin_str = bin_str.zfill((len(bin_str) + 7) // 8 * 8)
        byte_arr = bytearray()
        for i in range(0, len(padded_bin_str), 8):
            chunk = padded_bin_str[i:i+8]
            byte_arr.append(int(chunk, 2))
        return byte_arr

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            String representation
        """
        return f"<NumpyMegaBinary {self.to_string()}>"

# Example usage (optional, for testing)
if __name__ == "__main__":
    a = NumpyMegaBinary("10101100")
    b = NumpyMegaBinary("01010101")
    zero = NumpyMegaBinary("0")
    len8 = NumpyMegaBinary("1000") # Length 8
    hp2 = NumpyMegaBinary("10")   # Half period 2
    duty4 = NumpyMegaBinary("100") # 4 ones (duty cycle 0.5 for length 8)
    shift2 = NumpyMegaBinary("10") # Shift 2

    print(f"a = {a}")
    print(f"b = {b}")

    print(f"a & b = {a.bitwise_and(b)}")
    print(f"a | b = {a.bitwise_or(b)}")
    print(f"a ^ b = {a.bitwise_xor(b)}")
    print(f"~a (bitwise) = {a.bitwise_not()}") # Note: NOT behavior depends on interpretation

    print(f"a + b = {a.add(b)}")
    print(f"a * b = {a.mul(b)}")
    # print(f"a / b = {a.div(b)}") # Integer division

    print(f"a << 2 = {a.shift_left(shift2)}")
    print(f"a >> 2 = {a.shift_right(shift2)}")

    print(f"a.get_bit(3) = {a.get_bit(NumpyMegaBinary('11'))}") # Get 4th bit (pos 3)
    a_copy = a.copy()
    a_copy.set_bit(NumpyMegaBinary('0'), True) # Set LSB
    print(f"a after set_bit(0, True) = {a_copy}")

    print(f"Blocky sin(len=8, hp=2) = {NumpyMegaBinary.generate_blocky_sin(len8, hp2)}")
    print(f"Duty cycle(len=8, num_ones=4) = {NumpyMegaBinary.create_duty_cycle(len8, duty4)}")

    print(f"Interfere([a,b], XOR) = {NumpyMegaBinary.interfere([a, b], InterferenceMode.XOR)}")
    print(f"Propagate(a, shift=2) = {a.propagate(shift2)}")

    print(f"a.to_bits() = {a.to_bits()}")
    print(f"a.to_bits_bigendian() = {a.to_bits_bigendian()}")
    print(f"a.to_bytes() = {a.to_bytes()}")
    print(f"zero.is_zero() = {zero.is_zero()}")
    print(f"a.is_zero() = {a.is_zero()}")