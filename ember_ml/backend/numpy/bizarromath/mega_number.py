"""
NumPy implementation of MegaNumber, the foundation for BizarroMath arbitrary precision.

This module provides the NumpyMegaNumber class, using numpy.ndarray with dtype=int16
as the underlying representation for chunk-based (limb-based) arithmetic.
"""

import numpy as np
from typing import Tuple, Union, List, Optional, Any

# Import backend types
from ember_ml.backend.numpy.types import TensorLike

class NumpyMegaNumber:
    """
    A chunk-based big integer (or float) with HPC-limb arithmetic,
    using NumPy arrays with int16 dtype to mimic BigBase65536 logic.
    """

    # Constants as NumPy arrays
    _global_chunk_size = np.array(16, dtype=np.int16)  # bits per limb
    _base = np.array(65536, dtype=np.int32)  # 2^16
    _mask = np.array(65535, dtype=np.int32)  # 2^16 - 1

    # Optional thresholds for advanced multiplication
    _MUL_THRESHOLD_KARATSUBA = np.array(32, dtype=np.int16)
    _MUL_THRESHOLD_TOOM = np.array(128, dtype=np.int16)

    _max_precision_bits = None
    _log2_of_10_cache = None  # class-level for caching log2(10)

    def __init__(
        self,
        value: Union[str, 'NumpyMegaNumber', np.ndarray] = None,
        mantissa: Optional[np.ndarray] = None,
        exponent: Optional[np.ndarray] = None,
        negative: bool = False,
        is_float: bool = False,
        exponent_negative: bool = False,
        keep_leading_zeros: bool = False
    ):
        """
        Initialize a HPC-limb object using NumPy arrays.

        Args:
            value: Initial value, can be:
                - String (decimal or binary)
                - NumpyMegaNumber
                - NumPy array of limbs
            mantissa: NumPy array of limbs
            exponent: NumPy array of limbs
            negative: Sign flag
            is_float: Float flag
            exponent_negative: Exponent sign flag
            keep_leading_zeros: Whether to keep leading zeros
        """
        if mantissa is None:
            mantissa = np.array([0], dtype=np.int16)
        if exponent is None:
            exponent = np.array([0], dtype=np.int16)

        self.mantissa = mantissa
        self.exponent = exponent
        self.negative = negative
        self.is_float = is_float
        self.exponent_negative = exponent_negative
        self._keep_leading_zeros = keep_leading_zeros

        if isinstance(value, str):
            # Parse decimal string
            # TODO: Implement robust string parsing (decimal/binary detection)
            # For now, assume decimal if not MLXMegaNumber or mx.array
            tmp = NumpyMegaNumber.from_decimal_string(value)
            self.mantissa = tmp.mantissa
            self.exponent = tmp.exponent
            self.negative = tmp.negative
            self.is_float = tmp.is_float
            self.exponent_negative = tmp.exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        elif isinstance(value, NumpyMegaNumber):
            # Copy
            self.mantissa = np.array(value.mantissa, dtype=np.int16)
            self.exponent = np.array(value.exponent, dtype=np.int16)
            self.negative = value.negative
            self.is_float = value.is_float
            self.exponent_negative = value.exponent_negative
            self._keep_leading_zeros = value._keep_leading_zeros # Use value's setting
        elif isinstance(value, np.ndarray):
            # Interpret as mantissa
            self.mantissa = value
            self.exponent = np.array([0], dtype=np.int16)
            self.negative = negative
            self.is_float = is_float
            self.exponent_negative = exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        else:
            # If nothing => user-supplied mantissa/exponent or default [0]
            pass

        # Normalize
        self._normalize()

    def _normalize(self):
        """
        If keep_leading_zeros=False => remove trailing zero-limbs from mantissa.
        If float => also remove trailing zeros from exponent. Keep at least 1 limb.
        If everything is zero => unify sign bits to false/positive.
        """
        if not self._keep_leading_zeros:
            # Trim mantissa
            while len(self.mantissa) > 1 and self.mantissa[-1] == 0:
                self.mantissa = self.mantissa[:-1]

            # Trim exponent if float
            if self.is_float:
                while len(self.exponent) > 1 and self.exponent[-1] == 0:
                    self.exponent = self.exponent[:-1]

            # If mantissa is entirely zero => unify sign
            if len(self.mantissa) == 1 and self.mantissa[0] == 0:
                self.negative = False
                self.exponent = np.array([0], dtype=np.int16)
                self.exponent_negative = False
        else:
            # If keep_leading_zeros => only unify if mantissa is all zero
            if np.all(self.mantissa == 0):
                self.negative = False
                self.exponent_negative = False # Keep exponent value if float

    @classmethod
    def from_decimal_string(cls, dec_str: str) -> "NumpyMegaNumber":
        """
        Convert decimal => HPC big-int or HPC float.
        We detect fractional by '.' => if present => treat as float, shifting exponent.

        Args:
            dec_str: Decimal string

        Returns:
            NumpyMegaNumber
        """
        s = dec_str.strip()
        if not s:
            return cls(mantissa=np.array([0], dtype=np.int16),
                       exponent=np.array([0], dtype=np.int16),
                       negative=False, is_float=False)

        negative = False
        if s.startswith('-'):
            negative = True
            s = s[1:].strip()

        # Detect fractional
        point_pos = s.find('.')
        frac_len = 0
        if point_pos >= 0:
            frac_len = len(s) - (point_pos + 1)
            s = s.replace('.', '')

        # Repeatedly multiply by 10 and add digit
        mant = np.array([0], dtype=np.int16)
        ten = np.array([10], dtype=np.int16) # Define ten here

        for ch in s:
            if ch < '0' or ch > '9':
                raise ValueError(f"Invalid digit '{ch}' in decimal string.")

            # Convert digit to NumPy array
            digit_val = np.array(int(ch), dtype=np.int16)

            # Multiply mant by 10
            mant = cls._mul_chunklists(
                mant,
                ten, # Use defined ten
                cls._global_chunk_size,
                cls._base
            )

            # Add digit using _add_chunklists for simplicity and consistency
            mant = cls._add_chunklists(mant, digit_val)


        exp_limb = np.array([0], dtype=np.int16)
        exponent_negative = False
        is_float = False

        # If we had fraction => shift exponent
        if frac_len > 0:
            is_float = True
            exponent_negative = True

            # Approximate: frac_len * log2(10) => bit shift exponent
            # Convert frac_len to NumPy array
            frac_len_np = np.array(frac_len, dtype=np.int16)

            # Multiply by log2(10) ≈ 3.32
            # TODO: Use a higher precision log2(10) if needed
            log2_10 = np.array(3.32192809, dtype=np.float32) # More precision
            bits_needed_float = np.multiply(np.array(frac_len, dtype=np.float32), log2_10)
            bits_needed = np.array(np.ceil(bits_needed_float), dtype=np.int32) # Use int32 for intermediate

            exp_limb = cls._int_to_chunklist(bits_needed, cls._global_chunk_size)

        obj = cls(
            mantissa=mant,
            exponent=exp_limb,
            negative=negative,
            is_float=is_float,
            exponent_negative=exponent_negative
        )
        obj._normalize()
        return obj

    @classmethod
    def from_binary_string(cls, bin_str: str) -> "NumpyMegaNumber":
        """
        Convert binary string => HPC big-int.

        Args:
            bin_str: Binary string (e.g., "1010" or "0b1010")

        Returns:
            NumpyMegaNumber
        """
        s = bin_str.strip()
        if s.startswith('0b'):
            s = s[2:]
        if not s:
            s = "0"

        # Convert to integer
        val = int(s, 2)

        # Convert to NumPy array (use int64 for potentially large integers)
        val_np = np.array(val, dtype=np.int64)

        # Convert to limbs
        limbs = cls._int_to_chunklist(val_np, cls._global_chunk_size)

        return cls(
            mantissa=limbs,
            exponent=np.array([0], dtype=np.int16),
            negative=False,
            is_float=False
        )

    def to_decimal_string(self, max_digits=None) -> str:
        """
        Convert to decimal string.

        Args:
            max_digits: Maximum number of digits to include

        Returns:
            Decimal string representation
        """
        # Handle zero
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"

        sign_str = "-" if self.negative else ""

        if not self.is_float:
            # Integer => repeated divmod 10
            tmp = np.array(self.mantissa)
            digits_rev = []

            zero = np.array([0], dtype=np.int16)
            ten = np.array(10, dtype=np.int16)

            while not (len(tmp) == 1 and tmp[0] == 0):
                tmp, r = self._divmod_small(tmp, ten)
                # Ensure remainder is treated as scalar Python int for string conversion
                digits_rev.append(str(int(r.item())))

            digits_rev.reverse()
            dec_str = "".join(digits_rev)

            # Truncation logic seems problematic for large numbers, reconsider if needed
            # if max_digits and len(dec_str) > max_digits:
            #     dec_str = f"...{dec_str[-max_digits:]}"

            return sign_str + dec_str
        else:
            # Float => exponent shift
            # If exponent_negative => we do mantissa // 2^(exponent), capturing remainder => fractional digits.
            # else => mantissa << exponent => integer.
            exp_int = self._chunklist_to_int(self.exponent)

            if self.exponent_negative:
                # Do integer part
                int_part, remainder = self._div_by_2exp(self.mantissa, exp_int)
                int_str = self._chunk_to_dec_str(int_part, max_digits)

                # If remainder=0 => done
                zero = np.array([0], dtype=np.int16)
                if self._compare_abs(remainder, zero) == 0:
                    return sign_str + int_str

                # Else => build fractional by repeatedly *10 // 2^exp_int
                frac_digits = []
                steps = max_digits or 50 # Limit fractional digits
                cur_rem = remainder

                ten = np.array([10], dtype=np.int16)
                # Precompute 2^exp_int for efficiency
                # Use int64 for potentially large intermediate values
                two_exp = np.power(np.array(2, dtype=np.int64), exp_int)

                for _ in range(steps):
                    # Multiply remainder by 10
                    cur_rem = self._mul_chunklists(
                        cur_rem,
                        ten,
                        self._global_chunk_size,
                        self._base
                    )

                    # Divide by 2^exp_int
                    q, cur_rem = self._div_chunk(cur_rem, self._int_to_chunklist(two_exp, self._global_chunk_size))

                    digit_val = self._chunklist_to_int(q)
                    frac_digits.append(str(int(digit_val.item()))) # Convert scalar NumPy array to Python int

                    if self._compare_abs(cur_rem, zero) == 0:
                        break

                return sign_str + int_str + "." + "".join(frac_digits)
            else:
                # Exponent positive => mantissa << exp_int
                shifted = self._mul_by_2exp(self.mantissa, exp_int)
                return sign_str + self._chunk_to_dec_str(shifted, max_digits)

    def _chunk_to_dec_str(self, chunks: np.ndarray, max_digits: Optional[int] = None) -> str:
        """
        Convert chunks to decimal string.

        Args:
            chunks: NumPy array of chunks
            max_digits: Maximum number of digits

        Returns:
            Decimal string
        """
        # Use to_decimal_string with a temporary NumpyMegaNumber
        tmp = NumpyMegaNumber(mantissa=chunks, is_float=False)
        return tmp.to_decimal_string(max_digits)

    def _div_by_2exp(self, limbs: np.ndarray, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integer division: limbs // 2^bits, remainder = limbs % 2^bits.

        Args:
            limbs: NumPy array of limbs
            bits: Number of bits to divide by (as NumPy array)

        Returns:
            Tuple of (quotient, remainder)
        """
        zero = np.array(0, dtype=np.int32) # Use int32 for comparison
        bits_int = bits # Assume bits is already a scalar array

        if np.all(np.less_equal(bits_int, zero)):
             return (np.array(limbs), np.array([0], dtype=np.int16))

        # Convert limbs to integer (use int64 for larger range)
        val_A = self._chunklist_to_int(limbs)

        # Calculate total bits in limbs
        total_bits = np.multiply(np.array(len(limbs), dtype=np.int32), np.array(16, dtype=np.int32)) # Use 16 directly

        if np.all(np.greater_equal(bits_int, total_bits)):
            # Everything is remainder
            return (np.array([0], dtype=np.int16), limbs)

        # Calculate remainder mask (use int64 for mask calculation)
        one = np.array(1, dtype=np.int64)
        remainder_mask = np.subtract(np.left_shift(one, bits_int), one)

        # Calculate remainder
        remainder_val = np.bitwise_and(val_A, remainder_mask)

        # Calculate quotient
        quotient_val = np.right_shift(val_A, bits_int)

        # Convert back to chunks
        quotient_part = self._int_to_chunklist(quotient_val, self._global_chunk_size)
        remainder_part = self._int_to_chunklist(remainder_val, self._global_chunk_size)

        return (quotient_part, remainder_part)

    def _mul_by_2exp(self, limbs: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """
        Multiply by 2^bits.

        Args:
            limbs: NumPy array of limbs
            bits: Number of bits to multiply by (as NumPy array)

        Returns:
            NumPy array of limbs
        """
        zero = np.array(0, dtype=np.int32) # Use int32 for comparison
        bits_int = bits # Assume bits is already a scalar array

        if np.all(np.less_equal(bits_int, zero)):
            return np.array(limbs)

        # Convert limbs to integer (use int64)
        val_A = self._chunklist_to_int(limbs)

        # Shift left
        val_shifted = np.left_shift(val_A, bits_int)

        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)

    def add(self, other: "NumpyMegaNumber") -> "NumpyMegaNumber":
        """
        Add two NumpyMegaNumbers.

        Args:
            other: Another NumpyMegaNumber

        Returns:
            Sum as NumpyMegaNumber
        """
        if self.is_float or other.is_float:
            # Ensure both are treated as float for alignment
            self_float = self.copy()
            other_float = other.copy()
            self_float.is_float = True
            other_float.is_float = True
            return self._add_float(other_float)

        # Integer addition
        if self.negative == other.negative:
            # Same sign => add
            sum_limb = self._add_chunklists(self.mantissa, other.mantissa)
            sign = self.negative
            out = NumpyMegaNumber(
                mantissa=sum_limb,
                exponent=np.array([0], dtype=np.int16),
                negative=sign
            )
            return out
        else:
            # Opposite sign => subtract smaller from bigger
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)

            if cmp_val == 0:
                # Zero
                return NumpyMegaNumber() # Return default zero
            elif cmp_val > 0:
                diff = self._sub_chunklists(self.mantissa, other.mantissa)
                return NumpyMegaNumber(
                    mantissa=diff,
                    exponent=np.array([0], dtype=np.int16),
                    negative=self.negative
                )
            else: # cmp_val < 0
                diff = self._sub_chunklists(other.mantissa, self.mantissa)
                return NumpyMegaNumber(
                    mantissa=diff,
                    exponent=np.array([0], dtype=np.int16),
                    negative=other.negative
                )

    def sub(self, other: "NumpyMegaNumber") -> "NumpyMegaNumber":
        """
        Subtract other from self.

        Args:
            other: Another NumpyMegaNumber

        Returns:
            Difference as NumpyMegaNumber
        """
        # a - b => a + (-b)
        negB = other.copy()
        negB.negative = not other.negative
        # Handle zero case: -0 is still 0
        if len(negB.mantissa) == 1 and negB.mantissa[0] == 0:
             negB.negative = False
        return self.add(negB)

    def mul(self, other: "NumpyMegaNumber") -> "NumpyMegaNumber":
        """
        Multiply two NumpyMegaNumbers.

        Args:
            other: Another NumpyMegaNumber

        Returns:
            Product as NumpyMegaNumber
        """
        # Handle zero multiplication
        if (len(self.mantissa) == 1 and self.mantissa[0] == 0) or \
           (len(other.mantissa) == 1 and other.mantissa[0] == 0):
            return NumpyMegaNumber() # Return default zero

        # Determine sign
        sign = (self.negative != other.negative)

        # Multiply mantissas
        out_limb = self._mul_chunklists(
            self.mantissa,
            other.mantissa,
            self._global_chunk_size,
            self._base
        )

        if not (self.is_float or other.is_float):
            # Integer multiply
            out = NumpyMegaNumber(
                mantissa=out_limb,
                exponent=np.array([0], dtype=np.int16),
                negative=sign
            )
        else:
            # Float multiply: add exponents
            eA = self._exp_as_int(self)
            eB = self._exp_as_int(other)
            sum_exp = np.add(eA, eB)

            zero = np.array(0, dtype=np.int32) # Use int32
            exp_neg = np.all(np.less(sum_exp, zero))
            sum_exp_abs = np.abs(sum_exp)

            new_exp = self._int_to_chunklist(sum_exp_abs, self._global_chunk_size) if np.any(np.not_equal(sum_exp_abs, zero)) else np.array([0], dtype=np.int16)

            out = NumpyMegaNumber(
                mantissa=out_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )

        out._normalize()
        return out

    def div(self, other: "NumpyMegaNumber") -> "NumpyMegaNumber":
        """
        Divide self by other.

        Args:
            other: Another NumpyMegaNumber

        Returns:
            Quotient as NumpyMegaNumber
        """
        # Check for division by zero
        if len(other.mantissa) == 1 and other.mantissa[0] == 0:
            raise ZeroDivisionError("division by zero")

        # Handle division by self
        if self._compare_abs(self.mantissa, other.mantissa) == 0 and \
           self.is_float == other.is_float and \
           self._compare_abs(self.exponent, other.exponent) == 0 and \
           self.exponent_negative == other.exponent_negative:
            sign = (self.negative != other.negative)
            return NumpyMegaNumber("1" if not sign else "-1")

        # Handle self is zero
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return NumpyMegaNumber() # Return zero

        sign = (self.negative != other.negative)

        if not (self.is_float or other.is_float):
            # Integer division
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)

            if cmp_val < 0:
                # Result = 0
                return NumpyMegaNumber() # Return zero
            # cmp_val == 0 handled above
            else: # cmp_val > 0
                q, _ = self._div_chunk(self.mantissa, other.mantissa)
                out = NumpyMegaNumber(
                    mantissa=q,
                    exponent=np.array([0], dtype=np.int16),
                    negative=sign
                )
                out._normalize()
                return out
        else:
            # Float division - requires careful precision handling
            # This implementation is simplified and might lack precision
            # Convert both to a common high precision representation first?
            # Or implement floating point division directly using limbs.
            # For now, a simplified approach:
            eA = self._exp_as_int(self)
            eB = self._exp_as_int(other)
            newExpVal = np.subtract(eA, eB)

            # Increase precision of A before division
            precision_increase = max(0, len(other.mantissa) * 16) # Heuristic
            mantA_shifted = self._mul_by_2exp(self.mantissa, np.array(precision_increase, dtype=np.int32))
            newExpVal = np.add(newExpVal, np.array(precision_increase, dtype=np.int32)) # Adjust exponent

            q_limb, _ = self._div_chunk(mantA_shifted, other.mantissa)

            zero = np.array(0, dtype=np.int32) # Use int32
            exp_neg = np.all(np.less(newExpVal, zero))
            newExpVal_abs = np.abs(newExpVal)

            new_exp = self._int_to_chunklist(newExpVal_abs, self._global_chunk_size) if np.any(np.not_equal(newExpVal_abs, zero)) else np.array([0], dtype=np.int16)

            out = NumpyMegaNumber(
                mantissa=q_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )
            out._normalize()
            return out


    def _add_float(self, other: "NumpyMegaNumber") -> "NumpyMegaNumber":
        """
        Add two NumpyMegaNumbers in float mode. Aligns exponents before adding.

        Args:
            other: Another NumpyMegaNumber (assumed is_float=True)

        Returns:
            Sum as NumpyMegaNumber (is_float=True)
        """
        eA = self._exp_as_int(self)
        eB = self._exp_as_int(other)

        # Align exponents by shifting the number with the smaller exponent
        if np.all(np.equal(eA, eB)):
            mantA, mantB = self.mantissa, other.mantissa
            final_exp = eA
        elif np.all(np.greater(eA, eB)):
            shift = np.subtract(eA, eB)
            mantA = self.mantissa
            mantB = self._shift_right(other.mantissa, shift)
            final_exp = eA
        else: # eB > eA
            shift = np.subtract(eB, eA)
            mantA = self._shift_right(self.mantissa, shift)
            mantB = other.mantissa
            final_exp = eB

        # Pad mantissas to the same length for addition/subtraction
        lenA, lenB = len(mantA), len(mantB)
        max_len = max(lenA, lenB)
        if lenA < max_len:
            mantA = np.pad(mantA, [(0, max_len - lenA)])
        if lenB < max_len:
            mantB = np.pad(mantB, [(0, max_len - lenB)])


        # Combine signs
        if self.negative == other.negative:
            sum_limb = self._add_chunklists(mantA, mantB)
            sign = self.negative
        else:
            c = self._compare_abs(mantA, mantB)
            if c == 0:
                return NumpyMegaNumber(is_float=True)  # Zero
            elif c > 0:
                sum_limb = self._sub_chunklists(mantA, mantB)
                sign = self.negative
            else: # c < 0
                sum_limb = self._sub_chunklists(mantB, mantA)
                sign = other.negative

        zero = np.array(0, dtype=np.int32) # Use int32
        exp_neg = np.all(np.less(final_exp, zero))
        final_exp_abs = np.abs(final_exp)

        exp_chunk = self._int_to_chunklist(final_exp_abs, self._global_chunk_size) if np.any(np.not_equal(final_exp_abs, zero)) else np.array([0], dtype=np.int16)

        out = NumpyMegaNumber(
            mantissa=sum_limb,
            exponent=exp_chunk,
            negative=sign,
            is_float=True,
            exponent_negative=exp_neg
        )
        out._normalize()
        return out

    def _exp_as_int(self, mn: "NumpyMegaNumber") -> np.ndarray:
        """
        Get exponent as integer (NumPy array).

        Args:
            mn: NumpyMegaNumber

        Returns:
            Exponent as NumPy array (int32)
        """
        # Use int64 for intermediate conversion if exponent can be large
        val = self._chunklist_to_int(mn.exponent)
        return np.negative(val) if mn.exponent_negative else val

    def _shift_right(self, limbs: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """
        Shift limbs right by shift bits.

        Args:
            limbs: NumPy array of limbs
            shift: Number of bits to shift (as NumPy array, int32)

        Returns:
            Shifted limbs
        """
        # Convert to integer (use int64)
        val = self._chunklist_to_int(limbs)

        # Shift right
        val_shifted = np.right_shift(val, shift)

        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)

    def compare_abs(self, other: "NumpyMegaNumber") -> int:
        """
        Compare absolute values.

        Args:
            other: Another NumpyMegaNumber

        Returns:
            1 if self > other, -1 if self < other, 0 if equal
        """
        # TODO: Handle float comparison by aligning exponents first
        if self.is_float or other.is_float:
             # Simplified comparison for floats - align exponents first
             eA = self._exp_as_int(self)
             eB = self._exp_as_int(other)
             if np.all(np.greater(eA, eB)): return 1
             if np.all(np.less(eA, eB)): return -1
             # If exponents are equal, compare mantissas
             return self._compare_abs(self.mantissa, other.mantissa)
        else:
            return self._compare_abs(self.mantissa, other.mantissa)


    @classmethod
    def _compare_abs(cls, A: np.ndarray, B: np.ndarray) -> int:
        """
        Compare absolute values of two NumPy arrays (mantissas).

        Args:
            A: First NumPy array
            B: Second NumPy array

        Returns:
            1 if A > B, -1 if A < B, 0 if equal
        """
        # Trim leading zeros for comparison if necessary (should be handled by normalize)
        # A = A[np.argmax(A != 0):] if np.any(A != 0) else np.array([0], dtype=np.int16)
        # B = B[np.argmax(B != 0):] if np.any(B != 0) else np.array([0], dtype=np.int16)

        lenA, lenB = len(A), len(B)
        if lenA > lenB: return 1
        if lenA < lenB: return -1

        # Compare from most significant limb
        for i in reversed(range(lenA)):
            if A[i] > B[i]: return 1
            if A[i] < B[i]: return -1
        return 0 # Equal

    @classmethod
    def _int_to_chunklist(cls, val: np.ndarray, csize: np.ndarray) -> np.ndarray:
        """
        Convert integer (int32 or int64) to chunk list (int16).

        Args:
            val: Integer as NumPy array (int32 or int64)
            csize: Chunk size (int16)

        Returns:
            NumPy array of chunks (int16)
        """
        # Create mask (use int64 for mask if val is int64)
        one = np.array(1, dtype=val.dtype)
        mask = np.subtract(np.left_shift(one, csize), one)

        out = []
        zero = np.array(0, dtype=val.dtype)

        if np.all(np.equal(val, zero)):
            return np.array([0], dtype=np.int16)

        # Convert to chunks
        current_val = val
        while np.any(np.greater(current_val, zero)):
            chunk = np.bitwise_and(current_val, mask)
            # Ensure chunk fits into int16 before appending
            out.append(chunk.astype(np.int16))
            current_val = np.right_shift(current_val, csize)

        return np.array(out, dtype=np.int16) if out else np.array([0], dtype=np.int16)


    @classmethod
    def _chunklist_to_int(cls, limbs: np.ndarray) -> np.ndarray:
        """
        Combine limbs => integer (int64), little-endian.

        Args:
            limbs: NumPy array of limbs (int16)

        Returns:
            Integer as NumPy array (int64)
        """
        val = np.array(0, dtype=np.int64) # Use int64 for result
        shift = np.array(0, dtype=np.int16)
        csize = cls._global_chunk_size

        for i in range(len(limbs)):
            # Convert limb to int64 before shifting
            limb_int64 = limbs[i].astype(np.int64)
            # Ensure mask is applied correctly if limb was negative in int16
            limb_int64 = np.bitwise_and(limb_int64, np.array(0xFFFF, dtype=np.int64))

            limb_shifted = np.left_shift(limb_int64, shift)
            val = np.add(val, limb_shifted)
            shift = np.add(shift, csize)

        return val

    @classmethod
    def _mul_chunklists(cls, A: np.ndarray, B: np.ndarray, csize: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Multiplication dispatcher: naive / Karatsuba / Toom.

        Args:
            A: First NumPy array of limbs
            B: Second NumPy array of limbs
            csize: Chunk size
            base: Base (2^csize)

        Returns:
            Product as NumPy array of limbs
        """
        # Get lengths
        la, lb = len(A), len(B)
        n = max(la, lb)

        # Choose multiplication algorithm based on size
        # Convert thresholds to Python ints for comparison
        karatsuba_threshold = int(cls._MUL_THRESHOLD_KARATSUBA.item())
        toom_threshold = int(cls._MUL_THRESHOLD_TOOM.item())

        if n < karatsuba_threshold:
            return cls._mul_naive_chunklists(A, B, csize, base)
        # TODO: Implement Karatsuba and Toom-Cook if needed for performance
        # elif n < toom_threshold:
        #     return cls._mul_karatsuba_chunklists(A, B, csize, base)
        else:
            # Fallback to naive for now
             return cls._mul_naive_chunklists(A, B, csize, base)
        #     return cls._mul_toom_chunklists(A, B, csize, base)


    @classmethod
    def _mul_naive_chunklists(cls, A: np.ndarray, B: np.ndarray, csize: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Naive multiplication of chunk lists. Uses int32 for intermediate products.

        Args:
            A: First NumPy array of limbs (int16)
            B: Second NumPy array of limbs (int16)
            csize: Chunk size (int16)
            base: Base (2^csize) (int32)

        Returns:
            Product as NumPy array of limbs (int16)
        """
        la, lb = len(A), len(B)
        # Output can have up to la + lb limbs
        out = np.zeros(la + lb, dtype=np.int32) # Use int32 for intermediate results

        # Ensure A and B are int32 for multiplication
        A_int32 = A.astype(np.int32)
        B_int32 = B.astype(np.int32)
        mask_int32 = cls._mask # Already int32

        for i in range(la):
            carry = np.array(0, dtype=np.int32)
            for j in range(lb):
                # Product of two int16 limbs can exceed int16, use int32
                product = np.multiply(A_int32[i], B_int32[j])
                # Add previous value at out[i+j] and carry
                current_sum = np.add(np.add(product, out[i + j]), carry)

                # Lower 16 bits go into output limb
                out_limb = np.bitwise_and(current_sum, mask_int32)
                # Update output array element
                out[i + j] = out_limb

                # Upper bits form the new carry
                carry = np.right_shift(current_sum, csize)

            # Propagate final carry
            idx = i + lb
            while np.any(np.greater(carry, np.array(0, dtype=np.int32))):
                 if idx >= len(out):
                     # Need to extend output array if carry propagates beyond initial size
                     out = np.pad(out, [(0, 1)]) # Pad with one zero
                 current_sum = np.add(out[idx], carry)
                 out_limb = np.bitwise_and(current_sum, mask_int32)
                 out[idx] = out_limb
                 carry = np.right_shift(current_sum, csize)
                 idx += 1


        # Trim leading zeros from the int32 result before converting back to int16
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return np.array([0], dtype=np.int16) # Result is zero

        # Convert relevant part back to int16
        return out[:first_non_zero + 1].astype(np.int16)


    @classmethod
    def _add_chunklists(cls, A: TensorLike, B: TensorLike) -> np.ndarray:
        """
        Add two chunk lists (int16).

        Args:
            A: TensorLike (will be converted to NumPy array)
            B: TensorLike (will be converted to NumPy array)

        Returns:
            Sum as NumPy array of limbs
        """
        A = np.array(A, ndmin=1) # Convert to NumPy array, ensure at least 1D
        B = np.array(B, ndmin=1) # Convert to NumPy array, ensure at least 1D

        la, lb = len(A), len(B)
        max_len = max(la, lb)

        # Pad shorter array with zeros
        if la < max_len:
            A = np.pad(A, [(0, max_len - la)])
        if lb < max_len:
            B = np.pad(B, [(0, max_len - lb)])

        # Use int32 for intermediate sum to handle carry
        out = np.zeros(max_len + 1, dtype=np.int32)
        carry = np.array(0, dtype=np.int32)
        base = cls._base # Use int32 base
        mask = cls._mask # Already int32 from our previous fix

        for i in range(max_len):
            # Add limbs and carry (use int32)
            s = np.add(np.add(A[i].astype(np.int32), B[i].astype(np.int32)), carry)

            # Lower 16 bits go into output limb
            out_limb = np.bitwise_and(s, mask)
            out[i] = out_limb

            # Upper bits form the new carry
            carry = np.right_shift(s, cls._global_chunk_size)

        # Set final carry if any
        out[max_len] = carry

        # Trim leading zeros from the int32 result
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return np.array([0], dtype=np.int16)

        # Convert relevant part back to int16
        return out[:first_non_zero + 1].astype(np.int16)


    @classmethod
    def _sub_chunklists(cls, A: TensorLike, B: TensorLike) -> np.ndarray:
        """
        Subtract B from A (A >= B). Assumes A and B are int16.

        Args:
            A: TensorLike (will be converted to NumPy array)
            B: TensorLike (will be converted to NumPy array)

        Returns:
            Difference as NumPy array of limbs
        """
        A = np.array(A, ndmin=1) # Convert to NumPy array, ensure at least 1D
        B = np.array(B, ndmin=1) # Convert to NumPy array, ensure at least 1D

        la, lb = len(A), len(B)
        # A is assumed >= B, so len(A) >= len(B)
        max_len = la

        # Pad B if necessary
        if lb < max_len:
            B = np.pad(B, [(0, max_len - lb)])

        # Use int32 for intermediate diff to handle borrow
        out = np.zeros(max_len, dtype=np.int32)
        borrow = np.array(0, dtype=np.int32)
        base = cls._base # Use int32 base
        mask = cls._mask # Already int32 from our previous fix

        for i in range(max_len):
            # Subtract limbs and borrow (use int32)
            diff = np.subtract(np.subtract(A[i].astype(np.int32), B[i].astype(np.int32)), borrow)

            # Check if borrow is needed
            if np.all(np.less(diff, np.array(0, dtype=np.int32))):
                diff = np.add(diff, base)
                borrow = np.array(1, dtype=np.int32)
            else:
                borrow = np.array(0, dtype=np.int32)

            # Lower 16 bits go into output limb
            out_limb = np.bitwise_and(diff, mask)
            out[i] = out_limb

        # Trim leading zeros from the int32 result
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return np.array([0], dtype=np.int16)

        # Convert relevant part back to int16
        return out[:first_non_zero + 1].astype(np.int16)


    @classmethod
    def _div_chunk(cls, A: TensorLike, B: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide A by B (chunk lists). Simplified version.

        Args:
            A: TensorLike (will be converted to NumPy array)
            B: TensorLike (will be converted to NumPy array)

        Returns:
            Tuple of (quotient, remainder) as NumPy arrays of limbs
        """
        A = np.array(A, ndmin=1) # Convert to NumPy array, ensure at least 1D
        B = np.array(B, ndmin=1) # Convert to NumPy array, ensure at least 1D

        # Convert to integers for division (use int64)
        val_A = cls._chunklist_to_int(A)
        val_B = cls._chunklist_to_int(B)

        if np.all(np.equal(val_B, np.array(0, dtype=np.int64))):
             raise ZeroDivisionError("division by zero in _div_chunk")

        # Perform integer division
        quotient_val = np.floor_divide(val_A, val_B)
        remainder_val = np.remainder(val_A, val_B)

        # Convert back to chunks
        quotient_limbs = cls._int_to_chunklist(quotient_val, cls._global_chunk_size)
        remainder_limbs = cls._int_to_chunklist(remainder_val, cls._global_chunk_size)

        return (quotient_limbs, remainder_limbs)


    @classmethod
    def _shiftleft_one_chunk(cls, limbs: np.ndarray) -> np.ndarray:
        """
        Shift limbs left by one chunk (equivalent to multiplying by base).

        Args:
            limbs: NumPy array of limbs

        Returns:
            Shifted limbs
        """
        if len(limbs) == 1 and limbs[0] == 0:
            return np.array([0], dtype=np.int16)
        # Prepend a zero limb
        return np.concatenate([np.array([0], dtype=np.int16), limbs])


    @classmethod
    def _divmod_small(cls, A: TensorLike, small_val: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide A (chunk list) by a small integer value.

        Args:
            A: TensorLike (will be converted to NumPy array)
            small_val: TensorLike (will be converted to NumPy array)

        Returns:
            Tuple of (quotient_limbs, remainder_scalar)
        """
        A = np.array(A, ndmin=1) # Convert to NumPy array, ensure at least 1D
        small_val = np.array(small_val) # Convert to NumPy array

        # Convert A to integer (use int64)
        val_A = cls._chunklist_to_int(A)
        # Convert small_val to int64
        divisor = small_val.astype(np.int64)

        if np.all(np.equal(divisor, np.array(0, dtype=np.int64))):
            raise ZeroDivisionError("division by zero in _divmod_small")

        # Perform division and get remainder
        quotient_val = np.floor_divide(val_A, divisor)
        remainder_val = np.remainder(val_A, divisor)

        # Convert quotient back to chunks
        quotient_limbs = cls._int_to_chunklist(quotient_val, cls._global_chunk_size)

        # Remainder is a scalar, convert back to int16
        remainder_scalar = remainder_val.astype(np.int16)

        return (quotient_limbs, remainder_scalar)


    def copy(self) -> "NumpyMegaNumber":
        """
        Create a copy of this NumpyMegaNumber.

        Returns:
            Copy of this NumpyMegaNumber
        """
        return NumpyMegaNumber(
            mantissa=np.array(self.mantissa), # Ensure copy
            exponent=np.array(self.exponent), # Ensure copy
            negative=self.negative,
            is_float=self.is_float,
            exponent_negative=self.exponent_negative,
            keep_leading_zeros=self._keep_leading_zeros
        )

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            String representation
        """
        # Limit displayed digits for brevity in repr
        return f"<NumpyMegaNumber {self.to_decimal_string(max_digits=50)}>"

# Example usage (optional, for testing)
if __name__ == "__main__":
    a = NumpyMegaNumber("12345678901234567890")
    b = NumpyMegaNumber("98765432109876543210")
    c = NumpyMegaNumber("-123.456")
    d = NumpyMegaNumber("0.000789")

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")

    print(f"a + b = {a.add(b)}")
    print(f"a * b = {a.mul(b)}")
    # print(f"b / a = {b.div(a)}") # Division might be slow/imprecise
    print(f"c + d = {c.add(d)}")
    print(f"c * d = {c.mul(d)}")
    # print(f"c / d = {c.div(d)}")

    # Binary string conversion
    bin_a = NumpyMegaNumber.from_binary_string("1111000011110000")
    print(f"Binary '1111000011110000' = {bin_a}")
    print(f"Decimal of bin_a = {bin_a.to_decimal_string()}")
