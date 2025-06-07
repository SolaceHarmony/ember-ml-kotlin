"""
MLX implementation of MegaNumber, the foundation for BizarroMath arbitrary precision.

This module provides the MLXMegaNumber class, using mlx.array with dtype=int16
as the underlying representation for chunk-based (limb-based) arithmetic.
"""

import mlx.core as mx
from typing import Tuple, Union, List, Optional, Any

class MLXMegaNumber:
    """
    A chunk-based big integer (or float) with HPC-limb arithmetic,
    using MLX arrays with int16 dtype to mimic BigBase65536 logic.
    """

    # Constants as MLX arrays
    _global_chunk_size = mx.array(16, dtype=mx.int16)  # bits per limb
    _base = mx.array(65536, dtype=mx.int32)  # 2^16
    _mask = mx.array(65535, dtype=mx.int16)  # 2^16 - 1

    # Optional thresholds for advanced multiplication
    _MUL_THRESHOLD_KARATSUBA = mx.array(32, dtype=mx.int16)
    _MUL_THRESHOLD_TOOM = mx.array(128, dtype=mx.int16)

    _max_precision_bits = None
    _log2_of_10_cache = None  # class-level for caching log2(10)

    def __init__(
        self,
        value: Union[str, 'MLXMegaNumber', mx.array] = None,
        mantissa: Optional[mx.array] = None,
        exponent: Optional[mx.array] = None,
        negative: bool = False,
        is_float: bool = False,
        exponent_negative: bool = False,
        keep_leading_zeros: bool = False
    ):
        """
        Initialize a HPC-limb object using MLX arrays.

        Args:
            value: Initial value, can be:
                - String (decimal or binary)
                - MLXMegaNumber
                - MLX array of limbs
            mantissa: MLX array of limbs
            exponent: MLX array of limbs
            negative: Sign flag
            is_float: Float flag
            exponent_negative: Exponent sign flag
            keep_leading_zeros: Whether to keep leading zeros
        """
        if mantissa is None:
            mantissa = mx.array([0], dtype=mx.int16)
        if exponent is None:
            exponent = mx.array([0], dtype=mx.int16)

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
            tmp = MLXMegaNumber.from_decimal_string(value)
            self.mantissa = tmp.mantissa
            self.exponent = tmp.exponent
            self.negative = tmp.negative
            self.is_float = tmp.is_float
            self.exponent_negative = tmp.exponent_negative
            self._keep_leading_zeros = keep_leading_zeros
        elif isinstance(value, MLXMegaNumber):
            # Copy
            self.mantissa = mx.array(value.mantissa, dtype=mx.int16)
            self.exponent = mx.array(value.exponent, dtype=mx.int16)
            self.negative = value.negative
            self.is_float = value.is_float
            self.exponent_negative = value.exponent_negative
            self._keep_leading_zeros = value._keep_leading_zeros # Use value's setting
        elif isinstance(value, mx.array):
            # Interpret as mantissa
            self.mantissa = value
            self.exponent = mx.array([0], dtype=mx.int16)
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
                self.exponent = mx.array([0], dtype=mx.int16)
                self.exponent_negative = False
        else:
            # If keep_leading_zeros => only unify if mantissa is all zero
            if mx.all(self.mantissa == 0):
                self.negative = False
                self.exponent_negative = False # Keep exponent value if float

    @classmethod
    def from_decimal_string(cls, dec_str: str) -> "MLXMegaNumber":
        """
        Convert decimal => HPC big-int or HPC float.
        We detect fractional by '.' => if present => treat as float, shifting exponent.

        Args:
            dec_str: Decimal string

        Returns:
            MLXMegaNumber
        """
        s = dec_str.strip()
        if not s:
            return cls(mantissa=mx.array([0], dtype=mx.int16),
                       exponent=mx.array([0], dtype=mx.int16),
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
        mant = mx.array([0], dtype=mx.int16)
        ten = mx.array([10], dtype=mx.int16) # Define ten here

        for ch in s:
            if ch < '0' or ch > '9':
                raise ValueError(f"Invalid digit '{ch}' in decimal string.")

            # Convert digit to MLX array
            digit_val = mx.array(int(ch), dtype=mx.int16)

            # Multiply mant by 10
            mant = cls._mul_chunklists(
                mant,
                ten, # Use defined ten
                cls._global_chunk_size,
                cls._base
            )

            # Add digit using _add_chunklists for simplicity and consistency
            mant = cls._add_chunklists(mant, digit_val)


        exp_limb = mx.array([0], dtype=mx.int16)
        exponent_negative = False
        is_float = False

        # If we had fraction => shift exponent
        if frac_len > 0:
            is_float = True
            exponent_negative = True

            # Approximate: frac_len * log2(10) => bit shift exponent
            # Convert frac_len to MLX array
            frac_len_mx = mx.array(frac_len, dtype=mx.int16)

            # Multiply by log2(10) ≈ 3.32
            # TODO: Use a higher precision log2(10) if needed
            log2_10 = mx.array(3.32192809, dtype=mx.float32) # More precision
            bits_needed_float = mx.multiply(mx.array(frac_len, dtype=mx.float32), log2_10)
            bits_needed = mx.array(mx.ceil(bits_needed_float), dtype=mx.int32) # Use int32 for intermediate

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
    def from_binary_string(cls, bin_str: str) -> "MLXMegaNumber":
        """
        Convert binary string => HPC big-int.

        Args:
            bin_str: Binary string (e.g., "1010" or "0b1010")

        Returns:
            MLXMegaNumber
        """
        s = bin_str.strip()
        if s.startswith('0b'):
            s = s[2:]
        if not s:
            s = "0"

        # Convert to integer
        val = int(s, 2)

        # Convert to MLX array (use int64 for potentially large integers)
        val_mx = mx.array(val, dtype=mx.int64)

        # Convert to limbs
        limbs = cls._int_to_chunklist(val_mx, cls._global_chunk_size)

        return cls(
            mantissa=limbs,
            exponent=mx.array([0], dtype=mx.int16),
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
            tmp = mx.array(self.mantissa)
            digits_rev = []

            zero = mx.array([0], dtype=mx.int16)
            ten = mx.array(10, dtype=mx.int16)

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
                zero = mx.array([0], dtype=mx.int16)
                if self._compare_abs(remainder, zero) == 0:
                    return sign_str + int_str

                # Else => build fractional by repeatedly *10 // 2^exp_int
                frac_digits = []
                steps = max_digits or 50 # Limit fractional digits
                cur_rem = remainder

                ten = mx.array([10], dtype=mx.int16)
                # Precompute 2^exp_int for efficiency
                # Use int64 for potentially large intermediate values
                two_exp = mx.power(mx.array(2, dtype=mx.int64), exp_int)

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
                    frac_digits.append(str(int(digit_val.item()))) # Convert scalar MLX array to Python int

                    if self._compare_abs(cur_rem, zero) == 0:
                        break

                return sign_str + int_str + "." + "".join(frac_digits)
            else:
                # Exponent positive => mantissa << exp_int
                shifted = self._mul_by_2exp(self.mantissa, exp_int)
                return sign_str + self._chunk_to_dec_str(shifted, max_digits)

    def _chunk_to_dec_str(self, chunks: mx.array, max_digits: Optional[int] = None) -> str:
        """
        Convert chunks to decimal string.

        Args:
            chunks: MLX array of chunks
            max_digits: Maximum number of digits

        Returns:
            Decimal string
        """
        # Use to_decimal_string with a temporary MLXMegaNumber
        tmp = MLXMegaNumber(mantissa=chunks, is_float=False)
        return tmp.to_decimal_string(max_digits)

    def _div_by_2exp(self, limbs: mx.array, bits: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Integer division: limbs // 2^bits, remainder = limbs % 2^bits.

        Args:
            limbs: MLX array of limbs
            bits: Number of bits to divide by (as MLX array)

        Returns:
            Tuple of (quotient, remainder)
        """
        zero = mx.array(0, dtype=mx.int32) # Use int32 for comparison
        bits_int = bits # Assume bits is already a scalar array

        if mx.all(mx.less_equal(bits_int, zero)):
             return (mx.array(limbs), mx.array([0], dtype=mx.int16))

        # Convert limbs to integer (use int64 for larger range)
        val_A = self._chunklist_to_int(limbs)

        # Calculate total bits in limbs
        total_bits = mx.multiply(mx.array(len(limbs), dtype=mx.int32), mx.array(16, dtype=mx.int32)) # Use 16 directly

        if mx.all(mx.greater_equal(bits_int, total_bits)):
            # Everything is remainder
            return (mx.array([0], dtype=mx.int16), limbs)

        # Calculate remainder mask (use int64 for mask calculation)
        one = mx.array(1, dtype=mx.int64)
        remainder_mask = mx.subtract(mx.left_shift(one, bits_int), one)

        # Calculate remainder
        remainder_val = mx.bitwise_and(val_A, remainder_mask)

        # Calculate quotient
        quotient_val = mx.right_shift(val_A, bits_int)

        # Convert back to chunks
        quotient_part = self._int_to_chunklist(quotient_val, self._global_chunk_size)
        remainder_part = self._int_to_chunklist(remainder_val, self._global_chunk_size)

        return (quotient_part, remainder_part)

    def _mul_by_2exp(self, limbs: mx.array, bits: mx.array) -> mx.array:
        """
        Multiply by 2^bits.

        Args:
            limbs: MLX array of limbs
            bits: Number of bits to multiply by (as MLX array)

        Returns:
            MLX array of limbs
        """
        zero = mx.array(0, dtype=mx.int32) # Use int32 for comparison
        bits_int = bits # Assume bits is already a scalar array

        if mx.all(mx.less_equal(bits_int, zero)):
            return mx.array(limbs)

        # Convert limbs to integer (use int64)
        val_A = self._chunklist_to_int(limbs)

        # Shift left
        val_shifted = mx.left_shift(val_A, bits_int)

        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)

    def add(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Add two MLXMegaNumbers.

        Args:
            other: Another MLXMegaNumber

        Returns:
            Sum as MLXMegaNumber
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
            out = MLXMegaNumber(
                mantissa=sum_limb,
                exponent=mx.array([0], dtype=mx.int16),
                negative=sign
            )
            return out
        else:
            # Opposite sign => subtract smaller from bigger
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)

            if cmp_val == 0:
                # Zero
                return MLXMegaNumber() # Return default zero
            elif cmp_val > 0:
                diff = self._sub_chunklists(self.mantissa, other.mantissa)
                return MLXMegaNumber(
                    mantissa=diff,
                    exponent=mx.array([0], dtype=mx.int16),
                    negative=self.negative
                )
            else: # cmp_val < 0
                diff = self._sub_chunklists(other.mantissa, self.mantissa)
                return MLXMegaNumber(
                    mantissa=diff,
                    exponent=mx.array([0], dtype=mx.int16),
                    negative=other.negative
                )

    def sub(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Subtract other from self.

        Args:
            other: Another MLXMegaNumber

        Returns:
            Difference as MLXMegaNumber
        """
        # a - b => a + (-b)
        negB = other.copy()
        negB.negative = not other.negative
        # Handle zero case: -0 is still 0
        if len(negB.mantissa) == 1 and negB.mantissa[0] == 0:
             negB.negative = False
        return self.add(negB)

    def mul(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Multiply two MLXMegaNumbers.

        Args:
            other: Another MLXMegaNumber

        Returns:
            Product as MLXMegaNumber
        """
        # Handle zero multiplication
        if (len(self.mantissa) == 1 and self.mantissa[0] == 0) or \
           (len(other.mantissa) == 1 and other.mantissa[0] == 0):
            return MLXMegaNumber() # Return default zero

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
            out = MLXMegaNumber(
                mantissa=out_limb,
                exponent=mx.array([0], dtype=mx.int16),
                negative=sign
            )
        else:
            # Float multiply: add exponents
            eA = self._exp_as_int(self)
            eB = self._exp_as_int(other)
            sum_exp = mx.add(eA, eB)

            zero = mx.array(0, dtype=mx.int32) # Use int32
            exp_neg = mx.all(mx.less(sum_exp, zero))
            sum_exp_abs = mx.abs(sum_exp)

            new_exp = self._int_to_chunklist(sum_exp_abs, self._global_chunk_size) if mx.any(mx.not_equal(sum_exp_abs, zero)) else mx.array([0], dtype=mx.int16)

            out = MLXMegaNumber(
                mantissa=out_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )

        out._normalize()
        return out

    def div(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Divide self by other.

        Args:
            other: Another MLXMegaNumber

        Returns:
            Quotient as MLXMegaNumber
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
            return MLXMegaNumber("1" if not sign else "-1")

        # Handle self is zero
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return MLXMegaNumber() # Return zero

        sign = (self.negative != other.negative)

        if not (self.is_float or other.is_float):
            # Integer division
            cmp_val = self._compare_abs(self.mantissa, other.mantissa)

            if cmp_val < 0:
                # Result = 0
                return MLXMegaNumber() # Return zero
            # cmp_val == 0 handled above
            else: # cmp_val > 0
                q, _ = self._div_chunk(self.mantissa, other.mantissa)
                out = MLXMegaNumber(
                    mantissa=q,
                    exponent=mx.array([0], dtype=mx.int16),
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
            newExpVal = mx.subtract(eA, eB)

            # Increase precision of A before division
            precision_increase = max(0, len(other.mantissa) * 16) # Heuristic
            mantA_shifted = self._mul_by_2exp(self.mantissa, mx.array(precision_increase, dtype=mx.int32))
            newExpVal = mx.add(newExpVal, mx.array(precision_increase, dtype=mx.int32)) # Adjust exponent

            q_limb, _ = self._div_chunk(mantA_shifted, other.mantissa)

            zero = mx.array(0, dtype=mx.int32) # Use int32
            exp_neg = mx.all(mx.less(newExpVal, zero))
            newExpVal_abs = mx.abs(newExpVal)

            new_exp = self._int_to_chunklist(newExpVal_abs, self._global_chunk_size) if mx.any(mx.not_equal(newExpVal_abs, zero)) else mx.array([0], dtype=mx.int16)

            out = MLXMegaNumber(
                mantissa=q_limb,
                exponent=new_exp,
                negative=sign,
                is_float=True,
                exponent_negative=exp_neg
            )
            out._normalize()
            return out


    def _add_float(self, other: "MLXMegaNumber") -> "MLXMegaNumber":
        """
        Add two MLXMegaNumbers in float mode. Aligns exponents before adding.

        Args:
            other: Another MLXMegaNumber (assumed is_float=True)

        Returns:
            Sum as MLXMegaNumber (is_float=True)
        """
        eA = self._exp_as_int(self)
        eB = self._exp_as_int(other)

        # Align exponents by shifting the number with the smaller exponent
        if mx.all(mx.equal(eA, eB)):
            mantA, mantB = self.mantissa, other.mantissa
            final_exp = eA
        elif mx.all(mx.greater(eA, eB)):
            shift = mx.subtract(eA, eB)
            mantA = self.mantissa
            mantB = self._shift_right(other.mantissa, shift)
            final_exp = eA
        else: # eB > eA
            shift = mx.subtract(eB, eA)
            mantA = self._shift_right(self.mantissa, shift)
            mantB = other.mantissa
            final_exp = eB

        # Pad mantissas to the same length for addition/subtraction
        lenA, lenB = len(mantA), len(mantB)
        max_len = max(lenA, lenB)
        if lenA < max_len:
            mantA = mx.pad(mantA, [(0, max_len - lenA)])
        if lenB < max_len:
            mantB = mx.pad(mantB, [(0, max_len - lenB)])


        # Combine signs
        if self.negative == other.negative:
            sum_limb = self._add_chunklists(mantA, mantB)
            sign = self.negative
        else:
            c = self._compare_abs(mantA, mantB)
            if c == 0:
                return MLXMegaNumber(is_float=True)  # Zero
            elif c > 0:
                sum_limb = self._sub_chunklists(mantA, mantB)
                sign = self.negative
            else: # c < 0
                sum_limb = self._sub_chunklists(mantB, mantA)
                sign = other.negative

        zero = mx.array(0, dtype=mx.int32) # Use int32
        exp_neg = mx.all(mx.less(final_exp, zero))
        final_exp_abs = mx.abs(final_exp)

        exp_chunk = self._int_to_chunklist(final_exp_abs, self._global_chunk_size) if mx.any(mx.not_equal(final_exp_abs, zero)) else mx.array([0], dtype=mx.int16)

        out = MLXMegaNumber(
            mantissa=sum_limb,
            exponent=exp_chunk,
            negative=sign,
            is_float=True,
            exponent_negative=exp_neg
        )
        out._normalize()
        return out

    def _exp_as_int(self, mn: "MLXMegaNumber") -> mx.array:
        """
        Get exponent as integer (MLX array).

        Args:
            mn: MLXMegaNumber

        Returns:
            Exponent as MLX array (int32)
        """
        # Use int64 for intermediate conversion if exponent can be large
        val = self._chunklist_to_int(mn.exponent)
        return mx.negative(val) if mn.exponent_negative else val

    def _shift_right(self, limbs: mx.array, shift: mx.array) -> mx.array:
        """
        Shift limbs right by shift bits.

        Args:
            limbs: MLX array of limbs
            shift: Number of bits to shift (as MLX array, int32)

        Returns:
            Shifted limbs
        """
        # Convert to integer (use int64)
        val = self._chunklist_to_int(limbs)

        # Shift right
        val_shifted = mx.right_shift(val, shift)

        # Convert back to chunks
        return self._int_to_chunklist(val_shifted, self._global_chunk_size)

    def compare_abs(self, other: "MLXMegaNumber") -> int:
        """
        Compare absolute values.

        Args:
            other: Another MLXMegaNumber

        Returns:
            1 if self > other, -1 if self < other, 0 if equal
        """
        # TODO: Handle float comparison by aligning exponents first
        if self.is_float or other.is_float:
             # Simplified comparison for floats - align exponents first
             eA = self._exp_as_int(self)
             eB = self._exp_as_int(other)
             if mx.all(mx.greater(eA, eB)): return 1
             if mx.all(mx.less(eA, eB)): return -1
             # If exponents are equal, compare mantissas
             return self._compare_abs(self.mantissa, other.mantissa)
        else:
            return self._compare_abs(self.mantissa, other.mantissa)


    @classmethod
    def _compare_abs(cls, A: mx.array, B: mx.array) -> int:
        """
        Compare absolute values of two MLX arrays (mantissas).

        Args:
            A: First MLX array
            B: Second MLX array

        Returns:
            1 if A > B, -1 if A < B, 0 if equal
        """
        # Trim leading zeros for comparison if necessary (should be handled by normalize)
        # A = A[mx.argmax(A != 0):] if mx.any(A != 0) else mx.array([0], dtype=mx.int16)
        # B = B[mx.argmax(B != 0):] if mx.any(B != 0) else mx.array([0], dtype=mx.int16)

        lenA, lenB = len(A), len(B)
        if lenA > lenB: return 1
        if lenA < lenB: return -1

        # Compare from most significant limb
        for i in reversed(range(lenA)):
            if A[i] > B[i]: return 1
            if A[i] < B[i]: return -1
        return 0 # Equal

    @classmethod
    def _int_to_chunklist(cls, val: mx.array, csize: mx.array) -> mx.array:
        """
        Convert integer (int32 or int64) to chunk list (int16).

        Args:
            val: Integer as MLX array (int32 or int64)
            csize: Chunk size (int16)

        Returns:
            MLX array of chunks (int16)
        """
        # Create mask (use int64 for mask if val is int64)
        one = mx.array(1, dtype=val.dtype)
        mask = mx.subtract(mx.left_shift(one, csize), one)

        out = []
        zero = mx.array(0, dtype=val.dtype)

        if mx.all(mx.equal(val, zero)):
            return mx.array([0], dtype=mx.int16)

        # Convert to chunks
        current_val = val
        while mx.any(mx.greater(current_val, zero)):
            chunk = mx.bitwise_and(current_val, mask)
            # Ensure chunk fits into int16 before appending
            out.append(mx.astype(chunk, mx.int16))
            current_val = mx.right_shift(current_val, csize)

        return mx.array(out, dtype=mx.int16) if out else mx.array([0], dtype=mx.int16)


    @classmethod
    def _chunklist_to_int(cls, limbs: mx.array) -> mx.array:
        """
        Combine limbs => integer (int64), little-endian.

        Args:
            limbs: MLX array of limbs (int16)

        Returns:
            Integer as MLX array (int64)
        """
        val = mx.array(0, dtype=mx.int64) # Use int64 for result
        shift = mx.array(0, dtype=mx.int16)
        csize = cls._global_chunk_size

        for i in range(len(limbs)):
            # Convert limb to int64 before shifting
            limb_int64 = mx.astype(limbs[i], mx.int64)
            # Ensure mask is applied correctly if limb was negative in int16
            limb_int64 = mx.bitwise_and(limb_int64, mx.array(0xFFFF, dtype=mx.int64))

            limb_shifted = mx.left_shift(limb_int64, shift)
            val = mx.add(val, limb_shifted)
            shift = mx.add(shift, csize)

        return val

    @classmethod
    def _mul_chunklists(cls, A: mx.array, B: mx.array, csize: mx.array, base: mx.array) -> mx.array:
        """
        Multiplication dispatcher: naive / Karatsuba / Toom.

        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs
            csize: Chunk size
            base: Base (2^csize)

        Returns:
            Product as MLX array of limbs
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
    def _mul_naive_chunklists(cls, A: mx.array, B: mx.array, csize: mx.array, base: mx.array) -> mx.array:
        """
        Naive multiplication of chunk lists. Uses int32 for intermediate products.

        Args:
            A: First MLX array of limbs (int16)
            B: Second MLX array of limbs (int16)
            csize: Chunk size (int16)
            base: Base (2^csize) (int32)

        Returns:
            Product as MLX array of limbs (int16)
        """
        la, lb = len(A), len(B)
        # Output can have up to la + lb limbs
        out = mx.zeros(la + lb, dtype=mx.int32) # Use int32 for intermediate results

        # Ensure A and B are int32 for multiplication
        A_int32 = mx.astype(A, mx.int32)
        B_int32 = mx.astype(B, mx.int32)
        mask_int32 = mx.astype(cls._mask, mx.int32) # Mask for extracting lower 16 bits

        for i in range(la):
            carry = mx.array(0, dtype=mx.int32)
            for j in range(lb):
                # Product of two int16 limbs can exceed int16, use int32
                product = mx.multiply(A_int32[i], B_int32[j])
                # Add previous value at out[i+j] and carry
                current_sum = mx.add(mx.add(product, out[i + j]), carry)

                # Lower 16 bits go into output limb
                out_limb = mx.bitwise_and(current_sum, mask_int32)
                # Update output array element
                out = out.at[i + j].set(out_limb)

                # Upper bits form the new carry
                carry = mx.right_shift(current_sum, csize)

            # Propagate final carry
            idx = i + lb
            while mx.any(mx.greater(carry, mx.array(0, dtype=mx.int32))):
                 if idx >= len(out):
                     # Need to extend output array if carry propagates beyond initial size
                     out = mx.pad(out, [(0, 1)]) # Pad with one zero
                 current_sum = mx.add(out[idx], carry)
                 out_limb = mx.bitwise_and(current_sum, mask_int32)
                 out = out.at[idx].set(out_limb)
                 carry = mx.right_shift(current_sum, csize)
                 idx += 1


        # Trim leading zeros from the int32 result before converting back to int16
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return mx.array([0], dtype=mx.int16) # Result is zero

        # Convert relevant part back to int16
        return mx.astype(out[:first_non_zero + 1], mx.int16)


    @classmethod
    def _add_chunklists(cls, A: mx.array, B: mx.array) -> mx.array:
        """
        Add two chunk lists (int16).

        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs

        Returns:
            Sum as MLX array of limbs
        """
        la, lb = len(A), len(B)
        max_len = max(la, lb)

        # Pad shorter array with zeros
        if la < max_len:
            A = mx.pad(A, [(0, max_len - la)])
        if lb < max_len:
            B = mx.pad(B, [(0, max_len - lb)])

        # Use int32 for intermediate sum to handle carry
        out = mx.zeros(max_len + 1, dtype=mx.int32)
        carry = mx.array(0, dtype=mx.int32)
        base = cls._base # Use int32 base
        mask = mx.astype(cls._mask, mx.int32) # Use int32 mask

        for i in range(max_len):
            # Add limbs and carry (use int32)
            s = mx.add(mx.add(mx.astype(A[i], mx.int32), mx.astype(B[i], mx.int32)), carry)

            # Lower 16 bits go into output limb
            out_limb = mx.bitwise_and(s, mask)
            out = out.at[i].set(out_limb)

            # Upper bits form the new carry
            carry = mx.right_shift(s, cls._global_chunk_size)

        # Set final carry if any
        out = out.at[max_len].set(carry)

        # Trim leading zeros from the int32 result
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return mx.array([0], dtype=mx.int16)

        # Convert relevant part back to int16
        return mx.astype(out[:first_non_zero + 1], mx.int16)


    @classmethod
    def _sub_chunklists(cls, A: mx.array, B: mx.array) -> mx.array:
        """
        Subtract B from A (A >= B). Assumes A and B are int16.

        Args:
            A: First MLX array of limbs
            B: Second MLX array of limbs

        Returns:
            Difference as MLX array of limbs
        """
        la, lb = len(A), len(B)
        # A is assumed >= B, so len(A) >= len(B)
        max_len = la

        # Pad B if necessary
        if lb < max_len:
            B = mx.pad(B, [(0, max_len - lb)])

        # Use int32 for intermediate diff to handle borrow
        out = mx.zeros(max_len, dtype=mx.int32)
        borrow = mx.array(0, dtype=mx.int32)
        base = cls._base # Use int32 base
        mask = mx.astype(cls._mask, mx.int32) # Use int32 mask

        for i in range(max_len):
            # Subtract limbs and borrow (use int32)
            diff = mx.subtract(mx.subtract(mx.astype(A[i], mx.int32), mx.astype(B[i], mx.int32)), borrow)

            # Check if borrow is needed
            if mx.all(mx.less(diff, mx.array(0, dtype=mx.int32))):
                diff = mx.add(diff, base)
                borrow = mx.array(1, dtype=mx.int32)
            else:
                borrow = mx.array(0, dtype=mx.int32)

            # Lower 16 bits go into output limb
            out_limb = mx.bitwise_and(diff, mask)
            out = out.at[i].set(out_limb)

        # Trim leading zeros from the int32 result
        first_non_zero = -1
        for k in range(len(out) - 1, -1, -1):
            if out[k] != 0:
                first_non_zero = k
                break

        if first_non_zero == -1:
            return mx.array([0], dtype=mx.int16)

        # Convert relevant part back to int16
        return mx.astype(out[:first_non_zero + 1], mx.int16)


    @classmethod
    def _div_chunk(cls, A: mx.array, B: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Divide A by B (chunk lists). Simplified version.

        Args:
            A: Dividend as MLX array of limbs
            B: Divisor as MLX array of limbs

        Returns:
            Tuple of (quotient, remainder) as MLX arrays of limbs
        """
        # Convert to integers for division (use int64)
        val_A = cls._chunklist_to_int(A)
        val_B = cls._chunklist_to_int(B)

        if mx.all(mx.equal(val_B, mx.array(0, dtype=mx.int64))):
             raise ZeroDivisionError("division by zero in _div_chunk")

        # Perform integer division
        quotient_val = mx.floor_divide(val_A, val_B)
        remainder_val = mx.remainder(val_A, val_B)

        # Convert back to chunks
        quotient_limbs = cls._int_to_chunklist(quotient_val, cls._global_chunk_size)
        remainder_limbs = cls._int_to_chunklist(remainder_val, cls._global_chunk_size)

        return (quotient_limbs, remainder_limbs)


    @classmethod
    def _shiftleft_one_chunk(cls, limbs: mx.array) -> mx.array:
        """
        Shift limbs left by one chunk (equivalent to multiplying by base).

        Args:
            limbs: MLX array of limbs

        Returns:
            Shifted limbs
        """
        if len(limbs) == 1 and limbs[0] == 0:
            return mx.array([0], dtype=mx.int16)
        # Prepend a zero limb
        return mx.concatenate([mx.array([0], dtype=mx.int16), limbs])


    @classmethod
    def _divmod_small(cls, A: mx.array, small_val: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Divide A (chunk list) by a small integer value.

        Args:
            A: Dividend as MLX array of limbs
            small_val: Divisor as MLX array (scalar int16)

        Returns:
            Tuple of (quotient_limbs, remainder_scalar)
        """
        # Convert A to integer (use int64)
        val_A = cls._chunklist_to_int(A)
        # Convert small_val to int64
        divisor = mx.astype(small_val, mx.int64)

        if mx.all(mx.equal(divisor, mx.array(0, dtype=mx.int64))):
            raise ZeroDivisionError("division by zero in _divmod_small")

        # Perform division and get remainder
        quotient_val = mx.floor_divide(val_A, divisor)
        remainder_val = mx.remainder(val_A, divisor)

        # Convert quotient back to chunks
        quotient_limbs = cls._int_to_chunklist(quotient_val, cls._global_chunk_size)

        # Remainder is a scalar, convert back to int16
        remainder_scalar = mx.astype(remainder_val, mx.int16)

        return (quotient_limbs, remainder_scalar)


    def copy(self) -> "MLXMegaNumber":
        """
        Create a copy of this MLXMegaNumber.

        Returns:
            Copy of this MLXMegaNumber
        """
        return MLXMegaNumber(
            mantissa=mx.array(self.mantissa), # Ensure copy
            exponent=mx.array(self.exponent), # Ensure copy
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
        return f"<MLXMegaNumber {self.to_decimal_string(max_digits=50)}>"

# Example usage (optional, for testing)
if __name__ == "__main__":
    a = MLXMegaNumber("12345678901234567890")
    b = MLXMegaNumber("98765432109876543210")
    c = MLXMegaNumber("-123.456")
    d = MLXMegaNumber("0.000789")

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
    bin_a = MLXMegaNumber.from_binary_string("1111000011110000")
    print(f"Binary '1111000011110000' = {bin_a}")
    print(f"Decimal of bin_a = {bin_a.to_decimal_string()}")
