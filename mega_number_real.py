#!/usr/bin/env python3
import time
import random
import math
import os
import pickle
import array
from typing import Tuple, Union
def choose_array_type():
    """
    Try to use 64-bit unsigned limbs ('Q').
    If that's not supported (e.g., on 32-bit ARM),
    fall back to 32-bit unsigned limbs ('L').
    """
    try:
        test = array.array('Q', [0])  # see if 'Q' is valid
        return ('Q', 64)
    except (ValueError, OverflowError):
        return ('L', 32)  # fallback to 32-bit

class MegaNumber:
    """
    Chunk-based big-int with decimal I/O (plus optional float exponent).
    Uses array.array(cls._chunk_code) to store limbs, each 64 bits, instead of Python lists.
    """
    # Use the detection logic above:
    _chunk_code, _global_chunk_size = choose_array_type()
    _base = 1 << _global_chunk_size
    _mask = (1 << _global_chunk_size) - 1

    _max_precision_bits = None

    # Thresholds for picking naive vs. Karatsuba vs. Toom-3
    _MUL_THRESHOLD_KARATSUBA = 32
    _MUL_THRESHOLD_TOOM = 128

    def __init__(
            self,
            mantissa: array.array = None,
            exponent: array.array = None,
            negative: bool = False,
            is_float: bool = False,
            exponent_negative: bool = False,
            keep_leading_zeros=False

    ):
        # Default arrays
        if mantissa is None:
            # Instead of array.array(cls._chunk_code ...), do array.array(self._chunk_code, ...)
            mantissa = array.array(self._chunk_code, [0])
        if exponent is None:
            exponent = array.array(self._chunk_code, [0])

        self.mantissa = mantissa
        self.exponent = exponent
        self.negative = negative
        self.is_float = is_float
        self.exponent_negative = exponent_negative
        self._keep_leading_zeros = keep_leading_zeros

        # If user wants typical big-int usage => _normalize() as usual
        # If keep_leading_zeros => skip or partially skip
        self._normalize()

    @classmethod
    def _auto_pick_chunk_size(cls, candidates=None, test_bit_len=1024, trials=10):
        """
        Benchmarks various chunk sizes to pick one that is fastest for multiplication.
        """
        if candidates is None:
            candidates = [8, 16, 32, 64]
        best_csize = None
        best_time = float('inf')
        for csize in candidates:
            t = cls._benchmark_mul(csize, test_bit_len, trials)
            if t < best_time:
                best_time = t
                best_csize = csize
        cls._global_chunk_size = best_csize
        cls._base = 1 << best_csize
        cls._mask = cls._base - 1

    @classmethod
    def _benchmark_mul(cls, csize, bit_len, trials):
        """
        Simple benchmark to measure chunk-based multiplication speed.
        """
        start = time.time()
        base = 1 << csize
        for _ in range(trials):
            A_val = random.getrandbits(bit_len)
            B_val = random.getrandbits(bit_len)
            A_limb = cls._int_to_chunklist(A_val, csize)
            B_limb = cls._int_to_chunklist(B_val, csize)
            for __ in range(3):
                _ = cls._mul_chunklists(A_limb, B_limb, csize, base)
        return time.time() - start

    def _normalize(self):
        """
        If keep_leading_zeros=True => skip removing trailing zero-limbs
        from mantissa (and exponent, if float).
        This ensures HPC-limb usage doesn't lose capacity for wave logic
        or fixed bit-length representation.
        """
        # 1) if user allows trimming => do typical big-int style:
        if not self._keep_leading_zeros:
            # Trim zero-limbs from mantissa, but keep at least one limb
            while len(self.mantissa) > 1 and self.mantissa[-1] == 0:
                self.mantissa.pop()

            # For float usage, might also trim exponent
            if self.is_float:
                while len(self.exponent) > 1 and self.exponent[-1] == 0:
                    self.exponent.pop()

            # If mantissa is all zero => unify sign bits
            if len(self.mantissa) == 1 and self.mantissa[0] == 0:
                self.negative = False
                self.exponent = [0]
                self.exponent_negative = False

        else:
            # 2) keep_leading_zeros=True => do minimal or no trimming
            # we might still unify if the entire mantissa is zero
            # but skip removing partial zero-limbs for HPC wave logic
            if all(x == 0 for x in self.mantissa):
                # if truly zero => unify sign bits, keep same # of limbs
                self.negative = False
                self.exponent_negative = False
                # If you want to preserve entire capacity, skip removing limbs
                # But sometimes you might want to keep exactly the same length.

    @property
    def max_precision_bits(self):
        return self._max_precision_bits

    def _check_precision_limit(self, num: "MegaNumber"):
        """
        Optional check if we exceed a user-defined max bit precision.
        """
        if self._max_precision_bits is not None:
            total_bits = len(num.mantissa) * self._global_chunk_size
            if total_bits > self._max_precision_bits:
                raise ValueError("Precision exceeded!")

    @classmethod
    def dynamic_precision_test(cls, operation='mul', threshold_seconds=2.0, hard_limit=6.0):
        """
        Optional routine to set _max_precision_bits by benchmarking.
        """
        if cls._max_precision_bits is not None:
            return cls._max_precision_bits

        cls._max_precision_bits = 999999  # Simplified
        return cls._max_precision_bits

    @classmethod
    def load_cached_precision(cls, cache_file="precision.pkl"):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cls._max_precision_bits = pickle.load(f)

    @classmethod
    def save_cached_precision(cls, cache_file="precision.pkl"):
        if cls._max_precision_bits is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(cls._max_precision_bits, f)

    # ----------------------------------------------------------------
    #    Constructors
    # ----------------------------------------------------------------
    @classmethod
    def from_decimal_string(cls, dec_str: str) -> "MegaNumber":
        """
        Parse a decimal string => integer or float MegaNumber.
        """
        if cls._global_chunk_size is None:
            cls._chunk_code, cls._global_chunk_size = choose_array_type()
            cls._auto_detect_done = True

        s = dec_str.strip()
        if not s:
            return cls(array.array(cls._chunk_code, [0]), array.array(cls._chunk_code, [0]), negative=False, is_float=False)

        negative = False
        if s.startswith('-'):
            negative = True
            s = s[1:].strip()

        # detect fractional
        point_pos = s.find('.')
        frac_len = 0
        if point_pos >= 0:
            frac_len = len(s) - (point_pos + 1)
            s = s.replace('.', '')

        # Repeated multiply-by-10, add digit
        mant_limb = array.array(cls._chunk_code, [0])
        for ch in s:
            if not ('0' <= ch <= '9'):
                raise ValueError(f"Invalid decimal digit {ch}")

            # Convert digit string to integer
            digit_val = int(ch)

            # multiply by 10
            mant_limb = cls._mul_chunklists(
                mant_limb,
                array.array(cls._chunk_code, [10]),
                cls._global_chunk_size,
                cls._base
            )
            # add digit
            carry = digit_val
            idx = 0
            while carry != 0 or idx < len(mant_limb):
                if idx == len(mant_limb):
                    mant_limb.append(0)
                ssum = mant_limb[idx] + carry
                mant_limb[idx] = ssum & cls._mask
                carry = ssum >> cls._global_chunk_size
                idx += 1

        exp_limb = array.array(cls._chunk_code, [0])
        exponent_negative = False
        is_float = False

        # approximate shift in binary exponent for fractional part
        if frac_len > 0:
            shift_bits = int(math.ceil(frac_len * math.log2(10)))
            exp_limb = cls._int_to_chunklist(shift_bits, cls._global_chunk_size)
            exponent_negative = True
            is_float = True

        obj = cls(
            mantissa=mant_limb,
            exponent=exp_limb,
            negative=negative,
            is_float=is_float,
            exponent_negative=exponent_negative
        )
        obj._normalize()
        if cls == MegaNumber:
            return obj._to_subclass()
        return obj

    def _to_subclass(self) -> "MegaNumber":
        """
        Convert MegaNumber to appropriate subclass based on is_float flag.
        """
        if self.is_float:
            from .mega_float import MegaFloat
            if isinstance(self, MegaFloat):
                return self
            return MegaFloat(
                value=None,  # Avoid string parsing path
                mantissa=self.mantissa[:],
                exponent=self.exponent[:],
                negative=self.negative,
                is_float=True,
                exponent_negative=self.exponent_negative
            )
        else:
            from .mega_integer import MegaInteger
            if isinstance(self, MegaInteger):
                return self
            return MegaInteger(
                value=None,  # Avoid string parsing path
                mantissa=self.mantissa[:],
                exponent=self.exponent[:],
                negative=self.negative,
                is_float=False,
                exponent_negative=self.exponent_negative
            )

    @classmethod
    def from_binary_string(cls, bin_str: str) -> "MegaNumber":
        """
        Parse an unsigned binary string => MegaNumber.
        """
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True

        s = bin_str.strip()
        if not s:
            return cls(array.array(cls._chunk_code, [0]), array.array(cls._chunk_code, [0]), negative=False, is_float=False)

        # Convert binary string to integer
        int_val = int(s, 2)
        mant_limb = cls._int_to_chunklist(int_val, cls._global_chunk_size)

        obj = cls(
            mantissa=mant_limb,
            exponent=array.array(cls._chunk_code, [0]),
            negative=False,
            is_float=False,
            exponent_negative=False
        )
        obj._normalize()
        return obj

    @classmethod
    def from_int(cls, val: int) -> "MegaNumber":
        """
        Convert a Python int => HPC MegaNumber (using array.array(cls._chunk_code)).
        """
        if val == 0:
            return cls(array.array(cls._chunk_code, [0]), array.array(cls._chunk_code, [0]), negative=False)
        negative = (val < 0)
        val_abs = abs(val)
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        limbs = cls._int_to_chunklist(val_abs, cls._global_chunk_size)
        return cls(
            mantissa=limbs,
            exponent=array.array(cls._chunk_code, [0]),
            negative=negative
        )

    # ----------------------------------------------------------------
    #   Output
    # ----------------------------------------------------------------
    def to_decimal_string(self, max_digits=None) -> str:
        """Convert to decimal string (or 'mantissa * 2^exponent' if float)."""
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return "0"

        sign_str = "-" if self.negative else ""
        is_exp_nonzero = (len(self.exponent) > 1 or self.exponent[0] != 0)
        exp_is_neg = self.exponent_negative

        if not is_exp_nonzero and not exp_is_neg:
            # purely integer
            dec_str = self._chunk_to_dec_str(self.mantissa, max_digits)
            return sign_str + dec_str
        else:
            # float => "mant * 2^exponent"
            mant_str = self._chunk_to_dec_str(self.mantissa, max_digits)
            e_val = self._chunklist_to_small_int(self.exponent, self._global_chunk_size)
            if exp_is_neg:
                e_val = -e_val
            return f"{sign_str}{mant_str} * 2^{e_val}"
    @classmethod
    def _chunk_to_dec_str(cls, limbs: array.array, max_digits=None) -> str:
        """
        Convert an array('Q') of limb chunks => decimal string by repeatedly
        calling _divmod_small(..., 10).
        """
        if len(limbs) == 1 and limbs[0] == 0:
            return "0"

        # Make a copy so we can mutate temp
        from copy import copy
        temp = copy(limbs)

        digits = []
        while not (len(temp) == 1 and temp[0] == 0):
            temp, remainder = cls._divmod_small(temp, 10)
            digits.append(str(remainder))

        digits.reverse()
        full_str = "".join(digits)

        if max_digits is None or max_digits >= len(full_str):
            return full_str
        else:
            # truncated
            return f"...{full_str[-max_digits:]}"
    # ----------------------------------------------------------------
    #   Arithmetic
    # ----------------------------------------------------------------
    def add(self, other: "MegaNumber") -> "MegaNumber":
        """
        If float => align exponents. Else do signed integer addition.
        """
        if self.is_float or other.is_float:
            return self._add_float(other)

        # integer mode
        if self.negative == other.negative:
            # same sign => add magnitudes
            sum_limb = self._add_chunklists(self.mantissa, other.mantissa)
            sign = self.negative
            result = MegaNumber(sum_limb, array.array(self._chunk_code, [0]), sign)
        else:
            c = self._compare_abs(self.mantissa, other.mantissa)
            if c == 0:
                return MegaNumber()
            elif c > 0:
                diff = self._sub_chunklists(self.mantissa, other.mantissa)
                result = MegaNumber(diff, array.array(self._chunk_code, [0]), self.negative)
            else:
                diff = self._sub_chunklists(other.mantissa, self.mantissa)
                result = MegaNumber(diff, array.array(self._chunk_code, [0]), other.negative)

        self._check_precision_limit(result)
        return result

    def sub(self, other: "MegaNumber") -> "MegaNumber":
        # a - b => a + (-b)
        neg_other = MegaNumber(
            other.mantissa[:],
            other.exponent[:],
            not other.negative,
            other.is_float,
            other.exponent_negative
        )
        return self.add(neg_other)

    def mul(self, other: "MegaNumber") -> "MegaNumber":
        """
        If float => combine exponents, else do integer multiply.
        """
        if not (self.is_float or other.is_float):
            # integer multiply
            sign = (self.negative != other.negative)
            out_limb = self._mul_chunklists(
                self.mantissa, other.mantissa,
                self._global_chunk_size, self._base
            )
            out = MegaNumber(out_limb, array.array(self._chunk_code, [0]), sign)
            out._normalize()
            return out

        # float multiply
        combined_sign = (self.negative != other.negative)
        expA = self._exp_as_int(self)
        expB = self._exp_as_int(other)
        sum_exp = expA + expB
        out_limb = self._mul_chunklists(
            self.mantissa, other.mantissa,
            self._global_chunk_size, self._base
        )
        exp_neg = (sum_exp < 0)
        sum_exp_abs = abs(sum_exp)
        new_exponent = (
            self._int_to_chunklist(sum_exp_abs, self._global_chunk_size)
            if sum_exp_abs else array.array(self._chunk_code, [0])
        )
        result = MegaNumber(
            out_limb, new_exponent, combined_sign,
            is_float=True, exponent_negative=exp_neg
        )
        result._normalize()
        self._check_precision_limit(result)
        return result

    def div(self, other: "MegaNumber") -> "MegaNumber":
        """
        If float => subtract exponents, else integer divide.
        """
        if not (self.is_float or other.is_float):
            # integer division
            if len(other.mantissa) == 1 and other.mantissa[0] == 0:
                raise ZeroDivisionError("division by zero")

            sign = (self.negative != other.negative)
            c = self._compare_abs(self.mantissa, other.mantissa)
            if c < 0:
                return MegaNumber(array.array(self._chunk_code, [0]), array.array(self._chunk_code, [0]), False)
            elif c == 0:
                return MegaNumber(array.array(self._chunk_code, [1]), array.array(self._chunk_code, [0]), sign)
            else:
                q, _ = self._div_chunk(self.mantissa, other.mantissa)
                out = MegaNumber(q, array.array(self._chunk_code, [0]), sign)
                out._normalize()
                return out

        # float division
        combined_sign = (self.negative != other.negative)
        expA = self._exp_as_int(self)
        expB = self._exp_as_int(other)
        new_exponent_val = expA - expB

        if len(other.mantissa) == 1 and other.mantissa[0] == 0:
            raise ZeroDivisionError("division by zero")

        cmp_val = self._compare_abs(self.mantissa, other.mantissa)
        if cmp_val < 0:
            q_limb = array.array(self._chunk_code, [0])
        elif cmp_val == 0:
            q_limb = array.array(self._chunk_code, [1])
        else:
            q_limb, _ = self._div_chunk(self.mantissa, other.mantissa)

        exp_neg = (new_exponent_val < 0)
        new_exponent_val = abs(new_exponent_val)
        new_exponent = (
            self._int_to_chunklist(new_exponent_val, self._global_chunk_size)
            if new_exponent_val != 0 else array.array(self._chunk_code, [0])
        )

        result = MegaNumber(
            mantissa=q_limb,
            exponent=new_exponent,
            negative=combined_sign,
            is_float=True,
            exponent_negative=exp_neg
        )
        result._normalize()
        self._check_precision_limit(result)
        return result

    def pow(self, exponent: "MegaNumber") -> "MegaNumber":
        """
        Repeated-squaring for exponent>=0 integer.
        """
        if exponent.negative:
            raise NotImplementedError("Negative exponent not supported in pow().")

        # if exponent=0 => return 1
        if len(exponent.mantissa) == 1 and exponent.mantissa[0] == 0:
            return MegaNumber.from_int(1)

        base_copy = self.copy()
        result = MegaNumber.from_int(1)
        e = exponent.copy()

        # We'll do exponentiation by squaring:
        # while e > 0:
        #   if (e % 2)==1 => result *= base_copy
        #   base_copy *= base_copy
        #   e //= 2
        while not (len(e.mantissa) == 1 and e.mantissa[0] == 0):
            # check if e is odd => e.mantissa[0] & 1
            if (e.mantissa[0] & 1) == 1:
                result = result.mul(base_copy)
            # square base
            base_copy = base_copy.mul(base_copy)
            # e >>= 1 => shift exponent by 1 bit
            self._shr1_inplace(e)

        return result

    # ----------------------------------------------------------------
    #   Floats & Shifts
    # ----------------------------------------------------------------
    def _add_float(self, other: "MegaNumber") -> "MegaNumber":
        """
        Minimal float addition logic: align exponents, add mantissas.
        """
        def exp_as_int(mn: MegaNumber):
            return self._exp_as_int(mn)

        expA = exp_as_int(self)
        expB = exp_as_int(other)
        if expA == expB:
            mantA, mantB = self.mantissa, other.mantissa
            final_exp = expA
        elif expA > expB:
            shift = expA - expB
            mantA = self.mantissa
            mantB = self._shift_right(other.mantissa, shift)
            final_exp = expA
        else:
            shift = expB - expA
            mantA = self._shift_right(self.mantissa, shift)
            mantB = other.mantissa
            final_exp = expB

        # combine sign
        if self.negative == other.negative:
            sum_limb = self._add_chunklists(mantA, mantB)
            sign = self.negative
        else:
            c = self._compare_abs(mantA, mantB)
            if c == 0:
                return MegaNumber(is_float=True)
            elif c > 0:
                sum_limb = self._sub_chunklists(mantA, mantB)
                sign = self.negative
            else:
                sum_limb = self._sub_chunklists(mantB, mantA)
                sign = other.negative

        exp_neg = (final_exp < 0)
        final_exp_abs = abs(final_exp)
        exp_chunk = self._int_to_chunklist(final_exp_abs, self._global_chunk_size) if final_exp_abs else array.array(self._chunk_code, [0])

        out = MegaNumber(
            mantissa=sum_limb,
            exponent=exp_chunk,
            negative=sign,
            is_float=True,
            exponent_negative=exp_neg
        )
        out._normalize()
        self._check_precision_limit(out)
        return out

    def _exp_as_int(self, mn: "MegaNumber") -> int:
        val = self._chunklist_to_int(mn.exponent)
        return -val if mn.exponent_negative else val

    def _shr1_inplace(self, x: "MegaNumber"):
        """
        HPC integer right shift by 1 bit in x's mantissa. (No HPC fraction wrappers)
        """
        csize = self._global_chunk_size
        limbs = x.mantissa
        carry = 0
        # same logic as _div2 but in-place
        for i in reversed(range(len(limbs))):
            val = (carry << csize) + limbs[i]
            q = val >> 1
            carry = val & 1
            limbs[i] = q
        # remove trailing zeros
        while len(limbs) > 1 and limbs[-1] == 0:
            limbs.pop()
        # if it becomes zero => unify sign
        if len(limbs) == 1 and limbs[0] == 0:
            x.negative = False

    def sqrt(self) -> "MegaNumber":
        """
        HPC sqrt for integer or float.
        No references to fraction wrappers.
        """
        if self.negative:
            raise ValueError("Cannot sqrt negative.")
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            # Just return 0 (matching sign/is_float).
            return MegaNumber(
                array.array(self._chunk_code, [0]),
                array.array(self._chunk_code, [0]),
                negative=False,
                is_float=self.is_float
            )

        if not self.is_float:
            # integer sqrt
            A = array.array(self._chunk_code, self.mantissa)  # copy mantissa
            low = array.array(self._chunk_code, [0])
            high = array.array(self._chunk_code, A)  # another copy
            csize = self._global_chunk_size
            base = self._base
            while True:
                sum_lh = self._add_chunklists(low, high)
                mid = self._div2(sum_lh)
                c_lo = self._compare_abs(mid, low)
                c_hi = self._compare_abs(mid, high)
                if c_lo == 0 or c_hi == 0:
                    return MegaNumber(mid, array.array(self._chunk_code, [0]), False)
                mid_sqr = self._mul_chunklists(mid, mid, csize, base)
                c_cmp = self._compare_abs(mid_sqr, A)
                if c_cmp == 0:
                    return MegaNumber(mid, array.array(self._chunk_code, [0]), False)
                elif c_cmp < 0:
                    low = mid
                else:
                    high = mid
            else:
                # float sqrt => factor exponent out, do integer sqrt on mantissa, reapply half exponent
                return self._float_sqrt()

    def _float_sqrt(self) -> "MegaNumber":
        """
        HPC float sqrt => factor out exponent's parity, do integer sqrt on mantissa, reapply half exponent.
        """
        def exp_as_int(mn: MegaNumber):
            val = self._chunklist_to_int(mn.exponent)
            return -val if mn.exponent_negative else val

        total_exp = exp_as_int(self)
        remainder = total_exp & 1
        csize = self._global_chunk_size
        base = self._base

        # make a working copy of mantissa as array('Q')
        work_mantissa = array.array(self._chunk_code, self.mantissa)

        # if exponent is odd => multiply or divide by 2
        if remainder != 0:
            if total_exp > 0:
                carry = 0
                for i in range(len(work_mantissa)):
                    doubled = (work_mantissa[i] << 1) + carry
                    work_mantissa[i] = doubled & self._mask
                    carry = doubled >> csize
                if carry != 0:
                    work_mantissa.append(carry)
                total_exp -= 1
            else:
                carry = 0
                for i in reversed(range(len(work_mantissa))):
                    cur_val = (carry << csize) + work_mantissa[i]
                    work_mantissa[i] = cur_val >> 1
                    carry = cur_val & 1
                while len(work_mantissa) > 1 and work_mantissa[-1] == 0:
                    work_mantissa.pop()
                total_exp += 1

        # half of exponent
        half_exp = total_exp // 2
        # do integer sqrt on work_mantissa
        low = array.array(self._chunk_code, [0])
        high = array.array(self._chunk_code, work_mantissa)  # copy
        while True:
            sum_lh = self._add_chunklists(low, high)
            mid = self._div2(sum_lh)
            c_lo = self._compare_abs(mid, low)
            c_hi = self._compare_abs(mid, high)
            if c_lo == 0 or c_hi == 0:
                sqrt_mantissa = mid
                break
            mid_sqr = self._mul_chunklists(mid, mid, csize, base)
            c_cmp = self._compare_abs(mid_sqr, work_mantissa)
            if c_cmp == 0:
                sqrt_mantissa = mid
                break
            elif c_cmp < 0:
                low = mid
            else:
                high = mid

        exp_neg = (half_exp < 0)
        half_abs = abs(half_exp)
        new_exponent = (
            self._int_to_chunklist(half_abs, csize)
            if half_abs else array.array(self._chunk_code, [0])
        )

        out = MegaNumber(
            sqrt_mantissa,
            new_exponent,
            negative=False,
            is_float=True,
            exponent_negative=exp_neg
        )
        out._normalize()
        self._check_precision_limit(out)
        return out
    # ----------------------------------------------------------------
    #   Log/Exp Illustrations (approx)
    # ----------------------------------------------------------------
    def log2(self) -> "MegaNumber":
        """
        Approx binary log2 using Python float.
        """
        if self.negative:
            raise ValueError("Cannot compute log2 of a negative number.")
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            raise ValueError("Cannot compute log2 of zero.")

        # Convert mantissa to float
        mantissa_float = 0.0
        for i, chunk in enumerate(self.mantissa):
            mantissa_float += chunk * (2 ** (i * self._global_chunk_size))

        # Compute log2 using trigonometric equations
        log2_mantissa = math.log2(mantissa_float)

        # Adjust for the exponent
        exponent_val = self._chunklist_to_int(self.exponent)
        if self.exponent_negative:
            exponent_val = -exponent_val

        log2_result = log2_mantissa + exponent_val * self._global_chunk_size
        result_mantissa = self._int_to_chunklist(int(log2_result), self._global_chunk_size)
        out = MegaNumber(mantissa=result_mantissa, exponent=array.array(self._chunk_code, [0]), negative=False, is_float=True)
        out._normalize()
        return out

    def exp2(self) -> "MegaNumber":
        """
        Returns 2^self, approximate for float usage.
        """
        if self.negative:
            raise ValueError("Cannot compute exp2 of a negative number.")
        if len(self.mantissa) == 1 and self.mantissa[0] == 0:
            return MegaNumber(array.array(self._chunk_code, [1]), array.array(self._chunk_code, [0]), negative=False)

        # For small pure integer
        if (not self.is_float) and len(self.exponent) == 1 and self.exponent[0] == 0:
            val_int = 0
            shift = 0
            for chunk in self.mantissa:
                val_int += (chunk << shift)
                shift += self._global_chunk_size
            result = array.array(self._chunk_code, [1 << val_int])
            return MegaNumber(result, array.array(self._chunk_code, [0]), negative=False)

        # Float scenario => partial approximate
        val = self._chunklist_to_small_int(self.mantissa, self._global_chunk_size)
        if self.exponent_negative:
            val = -val
        if self.is_float:
            # fractional
            frac_val = val - int(val)
            int_val = int(val)

            int_part = array.array(self._chunk_code)
            if int_val >= 0:
                int_part.append(1 << int_val)
            else:
                int_part.append(1 >> -int_val)

            frac_part = self._exp2_frac(frac_val)
            product = self._mul_chunklists(int_part, frac_part, self._global_chunk_size, self._base)
            return MegaNumber(product, array.array(self._chunk_code, [0]), negative=False)

        # fallback
        product = array.array(self._chunk_code)
        product.append(1 << val)
        return MegaNumber(product, array.array(self._chunk_code, [0]), negative=False)

    def _exp2_frac(self, x: float) -> array.array:
        """
        Rudimentary Taylor for 2^x, 0<=x<1, storing result as array('Q').
        """
        terms = 10
        result = array.array(self._chunk_code, [1])  # first term
        factorial = 1.0
        power = 1.0
        ln2 = 0.693147

        for i in range(1, terms):
            power *= x * ln2
            factorial *= i
            # scale factor: 1<<csize
            term_val = int(power / factorial * (1 << self._global_chunk_size))
            # add in chunk-based
            result = self._add_chunklists(result, array.array(self._chunk_code, [term_val]))

        return result

    def exp(self) -> "MegaNumber":
        """
        Return exponent as MegaNumber (for a float), ignoring mantissa.
        """
        if len(self.exponent) == 1 and self.exponent[0] == 0:
            return MegaNumber(array.array(self._chunk_code, [0]), array.array(self._chunk_code, [0]))
        # Convert exponent-limbs => new MegaNumber
        out = MegaNumber(
            mantissa=array.array(self._chunk_code, self.exponent[:]),
            exponent=array.array(self._chunk_code, [0]),
            negative=self.exponent_negative,
            is_float=False
        )
        out._normalize()
        return out

    # ----------------------------------------------------------------
    #   SHIFT / SLICE helpers
    # ----------------------------------------------------------------
    def _shift_right(self, limbs: array.array, shift_bits: int) -> array.array:
        """
        Return a copy of 'limbs' right-shifted by shift_bits bits (approx).
        This is a naive approach.
        """
        if shift_bits <= 0:
            return array.array(self._chunk_code, limbs)
        # integer-based approach: // 2^shift_bits
        # We'll do repeated _shr1 if needed, or just do a single pass
        # for brevity, do repeated 1-bit shifts:
        tmp = array.array(self._chunk_code, limbs)
        for _ in range(shift_bits):
            carry = 0
            for i in reversed(range(len(tmp))):
                val = (carry << self._global_chunk_size) + tmp[i]
                tmp[i] = val >> 1
                carry = val & 1
            # trim
            while len(tmp) > 1 and tmp[-1] == 0:
                tmp.pop()
            if len(tmp) == 1 and tmp[0] == 0:
                break
        return tmp
    @classmethod
    def _shiftleft_one_chunk(cls, limbs: array.array) -> array.array:
        """
        Shift 'limbs' left by cls._global_chunk_size bits (i.e. multiply by 2^(chunk_size)),
        without storing 2^chunk_size in a single limb.

        We do this by inserting a zero limb at the 'front'.
        If chunk_size=64, that's effectively the same as * 2^64.
        """
        # Make an output array bigger by one limb
        out = array.array(cls._chunk_code, [0]*(len(limbs)+1))
        # Copy each limb up by one index
        for i in range(len(limbs)):
            out[i+1] = limbs[i]
        # Trim if you end up with trailing zero
        while len(out)>1 and out[-1] == 0:
            out.pop()
        return out
    # ----------------------------------------------------------------
    #   CHUNK UTILS
    # ----------------------------------------------------------------
    @classmethod
    def _int_to_chunklist(cls, val: int, csize: int) -> array.array:
        out = array.array(cls._chunk_code)
        if val == 0:
            out.append(0)
            return out
        while val > 0:
            out.append(val & ((1 << csize) - 1))
            val >>= csize
        return out

    @classmethod
    def _chunklist_to_small_int(cls, limbs: array.array, csize: int) -> float:
        """
        Return a float if the code sometimes does fractional logic.
        For pure integer usage, you might want an int, but we keep float
        to handle partial exponent code gracefully.
        """
        val = 0.0
        shift = 0
        for limb in limbs:
            val += (limb<<shift) # TODO - switch to MegaFloat val += float(limb) * (2 ** shift)
            shift += csize
        return val

    @classmethod
    def _chunklist_to_int(cls, limbs: array.array) -> int:
        """Combine chunk-limbs => a Python int."""
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        val = 0
        shift = 0
        for limb in limbs:
            val += limb << shift
            shift += cls._global_chunk_size
        return val

    @classmethod
    def _compare_abs(cls, A: array.array, B: array.array) -> int:
        """
        Compare absolute magnitude of two HPC-limb arrays A vs. B.
        Returns:
            -1 if A < B,
             0 if A == B,
             1 if A > B.
        """
        if len(A) > len(B):
            return 1
        if len(B) > len(A):
            return -1

        # Compare from the highest limb down
        for i in range(len(A) - 1, -1, -1):
            if A[i] > B[i]:
                return 1
            elif A[i] < B[i]:
                return -1
        return 0

    def compare_abs(self, other: "MegaNumber") -> int:
        """
        Compare absolute magnitude of self vs. other as HPC-limb objects.
        Internally calls the class method _compare_abs on both mantissas.

        Example usage:
            if self.compare_abs(other) > 0:
                ...
        """
        return type(self)._compare_abs(self.mantissa, other.mantissa)

    @classmethod
    def _mul_chunklists(cls, A: array.array, B: array.array, csize: int, base: int) -> array.array:
        """Dispatch: naive vs. Karatsuba vs. Toom, if implemented."""
        la, lb = len(A), len(B)
        n = max(la, lb)
        if n < cls._MUL_THRESHOLD_KARATSUBA:
            return cls._mul_naive_chunklists(A, B, csize, base)
        elif n < cls._MUL_THRESHOLD_TOOM:
            return cls._mul_karatsuba_chunklists(A, B, csize, base)
        else:
            return cls._mul_toom_chunklists(A, B, csize, base)
    @classmethod
    def _mul_naive_chunklists(cls, A: array.array, B: array.array, csize: int, base: int) -> array.array:
        # Implementation of naive O(n^2) multiply

        la, lb = len(A), len(B)
        out = array.array(cls._chunk_code, [0]*(la + lb))
        for i in range(la):
            carry = 0
            av = A[i]
            for j in range(lb):
                mul_val = av * B[j] + out[i+j] + carry
                out[i+j] = mul_val & (base-1)
                carry = mul_val >> csize
            if carry:
                out[i+lb] += carry
        # trim
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _mul_karatsuba_chunklists(cls, A: array.array, B: array.array, csize: int, base: int) -> array.array:
        """Placeholder Karatsuba: fallback to naive."""
        # TODO - implement Karatsuba
        return cls._mul_naive_chunklists(A, B, csize, base)

    @classmethod
    def _mul_toom_chunklists(cls, A: array.array, B: array.array, csize: int, base: int) -> array.array:
        """Placeholder Toom-3: fallback to naive."""
        # TODO - implement Toom-3
        return cls._mul_naive_chunklists(A, B, csize, base)

    @classmethod
    def _div_chunk(cls, A: array.array, B: array.array) -> Tuple[array.array, array.array]:
        """
        Long-division => (Q,R), chunk-based.
        """
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        if len(B) == 1 and B[0] == 0:
            raise ZeroDivisionError("divide by zero")

        c = cls._compare_abs(A, B)
        if c < 0:
            return (array.array(cls._chunk_code, [0]), A)
        if c == 0:
            return (array.array(cls._chunk_code, [1]), array.array(cls._chunk_code, [0]))

        Q = array.array(cls._chunk_code, [0]*len(A))
        R = array.array(cls._chunk_code, [0])
        base = 1 << cls._global_chunk_size

        for i in range(len(A)-1, -1, -1):
            R = cls._shiftleft_one_chunk(R)
            R = cls._add_chunklists(R, array.array(cls._chunk_code, [A[i]]))
            low, high = 0, base-1
            guess = 0
            while low <= high:
                mid = (low + high) >> 1
                mm = cls._mul_chunklists(B, array.array(cls._chunk_code, [mid]), cls._global_chunk_size, base)
                cmpv = cls._compare_abs(mm, R)
                if cmpv <= 0:
                    guess = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if guess != 0:
                mm = cls._mul_chunklists(B, array.array(cls._chunk_code, [guess]), cls._global_chunk_size, base)
                R = cls._sub_chunklists(R, mm)
            Q[i] = guess

        while len(Q) > 1 and Q[-1] == 0:
            Q.pop()
        while len(R) > 1 and R[-1] == 0:
            R.pop()
        return (Q, R)

    @classmethod
    def _divmod_small(cls, A: array.array, small_val: int) -> Tuple[array.array, int]:
        """
        Divmod by small_val => (quotient, remainder).
        """
        remainder = 0
        out = array.array(cls._chunk_code, A)  # make a copy
        csize = cls._global_chunk_size
        for i in reversed(range(len(out))):
            cur = (remainder << csize) + out[i]
            qd = cur // small_val
            remainder = cur % small_val
            out[i] = qd & cls._mask
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return (out, remainder)

    @classmethod
    def _add_chunklists(cls, A: array.array, B: array.array) -> array.array:
        if cls._global_chunk_size is None:
            cls._auto_pick_chunk_size()
            cls._auto_detect_done = True
        la, lb = len(A), len(B)
        max_len = max(la, lb) # TODO: handle overflow beyond _max_precision_bits more gracefully
        out = array.array(cls._chunk_code)
        carry = 0
        for i in range(max_len):
            av = A[i] if i < la else 0
            bv = B[i] if i < lb else 0
            s = av + bv + carry
            carry = s >> cls._global_chunk_size
            out.append(s & cls._mask)
        if carry:
            out.append(carry)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _sub_chunklists(cls, A: array.array, B: array.array) -> array.array:
        """
        Subtract B from A, assuming A >= B in absolute magnitude.
        """
        la, lb = len(A), len(B)
        max_len = max(la, lb)
        out = array.array(cls._chunk_code)
        carry = 0
        for i in range(max_len):
            av = A[i] if i < la else 0
            bv = B[i] if i < lb else 0
            diff = av - bv - carry
            if diff < 0:
                diff += cls._base
                carry = 1
            else:
                carry = 0
            out.append(diff & cls._mask)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    @classmethod
    def _div2(cls, limbs: array.array) -> array.array:
        """
        Right shift chunk-limbs by 1 bit => integer //2.
        """
        out = array.array(cls._chunk_code)
        carry = 0
        csize = cls._global_chunk_size
        for i in reversed(range(len(limbs))):
            val = (carry << csize) + limbs[i]
            q = val >> 1
            carry = val & 1
            out.append(q)
        out.reverse()
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out

    # ----------------------------------------------------------------
    #   Copy & Repr
    # ----------------------------------------------------------------
    def copy(self) -> "MegaNumber":
        """Create a copy with the same type/flags."""
        obj = type(self)(
            mantissa=array.array(self._chunk_code, self.mantissa),
            exponent=array.array(self._chunk_code, self.exponent),
            negative=self.negative,
            is_float=self.is_float,
            exponent_negative=self.exponent_negative
        )
        obj._normalize()
        return obj

    def __repr__(self):
        return f"<MegaNumber {self.to_decimal_string(50)}>"
