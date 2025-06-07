# hhtest/binary_ops_test.py

"""
Tests and implementations for binary wave operations based on the Hodgkin-Huxley
formalism document (docs/architecture/binary_hh_model.md), using Ember ML
abstractions directly.
"""

from typing import Union

import math

# --- Fixed-Point Representation (Q-Format) ---

# Let's choose Q7.8 format (1 sign, 7 integer, 8 fractional bits = 16 bits total)
# Alternatively, we could use standard integer types like int16/int32 and track
# the scaling factor separately. Q-format embeds the scaling.
TOTAL_BITS = 16
FRACTIONAL_BITS = 8
INTEGER_BITS = TOTAL_BITS - FRACTIONAL_BITS - 1 # Subtract 1 for sign bit
SCALE_FACTOR = 1 << FRACTIONAL_BITS # 2^FRACTIONAL_BITS
MIN_Q_VALUE = -(1 << (TOTAL_BITS - 1))
MAX_Q_VALUE = (1 << (TOTAL_BITS - 1)) - 1

def float_to_q(value: float, fractional_bits: int = FRACTIONAL_BITS) -> int:
    """Converts a float to its Q-format integer representation."""
    scale = 1 << fractional_bits
    scaled_value = round(value * scale)
    # Clamping to handle overflow/underflow for the chosen bit width
    total_bits = INTEGER_BITS + fractional_bits + 1
    min_val = -(1 << (total_bits - 1))
    max_val = (1 << (total_bits - 1)) - 1
    return max(min_val, min(max_val, scaled_value))

def q_to_float(q_value: int, fractional_bits: int = FRACTIONAL_BITS) -> float:
    """Converts a Q-format integer representation back to a float."""
    scale = 1 << fractional_bits
    return float(q_value) / scale


from ember_ml import ops
from ember_ml.ops import bitwise
from ember_ml.nn import tensor

# Define type alias for clarity
BinaryWave = tensor.EmberTensor

# --- Auxiliary Binary Functions (Placeholders/To Be Implemented) ---

# Updated bin_mult implementation
def bin_mult(a: BinaryWave, b: BinaryWave, fractional_bits: int = FRACTIONAL_BITS) -> BinaryWave:
    """
    Performs fixed-point multiplication (BIN_MULT) using integer ops.
    Assumes a and b are EmberTensors representing Q-format numbers.
    The result is scaled back to the original Q-format.
    """
    # Ensure inputs are tensors (though function signature implies they are)
    # Convert shift amount to tensor
    shift_tensor = tensor.convert_to_tensor(fractional_bits, dtype=tensor.int32) # Match dtype used in shift ops

    # Perform integer multiplication. Result has 2*fractional_bits scaling.
    # We might need to cast to a wider type temporarily if intermediate overflow is possible
    # e.g., int16 * int16 -> int32
    temp_result = ops.multiply(tensor.cast(a, tensor.int32), tensor.cast(b, tensor.int32))

    # Scale back by right-shifting by fractional_bits
    # This effectively divides by 2^fractional_bits
    scaled_result = bitwise.right_shift(temp_result, shift_tensor)

    # Cast back to original type (e.g., int16 for Q7.8)
    # Note: This cast might truncate/overflow if the result exceeds int16 range
    # Proper handling might require saturation logic depending on requirements.
    final_result = tensor.cast(scaled_result, a.dtype)

    return final_result

# Updated bin_pow implementation
def bin_pow(a: BinaryWave, p: int, fractional_bits: int = FRACTIONAL_BITS) -> BinaryWave:
    """
    Calculates the integer power p of a fixed-point number a (BIN_POW).
    Uses repeated binary multiplication (bin_mult).
    """
    if not isinstance(p, int) or p < 0:
        raise ValueError("Power must be a non-negative integer.")

    # Determine the Q-format representation of '1'
    # 1.0 * 2^fractional_bits
    one_q = tensor.convert_to_tensor(1 << fractional_bits, dtype=a.dtype)

    if p == 0:
        # Return 1 in the correct Q-format
        return one_q

    # Start with the base
    result = a
    # Multiply p-1 times
    for _ in range(p - 1):
        result = bin_mult(result, a, fractional_bits=fractional_bits)

    return result


# Standard HH Rate Equations (for generating test LUT data)
# Based on typical squid axon parameters at 6.3°C
def alpha_m(V_mV): return 0.1 * (V_mV + 40.0) / (1.0 - ops.exp(-(V_mV + 40.0) / 10.0))
def beta_m(V_mV):  return 4.0 * ops.exp(-(V_mV + 65.0) / 18.0)
def alpha_h(V_mV): return 0.07 * ops.exp(-(V_mV + 65.0) / 20.0)
def beta_h(V_mV):  return 1.0 / (1.0 + ops.exp(-(V_mV + 35.0) / 10.0))
def alpha_n(V_mV): return 0.01 * (V_mV + 55.0) / (1.0 - ops.exp(-(V_mV + 55.0) / 10.0))
def beta_n(V_mV):  return 0.125 * ops.exp(-(V_mV + 65.0) / 80.0)

# Safe versions of rate functions for LUT generation
# Safe versions of rate functions for LUT generation using ops abstraction
def safe_alpha_m(V_mV_t):
    """Safe implementation of alpha_m using ops abstraction."""
    # Constants as tensors
    const_40 = tensor.convert_to_tensor(40.0, dtype=V_mV_t.dtype)
    const_10 = tensor.convert_to_tensor(10.0, dtype=V_mV_t.dtype)
    const_01 = tensor.convert_to_tensor(0.1, dtype=V_mV_t.dtype)
    const_1 = tensor.convert_to_tensor(1.0, dtype=V_mV_t.dtype)
    epsilon = tensor.convert_to_tensor(1e-6, dtype=V_mV_t.dtype)
    
    # V_mV + 40.0
    V_plus_40 = ops.add(V_mV_t, const_40)
    
    # |V_mV + 40.0| < 1e-6
    abs_V_plus_40 = ops.abs(V_plus_40)
    is_close_to_zero = ops.less(abs_V_plus_40, epsilon)
    
    # -(V_mV + 40.0) / 10.0
    neg_V_plus_40_div_10 = ops.divide(ops.negative(V_plus_40), const_10)
    
    # exp(-(V_mV + 40.0) / 10.0)
    exp_term = ops.exp(neg_V_plus_40_div_10)
    
    # 1.0 - exp(-(V_mV + 40.0) / 10.0)
    one_minus_exp = ops.subtract(const_1, exp_term)
    
    # 0.1 * (V_mV + 40.0) / (1.0 - exp(-(V_mV + 40.0) / 10.0))
    safe_denominator = ops.where(
        ops.equal(one_minus_exp, tensor.convert_to_tensor(0.0, dtype=V_mV_t.dtype)),
        tensor.convert_to_tensor(1.0, dtype=V_mV_t.dtype),  # Avoid division by zero
        one_minus_exp
    )
    
    normal_result = ops.divide(ops.multiply(const_01, V_plus_40), safe_denominator)
    
    # Return 1.0 for values where |V_mV + 40.0| < 1e-6
    return ops.where(is_close_to_zero, const_1, normal_result)

def safe_alpha_n(V_mV_t):
    """Safe implementation of alpha_n using ops abstraction."""
    # Constants as tensors
    const_55 = tensor.convert_to_tensor(55.0, dtype=V_mV_t.dtype)
    const_10 = tensor.convert_to_tensor(10.0, dtype=V_mV_t.dtype)
    const_001 = tensor.convert_to_tensor(0.01, dtype=V_mV_t.dtype)
    const_1 = tensor.convert_to_tensor(1.0, dtype=V_mV_t.dtype)
    const_01 = tensor.convert_to_tensor(0.1, dtype=V_mV_t.dtype)
    epsilon = tensor.convert_to_tensor(1e-6, dtype=V_mV_t.dtype)
    
    # V_mV + 55.0
    V_plus_55 = ops.add(V_mV_t, const_55)
    
    # |V_mV + 55.0| < 1e-6
    abs_V_plus_55 = ops.abs(V_plus_55)
    is_close_to_zero = ops.less(abs_V_plus_55, epsilon)
    
    # -(V_mV + 55.0) / 10.0
    neg_V_plus_55_div_10 = ops.divide(ops.negative(V_plus_55), const_10)
    
    # exp(-(V_mV + 55.0) / 10.0)
    exp_term = ops.exp(neg_V_plus_55_div_10)
    
    # 1.0 - exp(-(V_mV + 55.0) / 10.0)
    one_minus_exp = ops.subtract(const_1, exp_term)
    
    # 0.01 * (V_mV + 55.0) / (1.0 - exp(-(V_mV + 55.0) / 10.0))
    safe_denominator = ops.where(
        ops.equal(one_minus_exp, tensor.convert_to_tensor(0.0, dtype=V_mV_t.dtype)),
        tensor.convert_to_tensor(1.0, dtype=V_mV_t.dtype),  # Avoid division by zero
        one_minus_exp
    )
    
    normal_result = ops.divide(ops.multiply(const_001, V_plus_55), safe_denominator)
    
    # Return 0.1 for values where |V_mV + 55.0| < 1e-6
    return ops.where(is_close_to_zero, const_01, normal_result)

def safe_beta_m(V_mV_t):
    """Safe implementation of beta_m using ops abstraction."""
    # Constants as tensors
    const_4 = tensor.convert_to_tensor(4.0, dtype=V_mV_t.dtype)
    const_65 = tensor.convert_to_tensor(65.0, dtype=V_mV_t.dtype)
    const_18 = tensor.convert_to_tensor(18.0, dtype=V_mV_t.dtype)
    
    # V_mV + 65.0
    V_plus_65 = ops.add(V_mV_t, const_65)
    
    # -(V_mV + 65.0) / 18.0
    neg_V_plus_65_div_18 = ops.divide(ops.negative(V_plus_65), const_18)
    
    # 4.0 * exp(-(V_mV + 65.0) / 18.0)
    return ops.multiply(const_4, ops.exp(neg_V_plus_65_div_18))

def safe_alpha_h(V_mV_t):
    """Safe implementation of alpha_h using ops abstraction."""
    # Constants as tensors
    const_007 = tensor.convert_to_tensor(0.07, dtype=V_mV_t.dtype)
    const_65 = tensor.convert_to_tensor(65.0, dtype=V_mV_t.dtype)
    const_20 = tensor.convert_to_tensor(20.0, dtype=V_mV_t.dtype)
    
    # V_mV + 65.0
    V_plus_65 = ops.add(V_mV_t, const_65)
    
    # -(V_mV + 65.0) / 20.0
    neg_V_plus_65_div_20 = ops.divide(ops.negative(V_plus_65), const_20)
    
    # 0.07 * exp(-(V_mV + 65.0) / 20.0)
    return ops.multiply(const_007, ops.exp(neg_V_plus_65_div_20))

def safe_beta_h(V_mV_t):
    """Safe implementation of beta_h using ops abstraction."""
    # Constants as tensors
    const_1 = tensor.convert_to_tensor(1.0, dtype=V_mV_t.dtype)
    const_35 = tensor.convert_to_tensor(35.0, dtype=V_mV_t.dtype)
    const_10 = tensor.convert_to_tensor(10.0, dtype=V_mV_t.dtype)
    
    # V_mV + 35.0
    V_plus_35 = ops.add(V_mV_t, const_35)
    
    # -(V_mV + 35.0) / 10.0
    neg_V_plus_35_div_10 = ops.divide(ops.negative(V_plus_35), const_10)
    
    # exp(-(V_mV + 35.0) / 10.0)
    exp_term = ops.exp(neg_V_plus_35_div_10)
    
    # 1.0 + exp(-(V_mV + 35.0) / 10.0)
    one_plus_exp = ops.add(const_1, exp_term)
    
    # 1.0 / (1.0 + exp(-(V_mV + 35.0) / 10.0))
    return ops.divide(const_1, one_plus_exp)

def safe_beta_n(V_mV_t):
    """Safe implementation of beta_n using ops abstraction."""
    # Constants as tensors
    const_0125 = tensor.convert_to_tensor(0.125, dtype=V_mV_t.dtype)
    const_65 = tensor.convert_to_tensor(65.0, dtype=V_mV_t.dtype)
    const_80 = tensor.convert_to_tensor(80.0, dtype=V_mV_t.dtype)
    
    # V_mV + 65.0
    V_plus_65 = ops.add(V_mV_t, const_65)
    
    # -(V_mV + 65.0) / 80.0
    neg_V_plus_65_div_80 = ops.divide(ops.negative(V_plus_65), const_80)
    
    # 0.125 * exp(-(V_mV + 65.0) / 80.0)
    return ops.multiply(const_0125, ops.exp(neg_V_plus_65_div_80))

# Placeholder LUT Generation (Illustrative - needs proper implementation)
def generate_lut(rate_func, v_min_mv, v_max_mv, num_entries, fractional_bits=FRACTIONAL_BITS):
    # Generate voltage points (float)
    voltages_mv = tensor.linspace(v_min_mv, v_max_mv, num_entries)
    # Calculate rates (float)
    # Calculate rates (float) using the rate function
    rates_float = rate_func(voltages_mv)
    
    # We need to convert each element of rates_float to Q format
    # Since we can't directly iterate over tensor elements, we'll use a map-like approach
    
    # Create a function to convert a single float value to Q format
    def convert_to_q_format(idx):
        # Extract a single element using tensor.gather
        single_value = tensor.gather(rates_float, tensor.convert_to_tensor(idx, dtype=tensor.int32))
        # Convert to Python float (this is unavoidable for the Q-format conversion)
        float_val = tensor.to_numpy(single_value).item()
        # Convert to Q format
        q_val = float_to_q(float_val, fractional_bits)
        return q_val
    
    # Apply the conversion to each element
    num_elements = rates_float.shape[0]
    rates_q_list = [convert_to_q_format(i) for i in range(num_elements)]
    rates_q = tensor.convert_to_tensor(rates_q_list, dtype=tensor.int16)
    # Store voltage step info for indexing
    v_step_mv = (v_max_mv - v_min_mv) / (num_entries - 1)
    v_min_q = float_to_q(v_min_mv, fractional_bits)
    v_step_q = float_to_q(v_step_mv, fractional_bits) # Approximate step in Q format
    # For simpler indexing, we might store v_min_q and calculate step dynamically
    # Or precompute Q voltages corresponding to indices
    return rates_q, float_to_q(v_min_mv), float_to_q(v_max_mv), num_entries # Return Q tensor and range info

# Global LUT dictionary (populated by a setup function later)
HH_LUTS = {}
LUT_PARAMS = {} # To store range, steps etc.

# Updated bin_rate implementation (Basic Lookup, No Interpolation)
def bin_rate(lut_key: str, v_q: BinaryWave, fractional_bits: int = FRACTIONAL_BITS) -> BinaryWave:
    """
    Looks up rate constants from pre-computed LUTs (BIN_RATE).
    Enhanced version with linear interpolation for improved accuracy.
    Assumes v_q is a Q-format tensor.
    """
    if lut_key not in HH_LUTS:
        raise ValueError(f"LUT for '{lut_key}' not found. Generate LUTs first.")

    lut_tensor = HH_LUTS[lut_key]
    v_min_q, v_max_q, num_entries = LUT_PARAMS[lut_key]

    # Calculate approximate index based on voltage range
    # 1. Scale voltage to range [0, 1] relative to LUT min/max Q values
    v_range_q = v_max_q - v_min_q
    # Ensure v_range_q is not zero and handle potential type issues
    if v_range_q == 0: v_range_q = 1 # Avoid division by zero
    
    # Convert min/max/range to tensors with same dtype as v_q
    v_min_q_tensor = tensor.convert_to_tensor(v_min_q, dtype=v_q.dtype)
    v_range_q_tensor = tensor.convert_to_tensor(v_range_q, dtype=v_q.dtype)
    num_entries_minus_one = tensor.convert_to_tensor(num_entries - 1, dtype=tensor.float32)

    # Calculate floating-point index for interpolation
    # THIS IS NOT BACKEND-PURE but illustrates the logic. Replace with fixed-point ops.
    v_offset = ops.subtract(v_q, v_min_q_tensor)
    v_offset_float = tensor.cast(v_offset, tensor.float32)
    # Use ops.multiply instead of * operator
    idx_f_numerator = ops.multiply(v_offset_float, num_entries_minus_one)
    # Convert v_range_q to a tensor with float32 dtype
    v_range_q_float = tensor.convert_to_tensor(float(v_range_q), dtype=tensor.float32)
    # Use ops.divide instead of / operator
    idx_f = ops.divide(idx_f_numerator, v_range_q_float)
    
    # Get lower and upper indices for interpolation
    # Use ops.floor and ops.ceil for proper operations
    idx_lower = tensor.cast(ops.floor(idx_f), tensor.int32)
    idx_upper = tensor.cast(ops.ceil(idx_f), tensor.int32)
    
    # Clamp indices to valid range [0, num_entries-1]
    # Implement element-wise min/max using ops.where
    zero = tensor.convert_to_tensor(0, dtype=idx_lower.dtype)
    max_idx = tensor.convert_to_tensor(num_entries - 1, dtype=idx_lower.dtype)
    
    # Clamp idx_lower between 0 and num_entries - 1
    # If idx_lower < 0, use 0, otherwise use idx_lower
    idx_lower = ops.where(ops.less(idx_lower, zero), zero, idx_lower)
    # If idx_lower > max_idx, use max_idx, otherwise use idx_lower
    idx_lower = ops.where(ops.greater(idx_lower, max_idx), max_idx, idx_lower)
    
    # Clamp idx_upper between 0 and num_entries - 1
    # If idx_upper < 0, use 0, otherwise use idx_upper
    idx_upper = ops.where(ops.less(idx_upper, zero), zero, idx_upper)
    # If idx_upper > max_idx, use max_idx, otherwise use idx_upper
    idx_upper = ops.where(ops.greater(idx_upper, max_idx), max_idx, idx_upper)
    
    # Calculate interpolation weight (fraction between lower and upper)
    # Calculate weight as the fractional part (idx_f - floor(idx_f))
    weight = ops.subtract(idx_f, ops.floor(idx_f))
    
    # If indices are the same (exact match or at boundaries), no interpolation needed
    if ops.all(ops.equal(idx_lower, idx_upper)):
        return tensor.gather(lut_tensor, idx_lower, axis=0)
    
    # Gather values from LUT for both indices
    # Use tensor.gather instead of ops.gather
    val_lower = tensor.gather(lut_tensor, idx_lower, axis=0)
    val_upper = tensor.gather(lut_tensor, idx_upper, axis=0)
    
    # Convert to float for interpolation
    val_lower_f = tensor.cast(val_lower, tensor.float32)
    val_upper_f = tensor.cast(val_upper, tensor.float32)
    
    # Linear interpolation: result = val_lower + weight * (val_upper - val_lower)
    interpolated_f = val_lower_f + weight * (val_upper_f - val_lower_f)
    
    # Convert back to original dtype
    # Implement rounding using floor and add 0.5
    interpolated_f_plus_half = ops.add(interpolated_f, tensor.convert_to_tensor(0.5, dtype=interpolated_f.dtype))
    interpolated = tensor.cast(ops.floor(interpolated_f_plus_half), val_lower.dtype)
    
    return interpolated

# --- Full HH Step ---

def binary_hh_step(V_q, m_q, h_q, n_q, Iext_q, dt_q_scaled_Cm, dt_q_scaled):
    """
    Performs one Forward Euler step of the Hodgkin-Huxley model using
    binary wave (fixed-point) operations.

    Args:
        V_q, m_q, h_q, n_q: Current state variables in Q-format.
        Iext_q: External current in Q-format.
        dt_q_scaled_Cm: Time step dt scaled by 1/Cm, represented in a suitable Q-format
                         for multiplication with current difference (e.g., Q?.?).
                         Alternatively, represent dt/Cm as a shift amount if possible.
                         For now, assume it's a Q-value requiring bin_mult.
        dt_q_scaled: Time step dt, represented in a suitable Q-format for
                      multiplication with gating variable rates.
                      Alternatively, represent dt as a shift amount if possible.
                      For now, assume it's a Q-value requiring bin_mult.

    Returns:
        Tuple[BinaryWave, BinaryWave, BinaryWave, BinaryWave]: Updated V, m, h, n in Q-format.
    """
    # --- Parameters (Assume these are globally available or passed in Q-format) ---
    # Using placeholder Q values. Real implementation needs these defined.
    gNa_q = float_to_q(120.0) # Example conductance
    gK_q  = float_to_q(36.0)
    gL_q  = float_to_q(0.3)
    ENa_q = float_to_q(50.0)  # Example reversal potential (mV)
    EK_q  = float_to_q(-77.0)
    EL_q  = float_to_q(-54.387)
    one_q = float_to_q(1.0) # Q-format representation of 1.0

    # Convert parameters to tensors
    gNa_q_t = tensor.convert_to_tensor(gNa_q, dtype=V_q.dtype)
    gK_q_t  = tensor.convert_to_tensor(gK_q, dtype=V_q.dtype)
    gL_q_t  = tensor.convert_to_tensor(gL_q, dtype=V_q.dtype)
    ENa_q_t = tensor.convert_to_tensor(ENa_q, dtype=V_q.dtype)
    EK_q_t  = tensor.convert_to_tensor(EK_q, dtype=V_q.dtype)
    EL_q_t  = tensor.convert_to_tensor(EL_q, dtype=V_q.dtype)
    one_q_t = tensor.convert_to_tensor(one_q, dtype=V_q.dtype)


    # --- 1. Gating Variable Updates ---
    # dx/dt = alpha(V)(1-x) - beta(V)x
    # x_new = x + dt * dx/dt

    # m
    alpha_m_q = bin_rate('alpha_m', V_q)
    beta_m_q  = bin_rate('beta_m', V_q) # Assumes LUT exists
    one_minus_m_q = ops.subtract(one_q_t, m_q)
    term1_m = bin_mult(alpha_m_q, one_minus_m_q)
    term2_m = bin_mult(beta_m_q, m_q)
    dmdt_q = ops.subtract(term1_m, term2_m)
    delta_m_q = bin_mult(dt_q_scaled, dmdt_q) # dt * dmdt
    m_new_q = ops.add(m_q, delta_m_q)

    # h
    alpha_h_q = bin_rate('alpha_h', V_q) # Assumes LUT exists
    beta_h_q  = bin_rate('beta_h', V_q)  # Assumes LUT exists
    one_minus_h_q = ops.subtract(one_q_t, h_q)
    term1_h = bin_mult(alpha_h_q, one_minus_h_q)
    term2_h = bin_mult(beta_h_q, h_q)
    dhdt_q = ops.subtract(term1_h, term2_h)
    delta_h_q = bin_mult(dt_q_scaled, dhdt_q) # dt * dhdt
    h_new_q = ops.add(h_q, delta_h_q)

    # n
    alpha_n_q = bin_rate('alpha_n', V_q) # Assumes LUT exists
    beta_n_q  = bin_rate('beta_n', V_q)  # Assumes LUT exists
    one_minus_n_q = ops.subtract(one_q_t, n_q)
    term1_n = bin_mult(alpha_n_q, one_minus_n_q)
    term2_n = bin_mult(beta_n_q, n_q)
    dndt_q = ops.subtract(term1_n, term2_n)
    delta_n_q = bin_mult(dt_q_scaled, dndt_q) # dt * dndt
    n_new_q = ops.add(n_q, delta_n_q)

    # --- 2. Membrane Potential Update ---
    # dV/dt = (1/Cm) * [Iext - INa - IK - IL]
    # V_new = V + dt * dV/dt = V + (dt/Cm) * [Iext - I_ionic]

    # Calculate ionic currents (I = g * (V-E))
    V_minus_ENa_q = ops.subtract(V_q, ENa_q_t)
    V_minus_EK_q  = ops.subtract(V_q, EK_q_t)
    V_minus_EL_q  = ops.subtract(V_q, EL_q_t)

    m3_q = bin_pow(m_q, 3)
    n4_q = bin_pow(n_q, 4)

    # INa = gNa * m^3 * h * (V - ENa)
    INa_term1 = bin_mult(m3_q, h_q)
    INa_term2 = bin_mult(gNa_q_t, INa_term1)
    INa_q     = bin_mult(INa_term2, V_minus_ENa_q)

    # IK = gK * n^4 * (V - EK)
    IK_term1 = bin_mult(gK_q_t, n4_q)
    IK_q     = bin_mult(IK_term1, V_minus_EK_q)

    # IL = gL * (V - EL)
    IL_q     = bin_mult(gL_q_t, V_minus_EL_q)

    # Total ionic current
    I_ionic_q = ops.add(ops.add(INa_q, IK_q), IL_q)

    # Bracket term: Iext - I_ionic
    bracket_V_q = ops.subtract(Iext_q, I_ionic_q)

    # Update V: V_new = V + (dt/Cm) * bracket
    delta_V_q = bin_mult(dt_q_scaled_Cm, bracket_V_q) # (dt/Cm) * bracket
    V_new_q = ops.add(V_q, delta_V_q)

    return V_new_q, m_new_q, h_new_q, n_new_q


# --- Initial Tests for Core Ops ---

if __name__ == "__main__":
    print("Testing Core Binary Operations via ember_ml.ops...")

    # Example tensors (using integers directly for base-2 representation)
    # Note: Specify dtype for integer operations
    val_a = tensor.convert_to_tensor(5, dtype=tensor.int32)  # Binary: ...0101
    val_b = tensor.convert_to_tensor(3, dtype=tensor.int32)  # Binary: ...0011
    shift_amount = 1
    k_tensor = tensor.convert_to_tensor(shift_amount, dtype=tensor.int32)

    # Test Addition
    add_res = ops.add(val_a, val_b)
    print(f"{val_a.item()} + {val_b.item()} = {add_res.item()} (Expected: 8)")
    assert ops.all(ops.equal(add_res, tensor.convert_to_tensor(8, dtype=tensor.int32)))

    # Test AND
    and_res = bitwise.bitwise_and(val_a, val_b)
    print(f"{val_a.item()} & {val_b.item()} = {and_res.item()} (Expected: 1)") # ...0101 & ...0011 = ...0001
    assert ops.all(ops.equal(and_res, tensor.convert_to_tensor(1, dtype=tensor.int32)))

    # Test OR
    or_res = bitwise.bitwise_or(val_a, val_b)
    print(f"{val_a.item()} | {val_b.item()} = {or_res.item()} (Expected: 7)") # ...0101 | ...0011 = ...0111
    assert ops.all(ops.equal(or_res, tensor.convert_to_tensor(7, dtype=tensor.int32)))

    # Test XOR
    xor_res = bitwise.bitwise_xor(val_a, val_b)
    print(f"{val_a.item()} ^ {val_b.item()} = {xor_res.item()} (Expected: 6)") # ...0101 ^ ...0011 = ...0110
    assert ops.all(ops.equal(xor_res, tensor.convert_to_tensor(6, dtype=tensor.int32)))

    # Test NOT (Bitwise Invert) - Result depends on integer width/representation
    # For signed 32-bit int, ~5 is -6
    not_res = bitwise.bitwise_not(val_a)
    print(f"~{val_a.item()} = {not_res.item()} (Expected: -6 for int32)")
    assert ops.all(ops.equal(not_res, tensor.convert_to_tensor(-6, dtype=tensor.int32)))

    # Test Right Shift
    rshift_res = bitwise.right_shift(val_a, k_tensor)
    print(f"{val_a.item()} >> {shift_amount} = {rshift_res.item()} (Expected: 2)") # ...0101 >> 1 = ...0010
    assert ops.all(ops.equal(rshift_res, tensor.convert_to_tensor(2, dtype=tensor.int32)))

    # Test Left Shift
    lshift_res = bitwise.left_shift(val_a, k_tensor)
    print(f"{val_a.item()} << {shift_amount} = {lshift_res.item()} (Expected: 10)") # ...0101 << 1 = ...1010
    assert ops.all(ops.equal(lshift_res, tensor.convert_to_tensor(10, dtype=tensor.int32)))

    print("\nBasic Binary Operation Tests Passed!")

    # --- Test Fixed-Point Conversion ---
    print("\nTesting Fixed-Point Conversion (Q7.8)...")

    float_val_1 = 0.5
    q_val_1 = float_to_q(float_val_1)
    float_conv_1 = q_to_float(q_val_1)
    # Expected Q7.8 for 0.5: 0.5 * 2^8 = 0.5 * 256 = 128
    print(f"Float {float_val_1} -> Q7.8 Int: {q_val_1} (Expected: {int(0.5 * SCALE_FACTOR)})")
    print(f"Q7.8 Int {q_val_1} -> Float: {float_conv_1} (Expected: {float_val_1})")
    assert q_val_1 == int(0.5 * SCALE_FACTOR)
    assert math.isclose(float_conv_1, float_val_1)

    float_val_2 = -1.25
    q_val_2 = float_to_q(float_val_2)
    float_conv_2 = q_to_float(q_val_2)
    # Expected Q7.8 for -1.25: -1.25 * 256 = -320
    print(f"Float {float_val_2} -> Q7.8 Int: {q_val_2} (Expected: {int(-1.25 * SCALE_FACTOR)})")
    print(f"Q7.8 Int {q_val_2} -> Float: {float_conv_2} (Expected: {float_val_2})")
    assert q_val_2 == int(-1.25 * SCALE_FACTOR)
    assert math.isclose(float_conv_2, float_val_2)

    float_val_3 = 10.75 # 10.75 * 256 = 2752
    q_val_3 = float_to_q(float_val_3)
    float_conv_3 = q_to_float(q_val_3)
    print(f"Float {float_val_3} -> Q7.8 Int: {q_val_3} (Expected: {int(10.75 * SCALE_FACTOR)})")
    print(f"Q7.8 Int {q_val_3} -> Float: {float_conv_3} (Expected: {float_val_3})")
    assert q_val_3 == int(10.75 * SCALE_FACTOR)
    assert math.isclose(float_conv_3, float_val_3)

    # Example of using fixed-point with ops (demonstrative)
    # We'd use the Q-integer values in the tensors
    q_tensor_a = tensor.convert_to_tensor(q_val_1, dtype=tensor.int16) # Use int16 for Q7.8
    q_tensor_b = tensor.convert_to_tensor(q_val_3, dtype=tensor.int16)

    # Addition in Q format: Just add the integers
    q_add_res = ops.add(q_tensor_a, q_tensor_b)
    float_add_res = q_to_float(q_add_res.item()) # .item() for scalar tensor
    print(f"Q Add: {q_val_1} + {q_val_3} = {q_add_res.item()} -> Float: {float_add_res} (Expected: {0.5 + 10.75})")
    assert math.isclose(float_add_res, float_val_1 + float_val_3)

    print("\nFixed-Point Conversion Tests Passed!")

    # --- Test Fixed-Point Multiplication ---
    print("\nTesting Fixed-Point Multiplication (bin_mult)...")

    # Use Q7.8 (int16) values from previous tests
    # float_val_1 = 0.5  -> q_val_1 = 128
    # float_val_3 = 10.75 -> q_val_3 = 2752
    q_tensor_1 = tensor.convert_to_tensor(q_val_1, dtype=tensor.int16)
    q_tensor_3 = tensor.convert_to_tensor(q_val_3, dtype=tensor.int16)

    # Test 1: 0.5 * 10.75 = 5.375
    expected_float_1 = 0.5 * 10.75
    expected_q_1 = float_to_q(expected_float_1)
    q_mult_res_1 = bin_mult(q_tensor_1, q_tensor_3)
    float_mult_res_1 = q_to_float(q_mult_res_1.item())
    print(f"Q Mult: {q_val_1} * {q_val_3} = {q_mult_res_1.item()} -> Float: {float_mult_res_1}")
    print(f"Expected Float: {expected_float_1} -> Q: {expected_q_1}")
    # Use ops.all for tensor comparison if result is not scalar
    assert ops.all(ops.equal(q_mult_res_1, tensor.convert_to_tensor(expected_q_1, dtype=tensor.int16)))
    # assert math.isclose(float_mult_res_1, expected_float_1, rel_tol=1e-5) # Check float result too

    # Test 2: -1.25 * 0.5 = -0.625
    # float_val_2 = -1.25 -> q_val_2 = -320
    q_tensor_2 = tensor.convert_to_tensor(q_val_2, dtype=tensor.int16)
    expected_float_2 = -1.25 * 0.5
    expected_q_2 = float_to_q(expected_float_2)
    q_mult_res_2 = bin_mult(q_tensor_2, q_tensor_1)
    float_mult_res_2 = q_to_float(q_mult_res_2.item())
    print(f"Q Mult: {q_val_2} * {q_val_1} = {q_mult_res_2.item()} -> Float: {float_mult_res_2}")
    print(f"Expected Float: {expected_float_2} -> Q: {expected_q_2}")
    assert ops.all(ops.equal(q_mult_res_2, tensor.convert_to_tensor(expected_q_2, dtype=tensor.int16)))
    # assert math.isclose(float_mult_res_2, expected_float_2, rel_tol=1e-5)

    # Test 3: -1.25 * 10.75 = -13.4375
    expected_float_3 = -1.25 * 10.75
    expected_q_3 = float_to_q(expected_float_3)
    q_mult_res_3 = bin_mult(q_tensor_2, q_tensor_3)
    float_mult_res_3 = q_to_float(q_mult_res_3.item())
    print(f"Q Mult: {q_val_2} * {q_val_3} = {q_mult_res_3.item()} -> Float: {float_mult_res_3}")
    print(f"Expected Float: {expected_float_3} -> Q: {expected_q_3}")
    assert ops.all(ops.equal(q_mult_res_3, tensor.convert_to_tensor(expected_q_3, dtype=tensor.int16)))
    # assert math.isclose(float_mult_res_3, expected_float_3, rel_tol=1e-5)

    print("\nFixed-Point Multiplication Tests Passed!")

    # --- Test BIN_RATE with Interpolation ---
    print("\nTesting BIN_RATE with Interpolation...")
    
    # 1. Generate LUTs
    V_MIN_MV = -100.0
    V_MAX_MV = 50.0
    NUM_ENTRIES = 20  # Use fewer entries to make interpolation more noticeable
    
    # Define a simple linear function for testing interpolation
    def linear_func(V_mV):
        # Convert scalar values to tensors
        scale = tensor.convert_to_tensor(0.1, dtype=V_mV.dtype)
        offset = tensor.convert_to_tensor(5.0, dtype=V_mV.dtype)
        # Use ops.multiply and ops.add for tensor operations
        return ops.add(ops.multiply(scale, V_mV), offset)  # f(V) = 0.1*V + 5
    
    # Generate LUT
    lut_linear_q, v_min_q, v_max_q, num_entries = generate_lut(linear_func, V_MIN_MV, V_MAX_MV, NUM_ENTRIES)
    HH_LUTS['linear'] = lut_linear_q
    LUT_PARAMS['linear'] = (v_min_q, v_max_q, num_entries)
    
    # 2. Test interpolation at various points
    test_voltages = [-90.0, -75.0, -50.0, -25.0, 0.0, 25.0, 40.0]
    
    print("Testing interpolation accuracy:")
    print("Voltage | Exact Value | Interpolated Value | Error")
    print("--------|-------------|-------------------|------")
    
    for v_mv in test_voltages:
        # Calculate exact value using the function
        exact_value = linear_func(tensor.convert_to_tensor(v_mv, dtype=tensor.float32)).item()
        
        # Convert voltage to Q format
        v_q = float_to_q(v_mv)
        v_q_tensor = tensor.convert_to_tensor(v_q, dtype=tensor.int16)
        
        # Get interpolated value using bin_rate
        interp_value_q = bin_rate('linear', v_q_tensor)
        interp_value = q_to_float(interp_value_q.item())
        
        # Calculate error
        error = abs(interp_value - exact_value)
        
        print(f"{v_mv:7.1f} | {exact_value:11.6f} | {interp_value:17.6f} | {error:.6f}")
    
    print("\nBIN_RATE Interpolation Tests Complete.")
    
    # --- Test Full HH Step with Interpolation ---
    print("\nTesting Full HH Step with Interpolation...")
    
    # Generate all required LUTs for HH model
    print("Generating LUTs for HH model...")
    V_MIN_MV = -100.0
    V_MAX_MV = 50.0
    NUM_ENTRIES = 256
    
    rate_funcs = {
        'alpha_m': safe_alpha_m,
        'beta_m': safe_beta_m,
        'alpha_h': safe_alpha_h,
        'beta_h': safe_beta_h,
        'alpha_n': safe_alpha_n,
        'beta_n': safe_beta_n
    }
    
    for key, func in rate_funcs.items():
        lut_q, v_min_q, v_max_q, num_entries = generate_lut(func, V_MIN_MV, V_MAX_MV, NUM_ENTRIES)
        HH_LUTS[key] = lut_q
        LUT_PARAMS[key] = (v_min_q, v_max_q, num_entries)
    
    print("LUTs Generated.")
    
    # --- Simulation Parameters ---
    Cm = 1.0  # uF/cm^2
    dt = 0.01 # ms - choose small dt for Euler stability
    
    # Represent dt and dt/Cm in Q7.8
    dt_q = float_to_q(dt)
    dt_div_Cm_q = float_to_q(dt / Cm)
    dt_q_scaled_Cm_t = tensor.convert_to_tensor(dt_div_Cm_q, dtype=tensor.int16)
    dt_q_scaled_t = tensor.convert_to_tensor(dt_q, dtype=tensor.int16)
    
    # --- Initial State ---
    V_init_mv = -65.0
    m_init = 0.0529
    h_init = 0.5961
    n_init = 0.3177
    I_ext_val = 10.0 # uA/cm^2
    
    # Convert initial state to Q7.8
    V_init_q = float_to_q(V_init_mv)
    m_init_q = float_to_q(m_init)
    h_init_q = float_to_q(h_init)
    n_init_q = float_to_q(n_init)
    Iext_init_q = float_to_q(I_ext_val)
    
    # Create tensors for initial state
    V_q_t = tensor.convert_to_tensor(V_init_q, dtype=tensor.int16)
    m_q_t = tensor.convert_to_tensor(m_init_q, dtype=tensor.int16)
    h_q_t = tensor.convert_to_tensor(h_init_q, dtype=tensor.int16)
    n_q_t = tensor.convert_to_tensor(n_init_q, dtype=tensor.int16)
    Iext_q_t = tensor.convert_to_tensor(Iext_init_q, dtype=tensor.int16)
    
    # --- Reference Float Calculation (One Step) ---
    V_f, m_f, h_f, n_f = V_init_mv, m_init, h_init, n_init
    gNa_f, gK_f, gL_f = 120.0, 36.0, 0.3
    ENa_f, EK_f, EL_f = 50.0, -77.0, -54.387
    
    # Use try-except for potential division by zero in float rate calcs
    try:
        alpha_m_f = 0.1*(V_f+40)/(1-math.exp(-(V_f+40)/10)) if abs(V_f + 40.0) > 1e-6 else 1.0
    except OverflowError: alpha_m_f = 0.0
    beta_m_f  = 4*math.exp(-(V_f+65)/18)
    alpha_h_f = 0.07*math.exp(-(V_f+65)/20)
    beta_h_f  = 1/(1+math.exp(-(V_f+35)/10))
    try:
        alpha_n_f = 0.01*(V_f+55)/(1-math.exp(-(V_f+55)/10)) if abs(V_f + 55.0) > 1e-6 else 0.1
    except OverflowError: alpha_n_f = 0.0
    beta_n_f  = 0.125*math.exp(-(V_f+65)/80)
    
    dmdt_f = alpha_m_f*(1-m_f) - beta_m_f*m_f
    dhdt_f = alpha_h_f*(1-h_f) - beta_h_f*h_f
    dndt_f = alpha_n_f*(1-n_f) - beta_n_f*n_f
    
    INa_f = gNa_f * (m_f**3) * h_f * (V_f - ENa_f)
    IK_f  = gK_f * (n_f**4) * (V_f - EK_f)
    IL_f  = gL_f * (V_f - EL_f)
    dVdt_f = (I_ext_val - INa_f - IK_f - IL_f) / Cm
    
    V_next_f = V_f + dt * dVdt_f
    m_next_f = m_f + dt * dmdt_f
    h_next_f = h_f + dt * dhdt_f
    n_next_f = n_f + dt * dndt_f
    
    print(f"\nReference Float Step:")
    print(f"  V_next={V_next_f:.4f}, m_next={m_next_f:.4f}, h_next={h_next_f:.4f}, n_next={n_next_f:.4f}")
    
    # --- Binary HH Step Calculation ---
    try:
        V_next_q_t, m_next_q_t, h_next_q_t, n_next_q_t = binary_hh_step(
            V_q_t, m_q_t, h_q_t, n_q_t, Iext_q_t, dt_q_scaled_Cm_t, dt_q_scaled_t
        )
        
        # Convert results back to float for comparison
        V_next_qf = q_to_float(V_next_q_t.item())
        m_next_qf = q_to_float(m_next_q_t.item())
        h_next_qf = q_to_float(h_next_q_t.item())
        n_next_qf = q_to_float(n_next_q_t.item())
        
        print(f"Binary HH Step (Q7.8) with Interpolation:")
        print(f"  V_next={V_next_qf:.4f}, m_next={m_next_qf:.4f}, h_next={h_next_qf:.4f}, n_next={n_next_qf:.4f}")
        
        # Compare results (allow tolerance)
        tolerance = 0.05 # Increased tolerance due to Q-format, LUT, Euler
        print(f"Comparing with tolerance: {tolerance}")
        v_close = math.isclose(V_next_qf, V_next_f, abs_tol=tolerance)
        m_close = math.isclose(m_next_qf, m_next_f, abs_tol=tolerance)
        h_close = math.isclose(h_next_qf, h_next_f, abs_tol=tolerance)
        n_close = math.isclose(n_next_qf, n_next_f, abs_tol=tolerance)
        print(f"  V close: {v_close}, m close: {m_close}, h close: {h_close}, n close: {n_close}")
        
        assert v_close
        assert m_close
        assert h_close
        assert n_close
        
        print("\nBinary HH Step test passed (within tolerance).")
        
    except Exception as e:
        print(f"Error during Binary HH Step test: {e}")
        raise
