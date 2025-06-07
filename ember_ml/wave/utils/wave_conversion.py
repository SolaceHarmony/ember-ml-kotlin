"""
Wave conversion utilities.

This module provides utilities for converting between different wave representations.
"""
from typing import Union, List, Tuple, Optional
from ember_ml.nn.tensor.types import TensorLike # Added import
from ember_ml.ops import linearalg
from ember_ml import ops # Moved import to top level
from typing import Any
from ember_ml.nn import tensor # Moved import to top level
# Define TensorLike as a type alias for better readability
DType = Any
def pcm_to_float(pcm_data: TensorLike, dtype: Any = tensor.float32) -> Any:
    """
    Convert PCM data to floating point representation.
    
    Args:
        pcm_data: PCM data as numpy array
        dtype: Output data type
        
    Returns:
        Floating point representation
    """
    pcm_data = tensor.convert_to_tensor(pcm_data)
    if 'int' in str(pcm_data.dtype):
        raise TypeError("PCM data must be integer type")
    
    dtype = tensor.dtype(dtype)
    if 'float' in str(dtype):
        raise TypeError("Output data type must be floating point")
    
    i = ops.bitwise.iinfo(pcm_data.dtype) # iinfo does not exist - use ops.bitwise.*
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    
    return (pcm_data.astype(dtype) - offset) / abs_max

def float_to_pcm(float_data: TensorLike, dtype: tensor.dtype = tensor.int16) -> TensorLike:
    """
    Convert floating point data to PCM representation.
    
    Args:
        float_data: Floating point data as numpy array
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    float_data = tensor.asarray(float_data)
    if float_data.dtype.kind != 'f':
        raise TypeError("Input data must be floating point")
    
    dtype = tensor.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("Output data type must be integer")
    
    i = ops.bitwise.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    
    return (float_data * abs_max + offset).clip(i.min, i.max).astype(dtype)

def pcm_to_db(pcm_data: TensorLike, ref: float = 1.0, min_db: float = -80.0) -> TensorLike:
    """
    Convert PCM data to decibels.
    
    Args:
        pcm_data: PCM data as numpy array
        ref: Reference value
        min_db: Minimum dB value
        
    Returns:
        Decibel representation
    """
    float_data = pcm_to_float(pcm_data)
    power = ops.abs(float_data) ** 2
    # Removed import from here: from ember_ml import ops
    db = 10 * ops.log10(stats.maximum(power, 1e-10) / ref)
    return stats.maximum(db, min_db)

def db_to_amplitude(db: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    Convert decibels to amplitude.
    
    Args:
        db: Decibel value
        
    Returns:
        Amplitude value
    """
    return 10 ** (db / 20)

def amplitude_to_db(amplitude: Union[float, TensorLike], min_db: float = -80.0) -> Union[float, TensorLike]:
    """
    Convert amplitude to decibels.
    
    Args:
        amplitude: Amplitude value
        min_db: Minimum dB value
        
    Returns:
        Decibel value
    """
    # Removed import from here: from ember_ml import ops
    db = 20 * ops.log10(stats.maximum(ops.abs(amplitude), 1e-10))
    return stats.maximum(db, min_db)

def pcm_to_binary(pcm_data: TensorLike, threshold: float = 0.0) -> TensorLike:
    """
    Convert PCM data to binary representation.
    
    Args:
        pcm_data: PCM data as numpy array
        threshold: Threshold for binarization
        
    Returns:
        Binary representation
    """
    float_data = pcm_to_float(pcm_data)
    return (float_data > threshold).astype(tensor.int8)

def binary_to_pcm(binary_data: TensorLike, amplitude: float = 1.0, dtype: tensor.dtype = tensor.int16) -> TensorLike:
    """
    Convert binary data to PCM representation.
    
    Args:
        binary_data: Binary data as numpy array
        amplitude: Amplitude of the PCM signal
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    float_data = binary_data.astype(tensor.float32) * 2 - 1
    float_data *= amplitude
    return float_to_pcm(float_data, dtype)

def pcm_to_phase(pcm_data: TensorLike) -> TensorLike:
    """
    Convert PCM data to phase representation.
    
    Args:
        pcm_data: PCM data as numpy array
        
    Returns:
        Phase representation
    """
    float_data = pcm_to_float(pcm_data)
    return tensor.angle(linearalg.fft(float_data))

def phase_to_pcm(phase_data: TensorLike, magnitude: Optional[TensorLike] = None, dtype: tensor.dtype = tensor.int16) -> TensorLike:
    """
    Convert phase data to PCM representation.
    
    Args:
        phase_data: Phase data as numpy array
        magnitude: Magnitude data as numpy array
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    if magnitude is None:
        # Need to import tensor here if not already imported
        from ember_ml.nn import tensor
        magnitude = tensor.ones_like(phase_data)

    complex_data = magnitude * ops.exp(1j * phase_data)
    float_data = ops.real(linearalg.fft.ifft(complex_data))
    return float_to_pcm(float_data, dtype)