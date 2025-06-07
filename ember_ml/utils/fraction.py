"""
Fraction utilities for the ember_ml library.

This module provides utilities for working with fractions.
"""

from fractions import Fraction
from typing import Union, Tuple, List

def to_fraction(value: Union[float, int, str]) -> Fraction:
    """
    Convert a value to a Fraction.
    
    Args:
        value: Value to convert
        
    Returns:
        Fraction representation
    """
    return Fraction(value)

def to_float(frac: Fraction) -> float:
    """
    Convert a Fraction to a float.
    
    Args:
        frac: Fraction to convert
        
    Returns:
        Float representation
    """
    return float(frac)

def simplify_fraction(frac: Fraction, max_denominator: int = 1000) -> Fraction:
    """
    Simplify a fraction to a given maximum denominator.
    
    Args:
        frac: Fraction to simplify
        max_denominator: Maximum denominator
        
    Returns:
        Simplified fraction
    """
    return frac.limit_denominator(max_denominator)

def fraction_to_ratio(frac: Fraction) -> Tuple[int, int]:
    """
    Convert a fraction to a ratio of integers.
    
    Args:
        frac: Fraction to convert
        
    Returns:
        Tuple of (numerator, denominator)
    """
    return frac.numerator, frac.denominator

def ratio_to_fraction(numerator: int, denominator: int) -> Fraction:
    """
    Convert a ratio of integers to a fraction.
    
    Args:
        numerator: Numerator of the ratio
        denominator: Denominator of the ratio
        
    Returns:
        Fraction representation
    """
    return Fraction(numerator, denominator)

def continued_fraction(value: float, max_terms: int = 10) -> List[int]:
    """
    Compute the continued fraction representation of a value.
    
    Args:
        value: Value to convert
        max_terms: Maximum number of terms
        
    Returns:
        List of continued fraction terms
    """
    terms = []
    for _ in range(max_terms):
        whole = int(value)
        terms.append(whole)
        frac = value - whole
        if abs(frac) < 1e-10:
            break
        value = 1 / frac
    return terms

def from_continued_fraction(terms: List[int]) -> Fraction:
    """
    Convert a continued fraction to a Fraction.
    
    Args:
        terms: Continued fraction terms
        
    Returns:
        Fraction representation
    """
    if not terms:
        return Fraction(0)
    
    result = Fraction(terms[-1])
    for term in reversed(terms[:-1]):
        result = Fraction(term) + Fraction(1, result)
    
    return result