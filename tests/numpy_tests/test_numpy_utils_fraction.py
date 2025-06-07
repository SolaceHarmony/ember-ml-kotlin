import pytest
import numpy as np # For comparison with known correct results
from fractions import Fraction # Import Python's Fraction for comparison

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.utils import fraction # Import fraction utilities
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

# Test cases for utils.fraction functions

def test_to_fraction():
    # Test to_fraction
    assert fraction.to_fraction(0.5) == Fraction(1, 2)
    assert fraction.to_fraction(1) == Fraction(1, 1)
    assert fraction.to_fraction("3/4") == Fraction(3, 4)
    assert fraction.to_fraction(0.75) == Fraction(3, 4)
    assert fraction.to_fraction(1.5) == Fraction(3, 2)

    # Test with values that might have floating point inaccuracies
    assert fraction.to_fraction(0.1) == Fraction(1, 10)
    assert fraction.to_fraction(1/3) == Fraction(1, 3)

def test_to_float():
    # Test to_float
    assert fraction.to_float(Fraction(1, 2)) == 0.5
    assert fraction.to_float(Fraction(3, 4)) == 0.75
    assert fraction.to_float(Fraction(5, 2)) == 2.5
    assert fraction.to_float(Fraction(1, 3)) == 1/3

def test_simplify_fraction():
    # Test simplify_fraction
    assert fraction.simplify_fraction(Fraction(2, 4), max_denominator=10) == Fraction(1, 2)
    assert fraction.simplify_fraction(Fraction(6, 9), max_denominator=5) == Fraction(2, 3)
    assert fraction.simplify_fraction(Fraction(10, 15), max_denominator=2) == Fraction(2, 3) # Cannot simplify further than 2/3 with max_denominator=2
    assert fraction.simplify_fraction(Fraction(7, 14), max_denominator=1) == Fraction(1, 2) # Simplifies to 1/2

def test_fraction_to_ratio():
    # Test fraction_to_ratio
    assert fraction.fraction_to_ratio(Fraction(1, 2)) == (1, 2)
    assert fraction.fraction_to_ratio(Fraction(3, 4)) == (3, 4)
    assert fraction.fraction_to_ratio(Fraction(5, 1)) == (5, 1)

def test_ratio_to_fraction():
    # Test ratio_to_fraction
    assert fraction.ratio_to_fraction(1, 2) == Fraction(1, 2)
    assert fraction.ratio_to_fraction(3, 4) == Fraction(3, 4)
    assert fraction.ratio_to_fraction(5, 1) == Fraction(5, 1)
    assert fraction.ratio_to_fraction(6, 8) == Fraction(3, 4) # Should simplify

def test_continued_fraction_from_continued_fraction():
    # Test continued_fraction and from_continued_fraction round trip
    value = 3.14159
    max_terms = 10
    terms = fraction.continued_fraction(value, max_terms)
    assert isinstance(terms, list)
    assert len(terms) <= max_terms

    reconstructed_fraction = fraction.from_continued_fraction(terms)
    assert isinstance(reconstructed_fraction, Fraction)

    # Check that the reconstructed fraction is close to the original value
    assert abs(fraction.to_float(reconstructed_fraction) - value) < 1e-9

    # Test with a simple fraction
    value_simple = 0.75
    terms_simple = fraction.continued_fraction(value_simple, 5)
    reconstructed_simple = fraction.from_continued_fraction(terms_simple)
    assert reconstructed_simple == Fraction(3, 4)