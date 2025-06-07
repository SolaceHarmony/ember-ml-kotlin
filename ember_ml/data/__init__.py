"""
Data handling components for ember_ml.

This module provides tools for loading and preprocessing data.
"""

from ember_ml.data.csv_loader import GenericCSVLoader
from ember_ml.data.type_detector import GenericTypeDetector

__all__ = ['GenericCSVLoader', 'GenericTypeDetector']