"""
BigQuery encoding utilities for Ember ML.

This module provides functions for encoding categorical features from BigQuery data
for use in Ember ML models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops

# Set up logging
logger = logging.getLogger(__name__)


def one_hot_encode(
    df: Any,
    column: str,
    unique_values: Optional[Any] = None,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    One-hot encode a categorical column.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        unique_values: Optional list of unique values (if None, will be computed)
        device: Device to place tensors on
        
    Returns:
        Tuple of (encoded_tensor, feature_names)
    """
    # Get column data
    col_data = df[column]
    
    # Get unique values if not provided
    if unique_values is None:
        unique_values = col_data.dropna().unique()
    
    # Initialize encoded data
    n_samples = len(df)
    n_categories = len(unique_values)
    encoded_data = tensor.zeros((n_samples, n_categories), device=device)
    
    # Generate feature names
    feature_names = []
    value_to_index = {}
    
    for i, value in enumerate(unique_values):
        # Create safe feature name
        if isinstance(value, str):
            safe_value = value.replace(' ', '_').replace('-', '_').replace('/', '_')
        else:
            safe_value = str(value)
        
        feature_name = f"{column}_{safe_value}"
        feature_names.append(feature_name)
        value_to_index[value] = i
    
    # Encode each row
    for i, value in enumerate(col_data):
        if pd.isna(value):
            # Skip NaN values
            continue
            
        if value in value_to_index:
            idx = value_to_index[value]
            # Set the corresponding column to 1
            indices = tensor.convert_to_tensor([[i, idx]], device=device)
            updates = tensor.convert_to_tensor([1.0], device=device)
            encoded_data = tensor.tensor_scatter_nd_update(encoded_data, indices, updates)
    
    logger.info(f"One-hot encoded column {column} with {n_categories} categories")
    return encoded_data, feature_names


def hash_encode(
    df: Any,
    column: str,
    n_components: int = 16,
    seed: int = 42,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Hash encode a high-cardinality categorical column.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        n_components: Number of hash components
        seed: Random seed
        device: Device to place tensors on
        
    Returns:
        Tuple of (encoded_tensor, feature_names)
    """
    # Get column data
    col_data = df[column]
    
    # Initialize encoded data
    n_samples = len(df)
    encoded_data = tensor.zeros((n_samples, n_components), device=device)
    
    # Generate feature names
    feature_names = [f"{column}_hash_{i}" for i in range(n_components)]
    
    # Set random seed
    tensor.set_seed(seed)
    
    # Encode each value
    for i, value in enumerate(col_data):
        if pd.isna(value):
            # Skip NaN values
            continue
            
        # Convert value to string
        str_value = str(value)
        
        # Generate hash value
        hash_val = hash(str_value + str(seed))
        
        # Use hash to generate pseudo-random values for each component
        for j in range(n_components):
            component_hash = hash(str_value + str(seed + j))
            # Scale to [0, 1]
            component_value = (component_hash % 10000) / 10000.0
            
            # Update the tensor at position [i, j]
            indices = tensor.convert_to_tensor([[i, j]], device=device)
            updates = tensor.convert_to_tensor([component_value], device=device)
            encoded_data = tensor.tensor_scatter_nd_update(encoded_data, indices, updates)
    
    # Reset seed
    tensor.set_seed(None)
    
    logger.info(f"Hash encoded column {column} with {n_components} components")
    return encoded_data, feature_names


def target_encode(
    df: Any,
    column: str,
    target_column: str,
    smoothing: float = 10.0,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Target encode a categorical column.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        target_column: Target column for encoding
        smoothing: Smoothing factor
        device: Device to place tensors on
        
    Returns:
        Tuple of (encoded_tensor, feature_names)
    """
    # Get column data
    col_data = df[column]
    target_data = df[target_column]
    
    # Calculate global mean
    global_mean = target_data.mean()
    
    # Calculate per-category means
    category_stats = df.groupby(column)[target_column].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed_means = (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) / (category_stats['count'] + smoothing)
    
    # Create mapping
    mapping = smoothed_means.to_dict()
    
    # Apply mapping
    encoded_values = col_data.map(mapping).fillna(global_mean).values
    
    # Convert to tensor
    encoded_tensor = tensor.convert_to_tensor(encoded_values, device=device)
    
    # Reshape to 2D
    encoded_tensor = tensor.reshape(encoded_tensor, (tensor.shape(encoded_tensor)[0], 1))
    
    # Generate feature name
    feature_name = [f"{column}_target_encoded"]
    
    logger.info(f"Target encoded column {column} using target {target_column}")
    return encoded_tensor, feature_name