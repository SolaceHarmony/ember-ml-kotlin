"""
BigQuery feature processing utilities for Ember ML.

This module provides functions for processing features from BigQuery data
for use in Ember ML models.
"""

import logging
from typing import Any, List, Optional, Tuple

# We need pandas for isna() checks in one_hot_encode and hash_encode
import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.ops import stats
from ember_ml.nn.features.bigquery.encoding import one_hot_encode, hash_encode

# Set up logging
logger = logging.getLogger(__name__)


def process_numeric_features(
    data: Any,
    columns: List[str],
    handle_missing: bool = True,
    handle_outliers: bool = True,
    normalize: bool = True,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Process numeric features from BigQuery data.
    
    Args:
        data: Input DataFrame
        columns: Numeric columns to process
        handle_missing: Whether to handle missing values
        handle_outliers: Whether to handle outliers
        normalize: Whether to normalize features
        device: Device to place tensors on
        
    Returns:
        Tuple of (processed_features_tensor, feature_names)
    """
    if not columns:
        logger.warning("No numeric columns to process")
        return None, []
    
    # Extract numeric columns
    numeric_data = data[columns]
    
    # Convert to tensor
    numeric_tensor = _dataframe_to_tensor(numeric_data, device)
    
    # Handle missing values
    if handle_missing:
        numeric_tensor = handle_missing_values(numeric_tensor)
    
    # Handle outliers if requested
    if handle_outliers is True:  # Explicit comparison to avoid callable error
        numeric_tensor = remove_outliers(numeric_tensor)
    
    # Normalize
    if normalize:
        numeric_tensor = normalize_robust(numeric_tensor)
    
    logger.info(f"Processed {len(columns)} numeric features")
    return numeric_tensor, columns


def process_categorical_features(
    data: Any,
    columns: List[str],
    handle_missing: bool = True,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Process categorical features from BigQuery data.
    
    Args:
        data: Input DataFrame
        columns: Categorical columns to process
        handle_missing: Whether to handle missing values
        device: Device to place tensors on
        
    Returns:
        Tuple of (processed_features_tensor, feature_names)
    """
    if not columns:
        logger.warning("No categorical columns to process")
        return None, []
    
    # Process each column separately
    encoded_features_list = []
    encoded_feature_names = []
    
    for col in columns:
        # Get unique values
        unique_values = data[col].dropna().unique()
        
        # Skip if too many categories
        if len(unique_values) > 100:
            logger.warning(f"Column {col} has {len(unique_values)} unique values, "
                           f"which exceeds the maximum of 100. Using hash encoding.")
            encoded_features, feature_names = hash_encode(data, col, n_components=16, device=device)
        else:
            # One-hot encode
            encoded_features, feature_names = one_hot_encode(data, col, unique_values, device=device)
        
        encoded_features_list.append(encoded_features)
        encoded_feature_names.extend(feature_names)
    
    # Combine all encoded features
    if encoded_features_list:
        encoded_data = tensor.concatenate(encoded_features_list, axis=1)
        logger.info(f"Processed {len(columns)} categorical features into {len(encoded_feature_names)} encoded features")
        return encoded_data, encoded_feature_names
    else:
        return None, []


def process_datetime_features(
    data: Any,
    columns: List[str],
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Process datetime features from BigQuery data.
    
    Args:
        data: Input DataFrame
        columns: Datetime columns to process
        device: Device to place tensors on
        
    Returns:
        Tuple of (processed_features_tensor, feature_names)
    """
    if not columns:
        logger.warning("No datetime columns to process")
        return None, []
    
    # Process each datetime column
    all_features = []
    all_feature_names = []
    
    for col in columns:
        try:
            # Extract date components
            components = {}
            components['year'] = data[col].dt.year
            components['month'] = data[col].dt.month
            components['day'] = data[col].dt.day
            components['day_of_week'] = data[col].dt.dayofweek
            components['hour'] = data[col].dt.hour
            
            # Process each component
            processed_components = []
            component_names = []
            
            for comp_name, comp_data in components.items():
                # Convert to tensor
                comp_tensor = tensor.convert_to_tensor(comp_data.values, device=device)
                
                # Apply cyclical encoding for cyclic features
                if comp_name in ['month', 'day_of_week', 'hour']:
                    # Get the maximum value for this cycle
                    if comp_name == 'month':
                        max_val = 12
                    elif comp_name == 'day_of_week':
                        max_val = 7
                    elif comp_name == 'hour':
                        max_val = 24
                    
                    # Apply sin/cos encoding
                    # Convert constants to tensors
                    two = tensor.convert_to_tensor(2.0, device=device)
                    pi = tensor.convert_to_tensor(3.14159, device=device)
                    max_val_tensor = tensor.convert_to_tensor(max_val, device=device)
                    
                    # Calculate using ops functions
                    angle = ops.divide(ops.multiply(ops.multiply(two, pi), comp_tensor), max_val_tensor)
                    sin_comp = ops.sin(angle)
                    cos_comp = ops.cos(angle)
                    
                    # Reshape to 2D
                    sin_comp = tensor.reshape(sin_comp, (tensor.shape(sin_comp)[0], 1))
                    cos_comp = tensor.reshape(cos_comp, (tensor.shape(cos_comp)[0], 1))
                    
                    # Add to processed components
                    processed_components.append(sin_comp)
                    processed_components.append(cos_comp)
                    component_names.append(f"{col}_{comp_name}_sin")
                    component_names.append(f"{col}_{comp_name}_cos")
                else:
                    # For non-cyclic features, normalize
                    min_val = stats.min(comp_tensor)
                    max_val = stats.max(comp_tensor)
                    range_val = ops.subtract(max_val, min_val)
                    
                    # Avoid division by zero
                    range_val = tensor.maximum(range_val, tensor.convert_to_tensor(1e-8, device=device))
                    
                    # Normalize
                    normalized_comp = ops.divide(ops.subtract(comp_tensor, min_val), range_val)
                    
                    # Reshape to 2D
                    normalized_comp = tensor.reshape(normalized_comp, (tensor.shape(normalized_comp)[0], 1))
                    
                    # Add to processed components
                    processed_components.append(normalized_comp)
                    component_names.append(f"{col}_{comp_name}")
            
            # Combine all components for this column
            if processed_components:
                column_features = tensor.concatenate(processed_components, axis=1)
                all_features.append(column_features)
                all_feature_names.extend(component_names)
        except Exception as e:
            logger.warning(f"Error processing datetime column {col}: {e}")
    
    # Combine all datetime features
    if all_features:
        combined_features = tensor.concatenate(all_features, axis=1)
        logger.info(f"Processed {len(columns)} datetime features into {len(all_feature_names)} encoded features")
        return combined_features, all_feature_names
    else:
        return None, []


def handle_missing_values(data: Any) -> Any:
    """
    Handle missing values in numeric data.
    
    Args:
        data: Input tensor
        
    Returns:
        Tensor with missing values handled
    """
    # Check for NaN values
    nan_mask = ops.isnan(data)
    
    # If no NaNs, return the original data
    # Check if any element is True using logical operations
    if ops.all(ops.logical_not(nan_mask)):
        return data
    
    # Calculate median for each feature
    # First, replace NaNs with zeros for calculation purposes
    data_no_nan = ops.where(nan_mask, tensor.zeros_like(data), data)
    
    # Calculate median for each feature (column)
    medians = []
    for i in range(tensor.shape(data)[1]):
        col_data = data_no_nan[:, i]
        # Use stats.median to compute the median
        median = stats.median(col_data)
        medians.append(median)
    
    # Convert to tensor
    medians_tensor = tensor.stack(medians)
    
    # Reshape medians to match data shape for broadcasting
    medians_tensor = tensor.reshape(medians_tensor, (1, -1))
    
    # Replace NaNs with medians
    # Create a broadcasted version of medians_tensor
    broadcasted_medians = tensor.tile(medians_tensor, [tensor.shape(data)[0], 1])
    
    return ops.where(nan_mask, broadcasted_medians, data)


def remove_outliers(data: Any) -> Any:
    """
    Handle outliers in numeric data using IQR method.
    
    Args:
        data: Input tensor
        
    Returns:
        Tensor with outliers handled
    """
    # Calculate quartiles for each feature
    q1_values = []
    q3_values = []
    
    for i in range(tensor.shape(data)[1]):
        col_data = data[:, i]
        # Use stats.percentile to compute quartiles
        q1 = stats.percentile(col_data, 25)
        q3 = stats.percentile(col_data, 75)
        
        q1_values.append(q1)
        q3_values.append(q3)
    
    # Convert to tensors
    q1_tensor = tensor.stack(q1_values)
    q3_tensor = tensor.stack(q3_values)
    
    # Calculate IQR
    iqr_tensor = ops.subtract(q3_tensor, q1_tensor)
    
    # Calculate bounds
    lower_bound = ops.subtract(q1_tensor, ops.multiply(iqr_tensor, 1.5))
    upper_bound = ops.add(q3_tensor, ops.multiply(iqr_tensor, 1.5))
    
    # Reshape bounds to match data shape for broadcasting
    lower_bound = tensor.reshape(lower_bound, (1, -1))
    upper_bound = tensor.reshape(upper_bound, (1, -1))
    
    # Clip values to bounds
    return ops.clip(data, lower_bound, upper_bound)


def normalize_robust(data: Any) -> Any:
    """
    Apply robust normalization to numeric data using quantiles.
    
    Args:
        data: Input tensor
        
    Returns:
        Normalized tensor
    """
    # Calculate min and max for each feature using 5th and 95th percentiles
    min_values = []
    max_values = []
    
    for i in range(tensor.shape(data)[1]):
        col_data = data[:, i]
        # Use stats.percentile to compute percentiles
        p05 = stats.percentile(col_data, 5)
        p95 = stats.percentile(col_data, 95)
        
        min_values.append(p05)
        max_values.append(p95)
    
    # Convert to tensors
    min_tensor = tensor.stack(min_values)
    max_tensor = tensor.stack(max_values)
    
    # Reshape min and max to match data shape for broadcasting
    min_tensor = tensor.reshape(min_tensor, (1, -1))
    max_tensor = tensor.reshape(max_tensor, (1, -1))
    
    # Calculate range
    range_tensor = ops.subtract(max_tensor, min_tensor)
    
    # Avoid division by zero
    epsilon = tensor.convert_to_tensor(1e-8, device=None)  # Use default device
    range_tensor = tensor.maximum(range_tensor, epsilon)
    
    # Normalize data
    normalized_data = ops.divide(ops.subtract(data, min_tensor), range_tensor)
    
    # Clip to ensure values are in [0, 1]
    return ops.clip(normalized_data, 0, 1)


def _dataframe_to_tensor(df: Any, device: Optional[str] = None) -> Any:
    """
    Convert a DataFrame to a tensor.
    
    Args:
        df: Input DataFrame
        device: Device to place tensor on
        
    Returns:
        Tensor representation of DataFrame
    """
    try:
        # For pandas-like DataFrames
        if isinstance(df, pd.DataFrame):
            data = df.values
        else:
            # For other DataFrame-like objects
            data = df.to_numpy()
    except AttributeError:
        # Fallback for other types
        data = df
    
    # Convert to tensor
    return tensor.convert_to_tensor(data, device=device)