"""
BigQuery visualization utilities for Ember ML.

This module provides functions for visualizing BigQuery data and feature processing
for use in Ember ML models.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ember_ml.nn import tensor

# Set up logging
logger = logging.getLogger(__name__)


def create_sample_table(
    data: Any,
    columns: List[str],
    table_id: str,
    title: str,
    max_rows: int = 5
) -> Dict[str, Any]:
    """
    Create a sample data table from DataFrame columns.
    
    Args:
        data: Input DataFrame
        columns: Columns to include in the table
        table_id: Unique identifier for the table
        title: Title of the table
        max_rows: Maximum number of rows for the sample table
        
    Returns:
        Dictionary with sample table data
    """
    try:
        # Select columns and take the head
        sample_df = data[columns].head(max_rows).copy()
        
        # Convert to dictionary
        sample_data = {
            'title': title,
            'data': sample_df.to_dict('records'),
            'columns': columns
        }
        
        logger.debug(f"Created sample table '{table_id}' with title '{title}'")
        return sample_data
    
    except Exception as e:
        logger.error(f"Error creating sample table '{table_id}': {e}")
        return {
            'title': title,
            'data': [],
            'columns': columns,
            'error': str(e)
        }


def create_sample_table_from_tensor(
    tensor_data: Any,
    feature_names: List[str],
    table_id: str,
    title: str,
    max_rows: int = 5
) -> Dict[str, Any]:
    """
    Create a sample data table from a tensor.
    
    Args:
        tensor_data: Input tensor
        feature_names: Column names corresponding to tensor columns
        table_id: Unique identifier for the table
        title: Title of the table
        max_rows: Maximum number of rows for the sample table
        
    Returns:
        Dictionary with sample table data
    """
    try:
        # Take the first max_rows
        num_rows_to_take = min(tensor.shape(tensor_data)[0], max_rows)
        sample_data_tensor = tensor_data[:num_rows_to_take, :]
        
        # Convert tensor to numpy array
        sample_data_np = tensor.to_numpy(sample_data_tensor)
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(sample_data_np, columns=feature_names)
        
        # Convert to dictionary
        sample_data = {
            'title': title,
            'data': sample_df.to_dict('records'),
            'columns': feature_names
        }
        
        logger.debug(f"Created sample table '{table_id}' from tensor with title '{title}'")
        return sample_data
    
    except Exception as e:
        logger.error(f"Error creating sample table '{table_id}' from tensor: {e}")
        return {
            'title': title,
            'data': [],
            'columns': feature_names,
            'error': str(e)
        }


def capture_frame(
    data: Any,
    frame_id: str,
    frame_type: str
) -> Dict[str, Any]:
    """
    Capture a frame for the processing animation.
    
    Args:
        data: Tensor data at the current step
        frame_id: Unique identifier for the frame
        frame_type: Type of frame
        
    Returns:
        Dictionary with frame data
    """
    try:
        
        # Create frame data
        frame_data = {
            'id': frame_id,
            'type': frame_type,
            'shape': tensor.shape(data),
            'data_sample': data[:min(100, data.shape[0]), :].tolist()
        }
        
        logger.debug(f"Captured frame '{frame_id}' of type '{frame_type}'")
        return frame_data
    
    except Exception as e:
        logger.error(f"Error capturing frame '{frame_id}': {e}")
        return {
            'id': frame_id,
            'type': frame_type,
            'error': str(e)
        }


def generate_processing_animation(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate metadata for processing animation.
    
    Args:
        frames: List of frame data dictionaries
        
    Returns:
        Dictionary with animation metadata
    """
    try:
        # Create animation metadata
        animation_data = {
            'frame_count': len(frames),
            'frames': frames,
            'frame_types': list(set(frame['type'] for frame in frames if 'type' in frame))
        }
        
        logger.info(f"Generated animation metadata with {len(frames)} frames")
        return animation_data
    
    except Exception as e:
        logger.error(f"Error generating animation metadata: {e}")
        return {
            'error': str(e)
        }