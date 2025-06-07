"""
Animated feature processor with visualization capabilities for BigQuery data.

This module provides feature processing with animation and sample tables
for different data types. It's designed to work with the EmberML backend
abstraction system for optimal performance across different hardware.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple # Removed Union, Callable
import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.ops import stats

# Set up logging
logger = logging.getLogger(__name__)


class AnimatedFeatureProcessor:
    """
    Feature processor with animated visualization and sample data tables.
    
    This class processes different types of features (numeric, categorical,
    datetime, etc.) with support for animations and sample tables. It uses
    EmberML's backend-agnostic operations to ensure compatibility across
    different compute environments.
    """
    
    def __init__(
        self,
        visualization_enabled: bool = True,
        sample_tables_enabled: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the animated feature processor.
        
        Args:
            visualization_enabled: Whether to generate visualizations
            sample_tables_enabled: Whether to generate sample data tables
            device: Optional device to place tensors on
        """
        self.visualization_enabled = visualization_enabled
        self.sample_tables_enabled = sample_tables_enabled
        self.device = device
        
        
        # Storage for animation frames
        self.processing_frames: List[Dict[str, Any]] = []
        
        # Storage for sample tables
        self.sample_tables: Dict[str, Dict[str, Any]] = {}
    
    def process_numeric_features(self,
        df: Any,
        columns: List[str],
        with_imputation: bool = True,
        with_outlier_handling: bool = True,
        with_normalization: bool = True
    ) -> Any:
        """
        Process numeric features with animation and sample tables.
        
        Args:
            df: Input DataFrame
            columns: Numeric columns to process
            with_imputation: Whether to perform missing value imputation
            with_outlier_handling: Whether to handle outliers
            with_normalization: Whether to normalize features
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No numeric columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_numeric', 
                                      'Original Numeric Data')
        
        # Convert to tensor for consistent processing
        data = self._dataframe_to_tensor(df[columns])
        
        # Capture initial state for animation
        if self.visualization_enabled:
            self._capture_frame(data, "Initial", "Numeric Features")
        
        # Step 1: Handle missing values
        if with_imputation:
            data_with_imputed_values = self._handle_missing_values(data)
            if self.visualization_enabled:
                self._capture_frame(data_with_imputed_values, 
                                    "Missing Value Imputation", "Numeric Features")
                
            # Create a sample data table for imputed values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    data_with_imputed_values, columns, 'imputed_numeric',
                    'After Missing Value Imputation'
                )
        else:
            data_with_imputed_values = data
        
        # Step 2: Handle outliers using robust methods
        if with_outlier_handling:
            data_without_outliers = self._handle_outliers(data_with_imputed_values)
            if self.visualization_enabled:
                self._capture_frame(data_without_outliers, 
                                    "Outlier Removal", "Numeric Features")
            
            # Create a sample data table for outlier-handled values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    data_without_outliers, columns, 'outlier_handled_numeric',
                    'After Outlier Handling'
                )
        else:
            data_without_outliers = data_with_imputed_values
        
        # Step 3: Normalize features to [0,1] range
        if with_normalization:
            normalized_data = self._normalize_robust(data_without_outliers)
            if self.visualization_enabled:
                self._capture_frame(normalized_data, 
                                    "Robust Normalization", "Numeric Features")
            
            # Create a sample data table for normalized values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    normalized_data, columns, 'normalized_numeric',
                    'After Robust Normalization'
                )
        else:
            normalized_data = data_without_outliers
        
        # Generate animation if enabled
        if self.visualization_enabled:
            self._generate_processing_animation()
        
        return normalized_data
    
    def process_categorical_features(
        self,
        df: Any,
        columns: List[str],
        encoding: str = 'one_hot',
        max_categories_per_column: int = 100
    ) -> Any:
        """
        Process categorical features with animation and sample tables.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to process
            encoding: Encoding method ('one_hot', 'target', 'hash')
            max_categories_per_column: Maximum categories per column for one-hot
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No categorical columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_categorical', 
                                      'Original Categorical Data')
        
        # Process each column separately
        encoded_features_list = []
        encoded_feature_names = []
        
        for col in columns:
            # Get unique values and create mapping
            unique_values = df[col].dropna().unique()
            
            # Skip if too many categories for one-hot encoding
            if encoding == 'one_hot' and len(unique_values) > max_categories_per_column:
                logger.warning(f"Column {col} has {len(unique_values)} unique values, "
                               f"which exceeds the maximum of {max_categories_per_column}. "
                               f"Switching to hash encoding.")
                encoding = 'hash'
            
            # Apply encoding
            if encoding == 'one_hot':
                encoded_features, feature_names = self._one_hot_encode(df, col, unique_values)
            elif encoding == 'target':
                # For now, we'll use one-hot as fallback since target encoding
                # requires a target variable
                encoded_features, feature_names = self._one_hot_encode(df, col, unique_values)
            elif encoding == 'hash':
                encoded_features, feature_names = self._hash_encode(df, col, n_components=16)
            else:
                raise ValueError(f"Unsupported encoding method: {encoding}")
            
            encoded_features_list.append(encoded_features)
            encoded_feature_names.extend(feature_names)
        
        # Combine all encoded features
        if encoded_features_list:
            encoded_data = tensor.concatenate(encoded_features_list, axis=1)
            
            # Create a sample data table for encoded values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    encoded_data, encoded_feature_names, 'encoded_categorical',
                    f'After {encoding.capitalize()} Encoding'
                )
                
            # Generate animation if enabled
            if self.visualization_enabled:
                self._generate_processing_animation()
                
            return encoded_data
        else:
            return tensor.zeros((len(df), 0), device=self.device)
    
    def _dataframe_to_tensor(self, df: Any) -> Any:
        """
        Convert a DataFrame to a tensor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tensor representation of DataFrame
        """
        try:
            # For pandas-like DataFrames
            data = df.values
        except AttributeError:
            # For other DataFrame-like objects
            data = df.to_numpy()
        
        # Convert to tensor
        return tensor.convert_to_tensor(data, device=self.device)
    
    def _handle_missing_values(self, data: Any) -> Any:
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
        if ops.all(ops.logical_not(nan_mask)):
            return data
        
        # Calculate median for each feature
        # First, replace NaNs with zeros for calculation purposes
        data_no_nan = ops.where(nan_mask, tensor.zeros_like(data), data)
        
        # Calculate median for each feature (column)
        medians = []
        for i in range(tensor.shape(data)[1]):
            col_data = data_no_nan[:, i]
            # Sort the data
            sorted_data = stats.sort(col_data)
            # Get median
            n_scalar = tensor.shape(sorted_data)[0] # Assuming scalar int
            n_tensor = tensor.convert_to_tensor(n_scalar, device=self.device) # Use tensor for ops
            two_tensor = tensor.convert_to_tensor(2, dtype=tensor.dtype(n_tensor), device=self.device)
            
            # Use ops.mod for modulo check
            if ops.equal(ops.mod(n_tensor, two_tensor), tensor.zeros_like(n_tensor)):
                # Use ops.floor_divide for integer division, ensure int indices
                idx1 = ops.subtract(ops.floor_divide(n_tensor, two_tensor), tensor.ones_like(n_tensor))
                idx2 = ops.floor_divide(n_tensor, two_tensor)
                # Cast indices to int64 for safety if needed by backend indexing
                idx1_int = tensor.cast(idx1, dtype=tensor.int64)
                idx2_int = tensor.cast(idx2, dtype=tensor.int64)
                median = ops.divide(
                    ops.add(sorted_data[idx1_int], sorted_data[idx2_int]),
                    tensor.convert_to_tensor(2.0, device=self.device) # Ensure device
                )
            else:
                # Use ops.floor_divide for integer division
                idx = ops.floor_divide(n_tensor, two_tensor)
                idx_int = tensor.cast(idx, dtype=tensor.int64) # Cast index
                median = sorted_data[idx_int]
            medians.append(median)
        
        # Convert to tensor
        medians_tensor = tensor.stack(medians)
        
        # Reshape medians to match data shape for broadcasting
        medians_tensor = tensor.reshape(medians_tensor, (1, -1))
        
        # Replace NaNs with medians
        # Create a broadcasted version of medians_tensor
        broadcasted_medians = tensor.tile(medians_tensor, [tensor.shape(data)[0], 1])
        
        return ops.where(nan_mask, broadcasted_medians, data)
    
    def _handle_outliers(self, data: Any) -> Any:
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
            # Sort the data
            sorted_data = stats.sort(col_data)
            n_scalar = tensor.shape(sorted_data)[0] # Assuming scalar int
            n_tensor = tensor.convert_to_tensor(n_scalar, device=self.device) # Use tensor for ops
            four_tensor = tensor.convert_to_tensor(4, dtype=tensor.dtype(n_tensor), device=self.device)
            three_tensor = tensor.convert_to_tensor(3, dtype=tensor.dtype(n_tensor), device=self.device)
            
            # Calculate quartile indices using ops
            q1_idx_tensor = ops.floor_divide(n_tensor, four_tensor)
            q3_idx_tensor = ops.floor_divide(ops.multiply(three_tensor, n_tensor), four_tensor)
            
            # Cast indices to int64 for safety
            q1_idx = tensor.cast(q1_idx_tensor, dtype=tensor.int64)
            q3_idx = tensor.cast(q3_idx_tensor, dtype=tensor.int64)

            # Get quartile values
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            
            q1_values.append(q1)
            q3_values.append(q3)
        
        # Convert to tensors
        q1_tensor = tensor.stack(q1_values)
        q3_tensor = tensor.stack(q3_values)
        
        # Calculate IQR
        iqr_tensor = ops.subtract(q3_tensor, q1_tensor)
        one_point_five = tensor.convert_to_tensor(1.5, dtype=tensor.dtype(iqr_tensor), device=self.device) # Ensure same dtype and device

        # Calculate bounds using ops.multiply
        lower_bound = ops.subtract(q1_tensor, ops.multiply(iqr_tensor, one_point_five))
        upper_bound = ops.add(q3_tensor, ops.multiply(iqr_tensor, one_point_five))
        
        # Reshape bounds to match data shape for broadcasting
        lower_bound = tensor.reshape(lower_bound, (1, -1))
        upper_bound = tensor.reshape(upper_bound, (1, -1))
        
        # Clip values to bounds
        return ops.clip(data, lower_bound, upper_bound)
    
    def _normalize_robust(self, data: Any) -> Any:
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
            # Sort the data
            sorted_data = stats.sort(col_data)
            n_scalar = tensor.shape(sorted_data)[0] # Assuming this returns a scalar (int or 0-d tensor)
            n_tensor = tensor.convert_to_tensor(n_scalar, dtype=tensor.float32, device=self.device) # Ensure float for multiplication
            
            # Calculate percentile indices using tensor ops
            idx_05_float = ops.multiply(tensor.convert_to_tensor(0.05, device=self.device), n_tensor)
            idx_95_float = ops.multiply(tensor.convert_to_tensor(0.95, device=self.device), n_tensor)
            
            # Cast to integer type suitable for indexing
            idx_05_int = tensor.cast(idx_05_float, dtype=tensor.int64) # Use int64 for safety
            idx_95_int = tensor.cast(idx_95_float, dtype=tensor.int64)
            
            # Ensure indices are within bounds [0, n-1] using tensor ops
            zero_tensor = tensor.zeros((), dtype=tensor.int64, device=self.device)
            # Need n as int64 for comparison/min
            n_int64_tensor = tensor.cast(n_tensor, dtype=tensor.int64)
            n_minus_1_tensor = ops.subtract(n_int64_tensor, tensor.ones((), dtype=tensor.int64, device=self.device))

            # Use ops.where for element-wise max/min
            p05_idx = ops.where(ops.less(idx_05_int, zero_tensor), zero_tensor, idx_05_int)
            # Ensure p95_idx doesn't exceed n-1
            p95_idx = ops.where(ops.greater(idx_95_int, n_minus_1_tensor), n_minus_1_tensor, idx_95_int)
            
            # Get percentile values (indexing expects scalar int or 0-d int tensor)
            # Convert tensor indices back to scalar integers if necessary for indexing
            # Assuming the backend tensor indexing supports 0-d tensors directly
            p05 = sorted_data[p05_idx]
            p95 = sorted_data[p95_idx]
            
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
        epsilon = tensor.convert_to_tensor(1e-8, device=self.device)
        range_tensor = tensor.maximum(range_tensor, epsilon)
        
        # Normalize data
        normalized_data = ops.divide(
            ops.subtract(data, min_tensor),
            range_tensor
        )
        
        # Clip to ensure values are in [0, 1]
        return ops.clip(normalized_data, 0, 1)
    
    def _one_hot_encode(
        self,
        df: Any,
        column: str,
        unique_values: Any
    ) -> Tuple[Any, List[str]]:
        """
        One-hot encode a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column to encode
            unique_values: Unique values in the column
            
        Returns:
            Tuple of (encoded_tensor, feature_names)
        """
        # Get column data
        col_data = df[column]
        
        # Initialize encoded data
        n_samples = len(df)
        n_categories = len(unique_values)
        encoded_data = tensor.zeros((n_samples, n_categories), device=self.device)
        
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
                encoded_data = tensor.tensor_scatter_nd_update(
                    encoded_data,
                    tensor.reshape(tensor.convert_to_tensor([[i, idx]]), (1, 2)),
                    tensor.reshape(tensor.convert_to_tensor([1.0]), (1,))
                )
        
        return encoded_data, feature_names
    
    def _hash_encode(
        self,
        df: Any,
        column: str,
        n_components: int = 16,
        seed: int = 42
    ) -> Tuple[Any, List[str]]:
        """
        Hash encode a high-cardinality categorical or identifier column.
        
        Args:
            df: Input DataFrame
            column: Column to encode
            n_components: Number of hash components
            seed: Random seed
            
        Returns:
            Tuple of (encoded_tensor, feature_names)
        """
        # Get column data
        col_data = df[column]
        
        # Initialize encoded data
        n_samples = len(df)
        encoded_data = tensor.zeros((n_samples, n_components), device=self.device)
        
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
            
            # Generate hash value using string concatenation
            # We need to use Python's + operator for string concatenation
            # This is acceptable since we're not operating on tensors
            str_seed = str(seed)
            hash_val = hash(str_value + str_seed)
            
            # Use hash to generate pseudo-random values for each component
            for j in range(n_components):
                # Convert j to string and concatenate
                seed_plus_j = str(seed + j)  # This is acceptable for string operations
                component_hash = hash(str_value + seed_plus_j)
                # Scale to [0, 1] using ops functions
                component_hash_tensor = tensor.convert_to_tensor(component_hash, device=self.device)
                modulo = ops.mod(component_hash_tensor, tensor.convert_to_tensor(10000, device=self.device))
                component_value = ops.divide(modulo, tensor.convert_to_tensor(10000.0, device=self.device))
                
                # Update the tensor at position [i, j]
                encoded_data = tensor.tensor_scatter_nd_update(
                    encoded_data,
                    tensor.reshape(tensor.convert_to_tensor([[i, j]]), (1, 2)),
                    tensor.reshape(tensor.convert_to_tensor([component_value]), (1,))
                )
        
        # Reset seed
        tensor.set_seed(None)
        
        return encoded_data, feature_names
    
    def _create_sample_table(
        self,
        df: Any,
        columns: List[str],
        table_id: str,
        title: str,
        max_rows: int = 5
    ) -> None:
        """
        Create a sample data table from DataFrame columns.

        Args:
            df: Input DataFrame
            columns: Columns to include in the table
            table_id: Unique identifier for the table
            title: Title of the table
            max_rows: Maximum number of rows for the sample table
        """
        if not self.sample_tables_enabled or not columns:
            return

        try:
            # Select columns and take the head
            sample_df = df[columns].head(max_rows).copy()

            # Convert potential tensor columns back for display if needed
            for col in columns:
                try:
                    first_element = sample_df[col].iloc[0]
                    if hasattr(first_element, '__class__') and first_element.__class__.__name__ == 'EmberTensor':
                        try:
                            sample_df[col] = sample_df[col].apply(lambda x: tensor.to_numpy(x) if hasattr(x, '__class__') and x.__class__.__name__ == 'EmberTensor' else x)
                        except Exception as e_inner:
                            logger.debug(f"Could not convert column {col} to numpy for sample table: {e_inner}")
                except IndexError:
                    logger.debug(f"Column {col} is empty, skipping numpy conversion check.")
                except Exception as e_outer:
                     logger.debug(f"Error checking column {col} for tensor type: {e_outer}")

            # Store the sample table
            self.sample_tables[table_id] = {
                'title': title,
                'data': sample_df.to_dict(orient='records'),
                'columns': columns
            }
            logger.debug(f"Created sample table '{table_id}' with title '{title}'")

        except Exception as e:
            logger.error(f"Error creating sample table '{table_id}': {e}", exc_info=True)

    def _create_sample_table_from_tensor(
        self,
        data: Any,
        columns: List[str],
        table_id: str,
        title: str,
        max_rows: int = 5
    ) -> None:
        """
        Create a sample data table from a tensor.

        Args:
            data: Input tensor
            columns: Column names corresponding to tensor columns
            table_id: Unique identifier for the table
            title: Title of the table
            max_rows: Maximum number of rows for the sample table
        """
        if not self.sample_tables_enabled or tensor.shape(data)[1] == 0:
            return

        try:
            # Take the first max_rows
            num_rows_to_take = min(tensor.shape(data)[0], max_rows)
            sample_data_tensor = data[:num_rows_to_take, :]

            # Convert tensor slice to a pandas DataFrame
            try:
                sample_data_np = tensor.to_numpy(sample_data_tensor)
            except Exception as e:
                 logger.warning(f"Could not convert tensor to NumPy for sample table {table_id}, falling back: {e}")
                 sample_data_np = [[tensor.to_numpy(item) for item in row] for row in sample_data_tensor]

            sample_df = pd.DataFrame(sample_data_np, columns=columns)

            # Store the sample table
            self.sample_tables[table_id] = {
                'title': title,
                'data': sample_df.to_dict(orient='records'),
                'columns': columns
            }
            logger.debug(f"Created sample table '{table_id}' from tensor with title '{title}'")

        except Exception as e:
            logger.error(f"Error creating sample table '{table_id}' from tensor: {e}", exc_info=True)

    def _capture_frame(
        self,
        data: Any,
        step_description: str,
        feature_type: str
    ) -> None:
        """
        Capture a frame for the processing animation.

        Args:
            data: Tensor data at the current step
            step_description: Description of the processing step
            feature_type: Type of feature being processed
        """
        if not self.visualization_enabled or tensor.shape(data)[1] == 0:
            return

        frame_data = {
            'step': step_description,
            'feature_type': feature_type,
            # Store a sample for visualization
            'data_sample': (data[:100, :].tolist() if hasattr(data, 'shape') and len(data.shape) > 1 
                            else [row[:] for row in data[:100]]),
            'shape': tensor.shape(data)
        }
        self.processing_frames.append(frame_data)
        logger.debug(f"Captured animation frame for step: {step_description}")


    def _generate_processing_animation(self) -> None:
        """
        Generate or prepare the processing animation from captured frames. (Placeholder)
        """
        if not self.visualization_enabled or not self.processing_frames:
            return

        logger.info(f"Animation frames captured: {len(self.processing_frames)}. "
                    f"Ready for animation generation (implementation pending).")
        # Placeholder for actual animation generation logic

    def get_processing_frames(self) -> List[Dict[str, Any]]:
        """Returns the captured animation frames."""
        return self.processing_frames

    def get_sample_tables(self) -> Dict[str, Dict[str, Any]]:
        """Returns the generated sample tables."""
        return self.sample_tables

    def clear_artifacts(self) -> None:
        """Clears stored animation frames and sample tables."""
        self.processing_frames = []
        self.sample_tables = {}
        logger.debug("Cleared animation frames and sample tables.")
