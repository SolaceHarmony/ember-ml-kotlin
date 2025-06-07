"""
Generic Feature Extraction Library

This module provides tools for generic feature extraction from CSV files,
maintaining the ability to process data regardless of column count or specific meanings.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from ember_ml import ops
from ember_ml.nn.tensor.types import TensorLike


class GenericCSVLoader:
    """
    Flexible CSV loader that preserves the schema-agnostic approach.
    
    This class handles loading CSV files with various compression formats
    and automatically detects data types.
    """
    
    def __init__(self, compression_support: bool = True):
        """
        Initialize the CSV loader with optional compression support.
        
        Args:
            compression_support: Whether to support compressed files (.gz, .zip)
        """
        self.compression_support = compression_support
        
    def load_csv(self, 
                file_path: str, 
                header_file: Optional[str] = None, 
                index_col: Optional[str] = None,
                datetime_cols: Optional[List[str]] = None,
                encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load CSV data with automatic type detection.
        
        Args:
            file_path: Path to the CSV file
            header_file: Optional path to header definition file
            index_col: Optional column to use as index (e.g., timestamp)
            datetime_cols: Optional list of columns to parse as datetime
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with appropriate data types
        """
        # Determine compression type from extension if supported
        compression = None
        if self.compression_support:
            if file_path.endswith('.gz'):
                compression = 'gzip'
            elif file_path.endswith('.zip'):
                compression = 'zip'
        
        # Load header definitions if provided
        column_types = {}
        if header_file and os.path.exists(header_file):
            column_types = self._parse_header_file(header_file)
        
        # Load CSV with pandas
        try:
            df = pd.read_csv(file_path, compression=compression, encoding=encoding)
            print(f"Successfully loaded {file_path} with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
        
        # Apply column types if defined
        for col, dtype in column_types.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        
        # Convert datetime columns if specified
        if datetime_cols:
            for col in datetime_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        print(f"Converted {col} to datetime")
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to datetime: {e}")
        
        # Set index if specified
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
            print(f"Set {index_col} as index")
        
        return df
    
    def _parse_header_file(self, header_file: str) -> Dict[str, str]:
        """
        Parse header definition file to get column types.
        
        Args:
            header_file: Path to header definition file
            
        Returns:
            Dictionary mapping column names to data types
        """
        column_types = {}
        
        try:
            with open(header_file, 'r') as f:
                lines = f.readlines()
                
            # Simple parsing: assume each line contains a column name
            # We'll just store the column names without specific types
            # and let pandas infer the types
            for line in lines:
                col_name = line.strip()
                if col_name:  # Skip empty lines
                    column_types[col_name] = None
                    
            print(f"Parsed {len(column_types)} column names from {header_file}")
                    
        except Exception as e:
            print(f"Error parsing header file: {e}")
            
        return column_types


class GenericTypeDetector:
    """
    Detects column types in a dataframe without prior knowledge.
    
    This class categorizes columns into numeric, datetime, categorical, and boolean types.
    """
    
    def __init__(self):
        """Initialize the type detector."""
        pass
        
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect column types in a dataframe.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of lists categorizing columns by type
        """
        numeric_cols = []
        datetime_cols = []
        categorical_cols = []
        boolean_cols = []
        
        for col_name, col_type in df.dtypes.items():
            if self._is_numeric_type(col_type):
                numeric_cols.append(col_name)
            elif self._is_datetime_type(col_type):
                datetime_cols.append(col_name)
            elif self._is_boolean_type(col_type):
                boolean_cols.append(col_name)
            else:
                categorical_cols.append(col_name)
        
        # Additional heuristics for categorical columns
        for col in numeric_cols[:]:  # Use a copy to avoid modification during iteration
            # If a numeric column has few unique values relative to its size,
            # it might be better treated as categorical
            if col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    numeric_cols.remove(col)
                    categorical_cols.append(col)
                    print(f"Reclassified {col} from numeric to categorical (unique ratio: {unique_ratio:.4f})")
        
        result = {
            'numeric': numeric_cols,
            'datetime': datetime_cols,
            'categorical': categorical_cols,
            'boolean': boolean_cols
        }
        
        print(f"Detected {len(numeric_cols)} numeric, {len(datetime_cols)} datetime, "
              f"{len(categorical_cols)} categorical, and {len(boolean_cols)} boolean columns")
        
        return result
    
    def _is_numeric_type(self, col_type) -> bool:
        """
        Check if column type is numeric.
        
        Args:
            col_type: Column data type
            
        Returns:
            True if numeric, False otherwise
        """
        return pd.api.types.is_numeric_dtype(col_type)
    
    def _is_datetime_type(self, col_type) -> bool:
        """
        Check if column type is datetime.
        
        Args:
            col_type: Column data type
            
        Returns:
            True if datetime, False otherwise
        """
        return pd.api.types.is_datetime64_any_dtype(col_type)
    
    def _is_boolean_type(self, col_type) -> bool:
        """
        Check if column type is boolean.
        
        Args:
            col_type: Column data type
            
        Returns:
            True if boolean, False otherwise
        """
        return pd.api.types.is_bool_dtype(col_type)


class GenericFeatureEngineer:
    """
    Creates features based on detected column types.
    
    This class handles feature engineering for different column types:
    - Datetime columns: Creates cyclical features (hour, day, month)
    - Categorical columns: Performs one-hot encoding
    - Boolean columns: Ensures proper boolean type
    - Numeric columns: Leaves as-is
    """
    
    def __init__(self, max_categories: int = 100, handle_unknown: str = 'ignore'):
        """
        Initialize the feature engineer.
        
        Args:
            max_categories: Maximum number of categories to one-hot encode
            handle_unknown: Strategy for handling unknown categories ('ignore' or 'error')
        """
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.categorical_mappings = {}  # Store mappings for categorical columns
        
    def engineer_features(self,
                         df: pd.DataFrame,
                         column_types: Dict[str, List[str]],
                         drop_original: bool = True) -> pd.DataFrame:
        """
        Create features based on column types.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types from GenericTypeDetector
            drop_original: Whether to drop original columns after engineering
            
        Returns:
            DataFrame with engineered features
        """
        df_processed = df.copy()
        
        # Process datetime columns
        for col in column_types.get('datetime', []):
            if col in df_processed.columns:
                df_processed = self._create_datetime_features(df_processed, col)
                if drop_original:
                    df_processed = df_processed.drop(columns=[col])
        
        # Process categorical columns
        for col in column_types.get('categorical', []):
            if col in df_processed.columns:
                df_processed = self._encode_categorical(df_processed, col)
                if drop_original:
                    df_processed = df_processed.drop(columns=[col])
        
        # Process boolean columns - ensure they're properly typed
        for col in column_types.get('boolean', []):
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(float)
        
        # Return the processed dataframe
        return df_processed
    
    def _create_datetime_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Create cyclical features from datetime column.
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with added cyclical features
        """
        if col not in df.columns:
            return df
        
        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
                return df
        
        # Create cyclical features using sine and cosine transformations
        # Hour of day (0-23)
        df[f'{col}_sin_hour'] = ops.sin(2 * ops.pi * df[col].dt.hour / 23.0)
        df[f'{col}_cos_hour'] = ops.cos(2 * ops.pi * df[col].dt.hour / 23.0)
        
        # Day of week (0-6)
        df[f'{col}_sin_dayofweek'] = ops.sin(2 * ops.pi * df[col].dt.dayofweek / 6.0)
        df[f'{col}_cos_dayofweek'] = ops.cos(2 * ops.pi * df[col].dt.dayofweek / 6.0)
        
        # Day of month (1-31)
        df[f'{col}_sin_day'] = ops.sin(2 * ops.pi * (df[col].dt.day - 1) / 30.0)
        df[f'{col}_cos_day'] = ops.cos(2 * ops.pi * (df[col].dt.day - 1) / 30.0)
        
        # Month (1-12)
        df[f'{col}_sin_month'] = ops.sin(2 * ops.pi * (df[col].dt.month - 1) / 11.0)
        df[f'{col}_cos_month'] = ops.cos(2 * ops.pi * (df[col].dt.month - 1) / 11.0)
        
        print(f"Created cyclical features for datetime column '{col}'")
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        One-hot encode categorical column.
        
        Args:
            df: Input DataFrame
            col: Categorical column name
            
        Returns:
            DataFrame with one-hot encoded features
        """
        if col not in df.columns:
            return df
        
        # Check number of unique values
        n_unique = df[col].nunique()
        if n_unique > self.max_categories:
            print(f"Warning: Column '{col}' has {n_unique} unique values, "
                  f"which exceeds the maximum of {self.max_categories}. Skipping encoding.")
            return df
        
        # Handle missing values
        df[col] = df[col].fillna('MISSING')
        
        # Get unique values for this column
        unique_values = df[col].unique()
        
        # Store mapping for future use (e.g., with new data)
        self.categorical_mappings[col] = unique_values
        
        # Create one-hot encoded columns
        for value in unique_values:
            # Create safe column name
            safe_value = str(value).replace(' ', '_').replace('-', '_').replace('/', '_')
            new_col = f"{col}_{safe_value}"
            df[new_col] = (df[col] == value).astype(float)
        
        print(f"One-hot encoded categorical column '{col}' with {len(unique_values)} unique values")
        return df
    
    def get_feature_names(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> List[str]:
        """
        Get the names of all features after engineering.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            
        Returns:
            List of feature names after engineering
        """
        # Start with numeric columns (which don't change)
        feature_names = column_types.get('numeric', []).copy()
        
        # Add boolean columns
        feature_names.extend(column_types.get('boolean', []))
        
        # Add engineered datetime features
        for col in column_types.get('datetime', []):
            feature_names.extend([
                f'{col}_sin_hour', f'{col}_cos_hour',
                f'{col}_sin_dayofweek', f'{col}_cos_dayofweek',
                f'{col}_sin_day', f'{col}_cos_day',
                f'{col}_sin_month', f'{col}_cos_month'
            ])
        
        # Add one-hot encoded categorical features
        for col in column_types.get('categorical', []):
            if col in self.categorical_mappings:
                for value in self.categorical_mappings[col]:
                    safe_value = str(value).replace(' ', '_').replace('-', '_').replace('/', '_')
                    feature_names.append(f"{col}_{safe_value}")
            else:
                # If we haven't seen this column yet, estimate based on the dataframe
                if col in df.columns:
                    unique_values = df[col].fillna('MISSING').unique()
                    if len(unique_values) <= self.max_categories:
                        for value in unique_values:
                            safe_value = str(value).replace(' ', '_').replace('-', '_').replace('/', '_')
                            feature_names.append(f"{col}_{safe_value}")
        
        return feature_names


class TemporalStrideProcessor:
    """
    Processes data into multi-stride temporal representations.
    
    This class creates sliding windows with different strides and applies
    PCA for dimensionality reduction, enabling multi-scale temporal analysis.
    """
    
    def __init__(self, window_size: int = 5, stride_perspectives: List[int] = None,
                 pca_components: Optional[int] = None):
        """
        Initialize the temporal stride processor.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths to use
            pca_components: Number of PCA components (if None, will be calculated)
        """
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives or [1, 3, 5]
        self.pca_components = pca_components
        self.pca_models = {}  # Store PCA models for each stride
        
    def process_batch(self, data: TensorLike) -> Dict[int, TensorLike]:
        """
        Process data into multi-stride temporal representations.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        results = {}
        
        for stride in self.stride_perspectives:
            # Extract windows using stride length
            windows = self._create_strided_sequences(data, stride)
            
            if not windows:
                print(f"Warning: No windows created for stride {stride}")
                continue
                
            # Convert to array and apply PCA blending
            windows_array = tensor.convert_to_tensor(windows)
            results[stride] = self._apply_pca_blend(windows_array, stride)
            
            print(f"Created {len(windows)} windows with stride {stride}, "
                  f"shape after PCA: {results[stride].shape}")
            
        return results
    
    def _create_strided_sequences(self, data: TensorLike, stride: int) -> List[TensorLike]:
        """
        Create sequences with the given stride.
        
        Args:
            data: Input data array
            stride: Stride length
            
        Returns:
            List of windowed sequences
        """
        num_samples = len(data)
        windows = []
        
        # Skip if data is too small for even one window
        if num_samples < self.window_size:
            print(f"Warning: Data length ({num_samples}) is smaller than window size ({self.window_size})")
            return windows
        
        for i in range(0, num_samples - self.window_size + 1, stride):
            windows.append(data[i:i+self.window_size])
            
        return windows
    
    def _apply_pca_blend(self, window_batch: TensorLike, stride: int) -> TensorLike:
        """
        Apply PCA-based temporal blending.
        
        Args:
            window_batch: Batch of windows (batch_size x window_size x features)
            stride: Stride length
            
        Returns:
            PCA-transformed data
        """
        batch_size, window_size, feature_dim = window_batch.shape
        
        # Reshape for PCA: [batch_size, window_size * feature_dim]
        flat_windows = window_batch.reshape(batch_size, -1)
        
        # Ensure PCA is fit
        if stride not in self.pca_models:
            # Calculate appropriate number of components
            if self.pca_components is None:
                # Use half the flattened dimension, but cap at 32 components
                n_components = min(flat_windows.shape[1] // 2, 32)
                # Ensure we don't try to extract more components than samples
                n_components = min(n_components, batch_size - 1)
            else:
                n_components = min(self.pca_components, batch_size - 1, flat_windows.shape[1])
                
            print(f"Fitting PCA for stride {stride} with {n_components} components")
            self.pca_models[stride] = PCA(n_components=n_components)
            self.pca_models[stride].fit(flat_windows)
            
        # Transform the data
        return self.pca_models[stride].transform(flat_windows)
    
    def get_explained_variance(self, stride: int) -> Optional[float]:
        """
        Get the explained variance ratio for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Sum of explained variance ratios or None if PCA not fit
        """
        if stride in self.pca_models:
            return sum(self.pca_models[stride].explained_variance_ratio_)
        return None
    
    def get_feature_importance(self, stride: int) -> Optional[TensorLike]:
        """
        Get feature importance for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Array of feature importance scores or None if PCA not fit
        """
        if stride in self.pca_models:
            # Calculate feature importance as the sum of absolute component weights
            return ops.abs(self.pca_models[stride].components_).sum(axis=0)
        return None


# Simple test function to demonstrate usage
def test_feature_extraction(csv_path: str, header_file: Optional[str] = None,
                           window_size: int = 5, stride_perspectives: List[int] = None):
    """
    Test the complete feature extraction pipeline.
    
    Args:
        csv_path: Path to CSV file
        header_file: Optional path to header definition file
        window_size: Size of sliding window for temporal processing
        stride_perspectives: List of stride lengths to use
    """
    # Load CSV
    loader = GenericCSVLoader()
    df = loader.load_csv(csv_path, header_file)
    
    # Detect types
    detector = GenericTypeDetector()
    column_types = detector.detect_column_types(df)
    
    # Print sample of each type
    print("\nSample of each column type:")
    for type_name, cols in column_types.items():
        if cols:
            sample_col = cols[0]
            print(f"\n{type_name.capitalize()} column '{sample_col}':")
            print(df[sample_col].head())
    
    # Engineer features
    engineer = GenericFeatureEngineer()
    df_engineered = engineer.engineer_features(df, column_types)
    
    # Print information about engineered features
    print(f"\nOriginal dataframe shape: {df.shape}")
    print(f"Engineered dataframe shape: {df_engineered.shape}")
    
    # Print sample of engineered features
    print("\nSample of engineered features:")
    feature_categories = {
        'datetime': ['_sin_', '_cos_'],
        'categorical': column_types['categorical'],
        'numeric': column_types['numeric'],
        'boolean': column_types['boolean']
    }
    
    for category, patterns in feature_categories.items():
        if isinstance(patterns, list) and len(patterns) > 0:
            if category in ['datetime', 'categorical']:
                # For datetime and categorical, look for pattern in column names
                if category == 'datetime':
                    cols = [col for col in df_engineered.columns if any(pattern in col for pattern in patterns)]
                else:
                    cols = [col for col in df_engineered.columns
                           if any(col.startswith(f"{cat_col}_") for cat_col in patterns)]
            else:
                # For numeric and boolean, use the columns directly
                cols = patterns
                
            if cols:
                sample_col = cols[0]
                print(f"\n{category.capitalize()} feature '{sample_col}':")
                print(df_engineered[sample_col].head())
    
    # Apply temporal stride processing
    print("\n--- Temporal Stride Processing ---")
    
    # Get numeric features for temporal processing
    numeric_features = [col for col in df_engineered.columns
                       if pd.api.types.is_numeric_dtype(df_engineered[col].dtype)]
    
    if not numeric_features:
        print("No numeric features available for temporal processing")
        return df, column_types, df_engineered, None
    
    # Convert to numpy array for processing
    data = df_engineered[numeric_features].values
    
    # Initialize temporal processor
    processor = TemporalStrideProcessor(
        window_size=window_size,
        stride_perspectives=stride_perspectives
    )
    
    # Process data
    stride_perspectives = processor.process_batch(data)
    
    # Print information about stride perspectives
    print("\nStride Perspectives:")
    for stride, perspective_data in stride_perspectives.items():
        explained_variance = processor.get_explained_variance(stride)
        variance_str = f"{explained_variance:.2f}" if explained_variance is not None else "N/A"
        print(f"Stride {stride}: Shape {perspective_data.shape}, "
              f"Explained Variance: {variance_str}")
    
    return df, column_types, df_engineered, stride_perspectives


if __name__ == "__main__":
    # This will run if the script is executed directly
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generic Feature Extraction')
    parser.add_argument('csv_path', help='Path to CSV file')
    parser.add_argument('--header-file', help='Path to header definition file')
    parser.add_argument('--window-size', type=int, default=5, help='Size of sliding window for temporal processing')
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 3, 5],
                        help='List of stride lengths to use (e.g., --strides 1 3 5)')
    
    if len(sys.argv) > 1:
        args = parser.parse_args()
        test_feature_extraction(
            args.csv_path,
            args.header_file,
            window_size=args.window_size,
            stride_perspectives=args.strides
        )
    else:
        parser.print_help()