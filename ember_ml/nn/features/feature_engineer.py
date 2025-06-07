"""
Generic Feature Engineer

This module provides a class for creating features based on detected column types.
"""

import pandas as pd
from typing import Dict, List, Any
from ember_ml import ops
from ember_ml.nn import tensor
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
        self.categorical_mappings: Dict[str, Any] = {}  # Store mappings for categorical columns
        
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
        two_pi = ops.multiply(tensor.convert_to_tensor(2.0), ops.pi)
        
        # Hour of day (0-23)
        df[f'{col}_sin_hour'] = ops.sin(ops.multiply(two_pi, ops.divide(tensor.convert_to_tensor(df[col].dt.hour), tensor.convert_to_tensor(23.0))))
        df[f'{col}_cos_hour'] = ops.cos(ops.multiply(two_pi, ops.divide(tensor.convert_to_tensor(df[col].dt.hour), tensor.convert_to_tensor(23.0))))
        
        # Day of week (0-6)
        df[f'{col}_sin_dayofweek'] = ops.sin(ops.multiply(two_pi, ops.divide(tensor.convert_to_tensor(df[col].dt.dayofweek), tensor.convert_to_tensor(6.0))))
        df[f'{col}_cos_dayofweek'] = ops.cos(ops.multiply(two_pi, ops.divide(tensor.convert_to_tensor(df[col].dt.dayofweek), tensor.convert_to_tensor(6.0))))
        
        # Day of month (1-31)
        df[f'{col}_sin_day'] = ops.sin(ops.multiply(two_pi, ops.divide(ops.subtract(tensor.convert_to_tensor(df[col].dt.day), tensor.convert_to_tensor(1)), tensor.convert_to_tensor(30.0))))
        df[f'{col}_cos_day'] = ops.cos(ops.multiply(two_pi, ops.divide(ops.subtract(tensor.convert_to_tensor(df[col].dt.day), tensor.convert_to_tensor(1)), tensor.convert_to_tensor(30.0))))
        
        # Month (1-12)
        df[f'{col}_sin_month'] = ops.sin(ops.multiply(two_pi, ops.divide(ops.subtract(tensor.convert_to_tensor(df[col].dt.month), tensor.convert_to_tensor(1)), tensor.convert_to_tensor(11.0))))
        df[f'{col}_cos_month'] = ops.cos(ops.multiply(two_pi, ops.divide(ops.subtract(tensor.convert_to_tensor(df[col].dt.month), tensor.convert_to_tensor(1)), tensor.convert_to_tensor(11.0))))
        
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