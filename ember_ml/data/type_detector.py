"""
Generic Type Detector

This module provides a class for detecting column types in a dataframe without prior knowledge.
"""

import pandas as pd
from typing import Dict, List


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
                numeric_cols.append(str(col_name))
            elif self._is_datetime_type(col_type):
                datetime_cols.append(str(col_name))
            elif self._is_boolean_type(col_type):
                boolean_cols.append(str(col_name))
            else:
                categorical_cols.append(str(col_name))
        
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