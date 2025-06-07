"""
Generic CSV Loader

This module provides a flexible CSV loader that preserves the schema-agnostic approach.
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union, Any

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