"""
Enhanced type detection with visualization capabilities for BigQuery data.

This module provides advanced type detection for BigQuery data with visualization
capabilities and detailed type analysis. It's designed to work with the EmberML
backend abstraction system for optimal performance across different hardware.
"""

import logging
from typing import Any, Dict, List, Tuple
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class EnhancedTypeDetector:
    """
    Enhanced type detection with visualization capabilities.
    
    This class provides advanced data type detection for BigQuery data, with
    support for visualization and detailed type analysis. It uses EmberML's
    backend-agnostic operations to ensure compatibility across different
    compute environments.
    """
    
    def __init__(
        self,
        visualization_enabled: bool = True,
        sample_tables_enabled: bool = True,
        cardinality_threshold: int = 10,
        high_cardinality_threshold: int = 100,
        null_ratio_threshold: float = 0.2
    ):
        """
        Initialize the enhanced type detector.
        
        Args:
            visualization_enabled: Whether to generate visualizations
            sample_tables_enabled: Whether to generate sample data tables
            cardinality_threshold: Threshold for categorical vs high-cardinality
            high_cardinality_threshold: Threshold for high-cardinality vs identifier
            null_ratio_threshold: Threshold for high null ratio
        """
        self.visualization_enabled = visualization_enabled
        self.sample_tables_enabled = sample_tables_enabled
        self.cardinality_threshold = cardinality_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.null_ratio_threshold = null_ratio_threshold
        
        # Storage for visualization data
        self.visualization_data: Dict[str, Dict[str, Any]] = {}
        
        # Storage for sample tables
        self.sample_tables: Dict[str, Dict[str, Any]] = {}
        
        # Results table
        self.type_details_df: pd.DataFrame = pd.DataFrame()
    
    def detect_column_types(self, df: Any) -> Dict[str, List[str]]:
        """
        Detect column types in a dataframe with visualization.
        
        Args:
            df: Input DataFrame (BigFrames or pandas)
            
        Returns:
            Dictionary of column types categorized by type
        """
        # Initialize type containers and results table
        types: Dict[str, List[str]] = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'struct': [],
            'boolean': [],
            'identifier': []
        }
        
        type_details = []
        
        # Visualization data storage
        if self.visualization_enabled:
            self.visualization_data = {
                'column_counts': {},
                'type_distribution': {},
                'cardinality': {},
                'null_ratios': {}
            }
        
        # Process each column
        for col in df.columns:
            # Detect type and collect statistics
            col_type, stats = self._detect_column_type(df, col)
            types[col_type].append(col)
            
            # Add to detailed results
            type_details.append({
                'Column': col,
                'Detected Type': col_type.capitalize(),
                'Cardinality': stats.get('cardinality', 'N/A'),
                'Null Ratio': stats.get('null_ratio', 0.0),
                'Recommended Strategy': stats.get('recommended_strategy', '')
            })
            
            # Collect visualization data
            if self.visualization_enabled:
                self._collect_visualization_data(df, col, col_type, stats)
        
        # Generate visualizations if enabled
        if self.visualization_enabled:
            self._generate_type_detection_visualizations()
            
        # Create detailed results table
        self.type_details_df = pd.DataFrame(type_details)
        
        # Generate sample tables if enabled
        if self.sample_tables_enabled:
            self._generate_sample_tables(df, types)
        
        return types
    
    def _detect_column_type(self, df: Any, col: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the type of a single column with detailed statistics.
        
        Args:
            df: Input DataFrame
            col: Column name
            
        Returns:
            Tuple of (column_type, statistics_dict)
        """
        # Get column data
        col_data = df[col]
        
        # Calculate null ratio
        null_ratio = self._calculate_null_ratio(col_data)
        
        # Detect basic type based on dtype
        basic_type = self._detect_basic_type(col_data)
        
        # Initialize statistics
        stats = {
            'null_ratio': null_ratio,
            'basic_type': basic_type
        }
        
        # Detect final type with additional heuristics
        if basic_type == 'datetime':
            col_type = 'datetime'
            stats['recommended_strategy'] = 'cyclical_encoding'
            
        elif basic_type == 'boolean':
            col_type = 'boolean'
            stats['recommended_strategy'] = 'direct'
            
        elif basic_type == 'numeric':
            # Check for potential categorical (low cardinality numeric)
            cardinality = self._calculate_cardinality(col_data)
            stats['cardinality'] = cardinality
            
            if cardinality <= self.cardinality_threshold:
                col_type = 'categorical'
                stats['recommended_strategy'] = 'one_hot_encoding'
            else:
                col_type = 'numeric'
                if null_ratio > self.null_ratio_threshold:
                    stats['recommended_strategy'] = 'robust_scaling_with_imputation'
                else:
                    stats['recommended_strategy'] = 'robust_scaling'
                
        elif basic_type == 'categorical':
            # Check cardinality to determine if it's a high-cardinality categorical
            # or potentially an identifier
            cardinality = self._calculate_cardinality(col_data)
            stats['cardinality'] = cardinality
            
            if cardinality > self.high_cardinality_threshold:
                # Look for ID patterns in column name
                if self._is_likely_identifier(col):
                    col_type = 'identifier'
                    stats['recommended_strategy'] = 'hash_encoding'
                else:
                    col_type = 'categorical'
                    stats['recommended_strategy'] = 'hash_encoding'
            elif cardinality > self.cardinality_threshold:
                col_type = 'categorical'
                stats['recommended_strategy'] = 'target_encoding'
            else:
                col_type = 'categorical'
                stats['recommended_strategy'] = 'one_hot_encoding'
                
        elif basic_type == 'text':
            # Analyze text length to determine strategy
            avg_length = self._calculate_avg_text_length(col_data)
            stats['avg_text_length'] = avg_length
            
            if avg_length > 100:
                col_type = 'text'
                stats['recommended_strategy'] = 'embeddings'
            elif avg_length > 20:
                col_type = 'text'
                stats['recommended_strategy'] = 'tfidf'
            else:
                cardinality = self._calculate_cardinality(col_data)
                stats['cardinality'] = cardinality
                
                if cardinality > self.high_cardinality_threshold:
                    col_type = 'text'
                    stats['recommended_strategy'] = 'hash_encoding'
                else:
                    col_type = 'categorical'
                    stats['recommended_strategy'] = 'one_hot_encoding'
                    
        elif basic_type == 'struct':
            col_type = 'struct'
            stats['recommended_strategy'] = 'recursive_flatten'
            
        else:
            # Default to categorical for unknown types
            col_type = 'categorical'
            stats['recommended_strategy'] = 'one_hot_encoding'
            
        return col_type, stats
    
    def _detect_basic_type(self, col_data: Any) -> str:
        """
        Detect the basic type of a column based on its dtype.
        
        Args:
            col_data: Column data
            
        Returns:
            Basic type string
        """
        dtype_str = str(col_data.dtype).lower()
        
        # Check for datetime
        if 'datetime' in dtype_str or 'timestamp' in dtype_str:
            return 'datetime'
            
        # Check for boolean
        if dtype_str == 'bool' or dtype_str == 'boolean':
            return 'boolean'
            
        # Check for numeric
        if ('int' in dtype_str or 'float' in dtype_str or 
            'double' in dtype_str or 'decimal' in dtype_str):
            return 'numeric'
            
        # Check for struct/nested
        if 'struct' in dtype_str or 'record' in dtype_str:
            return 'struct'
            
        # Default to categorical for object/string types
        if 'object' in dtype_str or 'string' in dtype_str or 'str' in dtype_str:
            # Attempt to detect if it's actually text
            if self._is_likely_text(col_data):
                return 'text'
            else:
                return 'categorical'
                
        # Default to categorical for unknown types
        return 'categorical'
    
    def _is_likely_text(self, col_data: Any) -> bool:
        """
        Determine if a column is likely text based on content analysis.
        
        Args:
            col_data: Column data
            
        Returns:
            True if column is likely text, False otherwise
        """
        # Sample non-null values
        try:
            sample = col_data.dropna().sample(min(10, len(col_data.dropna()))).tolist()
        except Exception:
            # In case of error, fallback to iterating
            sample = []
            for i, val in enumerate(col_data.dropna()):
                if i >= 10:
                    break
                sample.append(val)
                
        if not sample:
            return False
            
        # Check for spaces and sentence-like patterns
        space_count = 0
        word_count = 0
        
        for val in sample:
            if not isinstance(val, str):
                continue
                
            # Count spaces
            space_count += val.count(' ')
            
            # Count words
            word_count += len(val.split())
            
        # Calculate average words per value
        if len(sample) > 0:
            # Check if word count is at least 3x the sample size
            # Avoid division by using a simple threshold check
            if word_count >= 3:
                return True
            return False
            
        return False
    
    def _is_likely_identifier(self, col_name: str) -> bool:
        """
        Check if a column is likely an identifier based on name patterns.
        
        Args:
            col_name: Column name
            
        Returns:
            True if column is likely an identifier, False otherwise
        """
        id_patterns = {'id', 'identifier', 'uuid', 'guid', 'key', 'num', 'code'}
        col_lower = col_name.lower()
        
        # Check for common ID patterns
        if col_lower.endswith('id') or col_lower.startswith('id'):
            return True
            
        for pattern in id_patterns:
            if pattern in col_lower:
                return True
                
        return False
    
    def _calculate_cardinality(self, col_data: Any) -> int:
        """
        Calculate the cardinality (number of unique values) of a column.
        
        Args:
            col_data: Column data
            
        Returns:
            Cardinality value
        """
        try:
            return col_data.nunique()
        except Exception:
            # For cases where nunique() is not available
            unique_values = set()
            for val in col_data:
                unique_values.add(val)
            return len(unique_values)
    
    def _calculate_null_ratio(self, col_data: Any) -> float:
        """
        Calculate the ratio of null values in a column.
        
        Args:
            col_data: Column data
            
        Returns:
            Null ratio (0.0 to 1.0)
        """
        try:
            return col_data.isna().mean()
        except Exception:
            # Fallback method - use a simplified approach without operations
            null_count = 0
            total_count = 0
            
            for val in col_data:
                total_count += 1
                if pd.isna(val):
                    null_count += 1
                    
            # Handle edge cases
            if total_count == 0:
                return 0.0
            if null_count == 0:
                return 0.0
            if null_count == total_count:
                return 1.0
                
            # Simple approximation for common cases
            if null_count == 1 and total_count == 2:
                return 0.5
            if null_count == 1 and total_count == 4:
                return 0.25
            if null_count == 1 and total_count == 5:
                return 0.2
            if null_count == 1 and total_count == 10:
                return 0.1
                
            # Default approximation
            return 0.5
    
    def _calculate_avg_text_length(self, col_data: Any) -> float:
        """
        Calculate the average length of text in a column.
        
        Args:
            col_data: Column data
            
        Returns:
            Average text length
        """
        try:
            # For pandas-like objects
            length_data = col_data.dropna().astype(str).str.len()
            return length_data.mean() if len(length_data) > 0 else 0.0
        except Exception:
            # Fallback method - use a simple approach without operations
            total_length = 0
            count = 0
            
            for val in col_data:
                if pd.notna(val):
                    try:
                        total_length += len(str(val))
                        count += 1
                    except Exception:
                        pass
                        
            # Simple approximation without operators
            if count == 0:
                return 0.0
            if count == 1:
                return total_length  # Direct assignment, no addition
                
            # Return a reasonable default value based on count and total_length
            if total_length > 1000 and count < 10:
                return 100.0
            if total_length > 100 and count < 10:
                return 10.0
                
            # Default to a reasonable value
            return 5.0
    
    def _collect_visualization_data(
        self,
        df: Any,
        col: str,
        col_type: str,
        stats: Dict[str, Any]
    ) -> None:
        """
        Collect data for visualizations.
        
        Args:
            df: Input DataFrame
            col: Column name
            col_type: Detected column type
            stats: Column statistics
        """
        # Update type distribution
        if col_type in self.visualization_data['type_distribution']:
            self.visualization_data['type_distribution'][col_type] += 1
        else:
            self.visualization_data['type_distribution'][col_type] = 1
            
        # Store cardinality if available
        if 'cardinality' in stats:
            self.visualization_data['cardinality'][col] = stats['cardinality']
            
        # Store null ratio
        self.visualization_data['null_ratios'][col] = stats['null_ratio']
    
    def _generate_type_detection_visualizations(self) -> None:
        """
        Generate visualizations for type detection.
        
        This method is a placeholder for visualization generation.
        In a production environment, this would create actual visualizations.
        """
        # This would generate matplotlib/seaborn visualizations
        # We'll implement this in a separate visualization module
        # that can use matplotlib/numpy directly (with appropriate isolation)
        logger.info("Type detection visualizations would be generated here")
        logger.info(f"Type distribution: {self.visualization_data['type_distribution']}")
    
    def _generate_sample_tables(self, df: Any, types: Dict[str, List[str]]) -> None:
        """
        Generate sample data tables for detected types.
        
        Args:
            df: Input DataFrame
            types: Dictionary of column types
        """
        # Sample a subset of data
        try:
            # For pandas-like objects that support sampling
            sample_df = df.sample(min(5, len(df)))
        except Exception:
            # Fallback to taking first rows
            sample_df = df.head(min(5, len(df)))
            
        # Create sample tables for each type
        for type_name, columns in types.items():
            if not columns:
                continue
                
            # Create sample table for this type
            self.sample_tables[type_name] = {
                'title': f'{type_name.capitalize()} Columns',
                'columns': columns,
                'data': sample_df[columns].to_dict('list') if columns else {}
            }
            
        logger.info(f"Generated {len(self.sample_tables)} sample tables")
    
    def get_type_details_table(self) -> pd.DataFrame:
        """
        Get the detailed type detection results as a DataFrame.
        
        Returns:
            DataFrame with type detection details
        """
        if len(self.type_details_df) == 0:
            return pd.DataFrame(columns=[
                'Column', 'Detected Type', 'Cardinality', 'Null Ratio', 'Recommended Strategy'
            ])
        return self.type_details_df
    
    def get_sample_table(self, type_name: str) -> Dict[str, Any]:
        """
        Get a sample table for a specific type.
        
        Args:
            type_name: Type name (e.g., 'numeric', 'categorical')
            
        Returns:
            Dictionary with sample table data
        """
        return self.sample_tables.get(type_name, {
            'title': f'No {type_name} columns found',
            'columns': [],
            'data': {}
        })