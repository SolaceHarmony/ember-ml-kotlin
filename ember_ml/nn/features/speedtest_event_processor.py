"""
Specialized processor for BigQuery speedtest event data.

This module provides a specialized processor for the "ctl_modem_speedtest_event" table
in BigQuery. It utilizes the enhanced type detection and feature processing capabilities
to optimize feature extraction for this specific table structure.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.nn.features.enhanced_type_detector import EnhancedTypeDetector
from ember_ml.nn.features.animated_feature_processor import AnimatedFeatureProcessor

# Set up logging
logger = logging.getLogger(__name__)


class SpeedtestEventProcessor:
    """
    Specialized processor for speedtest event data from BigQuery.
    
    This class provides optimized feature extraction for the
    "ctl_modem_speedtest_event" table in BigQuery, with visualization
    capabilities and sample data tables throughout the process.
    """
    
    def __init__(
        self,
        project_id: str = "[REDACTED_PROJECT_ID]",
        credentials_path: Optional[str] = None,
        visualization_enabled: bool = True,
        sample_tables_enabled: bool = True,
        device: Optional[str] = None,
        max_memory_gb: float = 16.0
    ):
        """
        Initialize the speedtest event processor.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to GCP credentials file
            visualization_enabled: Whether to enable visualizations
            sample_tables_enabled: Whether to enable sample data tables
            device: Device to place tensors on
            max_memory_gb: Maximum memory usage in GB
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.visualization_enabled = visualization_enabled
        self.sample_tables_enabled = sample_tables_enabled
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        # Set backend to MLX by default for best performance
        try:
            from ember_ml.ops import set_backend
            set_backend('mlx')
            logger.info("Using MLX backend for optimal performance")
        except Exception as e:
            logger.warning(f"Failed to set MLX backend: {e}. Using default backend.")
        
        # Initialize components
        self.type_detector = EnhancedTypeDetector(
            visualization_enabled=visualization_enabled,
            sample_tables_enabled=sample_tables_enabled
        )
        
        self.feature_processor = AnimatedFeatureProcessor(
            visualization_enabled=visualization_enabled,
            sample_tables_enabled=sample_tables_enabled,
            device=device
        )
        
        # Memory usage tracking
        self.memory_usage_data = []
        self.processing_start_time = None
        
        # Results storage
        self.column_types = {}
        self.feature_tensors = {}
        self.combined_features = None
        self.processing_metadata = {}
    
    def process(
        self,
        table_id: str = "TEST1.ctl_modem_speedtest_event",
        target_column: Optional[str] = "downloadLatency",
        limit: Optional[int] = None,
        force_categorical_columns: Optional[List[str]] = None,
        force_numeric_columns: Optional[List[str]] = None,
        force_datetime_columns: Optional[List[str]] = None,
        force_identifier_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process the speedtest event table.
        
        Args:
            table_id: BigQuery table ID
            target_column: Target column for supervised learning
            limit: Optional row limit for testing
            force_categorical_columns: Columns to force as categorical
            force_numeric_columns: Columns to force as numeric
            force_datetime_columns: Columns to force as datetime
            force_identifier_columns: Columns to force as identifiers
            
        Returns:
            Dictionary of processed features and metadata
        """
        # Start timing and memory tracking
        self.processing_start_time = time.time()
        self._track_memory_usage("Initial")
        
        # Load data from BigQuery
        df = self._load_bigquery_data(table_id, limit)
        if df is None or len(df) == 0:
            logger.error(f"Failed to load data from {table_id}")
            return self._empty_result()
        
        self._track_memory_usage("After Data Loading")
        
        # Detect column types
        self.column_types = self.type_detector.detect_column_types(df)
        
        # Apply forced column types
        self._apply_forced_column_types(
            force_categorical_columns,
            force_numeric_columns,
            force_datetime_columns,
            force_identifier_columns
        )
        
        self._track_memory_usage("After Type Detection")
        
        # Process each type of features
        self._process_all_feature_types(df, target_column)
        
        # Combine all features
        self._combine_features()
        
        # Prepare metadata
        self._prepare_metadata(df)
        
        # Final memory tracking
        self._track_memory_usage("Final")
        
        return self._create_result()
    
    def _load_bigquery_data(
        self,
        table_id: str,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data from BigQuery.
        
        Args:
            table_id: BigQuery table ID
            limit: Optional row limit
            
        Returns:
            DataFrame with loaded data or None if failed
        """
        try:
            # Import BigQuery libraries
            import bigframes.pandas as bf
            from google.cloud import bigquery
            
            # Set BigFrames options
            bf.options.bigquery.project = self.project_id
            
            # Construct query
            full_table_id = f"{self.project_id}.{table_id}"
            query = f"SELECT * FROM `{full_table_id}`"
            
            if limit is not None:
                query += f" LIMIT {limit}"
            
            logger.info(f"Loading data from {full_table_id}")
            if limit is not None:
                logger.info(f"Using limit: {limit}")
            
            # Execute query
            df = bf.read_gbq(query)
            
            logger.info(f"Loaded {len(df)} rows from {full_table_id}")
            
            return df
        except ImportError:
            logger.error("Failed to import BigQuery libraries. Install with: "
                         "pip install google-cloud-bigquery bigframes")
            return None
        except Exception as e:
            logger.error(f"Failed to load data from BigQuery: {e}")
            return None
    
    def _apply_forced_column_types(
        self,
        force_categorical_columns: Optional[List[str]] = None,
        force_numeric_columns: Optional[List[str]] = None,
        force_datetime_columns: Optional[List[str]] = None,
        force_identifier_columns: Optional[List[str]] = None
    ) -> None:
        """
        Apply forced column types.
        
        Args:
            force_categorical_columns: Columns to force as categorical
            force_numeric_columns: Columns to force as numeric
            force_datetime_columns: Columns to force as datetime
            force_identifier_columns: Columns to force as identifiers
        """
        if force_categorical_columns:
            self._move_columns_to_type(force_categorical_columns, 'categorical')
            
        if force_numeric_columns:
            self._move_columns_to_type(force_numeric_columns, 'numeric')
            
        if force_datetime_columns:
            self._move_columns_to_type(force_datetime_columns, 'datetime')
            
        if force_identifier_columns:
            self._move_columns_to_type(force_identifier_columns, 'identifier')
    
    def _move_columns_to_type(self, columns: List[str], target_type: str) -> None:
        """
        Move columns to a specific type.
        
        Args:
            columns: Columns to move
            target_type: Target type
        """
        for col in columns:
            # Find and remove column from current type
            for type_name, type_columns in self.column_types.items():
                if col in type_columns:
                    self.column_types[type_name].remove(col)
                    break
            
            # Add column to target type
            if col not in self.column_types.get(target_type, []):
                if target_type not in self.column_types:
                    self.column_types[target_type] = []
                self.column_types[target_type].append(col)
    
    def _process_all_feature_types(
        self,
        df: Any,
        target_column: Optional[str] = None
    ) -> None:
        """
        Process all feature types.
        
        Args:
            df: Input DataFrame
            target_column: Target column for supervised learning
        """
        # Process numeric features
        if 'numeric' in self.column_types and self.column_types['numeric']:
            # Remove target column from features if specified
            numeric_features = self.column_types['numeric'].copy()
            if target_column in numeric_features:
                numeric_features.remove(target_column)
                
            if numeric_features:
                logger.info(f"Processing {len(numeric_features)} numeric features")
                self.feature_tensors['numeric'] = self.feature_processor.process_numeric_features(
                    df, numeric_features
                )
                self._track_memory_usage("After Numeric Processing")
        
        # Process categorical features
        if 'categorical' in self.column_types and self.column_types['categorical']:
            logger.info(f"Processing {len(self.column_types['categorical'])} categorical features")
            self.feature_tensors['categorical'] = self.feature_processor.process_categorical_features(
                df, self.column_types['categorical']
            )
            self._track_memory_usage("After Categorical Processing")
        
        # Process datetime features
        if 'datetime' in self.column_types and self.column_types['datetime']:
            logger.info(f"Processing {len(self.column_types['datetime'])} datetime features")
            self.feature_tensors['datetime'] = self.feature_processor.process_datetime_features(
                df, self.column_types['datetime']
            )
            self._track_memory_usage("After Datetime Processing")
        
        # Process identifier features
        if 'identifier' in self.column_types and self.column_types['identifier']:
            logger.info(f"Processing {len(self.column_types['identifier'])} identifier features")
            self.feature_tensors['identifier'] = self.feature_processor.process_identifier_features(
                df, self.column_types['identifier']
            )
            self._track_memory_usage("After Identifier Processing")
        
        # Add any other feature types as needed
    
    def _combine_features(self) -> None:
        """Combine all processed features into a single tensor."""
        feature_tensors = []
        
        for feature_type, tensor_data in self.feature_tensors.items():
            if tensor_data is not None:
                feature_tensors.append(tensor_data)
        
        if feature_tensors:
            logger.info("Combining all features")
            self.combined_features = tensor.concatenate(feature_tensors, axis=1)
            logger.info(f"Combined features shape: {tensor.shape(self.combined_features)}")
        else:
            logger.warning("No features to combine")
            self.combined_features = None
    
    def _prepare_metadata(self, df: Any) -> None:
        """
        Prepare metadata for the processing result.
        
        Args:
            df: Input DataFrame
        """
        self.processing_metadata = {
            'original_shape': (len(df), len(df.columns)),
            'column_types': {k: len(v) for k, v in self.column_types.items()},
            'feature_shapes': {k: tensor.shape(v).tolist() if v is not None else None 
                              for k, v in self.feature_tensors.items()},
            'combined_shape': tensor.shape(self.combined_features).tolist() if self.combined_features is not None else None,
            'memory_usage': self.memory_usage_data,
            'processing_time': time.time() - self.processing_start_time
        }
    
    def _create_result(self) -> Dict[str, Any]:
        """
        Create the final result dictionary.
        
        Returns:
            Dictionary with processed features and metadata
        """
        return {
            'features': self.combined_features,
            'feature_tensors': self.feature_tensors,
            'column_types': self.column_types,
            'type_details': self.type_detector.get_type_details_table(),
            'sample_tables': {
                'type_detector': self.type_detector.sample_tables,
                'feature_processor': self.feature_processor.sample_tables
            },
            'metadata': self.processing_metadata
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result when processing fails.
        
        Returns:
            Empty result dictionary
        """
        return {
            'features': None,
            'feature_tensors': {},
            'column_types': {},
            'type_details': pd.DataFrame(),
            'sample_tables': {
                'type_detector': {},
                'feature_processor': {}
            },
            'metadata': {
                'error': 'Failed to load data',
                'processing_time': time.time() - self.processing_start_time if self.processing_start_time else 0
            }
        }
    
    def _track_memory_usage(self, stage_name: str) -> None:
        """
        Track memory usage at a processing stage.
        
        Args:
            stage_name: Name of the processing stage
        """
        try:
            import psutil
            
            if self.processing_start_time is None:
                self.processing_start_time = time.time()
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
            
            # Calculate percentage of max memory
            percent_of_max = (memory_gb / self.max_memory_gb) * 100 if self.max_memory_gb > 0 else 0
            
            # Record memory usage
            timestamp = time.time() - self.processing_start_time
            self.memory_usage_data.append({
                'stage': stage_name,
                'timestamp': timestamp,
                'memory_gb': memory_gb,
                'percent_of_max': percent_of_max
            })
            
            logger.info(f"Memory usage at {stage_name}: {memory_gb:.2f} GB "
                       f"({percent_of_max:.1f}% of max)")
        except ImportError:
            logger.warning("psutil not installed. Memory tracking disabled.")
        except Exception as e:
            logger.warning(f"Failed to track memory usage: {e}")
    
    def prepare_for_rbm(
        self,
        features: Optional[Any] = None,
        binary_visible: bool = True,
        binarization_threshold: float = 0.5
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Prepare features for RBM training.
        
        Args:
            features: Features tensor (uses combined_features if None)
            binary_visible: Whether to binarize features for binary RBM
            binarization_threshold: Threshold for binarization
            
        Returns:
            Tuple of (prepared_features, metadata)
        """
        if features is None:
            features = self.combined_features
            
        if features is None:
            logger.error("No features available for RBM preparation")
            return None, {'error': 'No features available'}
        
        logger.info("Preparing features for RBM")
        
        # Ensure all values are in [0,1] range
        # Features should already be normalized from previous processing
        
        # Apply binarization if needed
        if binary_visible:
            logger.info(f"Binarizing features with threshold {binarization_threshold}")
            threshold = tensor.convert_to_tensor(binarization_threshold, device=self.device)
            binary_features = tensor.cast(
                ops.greater(features, threshold),
                tensor.float32
            )
            
            return binary_features, {
                'binary_visible': True,
                'binarization_threshold': binarization_threshold,
                'shape': list(tensor.shape(binary_features))
            }
        else:
            return features, {
                'binary_visible': False,
                'shape': list(tensor.shape(features))
            }
    
    def get_memory_usage_table(self) -> pd.DataFrame:
        """
        Get memory usage data as a DataFrame.
        
        Returns:
            DataFrame with memory usage data
        """
        return pd.DataFrame(self.memory_usage_data)