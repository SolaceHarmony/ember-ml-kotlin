"""
Terabyte-Scale Feature Extractor for BigQuery (Purified Version)

This module provides tools for feature extraction from terabyte-sized BigQuery tables,
with efficient chunking and memory management, using ember_ml's backend system
for optimal performance across different hardware.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
import gc
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('terabyte_feature_extractor')

# Import ember_ml backend utilities
from ember_ml.utils import backend_utils
from ember_ml import ops

# Import BigFrames if available
try:
    import bigframes.pandas as bf
    from bigframes.ml.preprocessing import MaxAbsScaler
    BIGFRAMES_AVAILABLE = True
except ImportError:
    BIGFRAMES_AVAILABLE = False
    logger.warning("BigFrames not available. Install it to use TerabyteFeatureExtractor.")


class TerabyteFeatureExtractor:
    """
    Feature extractor optimized for terabyte-scale BigQuery tables.
    
    This class handles feature extraction from very large BigQuery tables by:
    - Processing data in manageable chunks
    - Optimizing memory usage
    - Providing progress tracking
    - Supporting distributed processing
    - Using backend-agnostic operations for optimal performance
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        verbose: bool = True,
        preferred_backend: Optional[str] = None
    ):
        """
        Initialize the terabyte-scale feature extractor.
        
        Args:
            project_id: GCP project ID (optional if using in BigQuery Studio)
            location: BigQuery location (default: "US")
            chunk_size: Number of rows to process per chunk
            max_memory_gb: Maximum memory usage in GB
            verbose: Whether to print progress information
            preferred_backend: Preferred computation backend ('mlx', 'torch', 'numpy')
        """
        if not BIGFRAMES_AVAILABLE:
            raise ImportError("BigFrames is not available. Please install it to use TerabyteFeatureExtractor.")
        
        # Initialize backend
        self.backend = backend_utils.set_preferred_backend(preferred_backend)
        logger.info(f"Using {self.backend} backend for computation")
        
        # Set random seed for reproducibility
        backend_utils.initialize_random_seed(42)
        
        self.project_id = project_id
        self.location = location
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        # Initialize preprocessing objects
        self.scaler = None
        self.imputer = None
        self.high_null_cols = None
        self.fixed_dummy_columns = None
        
        # Initialize BigFrames options
        bf.options.bigquery.location = location
        if project_id:
            bf.options.bigquery.project = project_id
        
        # Set up memory monitoring
        self.memory_usage = []
        
        logger.info(f"Initialized TerabyteFeatureExtractor with chunk_size={chunk_size}, max_memory_gb={max_memory_gb}")
    
    def setup_bigquery_connection(self, credentials_path: Optional[str] = None):
        """
        Set up an optimized BigQuery connection for terabyte-scale processing.
        
        Args:
            credentials_path: Optional path to service account credentials
        """
        # Close any existing sessions
        bf.close_session()
        
        # Set up credentials if provided
        if credentials_path:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            bf.options.bigquery.credentials = credentials
        
        # Configure for large-scale processing
        bf.options.bigquery.max_results = self.chunk_size  # Set max results to chunk size
        bf.options.bigquery.progress_bar = self.verbose    # Show progress for long-running operations
        
        logger.info("BigQuery connection set up successfully")
    
    def optimize_bigquery_query(
        self,
        table_id: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str:
        """
        Create an optimized BigQuery query for terabyte-scale tables.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            columns: List of columns to select (None for all)
            where_clause: Optional WHERE clause for filtering
            limit: Optional row limit
            offset: Optional offset for pagination
            
        Returns:
            Optimized query string
        """
        # Start with base query
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"
            
        query = f"SELECT {select_clause} FROM `{table_id}`"
        
        # Add filtering if provided
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # Add partitioning hint for large tables
        if not where_clause or "_PARTITIONTIME" not in where_clause:
            query = f"/*+ OPTIMIZE_FOR_LARGE_TABLES */ {query}"
        
        # Add limit and offset if provided
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        return query
    
    def get_table_row_count(self, table_id: str, where_clause: Optional[str] = None) -> int:
        """
        Get the approximate row count of a BigQuery table.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            where_clause: Optional WHERE clause for filtering
            
        Returns:
            Approximate row count
        """
        # Create count query
        count_query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
        if where_clause:
            count_query += f" WHERE {where_clause}"
        
        # Execute query
        try:
            row_count_df = bf.read_gbq(count_query)
            total_rows = row_count_df.iloc[0, 0]
            logger.info(f"Table {table_id} has approximately {total_rows} rows")
            return total_rows
        except Exception as e:
            logger.error(f"Error getting row count: {e}")
            # Fallback to metadata
            try:
                from google.cloud import bigquery
                client = bigquery.Client(project=self.project_id)
                dataset_id, table_name = table_id.split('.')
                table_ref = client.dataset(dataset_id).table(table_name)
                table = client.get_table(table_ref)
                logger.info(f"Table {table_id} has approximately {table.num_rows} rows (from metadata)")
                return table.num_rows
            except Exception as e2:
                logger.error(f"Error getting row count from metadata: {e2}")
                return 0
    
    def process_bigquery_in_chunks(
        self,
        table_id: str,
        processing_fn: Optional[callable] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> Union[List[Any], pd.DataFrame]:
        """
        Process a BigQuery table in chunks to handle terabyte-scale data.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            processing_fn: Function to apply to each chunk
            columns: List of columns to select (None for all)
            where_clause: Optional WHERE clause for filtering
            max_chunks: Maximum number of chunks to process (None for all)
            
        Returns:
            Combined results from all chunks or list of results
        """
        # Get total row count
        total_rows = self.get_table_row_count(table_id, where_clause)
        
        # Calculate number of chunks
        num_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        if max_chunks and max_chunks < num_chunks:
            num_chunks = max_chunks
            logger.info(f"Processing limited to {max_chunks} chunks")
        
        logger.info(f"Processing approximately {total_rows} rows in {num_chunks} chunks of {self.chunk_size}")
        
        # Process each chunk
        results = []
        start_time = time.time()
        
        for i in range(num_chunks):
            chunk_start_time = time.time()
            logger.info(f"Processing chunk {i+1}/{num_chunks}")
            
            # Create query for this chunk
            offset = i * self.chunk_size
            chunk_query = self.optimize_bigquery_query(
                table_id=table_id,
                columns=columns,
                where_clause=where_clause,
                limit=self.chunk_size,
                offset=offset
            )
            
            # Load chunk
            try:
                chunk_df = bf.read_gbq(chunk_query)
                
                # Process chunk if function provided
                if processing_fn:
                    result = processing_fn(chunk_df)
                    results.append(result)
                else:
                    results.append(chunk_df)
                
                # Log progress
                chunk_time = time.time() - chunk_start_time
                progress = (i + 1) / num_chunks * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * num_chunks
                remaining = estimated_total - elapsed
                
                logger.info(f"Chunk {i+1}/{num_chunks} processed in {chunk_time:.2f}s. "
                           f"Progress: {progress:.1f}%. Estimated time remaining: {remaining:.2f}s")
                
                # Monitor memory usage
                self._monitor_memory()
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with next chunk
                continue
        
        # Combine results if needed
        total_time = time.time() - start_time
        logger.info(f"Completed processing {num_chunks} chunks in {total_time:.2f}s")
        
        if processing_fn and results:
            if isinstance(results[0], pd.DataFrame):
                logger.info("Combining DataFrame results")
                return pd.concat(results, ignore_index=True)
            else:
                return results
        elif not processing_fn:
            logger.info("Combining DataFrame results")
            return pd.concat(results, ignore_index=True)
        else:
            return []
    
    def _monitor_memory(self):
        """Monitor memory usage and log warnings if approaching limit."""
        try:
            import psutil
            memory_info = psutil.Process().memory_info()
            memory_gb = memory_info.rss / 1024 / 1024 / 1024
            self.memory_usage.append(memory_gb)
            
            if memory_gb > self.max_memory_gb * 0.8:
                logger.warning(f"Memory usage is high: {memory_gb:.2f} GB (limit: {self.max_memory_gb} GB)")
            
            logger.info(f"Current memory usage: {memory_gb:.2f} GB")
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
    
    def _detect_column_types(self, schema) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Detect column types from schema.
        
        Args:
            schema: DataFrame schema (dtypes)
            
        Returns:
            Tuple of lists: (numeric_cols, datetime_cols, categorical_cols, struct_cols, boolean_cols)
        """
        numeric_cols = []
        datetime_cols = []
        categorical_cols = []
        struct_cols = []
        boolean_cols = []
        
        for col_name, col_type in schema.items():
            if self._is_numeric_type(col_type):
                numeric_cols.append(col_name)
            elif self._is_datetime_type(col_type):
                datetime_cols.append(col_name)
            elif hasattr(bf, 'BooleanDtype') and col_type == bf.BooleanDtype():
                boolean_cols.append(col_name)
            elif (hasattr(bf, 'StringDtype') and col_type == bf.StringDtype()):
                categorical_cols.append(col_name)
            elif "STRUCT" in str(col_type).upper():
                struct_cols.append(col_name)
        
        logger.info(f"Detected {len(numeric_cols)} numeric, {len(datetime_cols)} datetime, "
                   f"{len(categorical_cols)} categorical, {len(boolean_cols)} boolean, "
                   f"and {len(struct_cols)} struct columns")
        
        return numeric_cols, datetime_cols, categorical_cols, struct_cols, boolean_cols
    
    def _is_numeric_type(self, col_type) -> bool:
        """Check if column type is numeric."""
        try:
            if pd.api.types.is_numeric_dtype(col_type):
                return True
            if hasattr(col_type, "id"):
                import pyarrow.types as pat
                return pat.is_integer(col_type) or pat.is_floating(col_type)
            return False
        except TypeError:
            return False
    
    def _is_datetime_type(self, col_type) -> bool:
        """Check if column type is datetime."""
        try:
            if pd.api.types.is_datetime64_any_dtype(col_type):
                return True
            if hasattr(col_type, "id"):
                import pyarrow.types as pat
                return pat.is_timestamp(col_type) or pat.is_date(col_type)
            return False
        except TypeError:
            return False
    
    def prepare_data(
        self,
        table_id: str,
        target_column: Optional[str] = None,
        force_categorical_columns: List[str] = None,
        drop_columns: List[str] = None,
        high_null_threshold: float = 0.9,
        limit: Optional[int] = None,
        index_col: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> Tuple:
        """
        Prepares data from a BigQuery table with optimizations for terabyte-scale.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            target_column: Target variable name. Heuristics if None.
            force_categorical_columns: Always treat as categorical
            drop_columns: Columns to drop
            high_null_threshold: Drop columns with > this % nulls (after encoding)
            limit: Optional row limit for testing
            index_col: Optional index column
            max_chunks: Maximum number of chunks to process
            
        Returns:
            Tuple: (train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer)
        """
        # Clean up session and memory
        bf.close_session()
        gc.collect()
        
        # Initialize parameters
        if force_categorical_columns is None:
            force_categorical_columns = []
        if drop_columns is None:
            drop_columns = []
        
        # Define a function to process each chunk
        def process_chunk(chunk_df):
            # Apply type conversions if needed
            # This is a simplified version - in a real implementation,
            # you would apply more sophisticated processing
            return chunk_df
        
        # Process data in chunks
        logger.info(f"Starting data preparation for table {table_id}")
        
        # If limit is specified, adjust chunk size and max_chunks
        if limit:
            max_chunks = (limit + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Limiting to {limit} rows ({max_chunks} chunks)")
        
        # Read data in chunks
        df = self.process_bigquery_in_chunks(
            table_id=table_id,
            processing_fn=process_chunk,
            max_chunks=max_chunks
        )
        
        # Check if df is a list or DataFrame
        if isinstance(df, list):
            # If it's a list, try to convert it to a DataFrame
            if df and isinstance(df[0], pd.DataFrame):
                df = pd.concat(df, ignore_index=True)
            else:
                logger.error("Failed to process data: result is not a DataFrame")
                return None
        
        # Set index if specified
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        
        # Detect column types
        numeric_cols, datetime_cols, categorical_cols, struct_cols, boolean_cols = self._detect_column_types(df.dtypes)
        
        # Add force_categorical_columns to categorical_cols
        for col in force_categorical_columns:
            if col in df.columns and col not in categorical_cols:
                categorical_cols.append(col)
                # Remove from other type lists if present
                if col in numeric_cols:
                    numeric_cols.remove(col)
                if col in datetime_cols:
                    datetime_cols.remove(col)
                if col in boolean_cols:
                    boolean_cols.remove(col)
        
        # Split data
        train_df, val_df, test_df = self._split_data(df, index_col)
        
        # Process each split
        logger.info("Processing training data")
        train_result = self._prepare_dataframe(
            train_df, numeric_cols, datetime_cols, categorical_cols, 
            struct_cols, boolean_cols, drop_columns, target_column, is_train=True
        )
        
        if train_result is None:
            logger.error("Failed to process training data")
            return None
        
        train_df_processed, train_features, self.scaler, self.imputer, self.high_null_cols, self.fixed_dummy_columns = train_result
        
        logger.info("Processing validation data")
        val_result = self._prepare_dataframe(
            val_df, numeric_cols, datetime_cols, categorical_cols, 
            struct_cols, boolean_cols, drop_columns, target_column, is_train=False
        )
        
        if val_result is None:
            logger.error("Failed to process validation data")
            return None
        
        val_df_processed, val_features, _, _, _, _ = val_result
        
        logger.info("Processing test data")
        test_result = self._prepare_dataframe(
            test_df, numeric_cols, datetime_cols, categorical_cols, 
            struct_cols, boolean_cols, drop_columns, target_column, is_train=False
        )
        
        if test_result is None:
            logger.error("Failed to process test data")
            return None
        
        test_df_processed, test_features, _, _, _, _ = test_result
        
        # Clean up
        gc.collect()
        
        logger.info("Data preparation completed successfully")
        return (
            train_df_processed, val_df_processed, test_df_processed,
            train_features, val_features, test_features,
            self.scaler, self.imputer
        )
    
    def _split_data(self, df, index_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            index_col: Index column name
            
        Returns:
            Tuple: (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data with {len(df)} rows")
        
        try:
            if index_col and pd.api.types.is_datetime64_any_dtype(df.index):
                # Temporal split
                split_date = df.index.quantile(0.8)
                train_df = df[df.index <= split_date]
                temp_df = df[df.index > split_date]
                logger.info(f"Temporal split date: {split_date}")
            else:
                # Random split using backend-agnostic random generation
                random_values = backend_utils.random_uniform(len(df))
                random_values_np = backend_utils.tensor_to_numpy_safe(random_values)
                
                df['__split_rand'] = random_values_np
                train_df = df[df['__split_rand'] <= 0.8].drop(columns=['__split_rand'])
                temp_df = df[df['__split_rand'] > 0.8].drop(columns=['__split_rand'])
                logger.info("Random split ratios: 80/20 (train/temp)")
        except Exception as e:
            logger.error(f"Error during initial split: {e}")
            # Fallback to simple split
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            temp_df = df.iloc[train_size:]
            logger.info("Fallback to simple split: 80/20 (train/temp)")
        
        # Split temp into validation and test
        try:
            # Generate random values using backend-agnostic function
            random_values = backend_utils.random_uniform(len(temp_df))
            random_values_np = backend_utils.tensor_to_numpy_safe(random_values)
            
            temp_df['__split_rand2'] = random_values_np
            val_df = temp_df[temp_df['__split_rand2'] <= 0.5].drop(columns=['__split_rand2'])
            test_df = temp_df[temp_df['__split_rand2'] > 0.5].drop(columns=['__split_rand2'])
            logger.info("Validation/test split ratios: 50/50 from temp")
        except Exception as e:
            logger.error(f"Error during validation/test split: {e}")
            # Fallback to simple split
            val_size = len(temp_df) // 2
            val_df = temp_df.iloc[:val_size]
            test_df = temp_df.iloc[val_size:]
            logger.info("Fallback to simple split for validation/test")
        
        logger.info(f"Split result: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test rows")
        return train_df, val_df, test_df
    
    def _prepare_dataframe(
        self,
        df_split,
        numeric_cols,
        datetime_cols,
        categorical_cols,
        struct_cols,
        boolean_cols,
        drop_columns,
        target_column,
        is_train=False
    ):
        """
        Prepare a dataframe split with feature engineering.
        
        Args:
            df_split: DataFrame split
            numeric_cols: List of numeric columns
            datetime_cols: List of datetime columns
            categorical_cols: List of categorical columns
            struct_cols: List of struct columns
            boolean_cols: List of boolean columns
            drop_columns: List of columns to drop
            target_column: Target column name
            is_train: Whether this is the training split
            
        Returns:
            Tuple: (processed_df, features, scaler, imputer, high_null_cols, fixed_dummy_columns)
        """
        try:
            logger.info(f"Preparing dataframe (is_train={is_train}) with {len(df_split)} rows")
            
            # Filter drop_columns
            cols_to_drop = [col for col in drop_columns if col in df_split.columns]
            if cols_to_drop:
                df_split = df_split.drop(columns=cols_to_drop)
                logger.info(f"Dropped columns: {cols_to_drop}")
            
            # Flatten STRUCT columns
            for struct_col in struct_cols:
                if struct_col in df_split.columns:
                    logger.info(f"Flattening struct column: {struct_col}")
                    df_split = self._flatten_struct(df_split, struct_col)
            
            # Convert datetime columns
            for col in datetime_cols:
                if col in df_split.columns:
                    logger.info(f"Converting to datetime: {col}")
                    df_split[col] = pd.to_datetime(df_split[col])
            
            # Determine target column
            _target_column = target_column
            if _target_column is None:
                eligible = [col for col in numeric_cols if col in df_split.columns and col not in categorical_cols]
                if eligible:
                    _target_column = eligible[-1]
                    logger.warning(f"No target specified. Using '{_target_column}' as target.")
                else:
                    logger.error("No suitable target column found.")
                    return None
            
            # Initial features: numeric (excluding target) plus boolean columns
            features = [col for col in numeric_cols if col != _target_column and col in df_split.columns]
            features.extend([col for col in boolean_cols if col in df_split.columns])
            logger.info(f"Initial features: {len(features)} columns")
            
            # One-hot encode categorical columns
            if categorical_cols:
                logger.info(f"Encoding {len(categorical_cols)} categorical columns")
                df_dummy = pd.get_dummies(df_split, columns=categorical_cols, drop_first=False)
                df_dummy.columns = df_dummy.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
                
                if is_train:
                    fixed_dummy = list(df_dummy.columns)
                else:
                    if self.fixed_dummy_columns is not None:
                        # Add missing columns with zeros
                        for col in self.fixed_dummy_columns:
                            if col not in df_dummy.columns:
                                df_dummy[col] = 0
                        # Keep only columns in fixed_dummy_columns
                        df_dummy = df_dummy[self.fixed_dummy_columns]
                    fixed_dummy = self.fixed_dummy_columns
                
                df_split = df_dummy
            else:
                logger.info("No categorical columns to encode")
                fixed_dummy = self.fixed_dummy_columns
            
            # Add cyclical features for datetime columns
            for col in datetime_cols:
                if col in df_split.columns:
                    logger.info(f"Adding cyclical features for: {col}")
                    df_split = self._create_datetime_features(df_split, col)
            
            # Re-assess numeric columns
            current_numeric = [col for col in df_split.columns 
                              if pd.api.types.is_numeric_dtype(df_split[col].dtype)]
            features = [col for col in current_numeric if col != _target_column]
            logger.info(f"Final features: {len(features)} columns")
            
            # Handle scaling and imputation
            high_null_cols_to_drop_list = []
            if is_train:
                # Prepare for scaling and imputation
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                
                # Create imputer
                self.imputer = SimpleImputer(strategy='median')
                
                # Fit imputer on training data
                self.imputer.fit(df_split[features])
                
                # Create scaler
                self.scaler = StandardScaler()
                
                # Fit scaler on training data (after imputation)
                imputed_data = self.imputer.transform(df_split[features])
                self.scaler.fit(imputed_data)
                
                # Transform data
                scaled_data = self.scaler.transform(imputed_data)
                
                # Convert back to DataFrame
                scaled_df = pd.DataFrame(scaled_data, index=df_split.index, columns=features)
                
                # Replace original columns with scaled ones
                for col in features:
                    df_split[col] = scaled_df[col]
                
                logger.info("Scaler and imputer fitted and applied on training data")
            else:
                if self.imputer is None or self.scaler is None:
                    logger.error("Imputer and scaler must be fitted on training data first")
                    return None
                
                # Apply imputer and scaler from training
                imputed_data = self.imputer.transform(df_split[features])
                scaled_data = self.scaler.transform(imputed_data)
                
                # Convert back to DataFrame
                scaled_df = pd.DataFrame(scaled_data, index=df_split.index, columns=features)
                
                # Replace original columns with scaled ones
                for col in features:
                    df_split[col] = scaled_df[col]
                
                logger.info("Scaler and imputer applied on validation/test data")
            
            # Clean up
            gc.collect()
            
            logger.info(f"Dataframe prepared successfully with {len(features)} features")
            return df_split, features, self.scaler, self.imputer, high_null_cols_to_drop_list, fixed_dummy
            
        except Exception as e:
            logger.error(f"ERROR in prepare_dataframe (is_train={is_train}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _flatten_struct(self, df: pd.DataFrame, struct_col_name: str) -> pd.DataFrame:
        """
        Flattens a struct column in a DataFrame.
        
        Args:
            df: Input DataFrame
            struct_col_name: Name of the struct column to flatten
            
        Returns:
            DataFrame with flattened struct column
        """
        try:
            # Convert to dict first if needed
            if not isinstance(df[struct_col_name].iloc[0], dict):
                df[struct_col_name] = df[struct_col_name].apply(lambda x: {} if pd.isna(x) else x)
            
            # Use json_normalize for flattening
            import json
            from pandas import json_normalize
            
            # Convert to JSON strings then parse to ensure proper format
            json_strings = df[struct_col_name].apply(lambda x: json.dumps(x) if isinstance(x, dict) else "{}")
            parsed_json = json_strings.apply(json.loads)
            
            # Normalize the JSON
            struct_df = json_normalize(parsed_json)
            
            # Add prefix to column names
            struct_df = struct_df.add_prefix(f"{struct_col_name}_")
            
            # Concatenate with original DataFrame
            result_df = pd.concat([df.drop(columns=[struct_col_name]), struct_df], axis=1)
            
            return result_df
        except Exception as e:
            logger.error(f"Error flattening struct column {struct_col_name}: {e}")
            # Return original DataFrame if flattening fails
            return df
    
    def _create_datetime_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Create cyclical features from datetime column using backend-agnostic operations.
        
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
                logger.warning(f"Could not convert {col} to datetime: {e}")
                return df
        
        # Extract datetime components
        hours = df[col].dt.hour / 23.0
        days_of_week = df[col].dt.dayofweek / 6.0
        days_of_month = (df[col].dt.day - 1) / 30.0
        months = (df[col].dt.month - 1) / 11.0
        
        # Apply sine and cosine transformations using backend-agnostic functions
        hours_sin, hours_cos = backend_utils.sin_cos_transform(hours, period=1.0)
        dow_sin, dow_cos = backend_utils.sin_cos_transform(days_of_week, period=1.0)
        dom_sin, dom_cos = backend_utils.sin_cos_transform(days_of_month, period=1.0)
        month_sin, month_cos = backend_utils.sin_cos_transform(months, period=1.0)
        
        # Convert to numpy arrays for pandas
        df[f'{col}_sin_hour'] = backend_utils.tensor_to_numpy_safe(hours_sin)
        df[f'{col}_cos_hour'] = backend_utils.tensor_to_numpy_safe(hours_cos)
        df[f'{col}_sin_dayofweek'] = backend_utils.tensor_to_numpy_safe(dow_sin)
        df[f'{col}_cos_dayofweek'] = backend_utils.tensor_to_numpy_safe(dow_cos)
        df[f'{col}_sin_day'] = backend_utils.tensor_to_numpy_safe(dom_sin)
        df[f'{col}_cos_day'] = backend_utils.tensor_to_numpy_safe(dom_cos)
        df[f'{col}_sin_month'] = backend_utils.tensor_to_numpy_safe(month_sin)
        df[f'{col}_cos_month'] = backend_utils.tensor_to_numpy_safe(month_cos)
        
        logger.info(f"Created cyclical features for datetime column '{col}' using {self.backend} backend")
        return df


class TerabyteTemporalStrideProcessor:
    """
    Processes data into multi-stride temporal representations optimized for terabyte-scale data.
    
    This class creates sliding windows with different strides and applies
    PCA for dimensionality reduction, enabling multi-scale temporal analysis
    while optimizing for memory usage with large datasets.
    
    Uses ember_ml's backend system for optimal performance across different hardware.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        stride_perspectives: List[int] = None,
        pca_components: Optional[int] = None,
        batch_size: int = 10000,
        use_incremental_pca: bool = True,
        verbose: bool = True,
        preferred_backend: Optional[str] = None
    ):
        """
        Initialize the temporal stride processor.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths to use
            pca_components: Number of PCA components (if None, will be calculated)
            batch_size: Size of batches for processing
            use_incremental_pca: Whether to use incremental PCA for large datasets
            verbose: Whether to print progress information
            preferred_backend: Preferred computation backend ('mlx', 'torch', 'numpy')
        """
        # Initialize backend
        self.backend = backend_utils.set_preferred_backend(preferred_backend)
        logger.info(f"Using {self.backend} backend for computation")
        
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives or [1, 3, 5]
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.use_incremental_pca = use_incremental_pca
        self.verbose = verbose
        
        self.pca_models = {}  # Store PCA models for each stride
        self.state_buffer = None  # Buffer for stateful processing
        
        logger.info(f"Initialized TerabyteTemporalStrideProcessor with window_size={window_size}, "
                   f"stride_perspectives={stride_perspectives}, batch_size={batch_size}")
    
    def process_large_dataset(
        self,
        data_generator: Generator,
        maintain_state: bool = True
    ) -> Dict[int, Any]:
        """
        Process a large dataset using a generator to avoid loading all data into memory.
        
        Args:
            data_generator: Generator yielding batches of data
            maintain_state: Whether to maintain state between batches
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        results = {stride: [] for stride in self.stride_perspectives}
        
        for batch_idx, batch_data in enumerate(data_generator):
            logger.info(f"Processing batch {batch_idx+1} with {len(batch_data)} rows")
            
            # Convert batch data to tensor
            batch_tensor = backend_utils.convert_to_tensor_safe(batch_data)
            
            # If state buffer exists and maintain_state is True, prepend to current batch
            if maintain_state and self.state_buffer is not None:
                state_buffer_tensor = backend_utils.convert_to_tensor_safe(self.state_buffer)
                batch_tensor = backend_utils.vstack_safe([state_buffer_tensor, batch_tensor])
                logger.info(f"Added state buffer, new batch size: {batch_tensor.shape[0]}")
            
            # Update state buffer with latest data if maintain_state is True
            if maintain_state:
                buffer_size = max(self.stride_perspectives) * self.window_size
                # Convert to numpy for slicing, then back to tensor
                batch_np = backend_utils.tensor_to_numpy_safe(batch_tensor)
                self.state_buffer = batch_np[-buffer_size:].copy()
                logger.info(f"Updated state buffer with {len(self.state_buffer)} rows")
            
            # Process batch for each stride perspective
            batch_results = self.process_batch(batch_tensor)
            
            # Append results
            for stride, data in batch_results.items():
                results[stride].append(data)
            
            # Force garbage collection
            gc.collect()
        
        # Combine results for each stride
        combined_results = {}
        for stride, data_list in results.items():
            if data_list:
                # Convert all tensors to the same format
                tensors = [backend_utils.convert_to_tensor_safe(data) for data in data_list]
                combined_results[stride] = backend_utils.vstack_safe(tensors)
                
                # Get shape for logging
                shape_str = str(backend_utils.tensor_to_numpy_safe(combined_results[stride]).shape)
                logger.info(f"Combined results for stride {stride}: {shape_str}")
            else:
                logger.warning(f"No results for stride {stride}")
        
        return combined_results
    
    def process_batch(self, data: Any) -> Dict[int, Any]:
        """
        Process a batch of data into multi-stride temporal representations.
        
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
                logger.warning(f"No windows created for stride {stride}")
                continue
            
            # Convert to tensor array
            windows_tensor = backend_utils.convert_to_tensor_safe(windows)
            
            # Apply PCA
            if self.use_incremental_pca:
                results[stride] = self._apply_incremental_pca(windows_tensor, stride)
            else:
                results[stride] = self._apply_pca_blend(windows_tensor, stride)
            
            # Get shape for logging
            windows_shape = backend_utils.tensor_to_numpy_safe(windows_tensor).shape
            results_shape = backend_utils.tensor_to_numpy_safe(results[stride]).shape
            
            logger.info(f"Created {windows_shape[0]} windows with stride {stride}, "
                       f"shape after PCA: {results_shape}")
        
        return results
    
    def _create_strided_sequences(self, data: Any, stride: int) -> List[Any]:
        """
        Create sequences with the given stride.
        
        Args:
            data: Input data array
            stride: Stride length
            
        Returns:
            List of windowed sequences
        """
        # Convert to numpy for easier slicing
        data_np = backend_utils.tensor_to_numpy_safe(data)
        num_samples = len(data_np)
        windows = []
        
        # Skip if data is too small for even one window
        if num_samples < self.window_size:
            logger.warning(f"Data length ({num_samples}) is smaller than window size ({self.window_size})")
            return windows
        
        for i in range(0, num_samples - self.window_size + 1, stride):
            windows.append(data_np[i:i+self.window_size])
        
        return windows
    
    def _apply_pca_blend(self, window_batch: Any, stride: int) -> Any:
        """
        Apply PCA-based temporal blending.
        
        Args:
            window_batch: Batch of windows (batch_size x window_size x features)
            stride: Stride length
            
        Returns:
            PCA-transformed data
        """
        # Convert to numpy for sklearn PCA
        window_batch_tensor = backend_utils.tensor_to_numpy_safe(window_batch)
        batch_size, window_size, feature_dim = window_batch_tensor.shape
        
        # Reshape for PCA: [batch_size, window_size * feature_dim]
        flat_windows = window_batch_tensor.reshape(batch_size, -1)
        
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
            
            logger.info(f"Fitting PCA for stride {stride} with {n_components} components")
            
            from sklearn.decomposition import PCA
            self.pca_models[stride] = PCA(n_components=n_components)
            self.pca_models[stride].fit(flat_windows)
        
        # Transform the data
        transformed_np = self.pca_models[stride].transform(flat_windows)
        
        # Convert back to tensor
        return backend_utils.convert_to_tensor_safe(transformed_np)
    
    def _apply_incremental_pca(self, window_batch: Any, stride: int) -> Any:
        """
        Apply incremental PCA for large datasets.
        
        Args:
            window_batch: Batch of windows (batch_size x window_size x features)
            stride: Stride length
            
        Returns:
            PCA-transformed data
        """
        # Convert to numpy for sklearn PCA
        window_batch_tensor = backend_utils.tensor_to_numpy_safe(window_batch)
        batch_size, window_size, feature_dim = window_batch_tensor.shape
        
        # Reshape for PCA: [batch_size, window_size * feature_dim]
        flat_windows = window_batch_tensor.reshape(batch_size, -1)
        
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
            
            logger.info(f"Fitting Incremental PCA for stride {stride} with {n_components} components")
            
            from sklearn.decomposition import IncrementalPCA
            self.pca_models[stride] = IncrementalPCA(n_components=n_components, batch_size=self.batch_size)
        
        # Partial fit with this batch
        self.pca_models[stride].partial_fit(flat_windows)
        
        # Transform the data
        transformed_np = self.pca_models[stride].transform(flat_windows)
        
        # Convert back to tensor
        return backend_utils.convert_to_tensor_safe(transformed_np)
    
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
    
    def get_feature_importance(self, stride: int) -> Optional[Any]:
        """
        Get feature importance for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Array of feature importance scores or None if PCA not fit
        """
        if stride in self.pca_models:
            # Calculate feature importance as the sum of absolute component weights
            components = backend_utils.convert_to_tensor_safe(self.pca_models[stride].components_)
            
            # Use ops module for all backends
            importance = stats.sum(ops.abs(components), axis=0)
            
            # Convert to numpy for compatibility with pandas/sklearn
            return backend_utils.tensor_to_numpy_safe(importance)
        return None


# Example usage
if __name__ == "__main__":
    # Print backend information
    backend_utils.print_backend_info()
    
    # Create feature extractor
    extractor = TerabyteFeatureExtractor(
        project_id="your-project-id",
        location="US",
        chunk_size=100000,
        max_memory_gb=16.0,
        preferred_backend="mlx"  # Try to use MLX if available
    )
    
    # Set up BigQuery connection
    extractor.setup_bigquery_connection()
    
    # Prepare data
    result = extractor.prepare_data(
        table_id="your-dataset.your-table",
        target_column="your-target-column",
        force_categorical_columns=["category1", "category2"],
        limit=1000000  # For testing
    )
    
    if result:
        train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result
        
        print(f"Train shape: {train_df.shape}")
        print(f"Validation shape: {val_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Features: {train_features}")
        
        # Create temporal stride processor with the same backend
        processor = TerabyteTemporalStrideProcessor(
            window_size=5,
            stride_perspectives=[1, 3, 5],
            pca_components=32,
            batch_size=10000,
            use_incremental_pca=True,
            preferred_backend="mlx"  # Use the same backend as the extractor
        )
        
        # Define a generator to yield data in batches
        def data_generator(df, batch_size=10000):
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i+batch_size][train_features].values
        
        # Process data
        stride_perspectives = processor.process_large_dataset(
            data_generator(train_df, batch_size=10000)
        )
        
        print("Stride perspectives:")
        for stride, data in stride_perspectives.items():
            # Convert to numpy for printing
            data_tensor = backend_utils.tensor_to_numpy_safe(data)
            print(f"Stride {stride}: {data_tensor.shape}")