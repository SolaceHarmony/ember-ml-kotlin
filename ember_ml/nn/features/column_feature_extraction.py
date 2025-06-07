"""
Column-based Feature Extraction Library

This module provides tools for feature extraction on a per-column basis,
which can be useful for handling heterogeneous data types and structures.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ember_ml.nn.features.pca_features import PCA
from ember_ml.nn.features.tensor_features import one_hot
# Import ember_ml ops for backend-agnostic operations
from ember_ml import ops
from ember_ml.nn import tensor


from sklearn.pipeline import Pipeline


class ColumnFeatureExtractor:
    """
    Extracts features on a per-column basis with specialized processing for each data type.
    
    This class handles feature extraction with column-specific processing:
    - Numeric columns: Scaling, imputation, and optional PCA
    - Categorical columns: One-hot encoding or embedding
    - Datetime columns: Cyclical features and time-based features
    - Text columns: Basic text features (length, word count, etc.)
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'standard',
                 categorical_strategy: str = 'onehot',
                 datetime_strategy: str = 'cyclical',
                 text_strategy: str = 'basic',
                 max_categories: int = 100):
        """
        Initialize the column feature extractor.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('standard', 'robust', 'minmax')
            categorical_strategy: Strategy for categorical columns ('onehot', 'ordinal', 'target')
            datetime_strategy: Strategy for datetime columns ('cyclical', 'components', 'both')
            text_strategy: Strategy for text columns ('basic', 'tfidf', 'count')
            max_categories: Maximum number of categories for one-hot encoding
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.text_strategy = text_strategy
        self.max_categories = max_categories
        
        # Store column processors
        self.column_processors: Dict[str, Any] = {}
        self.column_types: Dict[str, str] = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'ColumnFeatureExtractor':
        """
        Fit the feature extractor to the data.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for target encoding
            
        Returns:
            Self for method chaining
        """
        # Detect column types
        self._detect_column_types(df)
        
        # Create and fit processors for each column
        for column, col_type in self.column_types.items():
            if column == target_column:
                continue
                
            if col_type == 'numeric':
                self._fit_numeric_processor(df, column)
            elif col_type == 'categorical':
                self._fit_categorical_processor(df, column, target_column)
            elif col_type == 'datetime':
                self._fit_datetime_processor(df, column)
            elif col_type == 'text':
                self._fit_text_processor(df, column)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted processors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with transformed features
        """
        if not self.fitted:
            raise ValueError("ColumnFeatureExtractor must be fitted before transform")
            
        result_dfs = []
        
        # Transform each column
        for column, processor in self.column_processors.items():
            if column in df.columns:
                col_type = self.column_types[column]
                
                if col_type == 'numeric':
                    transformed = self._transform_numeric(df, column, processor)
                elif col_type == 'categorical':
                    transformed = self._transform_categorical(df, column, processor)
                elif col_type == 'datetime':
                    transformed = self._transform_datetime(df, column, processor)
                elif col_type == 'text':
                    transformed = self._transform_text(df, column, processor)
                else:
                    continue
                    
                result_dfs.append(transformed)
        
        # Combine all transformed features
        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        else:
            return pd.DataFrame()
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None,
                     time_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit to the data, then transform it.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for target encoding
            time_column: Optional column to use for temporal ordering (used by subclasses)
            
        Returns:
            DataFrame with transformed features
        """
        return self.fit(df, target_column).transform(df)
    
    def _detect_column_types(self, df: pd.DataFrame) -> None:
        """
        Detect column types in the DataFrame.
        
        Args:
            df: Input DataFrame
        """
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self.column_types[column] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                self.column_types[column] = 'datetime'
            elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                # Check if it's likely text or categorical
                if df[column].nunique() > self.max_categories:
                    self.column_types[column] = 'text'
                else:
                    self.column_types[column] = 'categorical'
            else:
                # Default to categorical for other types
                self.column_types[column] = 'categorical'
    
    def _fit_numeric_processor(self, df: pd.DataFrame, column: str) -> None:
        """
        Fit a processor for a numeric column.
        
        Args:
            df: Input DataFrame
            column: Column name
        """
        steps = []
        
        # Add imputer
        steps.append(('imputer', SimpleImputer(strategy='median')))
        
        # Add scaler based on strategy
        if self.numeric_strategy == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif self.numeric_strategy == 'robust':
            from sklearn.preprocessing import RobustScaler
            steps.append(('scaler', RobustScaler()))
        elif self.numeric_strategy == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            steps.append(('scaler', MinMaxScaler()))
        
        # Create and fit pipeline
        pipeline = Pipeline(steps)
        pipeline.fit(df[[column]])
        
        self.column_processors[column] = pipeline
    
    def _fit_categorical_processor(self, df: pd.DataFrame, column: str, target_column: Optional[str]) -> None:
        """
        Fit a processor for a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name
            target_column: Target column for target encoding
        """
        if self.categorical_strategy == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(df[[column]])
            self.column_processors[column] = encoder
        elif self.categorical_strategy == 'ordinal':
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(df[[column]])
            self.column_processors[column] = encoder
        elif self.categorical_strategy == 'target' and target_column is not None:
            # Simple target encoding: replace category with mean of target
            encoding = df.groupby(column)[target_column].mean().to_dict()
            self.column_processors[column] = encoding
    
    def _fit_datetime_processor(self, df: pd.DataFrame, column: str) -> None:
        """
        Fit a processor for a datetime column.
        
        Args:
            df: Input DataFrame
            column: Column name
        """
        # No fitting needed for datetime columns, just store the strategy
        self.column_processors[column] = self.datetime_strategy
    
    def _fit_text_processor(self, df: pd.DataFrame, column: str) -> None:
        """
        Fit a processor for a text column.
        
        Args:
            df: Input DataFrame
            column: Column name
        """
        if self.text_strategy == 'basic':
            # No fitting needed for basic text features
            self.column_processors[column] = 'basic'
        elif self.text_strategy == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            # Fill NaN values with empty string
            text_data = df[column].fillna('').astype(str)
            vectorizer.fit(text_data)
            self.column_processors[column] = vectorizer
        elif self.text_strategy == 'count':
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=100)
            # Fill NaN values with empty string
            text_data = df[column].fillna('').astype(str)
            vectorizer.fit(text_data)
            self.column_processors[column] = vectorizer
    
    def _transform_numeric(self, df: pd.DataFrame, column: str, processor: Pipeline) -> pd.DataFrame:
        """
        Transform a numeric column.
        
        Args:
            df: Input DataFrame
            column: Column name
            processor: Fitted processor
            
        Returns:
            DataFrame with transformed features
        """
        transformed = processor.transform(df[[column]])
        return pd.DataFrame(transformed, index=df.index, columns=[f"{column}_scaled"])
    
    def _transform_categorical(self, df: pd.DataFrame, column: str, processor: Any) -> pd.DataFrame:
        """
        Transform a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name
            processor: Fitted processor
            
        Returns:
            DataFrame with transformed features
        """
        if isinstance(processor, OneHotEncoder):
            transformed = processor.transform(df[[column]])
            feature_names = [f"{column}_{cat}" for cat in processor.categories_[0]]
            return pd.DataFrame(transformed, index=df.index, columns=feature_names)
        elif isinstance(processor, dict):
            # Target encoding
            transformed = df[column].map(processor).fillna(0)
            return pd.DataFrame({f"{column}_target_encoded": transformed}, index=df.index)
        else:
            # Ordinal encoding
            transformed = processor.transform(df[[column]])
            return pd.DataFrame(transformed, index=df.index, columns=[f"{column}_ordinal"])
    
    def _transform_datetime(self, df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
        """
        Transform a datetime column.
        
        Args:
            df: Input DataFrame
            column: Column name
            strategy: Datetime transformation strategy
            
        Returns:
            DataFrame with transformed features
        """
        result = {}
        
        if strategy in ['cyclical', 'both']:
            # Cyclical encoding of time components
            dt = df[column].dt
            
            # Convert pandas datetime components to numpy arrays first
            hour_array = dt.hour.to_numpy()
            dayofweek_array = dt.dayofweek.to_numpy()
            day_array = dt.day.to_numpy()
            month_array = dt.month.to_numpy()
            
            # Use ops with numpy arrays
            # Hour of day (0-23)
            two_pi = ops.multiply(tensor.convert_to_tensor(2.0), tensor.convert_to_tensor(ops.pi))
            hour_tensor = tensor.convert_to_tensor(hour_array)
            result[f"{column}_sin_hour"] = ops.sin(ops.divide(ops.multiply(two_pi, hour_tensor), tensor.convert_to_tensor(23.0)))
            result[f"{column}_cos_hour"] = ops.cos(ops.divide(ops.multiply(two_pi, hour_tensor), tensor.convert_to_tensor(23.0)))
            
            # Day of week (0-6)
            dayofweek_tensor = tensor.convert_to_tensor(dayofweek_array)
            result[f"{column}_sin_dayofweek"] = ops.sin(ops.divide(ops.multiply(two_pi, dayofweek_tensor), tensor.convert_to_tensor(6.0)))
            result[f"{column}_cos_dayofweek"] = ops.cos(ops.divide(ops.multiply(two_pi, dayofweek_tensor), tensor.convert_to_tensor(6.0)))
            
            # Day of month (1-31)
            day_tensor = tensor.convert_to_tensor(day_array)
            day_minus_one = ops.subtract(day_tensor, tensor.convert_to_tensor(1.0))
            result[f"{column}_sin_day"] = ops.sin(ops.divide(ops.multiply(two_pi, day_minus_one), tensor.convert_to_tensor(30.0)))
            result[f"{column}_cos_day"] = ops.cos(ops.divide(ops.multiply(two_pi, day_minus_one), tensor.convert_to_tensor(30.0)))
            
            # Month (1-12)
            month_tensor = tensor.convert_to_tensor(month_array)
            month_minus_one = ops.subtract(month_tensor, tensor.convert_to_tensor(1.0))
            result[f"{column}_sin_month"] = ops.sin(ops.divide(ops.multiply(two_pi, month_minus_one), tensor.convert_to_tensor(11.0)))
            result[f"{column}_cos_month"] = ops.cos(ops.divide(ops.multiply(two_pi, month_minus_one), tensor.convert_to_tensor(11.0)))
        
        if strategy in ['components', 'both']:
            # Direct components
            dt = df[column].dt
            result[f"{column}_year"] = dt.year
            result[f"{column}_month"] = dt.month
            result[f"{column}_day"] = dt.day
            result[f"{column}_hour"] = dt.hour
            result[f"{column}_minute"] = dt.minute
            result[f"{column}_dayofweek"] = dt.dayofweek
            result[f"{column}_quarter"] = dt.quarter
            
            # Time since epoch - use tensor.cast instead of direct int64 cast
            timestamp_tensor = tensor.convert_to_tensor(df[column].astype('int64').to_numpy())
            divisor = tensor.convert_to_tensor(10**9)
            result[f"{column}_timestamp"] = tensor.cast(ops.floor_divide(timestamp_tensor, divisor), dtype=tensor.int32)
        
        return pd.DataFrame(result, index=df.index)
    
    def _transform_text(self, df: pd.DataFrame, column: str, processor: Any) -> pd.DataFrame:
        """
        Transform a text column.
        
        Args:
            df: Input DataFrame
            column: Column name
            processor: Fitted processor or strategy
            
        Returns:
            DataFrame with transformed features
        """
        # Fill NaN values with empty string
        text_data = df[column].fillna('').astype(str)
        
        if processor == 'basic':
            # Basic text features
            result = {
                f"{column}_length": text_data.str.len(),
                f"{column}_word_count": text_data.str.split().str.len(),
                f"{column}_char_per_word": ops.divide(
                    tensor.convert_to_tensor(text_data.str.len().to_numpy()),
                    ops.add(
                        tensor.convert_to_tensor(text_data.str.split().str.len().to_numpy()),
                        tensor.convert_to_tensor(1.0)
                    )
                ),
                f"{column}_uppercase_ratio": ops.divide(
                    tensor.convert_to_tensor(text_data.str.count(r'[A-Z]').to_numpy()),
                    ops.add(
                        tensor.convert_to_tensor(text_data.str.len().to_numpy()),
                        tensor.convert_to_tensor(1.0)
                    )
                ),
                f"{column}_digit_ratio": ops.divide(
                    tensor.convert_to_tensor(text_data.str.count(r'[0-9]').to_numpy()),
                    ops.add(
                        tensor.convert_to_tensor(text_data.str.len().to_numpy()),
                        tensor.convert_to_tensor(1.0)
                    )
                ),
                f"{column}_special_ratio": ops.divide(
                    tensor.convert_to_tensor(text_data.str.count(r'[^a-zA-Z0-9\s]').to_numpy()),
                    ops.add(
                        tensor.convert_to_tensor(text_data.str.len().to_numpy()),
                        tensor.convert_to_tensor(1.0)
                    )
                )
            }
            return pd.DataFrame(result, index=df.index)
        else:
            # TF-IDF or Count vectorization
            transformed = processor.transform(text_data)
            feature_names = [f"{column}_{feat}" for feat in processor.get_feature_names_out()]
            return pd.DataFrame(transformed.toarray(), index=df.index, columns=feature_names)


class ColumnPCAFeatureExtractor(ColumnFeatureExtractor):
    """
    Extends ColumnFeatureExtractor with PCA-based dimensionality reduction.
    
    This class applies PCA to each column type separately, then combines the results.
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'standard',
                 categorical_strategy: str = 'onehot',
                 datetime_strategy: str = 'cyclical',
                 text_strategy: str = 'basic',
                 max_categories: int = 100,
                 pca_components: Optional[int] = None,
                 pca_per_type: bool = True):
        """
        Initialize the column PCA feature extractor.
        
        Args:
            numeric_strategy: Strategy for numeric columns
            categorical_strategy: Strategy for categorical columns
            datetime_strategy: Strategy for datetime columns
            text_strategy: Strategy for text columns
            max_categories: Maximum number of categories for one-hot encoding
            pca_components: Number of PCA components (if None, will be calculated)
            pca_per_type: Whether to apply PCA separately to each column type
        """
        super().__init__(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            datetime_strategy=datetime_strategy,
            text_strategy=text_strategy,
            max_categories=max_categories
        )
        
        self.pca_components = pca_components
        self.pca_per_type = pca_per_type
        self.pca_models: Dict[str, Any] = {}
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'ColumnPCAFeatureExtractor':
        """
        Fit the feature extractor to the data.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for target encoding
            
        Returns:
            Self for method chaining
        """
        # First, fit the base column processors
        super().fit(df, target_column)
        
        # Transform the data using the base processors
        transformed = super().transform(df)
        
        if self.pca_per_type:
            # Group columns by type
            numeric_cols = [col for col in transformed.columns if col.startswith(tuple(
                f"{c}_" for c in self.column_types if self.column_types[c] == 'numeric'
            ))]
            
            categorical_cols = [col for col in transformed.columns if col.startswith(tuple(
                f"{c}_" for c in self.column_types if self.column_types[c] == 'categorical'
            ))]
            
            datetime_cols = [col for col in transformed.columns if col.startswith(tuple(
                f"{c}_" for c in self.column_types if self.column_types[c] == 'datetime'
            ))]
            
            text_cols = [col for col in transformed.columns if col.startswith(tuple(
                f"{c}_" for c in self.column_types if self.column_types[c] == 'text'
            ))]
            
            # Fit PCA for each group
            self._fit_pca_for_group(transformed, numeric_cols, 'numeric')
            self._fit_pca_for_group(transformed, categorical_cols, 'categorical')
            self._fit_pca_for_group(transformed, datetime_cols, 'datetime')
            self._fit_pca_for_group(transformed, text_cols, 'text')
        else:
            # Fit a single PCA for all columns
            self._fit_pca_for_group(transformed, transformed.columns, 'all')
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted processors and PCA.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with transformed features
        """
        # First, transform using the base processors
        transformed = super().transform(df)
        
        if self.pca_per_type:
            # Transform each group separately
            result_dfs = []
            
            for group_type, pca_model in self.pca_models.items():
                if group_type == 'numeric':
                    cols = [col for col in transformed.columns if col.startswith(tuple(
                        f"{c}_" for c in self.column_types if self.column_types[c] == 'numeric'
                    ))]
                elif group_type == 'categorical':
                    cols = [col for col in transformed.columns if col.startswith(tuple(
                        f"{c}_" for c in self.column_types if self.column_types[c] == 'categorical'
                    ))]
                elif group_type == 'datetime':
                    cols = [col for col in transformed.columns if col.startswith(tuple(
                        f"{c}_" for c in self.column_types if self.column_types[c] == 'datetime'
                    ))]
                elif group_type == 'text':
                    cols = [col for col in transformed.columns if col.startswith(tuple(
                        f"{c}_" for c in self.column_types if self.column_types[c] == 'text'
                    ))]
                else:
                    continue
                
                if cols and len(cols) > 1:  # Only apply PCA if we have multiple columns
                    pca_result = self._transform_pca(transformed[cols], pca_model, group_type)
                    result_dfs.append(pca_result)
            
            # Combine all PCA results
            if result_dfs:
                return pd.concat(result_dfs, axis=1)
            else:
                return pd.DataFrame()
        else:
            # Transform all columns with a single PCA
            if 'all' in self.pca_models and len(transformed.columns) > 1:
                return self._transform_pca(transformed, self.pca_models['all'], 'all')
            else:
                return transformed
    
    def _fit_pca_for_group(self, df: pd.DataFrame, columns, group_type: str) -> None:
        """
        Fit PCA for a group of columns.
        
        Args:
            df: Input DataFrame
            columns: List or Index of column names
            group_type: Type of the column group
        """
        if len(columns) <= 1:
            return  # Skip PCA for single columns
            
        # Select columns
        X = df[columns]
        
        # Calculate appropriate number of components
        if self.pca_components is None:
            n_components = tensor.cast(
                ops.floor_divide(
                    tensor.convert_to_tensor(X.shape[1]),
                    tensor.convert_to_tensor(2)
                ),
                dtype=tensor.int32
            )
            # Use tensor.minimum or construct minimum manually
            # First approach: create a tensor array with both values and use stats.min
            candidates1 = tensor.stack([n_components, tensor.convert_to_tensor(10)])
            n_components = stats.min(candidates1, axis=0)
            
            # For the second calculation
            one_tensor = tensor.convert_to_tensor(1)
            samples_minus_one = ops.subtract(tensor.convert_to_tensor(X.shape[0]), one_tensor)
            candidates2 = tensor.stack([n_components, samples_minus_one])
            n_components = stats.min(candidates2, axis=0)
            # Convert to tensor.int32 type first, then to Python scalar
            n_components = tensor.cast(n_components, tensor.int32)
            n_components = tensor.to_numpy(n_components).item()
        else:
            n_components = min(self.pca_components, X.shape[1], X.shape[0] - 1)
            
        # Fit PCA
        # Convert pandas DataFrame to NumPy array first, then to tensor before passing to PCA
        X_numpy = X.to_numpy()
        X_tensor = tensor.convert_to_tensor(X_numpy)
        pca = PCA()
        pca.fit(X_tensor, n_components=n_components)
        
        self.pca_models[group_type] = pca
    
    def _transform_pca(self, df: pd.DataFrame, pca_model: PCA, group_type: str) -> pd.DataFrame:
        """
        Transform data using a fitted PCA model.
        
        Args:
            df: Input DataFrame
            pca_model: Fitted PCA model
            group_type: Type of the column group
            
        Returns:
            DataFrame with PCA-transformed features
        """
        # Convert pandas DataFrame to NumPy array first, then to tensor
        df_numpy = df.to_numpy()
        df_tensor = tensor.convert_to_tensor(df_numpy)
        transformed = pca_model.transform(df_tensor)
        
        # Use tensor functions to work with the transformed data
        shape = tensor.shape(transformed)
        n_components = shape[1]
        
        # Convert to numpy array using tensor.to_numpy for backend compatibility
        transformed_np = tensor.to_numpy(transformed)
        columns = [f"pca_{group_type}_{i+1}" for i in range(n_components)]
        return pd.DataFrame(transformed_np, index=df.index, columns=columns)


class TemporalColumnFeatureExtractor(ColumnFeatureExtractor):
    """
    Extends ColumnFeatureExtractor with temporal processing capabilities.
    
    This class applies temporal processing to time series data, creating
    features that capture temporal patterns and relationships.
    """
    
    def __init__(self,
                 window_size: int = 5,
                 stride: int = 1,
                 numeric_strategy: str = 'standard',
                 categorical_strategy: str = 'onehot',
                 datetime_strategy: str = 'cyclical',
                 text_strategy: str = 'basic',
                 max_categories: int = 100):
        """
        Initialize the temporal column feature extractor.
        
        Args:
            window_size: Size of the sliding window
            stride: Stride length for window creation
            numeric_strategy: Strategy for numeric columns
            categorical_strategy: Strategy for categorical columns
            datetime_strategy: Strategy for datetime columns
            text_strategy: Strategy for text columns
            max_categories: Maximum number of categories for one-hot encoding
        """
        super().__init__(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            datetime_strategy=datetime_strategy,
            text_strategy=text_strategy,
            max_categories=max_categories
        )
        
        self.window_size = window_size
        self.stride = stride
        self.temporal_processors: Dict[str, Any] = {}
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None, 
            time_column: Optional[str] = None) -> 'TemporalColumnFeatureExtractor':
        """
        Fit the feature extractor to the data.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for target encoding
            time_column: Column to use for temporal ordering
            
        Returns:
            Self for method chaining
        """
        # First, fit the base column processors
        super().fit(df, target_column)
        
        # Sort by time column if provided
        if time_column is not None and time_column in df.columns:
            df = df.sort_values(time_column)
        
        # Transform the data using the base processors
        transformed = super().transform(df)
        
        # Fit temporal processors for numeric columns
        for column in transformed.columns:
            if pd.api.types.is_numeric_dtype(transformed[column]):
                self._fit_temporal_processor(transformed, column)
        
        return self
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None,
                     time_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit to the data, then transform it.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for target encoding
            time_column: Optional column to use for temporal ordering
            
        Returns:
            DataFrame with transformed features
        """
        return self.fit(df, target_column, time_column).transform(df, time_column)
    
    def transform(self, df: pd.DataFrame, time_column: Optional[str] = None) -> pd.DataFrame:
        """
        Transform the data using the fitted processors and temporal processing.
        
        Args:
            df: Input DataFrame
            time_column: Column to use for temporal ordering
            
        Returns:
            DataFrame with transformed features
        """
        # Sort by time column if provided
        if time_column is not None and time_column in df.columns:
            df = df.sort_values(time_column)
        
        # First, transform using the base processors
        transformed = super().transform(df)
        
        # Apply temporal processing
        result_dfs = [transformed]
        
        for column, processor in self.temporal_processors.items():
            if column in transformed.columns:
                temporal_features = self._apply_temporal_processing(transformed[column], processor, column)
                result_dfs.append(temporal_features)
        
        # Combine all results
        return pd.concat(result_dfs, axis=1)
    
    def _fit_temporal_processor(self, df: pd.DataFrame, column: str) -> None:
        """
        Fit a temporal processor for a column.
        
        Args:
            df: Input DataFrame
            column: Column name
        """
        # For now, we just store the column name
        # More complex processors could be added here
        self.temporal_processors[column] = {
            'window_size': self.window_size,
            'stride': self.stride
        }
    
    def _apply_temporal_processing(self, series: pd.Series, processor: Dict, column: str) -> pd.DataFrame:
        """
        Apply temporal processing to a series.
        
        Args:
            series: Input series
            processor: Processor configuration
            column: Column name
            
        Returns:
            DataFrame with temporal features
        """
        window_size = processor['window_size']
        stride = processor['stride']
        
        # Create windows
        windows = []
        for i in range(0, len(series) - window_size + 1, stride):
            windows.append(series.iloc[i:i+window_size].values)
        
        if not windows:
            return pd.DataFrame()
        
        # Convert windows to tensor using ops
        windows_array = tensor.convert_to_tensor(windows)
        
        # Create features
        result = {}
        
        # Basic statistics
        means = ops.stats.mean(windows_array, axis=1)
        result[f"{column}_window_mean"] = means
        
        # Calculate standard deviation manually: std = sqrt(mean((x - mean(x))^2))
        # Expand means to match windows_array shape for broadcasting
        expanded_means = tensor.expand_dims(means, axis=1)
        # Calculate squared differences
        squared_diffs = ops.square(ops.subtract(windows_array, expanded_means))
        # Calculate variance
        variances = ops.stats.mean(squared_diffs, axis=1)
        # Calculate standard deviation
        result[f"{column}_window_std"] = ops.sqrt(variances)
        
        result[f"{column}_window_min"] = stats.min(windows_array, axis=1)
        result[f"{column}_window_max"] = stats.max(windows_array, axis=1)
        
        # Trend features - use ops for polyfit-like functionality
        # Create x values (0, 1, 2, ..., window_size-1)
        x_values = tensor.arange(0, window_size, dtype=tensor.float32)
        
        # Calculate slope for each window
        slopes = []
        for i in range(windows_array.shape[0]):
            window = windows_array[i]
            # Calculate mean of x and y
            x_mean = ops.stats.mean(x_values)
            y_mean = ops.stats.mean(window)
            
            # Calculate numerator: sum((x - x_mean) * (y - y_mean))
            x_diff = ops.subtract(x_values, x_mean)
            y_diff = ops.subtract(window, y_mean)
            numerator = stats.sum(ops.multiply(x_diff, y_diff))
            
            # Calculate denominator: sum((x - x_mean)^2)
            denominator = stats.sum(ops.square(x_diff))
            
            # Calculate slope: numerator / denominator
            slope = ops.divide(numerator, denominator)
            slopes.append(slope)
        
        result[f"{column}_window_slope"] = tensor.stack(slopes)
        
        # Create index for the result
        index = series.index[window_size-1::stride]
        if len(index) > len(windows):
            index = index[:len(windows)]
        elif len(index) < len(windows):
            # Pad index if needed
            last_idx = index[-1] if len(index) > 0 else series.index[-1]
            pad_size = len(windows) - len(index)
            if hasattr(index, 'freq') and index.freq is not None:
                # For DatetimeIndex with frequency
                new_indices = pd.date_range(start=last_idx + index.freq, periods=pad_size, freq=index.freq)
            else:
                # For other index types
                if isinstance(last_idx, (int, float)):
                    new_indices = pd.RangeIndex(start=last_idx + 1, stop=last_idx + pad_size + 1)
                else:
                    new_indices = pd.Index([f"{last_idx}_{i+1}" for i in range(pad_size)])
            
            index = index.append(new_indices)
        
        return pd.DataFrame(result, index=index)