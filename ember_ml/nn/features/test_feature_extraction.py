"""
Test Feature Extraction

This module provides a test function to demonstrate the feature extraction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ember_ml.nn.features.generic_csv_loader import GenericCSVLoader
from ember_ml.nn.features.generic_type_detector import GenericTypeDetector
from ember_ml.nn.features.generic_feature_engineer import GenericFeatureEngineer
from ember_ml.nn.features.temporal_processor import TemporalStrideProcessor

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