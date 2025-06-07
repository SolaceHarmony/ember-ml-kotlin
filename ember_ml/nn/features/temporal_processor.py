"""
Temporal Stride Processor

This module provides a class for processing data into multi-stride temporal representations.
"""

from typing import Dict, List, Optional, Any
from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.features.pca_features import PCA
from ember_ml.nn import tensor
class TemporalStrideProcessor:
    """
    Processes data into multi-stride temporal representations.
    
    This class creates sliding windows with different strides and applies
    PCA for dimensionality reduction, enabling multi-scale temporal analysis.
    """
    
    def __init__(self, window_size: int = 5, stride_perspectives: Optional[List[int]] = None,
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
        self.pca_models: Dict[int, Any] = {}  # Store PCA models for each stride
        
    def process_batch(self, data) -> Dict[int, EmberTensor]:
        """
        Process data into multi-stride temporal representations.
        
        Args:
            data: Input data tensor (samples x features)
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        # Convert input to EmberTensor if it's not already
        if not isinstance(data, EmberTensor):
            data = EmberTensor(data)
            
        results = {}
        
        for stride in self.stride_perspectives:
            # Extract windows using stride length
            windows = self._create_strided_sequences(data, stride)
            
            if not windows:
                print(f"Warning: No windows created for stride {stride}")
                continue
                
            # Apply PCA blending
            results[stride] = self._apply_pca_blend(windows, stride)
            
            print(f"Created {len(windows)} windows with stride {stride}, "
                  f"shape after PCA: {results[stride].shape}")
            
        return results
    
    def _create_strided_sequences(self, data: EmberTensor, stride: int) -> List[EmberTensor]:
        """
        Create sequences with the given stride.
        
        Args:
            data: Input data tensor
            stride: Stride length
            
        Returns:
            List of windowed sequences
        """
        num_samples = data.shape[0]
        windows: List[EmberTensor] = []
        
        # Skip if data is too small for even one window
        if num_samples < self.window_size:
            print(f"Warning: Data length ({num_samples}) is smaller than window size ({self.window_size})")
            return windows
        
        # Calculate the number of complete windows
        num_windows_tensor = ops.floor_divide(
            ops.add(
                ops.subtract(num_samples, self.window_size),
                tensor.convert_to_tensor(1)
            ),
            tensor.convert_to_tensor(stride)
        )
        # Convert to int and add 1 for range
        num_windows_int = tensor.cast(num_windows_tensor, tensor.int32).numpy()
        num_windows = ops.add(num_windows_int, 1).numpy()
        
        for i in range(num_windows):
            # Calculate start and end indices
            start_idx = ops.multiply(tensor.convert_to_tensor(i), tensor.convert_to_tensor(stride)).numpy()
            end_idx = ops.add(start_idx, self.window_size).numpy()
            
            # Use data.numpy() only for slicing, then convert back to EmberTensor
            # This is one of the few cases where numpy conversion is allowed
            data_np = data.numpy()
            window_np = data_np[start_idx:end_idx]
            window = EmberTensor(window_np)
            windows.append(window)
            
        return windows
    
    def _apply_pca_blend(self, window_batch: List[EmberTensor], stride: int) -> EmberTensor:
        """
        Apply PCA-based temporal blending.
        
        Args:
            window_batch: Batch of windows (list of EmberTensors)
            stride: Stride length
            
        Returns:
            PCA-transformed data
        """
        # Convert windows to numpy arrays for sklearn compatibility
        # This is one of the few cases where numpy conversion is allowed
        window_arrays = [window.numpy() for window in window_batch]
        
        # Stack windows using tensor.stack for the first window to get shape
        first_window = window_batch[0]
        batch_size = len(window_batch)
        window_size = first_window.shape[0]
        feature_dim = first_window.shape[1] if len(first_window.shape) > 1 else 1
        
        # Reshape each window and stack them
        reshaped_windows = []
        for window in window_batch:
            # Flatten the window
            flat_window = window.reshape((ops.multiply(window_size, feature_dim),))
            reshaped_windows.append(flat_window)
        
        # Stack the reshaped windows
        stacked_windows = tensor.stack(reshaped_windows)
        
        # Convert to numpy for PCA processing
        # This is one of the few cases where numpy conversion is allowed
        # The conversion is necessary for compatibility with our PCA implementation
        flat_windows_tensor = stacked_windows 
        
        # Ensure PCA is fit
        if stride not in self.pca_models:
            # Calculate appropriate number of components
            if self.pca_components is None:
                # Use half the flattened dimension, but cap at 32 components
                flat_dim = flat_windows_tensor.shape[1]
                half_dim = ops.floor_divide(
                    tensor.convert_to_tensor(flat_dim),
                    tensor.convert_to_tensor(2)
                ).numpy()
                n_components = min(half_dim, 32)
                
                # Ensure we don't try to extract more components than samples
                batch_size_minus_one = ops.subtract(
                    tensor.convert_to_tensor(batch_size),
                    tensor.convert_to_tensor(1)
                ).numpy()
                n_components = min(n_components, batch_size_minus_one)
            else:
                batch_size_minus_one = ops.subtract(
                    tensor.convert_to_tensor(batch_size),
                    tensor.convert_to_tensor(1)
                ).numpy()
                n_components = min(
                    self.pca_components,
                    batch_size_minus_one,
                    flat_windows_tensor.shape[1]
                )
                
            print(f"Fitting PCA for stride {stride} with {n_components} components")
            # Use our backend-agnostic PCA implementation
            pca = PCA()
            pca.fit(
                flat_windows_tensor,
                n_components=n_components,
                whiten=False,
                center=True,
                svd_solver="auto"
            )
            self.pca_models[stride] = pca
            
        # Transform the data
        transformed = self.pca_models[stride].transform(flat_windows_tensor)
        transformed_np = transformed.numpy() if hasattr(transformed, 'numpy') else transformed
        
        # Convert back to EmberTensor
        return EmberTensor(transformed_np)
    
    def get_explained_variance(self, stride: int) -> Optional[EmberTensor]:
        """
        Get the explained variance ratio for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Sum of explained variance ratios or None if PCA not fit
        """
        if stride in self.pca_models:
            return EmberTensor(stats.sum(self.pca_models[stride].explained_variance_ratio_))
        return None
    
    def get_feature_importance(self, stride: int) -> Optional[EmberTensor]:
        """
        Get feature importance for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Tensor of feature importance scores or None if PCA not fit
        """
        if stride in self.pca_models:
            # Calculate feature importance as the sum of absolute component weights
            abs_components = ops.abs(self.pca_models[stride].components_)
            importance = stats.sum(abs_components, axis=0)
            return EmberTensor(importance)
        return None