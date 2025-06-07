"""
Backend-agnostic implementation of Principal Component Analysis (PCA).

This module provides a PCA implementation using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Optional, Dict, Any, Union, Tuple
from math import log

from ember_ml.nn import tensor
from ember_ml import ops as ops
from ember_ml.ops import stats
from ember_ml.nn import tensor
# Defer svd import to method call to respect dynamic backend
def _svd_flip(u, v):
    """Sign correction for SVD to ensure deterministic output.
    
    Adjusts the signs of the columns of u and rows of v such that
    the loadings in v are always positive.
    
    Args:
        u: Left singular vectors
        v: Right singular vectors (transposed)
        
    Returns:
        u_adjusted, v_adjusted: Adjusted singular vectors
    """
    # Columns of u, rows of v
    # u has shape (n_samples, n_components)
    # v has shape (n_components, n_features)
    n_samples, n_components = tensor.shape(u)

    # Find the row index of the maximum absolute value in each column of u
    max_abs_row_indices = stats.argmax(ops.abs(u), axis=0) # Shape: (n_components,)

    # Gather the actual elements with the largest absolute value in each column
    # We need elements u[max_abs_row_indices[i], i] for each column i.
    # Using a loop for clarity as vectorized element gathering via ops might be complex/unreliable.
    gathered_elements = []
    for i in range(n_components):
        # Extract scalar index for the row
        row_idx_tensor = max_abs_row_indices[i]
        # Ensure row_idx_tensor is scalar before calling item() if needed
        # Assuming item() works on 0-d tensors resulting from indexing
        try:
             row_idx = tensor.item(row_idx_tensor)
        except: # Broad except, consider refining based on potential errors
             # If item() fails maybe it's already a scalar or needs different handling
             row_idx = int(row_idx_tensor) # Fallback attempt

        col_idx = i
        # Index the specific element u[row_idx, col_idx]
        # Direct indexing might depend on backend tensor type returned by ops
        element = u[row_idx, col_idx]
        gathered_elements.append(element)

    # Compute signs of the gathered elements
    signs = ops.sign(tensor.stack(gathered_elements)) # Shape: (n_components,)


    # Apply signs to columns of u
    # Broadcasting 'signs' (n_components,) across rows of u (n_samples, n_components)
    u = ops.multiply(u, signs)

    # Apply signs to rows of v
    # Reshape signs to (n_components, 1) for broadcasting with v (n_components, n_features)
    signs_reshaped = tensor.reshape(signs, (n_components, 1))
    v = ops.multiply(v, signs_reshaped)
    return u, v


def _find_ncomponents(
    n_components: Optional[Union[int, float, str]],
    n_samples: int,
    n_features: int,
    explained_variance: Any,
    explained_variance_ratio: Any = None,
) -> int:
    """Find the number of components to keep.
    
    Args:
        n_components: Number of components specified by the user
        n_samples: Number of samples
        n_features: Number of features
        explained_variance: Explained variance of each component
        explained_variance_ratio: Explained variance ratio of each component
        
    Returns:
        Number of components to keep
    """
    if n_components is None:
        n_components = min(n_samples, n_features)
    elif isinstance(n_components, float) and 0 < n_components < 1.0:
        # Compute number of components that explain at least n_components of variance
        ratio_cumsum = ops.cumsum(explained_variance_ratio)
        n_components = ops.add(stats.sum(ops.less(ratio_cumsum, n_components)), 1)
    elif n_components == 'mle':
        # Minka's MLE for selecting number of components
        n_components = _infer_dimensions(explained_variance, n_samples)
    
    # Ensure n_components is an integer and within bounds
    n_components = int(n_components)
    n_components = min(n_components, min(n_samples, n_features))
    
    return n_components


def _infer_dimensions(explained_variance, n_samples):
    """Infer the dimensions using Minka's MLE.
    
    Args:
        explained_variance: Explained variance of each component
        n_samples: Number of samples
        
    Returns:
        Number of components to keep
    """
    # Implementation of Minka's MLE for dimensionality selection
    n_components = explained_variance.shape[0]
    ll = tensor.zeros((n_components,))
    
    for i in range(n_components):
        if i < n_components - 1:
            sigma2 = ops.stats.mean(explained_variance[i+1:])
            if sigma2 > 0:
                ll = tensor.tensor_scatter_nd_update(
                    ll,
                    [[i]],
                    [ops.multiply(
                        ops.multiply(-0.5, n_samples),
                        ops.add(
                            ops.add(
                                stats.sum(ops.log(explained_variance[:i+1])),
                                ops.multiply(n_components - i - 1, ops.log(sigma2))
                            ),
                            ops.add(
                                ops.divide(n_components - i - 1, n_components - i),
                                ops.add(
                                    ops.divide(stats.sum(explained_variance[:i+1]), sigma2),
                                    ops.divide(ops.multiply(n_components - i - 1, sigma2), sigma2)
                                )
                            )
                        )
                    )]
                )
        else:
            ll = tensor.tensor_scatter_nd_update(
                ll,
                [[i]],
                [ops.multiply(-0.5, ops.multiply(n_samples, stats.sum(ops.log(explained_variance))))]
            )

    return ops.add(ops.stats.argmax(ll), 1) # Use ops.stats.argmax


def _randomized_svd(
    X: Any,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: Union[int, str] = 'auto',
    power_iteration_normalizer: str = 'auto',
    random_state: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Randomized SVD implementation using ops abstraction layer.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of components to extract
        n_oversamples: Additional number of random vectors for more stable approximation
        n_iter: Number of power iterations
        power_iteration_normalizer: Normalization method for power iterations
        random_state: Random seed
        
    Returns:
        U, S, V: Left singular vectors, singular values, right singular vectors
    """
    n_samples, n_features = tensor.shape(X)
    
    # Set random seed if provided
    if random_state is not None:
        tensor.set_seed(random_state)
    
    # Handle n_iter parameter
    if n_iter == 'auto':
        # Heuristic: set n_iter based on matrix size
        if min(n_samples, n_features) <= 10:
            n_iter = 7
        else:
            n_iter = 4
    
    # Step 1: Sample random vectors
    n_random = min(n_components + n_oversamples, min(n_samples, n_features))
    Q = tensor.random_normal((n_features, n_random))
    
    # Step 2: Compute Y = X * Q
    Y = ops.matmul(X, Q)
    
    # Step 3: Perform power iterations to increase accuracy
    for _ in range(n_iter):
        if power_iteration_normalizer == 'auto' or power_iteration_normalizer == 'QR':
            Q, _ = ops.qr(Y)
        elif power_iteration_normalizer == 'LU':
            # LU normalization not directly available in ops, use QR instead
            Q, _ = ops.qr(Y)
        else:  # 'none'
            Q = Y
        
        # Project X onto Q
        Y = ops.matmul(X, ops.matmul(tensor.transpose(X), Q))
        
        if power_iteration_normalizer == 'auto' or power_iteration_normalizer == 'QR':
            Q, _ = ops.qr(Y)
        elif power_iteration_normalizer == 'LU':
            Q, _ = ops.qr(Y)
        else:  # 'none'
            Q = Y
    
    # Step 4: Compute QR decomposition of Y
    Q, _ = ops.qr(Y)
    
    # Step 5: Project X onto Q
    B = ops.matmul(tensor.transpose(Q), X)
    
    # Step 6: Compute SVD of the small matrix B
    Uhat, S, V = ops.svd(B)
    # --- Truncate Uhat, S, V before calculating final U ---
    Uhat = Uhat[:, :n_components] # Shape (num_features, n_components) e.g. (5, 3)
    S = S[:n_components]         # Shape (n_components,) e.g. (3,)
    V = V[:n_components, :]      # Shape (n_components, n_features) e.g. (3, 5) assuming V is Vh

    # Now calculate final U using truncated Uhat
    U = ops.matmul(Q, Uhat)      # Shape (n_samples, n_features) @ (n_features, n_components) = (n_samples, n_components) e.g. (100, 5) @ (5, 3) = (100, 3)

    V = V[:n_components, :] # Assuming V is Vh (transposed right singular vectors)

    return U, S, V


class PCA:
    """Principal Component Analysis (PCA) implementation using ops abstraction layer.
    
    This implementation is backend-agnostic and works with all backends (NumPy, PyTorch, MLX).
    It implements the PCAInterface from ember_ml.features.interfaces.
    """
    
    def __init__(self):
        """Initialize PCA."""
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.noise_variance_ = None
        self.whiten_ = False
    
    def fit(
        self,
        X: Any,
        n_components: Optional[Union[int, float, str]] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> "PCA":
        """
        Fit the PCA model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            n_components: Number of components to keep
            whiten: Whether to whiten the data
            center: Whether to center the data
            svd_solver: SVD solver to use
            
        Returns:
            Self
        """
        X_tensor = tensor.convert_to_tensor(X)
        self.n_samples_, self.n_features_ = tensor.shape(X_tensor)
        self.whiten_ = whiten
        
        # Choose SVD solver
        if svd_solver == "auto":
            if max(self.n_samples_, self.n_features_) <= 500:
                svd_solver = "full"
            elif n_components is not None and ops.less(
                n_components, 
                ops.multiply(0.8, tensor.convert_to_tensor(min(self.n_samples_, self.n_features_)))
            ):
                svd_solver = "randomized"
            else:
                svd_solver = "full"
        
        # Center data
        if center:
            self.mean_ = stats.mean(X_tensor, axis=0) # Corrected: Use stats.mean
            X_centered = ops.subtract(X_tensor, self.mean_)
        else:
            self.mean_ = tensor.zeros((self.n_features_,))
            X_centered = X_tensor
        
        # Perform SVD
        if svd_solver == "full":
            # Import svd dynamically inside the method
            from ember_ml.ops.linearalg import svd
            U, S, V = svd(X_centered) # Full SVD

            # --- Truncate results from full SVD ---
            if n_components is not None:
                 # Ensure n_components is calculated if it was 'mle' or float
                 _n_components_int = _find_ncomponents(n_components, self.n_samples_, self.n_features_, S) # Need S for MLE

                 U = U[:, :_n_components_int]
                 S = S[:_n_components_int]
                 V = V[:_n_components_int, :] # Assuming V is Vh
            # --- End Truncation ---
            # Explained variance
            denominator = ops.subtract(self.n_samples_, 1)
            # Use a_min and a_max for ops.clip, using float('inf') for upper bound
            denominator = ops.clip(denominator, 1e-8, float('inf'))  # Avoid division by zero
            explained_variance = ops.divide(ops.square(S), denominator)
            total_var = stats.sum(explained_variance)
            # Use a_min and a_max for ops.clip, using float('inf') for upper bound
            total_var = ops.clip(total_var, 1e-8, float('inf'))  # Avoid division by zero
            explained_variance_ratio = ops.divide(explained_variance, total_var)
        elif svd_solver == "randomized":
            if n_components is None:
                n_components = min(self.n_samples_, self.n_features_)
            elif not isinstance(n_components, int):
                raise ValueError("Randomized SVD only supports integer number of components")
            
            U, S, V = _randomized_svd(
                X_centered,
                n_components=n_components,
                n_oversamples=10,
                n_iter=7,
                power_iteration_normalizer='auto',
                random_state=None,
            )
            # Explained variance
            denominator = ops.subtract(self.n_samples_, 1)
            denominator = ops.clip(denominator, 1e-8, float('inf'))  # Avoid division by zero
            explained_variance = ops.divide(ops.square(S), denominator)
            
            # Calculate total variance with safeguards
            squared_sum = stats.sum(ops.square(X_centered))
            total_var = ops.divide(squared_sum, denominator)
            
            # Ensure non-zero denominator for ratio calculation
            total_var = ops.clip(total_var, 1e-8, float('inf'))  # Avoid division by zero
            explained_variance_ratio = ops.divide(explained_variance, total_var)
        elif svd_solver == "covariance_eigh":
            # Compute covariance matrix with safeguards
            denominator = ops.subtract(self.n_samples_, 1)
            denominator = ops.clip(denominator, 1e-8, float('inf'))  # Avoid division by zero
            cov = ops.divide(
                ops.matmul(tensor.transpose(X_centered), X_centered),
                denominator
            )
            # Eigendecomposition
            eigenvals, eigenvecs = ops.eigh(cov)
            # Sort in descending order
            idx = tensor.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            # Fix numerical errors
            eigenvals = ops.clip(eigenvals, 0.0, float('inf'))
            # Compute equivalent variables to full SVD output
            explained_variance = eigenvals
            total_var = stats.sum(explained_variance)
            # Ensure non-zero denominator for ratio calculation
            total_var = ops.clip(total_var, 1e-8, float('inf'))  # Avoid division by zero
            explained_variance_ratio = ops.divide(explained_variance, total_var)
            S = ops.sqrt(ops.multiply(eigenvals, ops.subtract(self.n_samples_, 1)))
            V = tensor.transpose(eigenvecs)
            U = None  # Not needed
        else:
            raise ValueError(f"Unrecognized svd_solver='{svd_solver}'")
        
        # Flip signs for deterministic output
        if U is not None and V is not None:
            U, V = _svd_flip(U, V)
        
        # Determine number of components
        self.n_components_ = _find_ncomponents(
            n_components=n_components,
            n_samples=self.n_samples_,
            n_features=self.n_features_,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
        )
        
        # Store results
        self.components_ = V[:self.n_components_]
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        
        # Compute noise variance
        if ops.less(self.n_components_, min(self.n_samples_, self.n_features_)):
            # Check if slice is empty before computing mean to avoid warning
            remaining_variance = explained_variance[self.n_components_:]
            if tensor.shape(remaining_variance)[0] > 0:
                self.noise_variance_ = stats.mean(remaining_variance)
            else:
                self.noise_variance_ = tensor.convert_to_tensor(0.0)
        else:
            self.noise_variance_ = tensor.convert_to_tensor(0.0)
        
        return self
    
    def transform(self, X: Any) -> Any:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            X_new: Transformed values of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        X_centered = ops.subtract(X_tensor, self.mean_)
        X_transformed = ops.matmul(X_centered, tensor.transpose(self.components_))
        
        if self.whiten_:
            # Avoid division by zero
            eps = 1e-8  # Small constant to avoid division by zero
            scale = ops.sqrt(ops.clip(self.explained_variance_, eps, float('inf')))
            X_transformed = ops.divide(X_transformed, scale)
        
        return X_transformed
    
    def fit_transform(
        self,
        X: Any,
        n_components: Optional[Union[int, float, str]] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> Any:
        """
        Fit the model and apply dimensionality reduction.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            n_components: Number of components to keep
            whiten: Whether to whiten the data
            center: Whether to center the data
            svd_solver: SVD solver to use
            
        Returns:
            X_new: Transformed values of shape (n_samples, n_components)
        """
        self.fit(
            X,
            n_components=n_components,
            whiten=whiten,
            center=center,
            svd_solver=svd_solver,
        )
        return self.transform(X)
    
    def inverse_transform(self, X: Any) -> Any:
        """
        Transform data back to its original space.
        
        Args:
            X: Input data of shape (n_samples, n_components)
            
        Returns:
            X_original: Original data of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        
        if self.whiten_:
            # Avoid division by zero
            eps = 1e-8  # Small constant to avoid division by zero
            scale = ops.sqrt(ops.clip(self.explained_variance_, eps, float('inf')))
            X_unwhitened = ops.multiply(X_tensor, scale)
        else:
            X_unwhitened = X_tensor
        
        # Apply matrix multiplication using ops.matmul
        X_original = ops.matmul(X_unwhitened, self.components_)
        # Add the mean back using ops.add
        X_original = ops.add(X_original, self.mean_)
        
        return X_original