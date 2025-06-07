"""
RBM-based Anomaly Detector

This module provides an anomaly detection system based on Restricted Boltzmann Machines.
It integrates with the generic feature extraction library to provide end-to-end
anomaly detection capabilities, including unsupervised categorization of anomalies.
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import our modules
from ember_ml.ops import stats
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.models.rbm.rbm_module import RBMModule
from ember_ml.nn.tensor.types import TensorLike
class RBMBasedAnomalyDetector:
    """
    Anomaly detection system based on Restricted Boltzmann Machines.
    
    This class uses an RBM to learn the normal patterns in data and
    then detect anomalies as patterns that deviate significantly from
    the learned normal patterns.
    
    The detector can work with both raw data and features extracted
    by the generic feature extraction library.
    """
    
    def __init__(
        self,
        n_hidden: int = 10,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        batch_size: int = 10,
        anomaly_threshold_percentile: float = 95.0,
        anomaly_score_method: str = 'reconstruction',
        track_states: bool = True
    ):
        """
        Initialize the RBM-based anomaly detector.
        
        Args:
            n_hidden: Number of hidden units in the RBM
            learning_rate: Learning rate for RBM training
            momentum: Momentum for RBM training
            weight_decay: Weight decay for RBM training
            batch_size: Batch size for RBM training
            anomaly_threshold_percentile: Percentile for anomaly threshold
            anomaly_score_method: Method for computing anomaly scores
                ('reconstruction' or 'free_energy')
            track_states: Whether to track RBM states for visualization
        """
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.anomaly_score_method = anomaly_score_method
        self.track_states = track_states
        
        # RBM model (initialized during fit)
        self.rbm = None
        
        # Preprocessing parameters
        self.feature_means = None
        self.feature_stds = None
        self.feature_mins = None
        self.feature_maxs = None
        self.scaling_method = 'standard'  # 'standard' or 'minmax'
        
        # Anomaly detection parameters
        self.anomaly_threshold = None
        self.anomaly_scores_mean = None
        self.anomaly_scores_std = None
        
        # Training metadata
        self.n_features = None
        self.training_time = 0
        self.is_fitted = False
    
    def preprocess(
        self,
        X: TensorLike,
        fit: bool = False,
        scaling_method: str = 'standard'
    ) -> TensorLike:
        """
        Preprocess data for RBM training or anomaly detection.
        
        Args:
            X: Input data [n_samples, n_features]
            fit: Whether to fit preprocessing parameters
            scaling_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Preprocessed data
        """
        if fit:
            self.scaling_method = scaling_method
            self.n_features = X.shape[1]
            
            if scaling_method == 'standard':
                # Compute mean and std for standardization
                self.feature_means = stats.mean(X, axis=0)
                self.feature_stds = stats.std(X, axis=0)
                self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero
            else:
                # Compute min and max for min-max scaling
                self.feature_mins = stats.min(X, axis=0)
                self.feature_maxs = stats.max(X, axis=0)
                # Avoid division by zero
                self.feature_maxs[self.feature_maxs == self.feature_mins] += 1e-8
        
        # Apply scaling
        if self.scaling_method == 'standard':
            X_scaled = ops.divide(ops.subtract(X,self.feature_means), self.feature_stds)
        else:
            X_scaled = ops.divide(ops.subtract(X, self.feature_mins), ops.subtract(self.feature_maxs, self.feature_mins))
        
        return X_scaled
    
    def fit(
        self,
        X: TensorLike,
        validation_data: Optional[TensorLike] = None,
        epochs: int = 50,
        k: int = 1,
        early_stopping_patience: int = 5,
        scaling_method: str = 'standard',
        verbose: bool = True
    ) -> 'RBMBasedAnomalyDetector':
        """
        Fit the anomaly detector to normal data.
        
        Args:
            X: Normal data [n_samples, n_features]
            validation_data: Optional validation data
            epochs: Number of training epochs
            k: Number of Gibbs sampling steps
            early_stopping_patience: Patience for early stopping
            scaling_method: Scaling method ('standard' or 'minmax')
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        # Preprocess data
        print(f"[DEBUG] Input data shape: {X.shape}, dtype: {X.dtype}")
        X_scaled = self.preprocess(X, fit=True, scaling_method=scaling_method)
        print(f"[DEBUG] Preprocessed data shape: {X_scaled.shape}, dtype: {X_scaled.dtype}")
        print(f"[DEBUG] Preprocessed data min: {X_scaled.min()}, max: {X_scaled.max()}")
        X_scaled = self.preprocess(X, fit=True, scaling_method=scaling_method)
        
        # Preprocess validation data if provided
        if validation_data is not None:
            validation_data_scaled = self.preprocess(validation_data, fit=False)
        else:
            validation_data_scaled = None
        
        # Initialize RBM
        print(f"[DEBUG] Initializing RBM with {self.n_features} visible units and {self.n_hidden} hidden units")
        print(f"[DEBUG] Learning rate: {self.learning_rate}, momentum: {self.momentum}, weight decay: {self.weight_decay}")
        self.rbm = RBMModule(
            n_visible=self.n_features,
            n_hidden=self.n_hidden,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            use_binary_states=False
        )
        print(f"[DEBUG] RBM initialized with weights shape: {tensor.shape(self.rbm.weights.data)}")
        
        # We need to implement a custom training loop since RBMModule doesn't have a built-in train method
        # This is a simplified version of what would be in a full implementation
        
        # Convert data to tensor with float32 dtype to match the model weights
        print(f"[DEBUG] Converting data to tensor with dtype float32")
        X_tensor = tensor.convert_to_tensor(X_scaled, dtype=tensor.float32)
        print(f"[DEBUG] Tensor shape: {tensor.shape(X_tensor)}, device: {getattr(X_tensor, 'device', 'N/A')}")
        if validation_data_scaled is not None:
            val_tensor = tensor.convert_to_tensor(validation_data_scaled, dtype=tensor.float32)
        
        # Track best validation error for early stopping
        best_val_error = float('inf')
        patience_counter = 0
        val_errors_list = []  # Track validation errors for convergence visualization
        
        # Create storage for training states (for visualization)
        training_states = []
        training_errors = []
        
        # Initialize momentum buffers
        weights_momentum = tensor.zeros_like(self.rbm.weights.data)
        visible_bias_momentum = tensor.zeros_like(self.rbm.visible_bias.data)
        hidden_bias_momentum = tensor.zeros_like(self.rbm.hidden_bias.data)
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            print(f"[DEBUG] Training epoch {epoch+1}/{epochs}")
            
            # Positive phase
            # Compute hidden probabilities and states from input data
            pos_hidden_probs = self.rbm.compute_hidden_probabilities(X_tensor)
            pos_hidden_states = self.rbm.sample_hidden_states(pos_hidden_probs)
            
            # Convert to numpy to get min/max
            pos_hidden_probs_np = tensor.to_numpy(pos_hidden_probs)
            print(f"[DEBUG] Hidden probabilities shape: {tensor.shape(pos_hidden_probs)}, min: {pos_hidden_probs.min()}, max: {pos_hidden_probs.max()}")
            print(f"[DEBUG] Hidden states shape: {tensor.shape(pos_hidden_states)}")
            
            # Compute positive associations (positive phase statistics)
            # Outer product of visible states and hidden probabilities
            pos_associations = ops.matmul(
                tensor.transpose(X_tensor),
                pos_hidden_probs
            )
            
            # Negative phase (reconstruction)
            # Start with the hidden states from positive phase
            neg_hidden_states = pos_hidden_states
            
            # Compute visible probabilities and states
            neg_visible_probs = self.rbm.compute_visible_probabilities(neg_hidden_states)
            print(f"[DEBUG] Visible probabilities shape: {tensor.shape(neg_visible_probs)}")
            neg_visible_states = self.rbm.sample_visible_states(neg_visible_probs)
            
            # Compute hidden probabilities and states from reconstructed visible states
            neg_hidden_probs = self.rbm.compute_hidden_probabilities(neg_visible_states)
            neg_hidden_states = self.rbm.sample_hidden_states(neg_hidden_probs)
            
            # Compute negative associations (negative phase statistics)
            neg_associations = ops.matmul(
                tensor.transpose(neg_visible_states),
                neg_hidden_probs
            )
            
            # Compute gradients
            batch_size = tensor.convert_to_tensor(tensor.shape(X_tensor)[0], dtype=tensor.float32)
            weight_gradient = ops.divide(
                ops.subtract(pos_associations, neg_associations),
                batch_size
            )
            
            visible_bias_gradient = stats.mean(
                ops.subtract(X_tensor, neg_visible_states),
                axis=0
            )
            
            hidden_bias_gradient = stats.mean(
                ops.subtract(pos_hidden_probs, neg_hidden_probs),
                axis=0
            )
            
            # Update with momentum and weight decay
            weights_momentum = ops.add(
                ops.multiply(self.rbm.momentum, weights_momentum),
                weight_gradient
            )
            
            visible_bias_momentum = ops.add(
                ops.multiply(self.rbm.momentum, visible_bias_momentum),
                visible_bias_gradient
            )
            
            hidden_bias_momentum = ops.add(
                ops.multiply(self.rbm.momentum, hidden_bias_momentum),
                hidden_bias_gradient
            )
            
            # Apply updates
            self.rbm.weights.data = ops.add(
                self.rbm.weights.data,
                ops.multiply(
                    self.rbm.learning_rate,
                    ops.subtract(
                        weights_momentum,
                        ops.multiply(self.rbm.weight_decay, self.rbm.weights.data)
                    )
                )
            )
            
            self.rbm.visible_bias.data = ops.add(
                self.rbm.visible_bias.data,
                ops.multiply(self.rbm.learning_rate, visible_bias_momentum)
            )
            
            self.rbm.hidden_bias.data = ops.add(
                self.rbm.hidden_bias.data,
                ops.multiply(self.rbm.learning_rate, hidden_bias_momentum)
            )
            
            # Compute reconstruction error
            train_error = self.rbm.reconstruction_error(X_tensor)
            print(f"[DEBUG] Reconstruction error: {tensor.to_numpy(train_error)}")
            
            # Store training state for visualization
            # Get a copy of the current weights
            if hasattr(self.rbm.weights, 'numpy'):
                weights = self.rbm.weights.numpy().copy()
            elif hasattr(self.rbm.weights, 'data') and hasattr(self.rbm.weights.data, 'numpy'):
                weights = self.rbm.weights.data
                print(f"[DEBUG] Weights data type: {type(weights)}")
            else:
                # Try to convert using tensor.to_numpy
                weights = tensor.to_numpy(self.rbm.weights).copy()
            
            print(f"[DEBUG] Current weights stats - min: {weights.min()}, max: {weights.max()}, mean: {weights.mean()}")
                
            training_states.append({
                'weights': weights,
                'error': float(train_error)
            })
            training_errors.append(float(train_error))
            
            # Validation error
            if validation_data_scaled is not None:
                val_error = self.rbm.reconstruction_error(val_tensor)
                val_errors_list.append(float(val_error))  # Store validation error
                
                # Early stopping
                if val_error < best_val_error:
                    best_val_error = val_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                val_str = f", val_error: {val_error:.4f}" if validation_data_scaled is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, train_error: {train_error:.4f}{val_str}")
        
        # Store training states and errors on the RBM for visualization
        self.rbm.training_states = training_states
        self.rbm.training_errors = tensor.convert_to_tensor(training_errors,dtype=tensor.float32)
        
        # Store validation errors if available
        if validation_data_scaled is not None:
            val_errors = [float(val_error) for val_error in val_errors_list]
            self.rbm.validation_errors = tensor.convert_to_tensor(val_errors)
        
        # Compute anomaly scores on training data
        # Make sure to use the tensor version with the right dtype
        print(f"[DEBUG] Computing anomaly scores using method: {self.anomaly_score_method}")
        anomaly_scores = self.rbm.anomaly_score(X_tensor, method=self.anomaly_score_method)
        print(f"[DEBUG] Anomaly scores tensor shape: {tensor.shape(anomaly_scores)}")
        # Convert back to numpy for percentile calculation
        anomaly_scores = tensor.to_numpy(anomaly_scores)
        # Compute anomaly threshold
        print(f"[DEBUG] Anomaly scores numpy array shape: {anomaly_scores.shape}")
        print(f"[DEBUG] Anomaly scores stats - min: {anomaly_scores.min()}, max: {anomaly_scores.max()}, mean: {anomaly_scores.mean()}")
        print(f"[DEBUG] Computing threshold at {self.anomaly_threshold_percentile}th percentile")
        self.anomaly_threshold = stats.percentile(
            anomaly_scores,
            self.anomaly_threshold_percentile
        )
        print(f"[DEBUG] Anomaly threshold set to: {self.anomaly_threshold}")
        
        # Compute statistics of anomaly scores
        self.anomaly_scores_mean = stats.mean(anomaly_scores)
        self.anomaly_scores_std = stats.std(anomaly_scores)
        print(f"[DEBUG] Anomaly scores mean: {self.anomaly_scores_mean}, std: {self.anomaly_scores_std}")
        self.anomaly_scores_std = stats.std(anomaly_scores)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        if verbose:
            print(f"Anomaly detector trained in {self.training_time:.2f} seconds")
            print(f"Anomaly threshold: {self.anomaly_threshold:.4f}")
            print(f"Anomaly scores mean: {self.anomaly_scores_mean:.4f}, std: {self.anomaly_scores_std:.4f}")
        
        return self
    
    def predict(self, X: TensorLike) -> TensorLike:
        """
        Predict whether samples are anomalies.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Boolean array indicating anomalies [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        print(f"[DEBUG] Predict - Input data shape: {X.shape}")
        X_scaled = self.preprocess(X, fit=False)
        print(f"[DEBUG] Predict - Preprocessed data shape: {X_scaled.shape}")
        
        # Convert to tensor with correct dtype
        X_tensor = tensor.convert_to_tensor(X_scaled, dtype=tensor.float32)
        print(f"[DEBUG] Predict - Tensor shape: {tensor.shape(X_tensor)}")
        
        # Compute anomaly scores
        print(f"[DEBUG] Predict - Computing anomaly scores using method: {self.anomaly_score_method}")
        anomaly_scores = self.rbm.anomaly_score(X_tensor, method=self.anomaly_score_method)
        anomaly_scores = tensor.to_numpy(anomaly_scores)
        print(f"[DEBUG] Predict - Anomaly scores stats - min: {anomaly_scores.min()}, max: {anomaly_scores.max()}, mean: {anomaly_scores.mean()}")
        
        # Determine anomalies
        print(f"[DEBUG] Predict - Using threshold: {self.anomaly_threshold}")
        anomalies = anomaly_scores > self.anomaly_threshold
        print(f"[DEBUG] Predict - Found {stats.sum(anomalies)} anomalies out of {len(anomalies)} samples ({stats.sum(anomalies)/len(anomalies)*100:.2f}%)")
        
        return anomalies
    
    def anomaly_score(self, X: TensorLike) -> TensorLike:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Anomaly scores [n_samples]
        """
        if not self.is_fitted or self.rbm is None:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Convert to tensor with correct dtype
        X_tensor = tensor.convert_to_tensor(X_scaled, dtype=tensor.float32)
        
        # Compute anomaly scores
        scores_tensor = self.rbm.anomaly_score(X_tensor, method=self.anomaly_score_method)
        
        # Convert back to numpy
        return tensor.to_numpy(scores_tensor)
    
    def anomaly_probability(self, X: TensorLike) -> TensorLike:
        """
        Compute probability of being an anomaly.
        
        This uses a sigmoid function to map anomaly scores to [0, 1].
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Anomaly probabilities [n_samples]
        """
        if not self.is_fitted or self.rbm is None:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Compute anomaly scores
        scores = self.anomaly_score(X)
        
        # Normalize scores
        normalized_scores = (scores - self.anomaly_scores_mean) / self.anomaly_scores_std
        
        # Map to [0, 1] using sigmoid
        return 1.0 / (1.0 + ops.exp(-normalized_scores))
    
    def reconstruct(self, X: TensorLike) -> TensorLike:
        """
        Reconstruct input data using the RBM.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Reconstructed data [n_samples, n_features]
        """
        if not self.is_fitted or self.rbm is None:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Convert to tensor with correct dtype
        X_tensor = tensor.convert_to_tensor(X_scaled, dtype=tensor.float32)
        
        # Reconstruct data
        X_reconstructed_tensor = self.rbm.reconstruct(X_tensor)
        
        # Convert back to numpy
        X_reconstructed = tensor.to_numpy(X_reconstructed_tensor)
        
        # Inverse scaling
        if self.scaling_method == 'standard':
            X_reconstructed = X_reconstructed * self.feature_stds + self.feature_means
        else:
            X_reconstructed = X_reconstructed * (self.feature_maxs - self.feature_mins) + self.feature_mins
        
        return X_reconstructed
    
    def save(self, filepath: str) -> None:
        """
        Save the anomaly detector to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.rbm is None:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'anomaly_threshold_percentile': self.anomaly_threshold_percentile,
            'anomaly_score_method': self.anomaly_score_method,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_mins': self.feature_mins,
            'feature_maxs': self.feature_maxs,
            'scaling_method': self.scaling_method,
            'anomaly_threshold': self.anomaly_threshold,
            'anomaly_scores_mean': self.anomaly_scores_mean,
            'anomaly_scores_std': self.anomaly_scores_std,
            'n_features': self.n_features,
            'training_time': self.training_time,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        # For now, we can't save the RBM separately since RBMModule doesn't have a save method
        # In a real implementation, we would need to implement save/load for RBMModule
        
        # Save model data
        ops.save(filepath, model_data, allow_pickle=True)
        print(f"Anomaly detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RBMBasedAnomalyDetector':
        """
        Load an anomaly detector from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded anomaly detector
        """
        # Load model data
        model_data = ops.load(filepath, allow_pickle=True).item()
        
        # Create detector
        detector = cls(
            n_hidden=model_data['n_hidden'],
            learning_rate=model_data['learning_rate'],
            momentum=model_data['momentum'],
            weight_decay=model_data['weight_decay'],
            batch_size=model_data['batch_size'],
            anomaly_threshold_percentile=model_data['anomaly_threshold_percentile'],
            anomaly_score_method=model_data['anomaly_score_method']
        )
        
        # Set model parameters
        detector.feature_means = model_data['feature_means']
        detector.feature_stds = model_data['feature_stds']
        detector.feature_mins = model_data['feature_mins']
        detector.feature_maxs = model_data['feature_maxs']
        detector.scaling_method = model_data['scaling_method']
        detector.anomaly_threshold = model_data['anomaly_threshold']
        detector.anomaly_scores_mean = model_data['anomaly_scores_mean']
        detector.anomaly_scores_std = model_data['anomaly_scores_std']
        detector.n_features = model_data['n_features']
        detector.training_time = model_data['training_time']
        detector.is_fitted = model_data['is_fitted']
        
        # For now, we can't load the RBM since RBMModule doesn't have a load method
        # In a real implementation, we would need to implement save/load for RBMModule
        # rbm_filepath = filepath + '.rbm'
        # detector.rbm = RBMModule.load(rbm_filepath)
        
        # Initialize a new RBM with the saved parameters
        detector.rbm = RBMModule(
            n_visible=detector.n_features,
            n_hidden=detector.n_hidden,
            learning_rate=detector.learning_rate,
            momentum=detector.momentum,
            weight_decay=detector.weight_decay,
            use_binary_states=False
        )
        
        return detector
    
    def summary(self) -> str:
        """
        Get a summary of the anomaly detector.
        
        Returns:
            Summary string
        """
        if not self.is_fitted or self.rbm is None:
            return "RBM-based Anomaly Detector (not fitted)"
        
        summary = [
            "RBM-based Anomaly Detector Summary",
            "==================================",
            f"Features: {self.n_features}",
            f"Hidden units: {self.n_hidden}",
            f"Scaling method: {self.scaling_method}",
            f"Anomaly score method: {self.anomaly_score_method}",
            f"Anomaly threshold: {self.anomaly_threshold:.4f} ({self.anomaly_threshold_percentile}th percentile)",
            f"Anomaly scores mean: {self.anomaly_scores_mean:.4f}",
            f"Anomaly scores std: {self.anomaly_scores_std:.4f}",
            f"Training time: {self.training_time:.2f} seconds",
            "",
            "RBM Info:",
            "---------",
            f"Visible units: {self.rbm.n_visible}",
            f"Hidden units: {self.rbm.n_hidden}",
            f"Learning rate: {self.rbm.learning_rate}",
            f"Momentum: {self.rbm.momentum}",
            f"Weight decay: {self.rbm.weight_decay}"
        ]
        
        return "\n".join(summary)
    
    def categorize_anomalies(
        self,
        X: TensorLike,
        anomaly_flags: Optional[TensorLike] = None,
        max_clusters: int = 10,
        min_samples_per_cluster: int = 2
    ) -> Tuple[TensorLike, Dict]:
        """
        Categorize anomalies based on hidden unit activation patterns.
        
        This method uses the hidden unit activations of the RBM to cluster
        anomalies into different categories, even without explicit labels.
        It automatically determines the optimal number of clusters using
        silhouette scores.
        
        Args:
            X: Input data to categorize [n_samples, n_features]
            anomaly_flags: Optional boolean array indicating which samples are anomalies
                If None, anomalies are detected using the predict method
            max_clusters: Maximum number of clusters to consider
            min_samples_per_cluster: Minimum samples required to form a cluster
                
        Returns:
            Tuple of (category_labels, cluster_info)
            - category_labels: Array with cluster labels for each sample (-1 for normal samples)
            - cluster_info: Dictionary with information about each cluster
        """
        if not self.is_fitted or self.rbm is None:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        print(f"[DEBUG] Categorizing anomalies - Input data shape: {X.shape}")
        X_scaled = self.preprocess(X, fit=False)
        
        # Detect anomalies if not provided
        if anomaly_flags is None:
            anomaly_flags = self.predict(X)
        
        # Get anomalous samples
        anomaly_indices = ops.where(anomaly_flags)[0]
        anomaly_samples = X_scaled[anomaly_indices]
        
        print(f"[DEBUG] Found {len(anomaly_indices)} anomalies to categorize")
        
        # If no anomalies found, return empty results
        if len(anomaly_indices) == 0:
            return tensor.zeros(len(X), dtype=int) - 1, {}
        
        # If too few anomalies for clustering, assign all to the same category
        if len(anomaly_indices) < min_samples_per_cluster:
            category_labels = tensor.zeros(len(X), dtype=int) - 1  # -1 for normal samples
            category_labels[anomaly_indices] = 0  # 0 for the single anomaly category
            return category_labels, {0: {"count": len(anomaly_indices), "indices": anomaly_indices}}
        
        # Convert anomaly samples to tensors
        anomaly_tensor = tensor.convert_to_tensor(anomaly_samples, dtype=tensor.float32)
        
        # Get hidden unit activations for anomalies
        hidden_probs = self.rbm.compute_hidden_probabilities(anomaly_tensor)
        hidden_activations = tensor.to_numpy(hidden_probs)
        
        print(f"[DEBUG] Hidden activations shape: {hidden_activations.shape}")
        
        # Determine optimal number of clusters
        max_k = min(max_clusters, len(anomaly_indices) // min_samples_per_cluster)
        max_k = max(2, max_k)  # At least 2 clusters if possible
        
        best_k = 2  # Default to 2 clusters
        best_score = -1
        
        # Try different numbers of clusters
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(hidden_activations)
            
            # Compute silhouette score if there are at least 2 clusters with samples
            if len(np.unique(cluster_labels)) >= 2:
                score = silhouette_score(hidden_activations, cluster_labels)
                print(f"[DEBUG] K={k}, Silhouette Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Cluster with optimal K
        print(f"[DEBUG] Using optimal K={best_k} clusters (silhouette score: {best_score:.4f})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        anomaly_cluster_labels = kmeans.fit_predict(hidden_activations)
        
        # Create final category labels for all samples
        category_labels = tensor.zeros(len(X), dtype=int) - 1  # -1 for normal samples
        category_labels[anomaly_indices] = anomaly_cluster_labels
        
        # Prepare cluster information
        cluster_info = {}
        for cluster_id in range(best_k):
            # Get indices of samples in this cluster
            cluster_mask = anomaly_cluster_labels == cluster_id
            cluster_indices = anomaly_indices[cluster_mask]
            
            # Get centroid of this cluster
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Get features that are most activated for this cluster
            feature_activations = X_scaled[cluster_indices].mean(axis=0)
            top_features_idx = np.argsort(ops.abs(feature_activations))[::-1][:3]  # Top 3 features
            
            # Store cluster information
            cluster_info[cluster_id] = {
                "count": len(cluster_indices),
                "indices": cluster_indices,
                "centroid": centroid,
                "top_features": top_features_idx,
                "feature_activations": feature_activations
            }
            
            print(f"[DEBUG] Cluster {cluster_id}: {len(cluster_indices)} anomalies, "
                  f"top features: {top_features_idx}")
        
        return category_labels, cluster_info


# Integration with generic feature extraction
def detect_anomalies_from_features(
    features_df: pd.DataFrame,
    n_hidden: int = 10,
    anomaly_threshold_percentile: float = 95.0,
    training_fraction: float = 0.8,
    epochs: int = 50,
    verbose: bool = True
) -> Tuple[RBMBasedAnomalyDetector, TensorLike, TensorLike]:
    """
    Detect anomalies from features extracted by the generic feature extraction library.
    
    Args:
        features_df: DataFrame with extracted features
        n_hidden: Number of hidden units in the RBM
        anomaly_threshold_percentile: Percentile for anomaly threshold
        training_fraction: Fraction of data to use for training
        epochs: Number of training epochs
        verbose: Whether to print progress
        
    Returns:
        Tuple of (anomaly detector, anomaly flags, anomaly scores)
    """
    # Convert features to numpy array
    features = features_df.values
    
    # Split into training and testing sets
    n_samples = len(features)
    n_train = int(n_samples * training_fraction)
    
    # Shuffle data
    indices = tensor.random_permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_features = features[train_indices]
    test_features = features[test_indices]
    
    # Create and fit anomaly detector
    detector = RBMBasedAnomalyDetector(
        n_hidden=n_hidden,
        anomaly_threshold_percentile=anomaly_threshold_percentile
    )
    
    detector.fit(
        X=train_features,
        epochs=epochs,
        verbose=verbose
    )
    
    # Detect anomalies
    anomaly_flags = detector.predict(test_features)
    anomaly_scores = detector.anomaly_score(test_features)
    
    if verbose:
        n_anomalies = stats.sum(anomaly_flags)
        print(f"Detected {n_anomalies} anomalies out of {len(test_features)} samples "
              f"({n_anomalies/len(test_features)*100:.2f}%)")
    
    return detector, anomaly_flags, anomaly_scores