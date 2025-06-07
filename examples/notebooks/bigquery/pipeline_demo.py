"""
Integrated Pipeline Demo

This script demonstrates the complete data processing pipeline:
1. Feature extraction from BigQuery using terabyte-scale feature extractor
2. Feature learning with Restricted Boltzmann Machines
3. Processing through CfC-based liquid neural network with LSTM gating
4. Motor neuron output for triggering deeper exploration
"""

import os
import numpy as np
import pandas as pd
# Removed: import tensorflow as tf
import logging
import time
import argparse
from ember_ml.nn.modules import Module # Keep Module, remove others that cause ImportError
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from ember_ml.nn import ops # Ensure ops is imported for tensor operations
from ember_ml.nn import tensor # Ensure tensor is imported for tensor operations
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline_demo')

# Import our components (using the purified backend-agnostic implementation)
from ember_ml.nn.features.terabyte_feature_extractor import TerabyteFeatureExtractor, TerabyteTemporalStrideProcessor
from ember_ml.models.optimized_rbm import OptimizedRBM
from ember_ml.models.stride_aware_cfc import (
    create_liquid_network_with_motor_neuron,
    create_lstm_gated_liquid_network,
    create_multi_stride_liquid_network
)

# Check if ncps is available
from ember_ml.nn.modules import NCP, AutoNCP

class IntegratedPipeline:
    """
    Integrated pipeline for processing terabyte-scale data through
    feature extraction, RBM, and liquid neural network components.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        rbm_hidden_units: int = 64,
        cfc_units: int = 128,
        lstm_units: int = 32,
        stride_perspectives: List[int] = [1, 3, 5],
        sparsity_level: float = 0.5,
        threshold: float = 0.5,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            project_id: GCP project ID (optional if using in BigQuery Studio)
            location: BigQuery location (default: "US")
            chunk_size: Number of rows to process per chunk
            max_memory_gb: Maximum memory usage in GB
            rbm_hidden_units: Number of hidden units in RBM
            cfc_units: Number of units in CfC circuit
            lstm_units: Number of units in LSTM gating
            stride_perspectives: List of stride lengths to use
            sparsity_level: Sparsity level for the connections
            threshold: Initial threshold for triggering exploration
            use_gpu: Whether to use GPU acceleration if available
            verbose: Whether to print progress information
        """
        self.project_id = project_id
        self.location = location
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.rbm_hidden_units = rbm_hidden_units
        self.cfc_units = cfc_units
        self.lstm_units = lstm_units
        self.stride_perspectives = stride_perspectives
        self.sparsity_level = sparsity_level
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Initialize components
        self.feature_extractor = None
        self.temporal_processor = None
        self.rbm = None
        self.liquid_network = None
        
        # For tracking processing
        self.feature_dim = None
        self.rbm_feature_dim = None
        self.processing_time = {}
        
        logger.info(f"Initialized IntegratedPipeline with rbm_hidden_units={rbm_hidden_units}, "
                   f"cfc_units={cfc_units}, lstm_units={lstm_units}")
    
    def initialize_feature_extractor(self, credentials_path: Optional[str] = None):
        """
        Initialize the feature extractor component.
        
        Args:
            credentials_path: Optional path to service account credentials
        """
        start_time = time.time()
        
        # Create feature extractor
        self.feature_extractor = TerabyteFeatureExtractor(
            project_id=self.project_id,
            location=self.location,
            chunk_size=self.chunk_size,
            max_memory_gb=self.max_memory_gb,
            verbose=self.verbose
        )
        
        # Set up BigQuery connection
        self.feature_extractor.setup_bigquery_connection(credentials_path)
        
        # Create temporal processor
        self.temporal_processor = TerabyteTemporalStrideProcessor(
            window_size=10,
            stride_perspectives=self.stride_perspectives,
            pca_components=32,
            batch_size=10000,
            use_incremental_pca=True,
            verbose=self.verbose
        )
        
        self.processing_time['feature_extractor_init'] = time.time() - start_time
        logger.info(f"Feature extractor initialized in {self.processing_time['feature_extractor_init']:.2f}s")
    
    def initialize_rbm(self, input_dim: int):
        """
        Initialize the RBM component.
        
        Args:
            input_dim: Dimension of input features
        """
        start_time = time.time()
        
        # Create RBM
        self.rbm = OptimizedRBM(
            n_visible=input_dim,
            n_hidden=self.rbm_hidden_units,
            learning_rate=0.01,
            momentum=0.5,
            weight_decay=0.0001,
            batch_size=100,
            use_binary_states=False,
            use_gpu=self.use_gpu,
            verbose=self.verbose
        )
        
        self.feature_dim = input_dim
        self.rbm_feature_dim = self.rbm_hidden_units
        
        self.processing_time['rbm_init'] = time.time() - start_time
        logger.info(f"RBM initialized in {self.processing_time['rbm_init']:.2f}s")
    
    def initialize_liquid_network(self, input_dim: int, network_type: str = 'standard'):
        """
        Initialize the liquid neural network component.
        
        Args:
            input_dim: Dimension of input features
            network_type: Type of network ('standard', 'lstm_gated', or 'multi_stride')
        """
        start_time = time.time()
        
        # Create liquid neural network based on type
        # NOTE: The check for NCPS_AVAILABLE was removed as the variable is undefined.
        # Ensure necessary NeuronMap types are available or handle imports appropriately.
        if network_type == 'lstm_gated':
            self.liquid_network = create_lstm_gated_liquid_network(
                input_dim=input_dim,
                units=self.cfc_units,
                lstm_units=self.lstm_units,
                output_dim=1,
                sparsity_level=self.sparsity_level,
                stride_length=self.stride_perspectives[0],
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True
            )
        elif network_type == 'multi_stride':
            self.liquid_network = create_multi_stride_liquid_network(
                input_dim=input_dim,
                stride_perspectives=self.stride_perspectives,
                units_per_stride=self.cfc_units // len(self.stride_perspectives),
                output_dim=1,
                sparsity_level=self.sparsity_level,
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True
            )
        else:  # standard
            self.liquid_network = create_liquid_network_with_motor_neuron(
                input_dim=input_dim,
                units=self.cfc_units,
                output_dim=1,
                sparsity_level=self.sparsity_level,
                stride_length=self.stride_perspectives[0],
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True,
                mixed_memory=True
            )
        
        self.processing_time['liquid_network_init'] = time.time() - start_time
        logger.info(f"Liquid network ({network_type}) initialized in {self.processing_time['liquid_network_init']:.2f}s")
    
    def extract_features(
        self,
        table_id: str,
        target_column: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple:
        """
        Extract features from BigQuery table.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            target_column: Target variable name
            limit: Optional row limit for testing
            
        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        start_time = time.time()
        
        # Check if feature extractor is initialized
        if self.feature_extractor is None:
            self.initialize_feature_extractor()
        
        # Prepare data
        logger.info(f"Extracting features from {table_id}")
        result = self.feature_extractor.prepare_data(
            table_id=table_id,
            target_column=target_column,
            limit=limit
        )
        
        if result is None:
            raise ValueError("Feature extraction failed")
        
        # Unpack results
        train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result
        
        # Update feature dimension
        self.feature_dim = len(train_features)
        
        # Initialize RBM if not already initialized
        if self.rbm is None:
            self.initialize_rbm(self.feature_dim)
        
        self.processing_time['feature_extraction'] = time.time() - start_time
        logger.info(f"Feature extraction completed in {self.processing_time['feature_extraction']:.2f}s")
        logger.info(f"Extracted {self.feature_dim} features")
        
        return train_df[train_features], val_df[val_features], test_df[test_features]
    
    def apply_temporal_processing(self, features_df: pd.DataFrame) -> Dict[int, TensorLike]:
        """
        Apply temporal processing to features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        start_time = time.time()
        
        # Check if temporal processor is initialized
        if self.temporal_processor is None:
            self.temporal_processor = TerabyteTemporalStrideProcessor(
                window_size=10,
                stride_perspectives=self.stride_perspectives,
                pca_components=32,
                batch_size=10000,
                use_incremental_pca=True,
                verbose=self.verbose
            )
        
        # Define a generator to yield data in batches
        def data_generator(df, batch_size=10000):
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i+batch_size].values
        
        # Process data
        logger.info(f"Applying temporal processing with strides {self.stride_perspectives}")
        stride_perspectives = self.temporal_processor.process_large_dataset(
            data_generator(features_df, batch_size=10000)
        )
        
        self.processing_time['temporal_processing'] = time.time() - start_time
        logger.info(f"Temporal processing completed in {self.processing_time['temporal_processing']:.2f}s")
        
        # Log stride perspective shapes
        for stride, data in stride_perspectives.items():
            logger.info(f"Stride {stride}: shape {data.shape}")
        
        return stride_perspectives
    
    def train_rbm(self, features: Union[TensorLike, pd.DataFrame], epochs: int = 10) -> OptimizedRBM:
        """
        Train RBM on features.
        
        Args:
            features: Feature array or DataFrame
            epochs: Number of training epochs
            
        Returns:
            Trained RBM
        """
        start_time = time.time()
        
        # Convert to numpy array if DataFrame
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Check if RBM is initialized
        if self.rbm is None:
            self.initialize_rbm(features.shape[1])
        
        # Define a generator to yield data in batches
        def data_generator(data, batch_size=100):
            # Shuffle data
            indices = tensor.random_permutation(len(data))
            data = data[indices]
            
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
        
        # Train RBM
        logger.info(f"Training RBM with {self.rbm_hidden_units} hidden units for {epochs} epochs")
        training_errors = self.rbm.train_in_chunks(
            data_generator(features, batch_size=100),
            epochs=epochs,
            k=1
        )
        
        self.processing_time['rbm_training'] = time.time() - start_time
        logger.info(f"RBM training completed in {self.processing_time['rbm_training']:.2f}s")
        logger.info(f"Final reconstruction error: {training_errors[-1]:.4f}")
        
        return self.rbm
    
    def extract_rbm_features(self, features: Union[TensorLike, pd.DataFrame]) -> TensorLike:
        """
        Extract features from trained RBM.
        
        Args:
            features: Feature array or DataFrame
            
        Returns:
            RBM features
        """
        start_time = time.time()
        
        # Convert to numpy array if DataFrame
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Check if RBM is trained
        if self.rbm is None:
            raise ValueError("RBM must be trained before extracting features")
        
        # Define a generator to yield data in batches
        def data_generator(data, batch_size=1000):
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
        
        # Extract features
        logger.info(f"Extracting RBM features from {len(features)} samples")
        rbm_features = self.rbm.transform_in_chunks(
            data_generator(features, batch_size=1000)
        )
        
        self.processing_time['rbm_feature_extraction'] = time.time() - start_time
        logger.info(f"RBM feature extraction completed in {self.processing_time['rbm_feature_extraction']:.2f}s")
        logger.info(f"Extracted {rbm_features.shape[1]} RBM features")
        
        return rbm_features
    
    def train_liquid_network(
        self,
        features: TensorLike,
        targets: TensorLike,
        validation_data: Optional[Tuple[TensorLike, TensorLike]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        network_type: str = 'standard',
        learning_rate: float = 0.001, # Add learning rate parameter
        early_stopping_patience: int = 10 # Add early stopping patience
    ) -> Module: # Corrected return type hint
        """
        Train liquid neural network on RBM features using Ember ML's backend-agnostic components.

        Args:
            features: Feature array
            targets: Target array
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            network_type: Type of network ('standard', 'lstm_gated', or 'multi_stride')
            learning_rate: Learning rate for the optimizer.
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped.

        Returns:
            Trained liquid neural network (ember_ml.nn.Module)
        """
        # Import necessary Ember ML training components
        from ember_ml.training import Adam, MSELoss
        from ember_ml.nn import tensor # Ensure tensor is imported
        from ember_ml import ops # Ensure ops is imported

        start_time = time.time()

        # Check if liquid network is initialized
        if self.liquid_network is None:
            # Ensure the initialized network is an ember_ml.nn.Module
            self.initialize_liquid_network(features.shape[1], network_type)
            if not isinstance(self.liquid_network, Module):
                 raise TypeError("Initialized liquid_network is not an ember_ml.nn.Module subclass.")

        # --- Data Preparation ---
        # Convert numpy arrays to backend tensors
        train_features_tensor = tensor.convert_to_tensor(features)
        train_targets_tensor = tensor.convert_to_tensor(targets)

        # Reshape features for sequence input if needed (assuming RNN input)
        if len(train_features_tensor.shape) == 2:
            # Add sequence dimension: (batch, features) -> (batch, 1, features)
            train_features_tensor = tensor.expand_dims(train_features_tensor, axis=1)

        val_features_tensor = None
        val_targets_tensor = None
        if validation_data is not None:
            val_features, val_targets = validation_data
            val_features_tensor = tensor.convert_to_tensor(val_features)
            val_targets_tensor = tensor.convert_to_tensor(val_targets)
            if len(val_features_tensor.shape) == 2:
                val_features_tensor = tensor.expand_dims(val_features_tensor, axis=1)
            validation_data_tensor = (val_features_tensor, val_targets_tensor)
        else:
            validation_data_tensor = None

        # --- Training Setup ---
        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = MSELoss() # Or use ops.mse directly

        # Define the loss function to be differentiated
        def compute_loss(model_params, batch_x, batch_y):
            # Temporarily update model state - need mechanism for this
            # This part is tricky without a built-in functional API like JAX/Haiku
            # Assuming model uses its internal state for now
            y_pred = self.liquid_network(batch_x, training=True)
            # Ensure y_pred and batch_y have compatible shapes for loss
            # E.g., if y_pred is (batch, 1, 1) and batch_y is (batch, 1), squeeze y_pred
            if y_pred.shape != batch_y.shape:
                 y_pred = tensor.squeeze(y_pred) # Adjust as needed
            return loss_fn(batch_y, y_pred)

        # Get the function that computes loss and gradients
        # Assuming model.trainable_variables holds the parameters to update
        value_and_grad_fn = ops.value_and_grad(compute_loss, argnums=0) # Differentiate wrt first arg (params)

        # --- Manual Training Loop ---
        logger.info(f"Starting manual training for {network_type} liquid network for {epochs} epochs")
        num_samples = tensor.shape(train_features_tensor)[0]
        train_losses = []
        val_losses = [] # Placeholder for validation loss tracking

        # --- Early Stopping Initialization ---
        best_val_loss = float('inf')
        patience_counter = 0 # For early stopping
        lr_patience_counter = 0 # For LR reduction
        best_weights = None
        early_stopping_triggered = False

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle data each epoch
            # Assuming tensor.random_permutation exists and works across backends
            # If not, alternative: indices = tensor.shuffle(tensor.arange(num_samples))
            shuffled_indices = tensor.random_permutation(num_samples)
            train_features_tensor = train_features_tensor[shuffled_indices]
            train_targets_tensor = train_targets_tensor[shuffled_indices]

            for i in range(0, num_samples, batch_size):
                batch_start = i
                batch_end = min(i + batch_size, num_samples)
                batch_x = train_features_tensor[batch_start:batch_end]
                batch_y = train_targets_tensor[batch_start:batch_end]

                # Compute loss and gradients for the batch
                # Pass model.trainable_variables - how state is updated needs clarification
                loss_val, grads = value_and_grad_fn(self.liquid_network.trainable_variables, batch_x, batch_y)

                # Apply gradients
                optimizer.apply_gradients(zip(grads, self.liquid_network.trainable_variables))

                epoch_loss += tensor.item(loss_val) # Convert scalar tensor loss to Python float
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            epoch_duration = time.time() - epoch_start_time

            # --- Validation Step (Placeholder) ---
            current_val_loss = None
            if validation_data_tensor is not None:
                 # Perform forward pass on validation data (no gradients)
                 val_preds = self.liquid_network(validation_data_tensor[0], training=False)
                 if val_preds.shape != validation_data_tensor[1].shape:
                      val_preds = tensor.squeeze(val_preds) # Adjust as needed
                 current_val_loss = tensor.item(loss_fn(validation_data_tensor[1], val_preds))
                 val_losses.append(current_val_loss)
                 logger.info(f"Epoch {epoch+1}/{epochs} - {epoch_duration:.2f}s - loss: {avg_epoch_loss:.4f} - val_loss: {current_val_loss:.4f}")

                 # --- Early Stopping Check ---
                 if current_val_loss < best_val_loss:
                     best_val_loss = current_val_loss
                     patience_counter = 0
                     lr_patience_counter = 0 # Reset LR patience on improvement
                     best_weights = self.liquid_network.state_dict() # Save best weights
                     logger.info(f"Validation loss improved. Saving best weights.")
                 else:
                     patience_counter += 1
                     lr_patience_counter += 1 # Increment LR patience counter
                     logger.info(f"Validation loss did not improve for {patience_counter} ES epoch(s) / {lr_patience_counter} LR epoch(s).")

                 # --- Learning Rate Reduction Check ---
                 if lr_patience_counter >= lr_reduction_patience:
                     current_lr = optimizer.learning_rate # Assuming this attribute exists
                     new_lr = max(current_lr * lr_reduction_factor, min_lr)
                     if new_lr < current_lr:
                         optimizer.learning_rate = new_lr # Assuming this attribute is writable
                         logger.info(f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}.")
                         lr_patience_counter = 0 # Reset LR patience after reduction
                     else:
                          logger.info(f"LR reduction patience met, but LR already at minimum ({min_lr:.6f}).")
                          # Optionally reset counter even if LR not reduced, or let it keep incrementing
                          lr_patience_counter = 0 # Resetting here to avoid repeated checks every epoch once min_lr is hit

            else:
                 logger.info(f"Epoch {epoch+1}/{epochs} - {epoch_duration:.2f}s - loss: {avg_epoch_loss:.4f}")
                 # Note: Early stopping based on training loss is less common but could be added here if needed.

            # --- Check Patience Counter ---
            if validation_data_tensor is not None and patience_counter >= early_stopping_patience:
                 logger.info(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {early_stopping_patience} epochs.")
                 early_stopping_triggered = True
                 break

        # --- Restore Best Weights After Loop ---
        if early_stopping_triggered and best_weights is not None:
             logger.info(f"Restoring model weights from epoch {epoch + 1 - early_stopping_patience} with best validation loss: {best_val_loss:.4f}")
             self.liquid_network.load_state_dict(best_weights)
        elif not early_stopping_triggered:
             logger.info("Training completed without early stopping.")
        # If stopped early but best_weights is None (e.g., val loss never improved), keep last weights.

        self.processing_time['liquid_network_training'] = time.time() - start_time
        logger.info(f"Manual liquid network training completed in {self.processing_time['liquid_network_training']:.2f}s")

        # Log final losses
        final_loss = train_losses[-1]
        final_val_loss = val_losses[-1] if val_losses else None
        logger.info(f"Final training loss: {final_loss:.4f}")
        if final_val_loss is not None:
            logger.info(f"Final validation loss: {final_val_loss:.4f}")

        return self.liquid_network # Return the trained Ember ML module
    
    def process_data(
        self,
        features: TensorLike,
        return_triggers: bool = True
    ) -> Union[TensorLike, Tuple[TensorLike, TensorLike]]:
        """
        Process data through the complete pipeline.
        
        Args:
            features: Feature array
            return_triggers: Whether to return trigger signals
            
        Returns:
            Motor neuron outputs and trigger signals
        """
        start_time = time.time()
        
        # Check if all components are initialized
        if self.rbm is None:
            raise ValueError("RBM must be trained before processing data")
        if self.liquid_network is None:
            raise ValueError("Liquid network must be trained before processing data")
        
        # Extract RBM features
        rbm_features = self.rbm.transform(features)
        
        # Convert features to tensor
        rbm_features_tensor = tensor.convert_to_tensor(rbm_features)

        # Reshape for sequence input if needed
        if len(rbm_features_tensor.shape) == 2:
            rbm_features_tensor = tensor.expand_dims(rbm_features_tensor, axis=1) # Use tensor.expand_dims

        # Process through liquid network (Ember ML Module call)
        # Ensure the network is an Ember Module
        if not isinstance(self.liquid_network, Module):
             raise TypeError("Liquid network is not a valid Ember ML Module for processing.")

        # Use standard call for inference
        raw_outputs = self.liquid_network(rbm_features_tensor, training=False)

        # Unpack outputs based on LiquidNetworkWithMotorNeuron structure
        # Assuming it returns (motor_outputs, [trigger_signals, threshold_values])
        if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2 and isinstance(raw_outputs[1], list):
             motor_outputs_tensor = raw_outputs[0]
             trigger_signals_tensor = raw_outputs[1][0] # Extract trigger signals
        else:
             # Fallback if the output structure is different (e.g., only motor outputs)
             motor_outputs_tensor = raw_outputs
             # Calculate triggers based on threshold if not returned explicitly
             threshold_tensor = tensor.full_like(motor_outputs_tensor, self.threshold)
             trigger_signals_tensor = tensor.cast(ops.greater(motor_outputs_tensor, threshold_tensor), tensor.float32)

        # Convert outputs to NumPy for logging/downstream use if needed
        motor_outputs_np = tensor.to_numpy(motor_outputs_tensor)
        trigger_signals_np = tensor.to_numpy(trigger_signals_tensor)

        self.processing_time['data_processing'] = time.time() - start_time
        logger.info(f"Data processing completed in {self.processing_time['data_processing']:.2f}s")
        
        # Return outputs based on return_triggers
        # Return NumPy arrays as expected by the rest of the script
        if return_triggers:
            return motor_outputs_np, trigger_signals_np
        else:
            return motor_outputs_np
    
    def save_model(self, directory: str):
        """
        Save all model components.
        
        Args:
            directory: Directory to save models
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save RBM
        if self.rbm is not None:
            rbm_path = os.path.join(directory, "rbm.npy")
            self.rbm.save(rbm_path)
            logger.info(f"RBM saved to {rbm_path}")
        
        # Save liquid network state dictionary using ops.save
        if self.liquid_network is not None and isinstance(self.liquid_network, Module):
            liquid_network_state_path = os.path.join(directory, "liquid_network_state.pkl") # Use a suitable extension like .pkl or .npz
            try:
                # Assuming ops.save handles serialization appropriately
                ops.save(self.liquid_network.state_dict(), liquid_network_state_path)
                logger.info(f"Liquid network state saved to {liquid_network_state_path}")
                # TODO: Consider saving model configuration (units, layers, etc.) separately if needed for re-instantiation
            except Exception as e:
                logger.error(f"Failed to save liquid network state: {e}")
        elif self.liquid_network is not None:
             logger.warning("Liquid network is not an Ember Module, cannot save state_dict.")

        # Save processing times
        processing_times_path = os.path.join(directory, "processing_times.csv")
        pd.DataFrame([self.processing_time]).to_csv(processing_times_path, index=False)
        logger.info(f"Processing times saved to {processing_times_path}")
    
    def load_model(self, directory: str, network_type: str = 'standard'):
        """
        Load all model components.
        
        Args:
            directory: Directory to load models from
            network_type: Type of liquid network
        """
        # Load RBM
        rbm_path = os.path.join(directory, "rbm.npy")
        if os.path.exists(rbm_path):
            self.rbm = OptimizedRBM.load(rbm_path, use_gpu=self.use_gpu)
            self.rbm_feature_dim = self.rbm.n_hidden
            logger.info(f"RBM loaded from {rbm_path}")
        
        # Load liquid network state dictionary using ops.load
        liquid_network_state_path = os.path.join(directory, "liquid_network_state.pkl")
        if os.path.exists(liquid_network_state_path):
            try:
                # 1. Instantiate the model structure
                # Requires knowing the input_dim and network_type used during saving
                # This information might need to be saved/loaded separately or inferred
                if self.rbm_feature_dim is None:
                     # Attempt to infer from loaded RBM if available
                     if self.rbm:
                         self.rbm_feature_dim = self.rbm.n_hidden
                     else:
                         raise ValueError("Cannot load liquid network state: RBM feature dimension unknown.")

                logger.info(f"Re-initializing liquid network (type: {network_type}) for loading state...")
                # Re-initialize the network structure before loading state
                self.initialize_liquid_network(self.rbm_feature_dim, network_type)
                if not isinstance(self.liquid_network, Module):
                     raise TypeError("Failed to initialize liquid_network as an ember_ml.nn.Module.")

                # 2. Load the state dictionary
                state_dict = ops.load(liquid_network_state_path)
                # Assuming ops.load returns the dictionary

                # 3. Load state into the instantiated model
                self.liquid_network.load_state_dict(state_dict)
                logger.info(f"Liquid network state loaded from {liquid_network_state_path}")

            except FileNotFoundError:
                 logger.warning(f"Liquid network state file not found at {liquid_network_state_path}. Skipping load.")
            except Exception as e:
                logger.error(f"Failed to load liquid network state: {e}")

        # Load processing times
        processing_times_path = os.path.join(directory, "processing_times.csv")
        if os.path.exists(processing_times_path):
            self.processing_time = pd.read_csv(processing_times_path).iloc[0].to_dict()
            logger.info(f"Processing times loaded from {processing_times_path}")
    
    def summary(self) -> str:
        """
        Get a summary of the pipeline.
        
        Returns:
            Summary string
        """
        summary = [
            "Integrated Pipeline Summary",
            "==========================",
            f"Feature dimension: {self.feature_dim}",
            f"RBM hidden units: {self.rbm_hidden_units}",
            f"CfC units: {self.cfc_units}",
            f"LSTM units: {self.lstm_units}",
            f"Stride perspectives: {self.stride_perspectives}",
            f"Sparsity level: {self.sparsity_level}",
            f"Threshold: {self.threshold}",
            f"GPU acceleration: {self.use_gpu}",
            "",
            "Processing Times:",
        ]
        
        for key, value in self.processing_time.items():
            summary.append(f"  {key}: {value:.2f}s")
        
        if self.rbm is not None:
            summary.append("")
            summary.append("RBM Summary:")
            summary.append(self.rbm.summary())
        
        if self.liquid_network is not None:
            summary.append("")
            summary.append("Liquid Network Summary:")
            summary.append(str(self.liquid_network.summary()))
        
        return "\n".join(summary)


def main():
    """Main function for the pipeline demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated Pipeline Demo")
    parser.add_argument("--project-id", type=str, help="GCP project ID")
    parser.add_argument("--table-id", type=str, help="BigQuery table ID (dataset.table)")
    parser.add_argument("--target-column", type=str, help="Target column name")
    parser.add_argument("--limit", type=int, default=10000, help="Row limit for testing")
    parser.add_argument("--rbm-hidden-units", type=int, default=64, help="Number of hidden units in RBM")
    parser.add_argument("--cfc-units", type=int, default=128, help="Number of units in CfC circuit")
    parser.add_argument("--lstm-units", type=int, default=32, help="Number of units in LSTM gating")
    parser.add_argument("--network-type", type=str, default="standard", 
                        choices=["standard", "lstm_gated", "multi_stride"],
                        help="Type of liquid network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--load-models", action="store_true", help="Load models from save-dir")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = IntegratedPipeline(
        project_id=args.project_id,
        rbm_hidden_units=args.rbm_hidden_units,
        cfc_units=args.cfc_units,
        lstm_units=args.lstm_units,
        use_gpu=not args.no_gpu,
        verbose=args.verbose
    )
    
    # Load models if requested
    if args.load_models:
        pipeline.load_model(args.save_dir, args.network_type)
    
    # Extract features if table_id is provided
    if args.table_id:
        train_features, val_features, test_features = pipeline.extract_features(
            table_id=args.table_id,
            target_column=args.target_column,
            limit=args.limit
        )
        
        # Apply temporal processing
        train_temporal = pipeline.apply_temporal_processing(train_features)
        val_temporal = pipeline.apply_temporal_processing(val_features)
        
        # Train RBM
        pipeline.train_rbm(train_features, epochs=args.epochs)
        
        # Extract RBM features
        train_rbm_features = pipeline.extract_rbm_features(train_features)
        val_rbm_features = pipeline.extract_rbm_features(val_features)
        
        # Create dummy targets for demonstration using tensor.random_uniform
        # In a real application, you would use actual targets
        from ember_ml.nn import tensor # Ensure tensor is imported
        train_targets = tensor.random_uniform((len(train_rbm_features), 1))
        val_targets = tensor.random_uniform((len(val_rbm_features), 1))
        # Convert back to numpy temporarily because model.fit expects it (TF dependency)
        train_targets = tensor.to_numpy(train_targets)
        val_targets = tensor.to_numpy(val_targets)

        # Train liquid network
        pipeline.train_liquid_network(
            features=train_rbm_features,
            targets=train_targets,
            validation_data=(val_rbm_features, val_targets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            network_type=args.network_type
        )
        
        # Process test data
        test_rbm_features = pipeline.extract_rbm_features(test_features)
        motor_outputs, trigger_signals = pipeline.process_data(test_rbm_features)
        
        # Print results
        logger.info(f"Processed {len(test_rbm_features)} test samples")
        logger.info(f"Motor neuron output range: {motor_outputs.min():.4f} to {motor_outputs.max():.4f}")
        logger.info(f"Trigger rate: {trigger_signals.mean():.4f}")
        
        # Save models
        pipeline.save_model(args.save_dir)
    
    # Print pipeline summary
    print(pipeline.summary())


if __name__ == "__main__":
    main()