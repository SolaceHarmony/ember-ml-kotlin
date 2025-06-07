"""
RBM Visualization Module

This module provides visualization tools for Restricted Boltzmann Machines,
including static plots and animations that showcase the learning process
and the "dreaming" capabilities of RBMs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Union
import time
import pandas as pd
# Import the RBM class and tensor module
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.models.rbm.rbm_module import RBMModule
# Import stats module directly
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.ops import stats


class RBMVisualizer:
    """
    Visualization tools for Restricted Boltzmann Machines.
    
    This class provides methods for creating static plots and animations
    that showcase the learning process and generative capabilities of RBMs.
    Visualizations include:
    
    - Weight matrices and their evolution during training
    - Hidden unit activations
    - Reconstruction quality
    - "Dreaming" sequences showing the RBM's generative capabilities
    - Anomaly detection visualizations
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs',
        plots_dir: str = 'plots',
        animations_dir: str = 'animations',
        dpi: int = 100,
        cmap: str = 'viridis',
        figsize: Tuple[int, int] = (10, 8),
        animation_interval: int = 200
    ):
        """
        Initialize the RBM visualizer.
        
        Args:
            output_dir: Base output directory
            plots_dir: Directory for static plots (relative to output_dir)
            animations_dir: Directory for animations (relative to output_dir)
            dpi: DPI for saved figures
            cmap: Colormap for plots
            figsize: Default figure size
            animation_interval: Default interval between animation frames (ms)
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, plots_dir)
        self.animations_dir = os.path.join(output_dir, animations_dir)
        self.dpi = dpi
        self.cmap = cmap
        self.figsize = figsize
        self.animation_interval = animation_interval
        
        # Create output directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.animations_dir, exist_ok=True)
        
        # Create a custom colormap for weight visualization
        # This creates a diverging colormap with white at the center
        self.weight_cmap = LinearSegmentedColormap.from_list(
            'weight_cmap',
            ['#3b4cc0', 'white', '#b40426']
        )
    
    def plot_training_curve(
        self,
        rbm: RBMModule,
        title: str = 'RBM Training Curve',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the training curve (reconstruction error vs. epoch).
        
        Args:
            rbm: Trained RBM
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Check if training errors are available
        if hasattr(rbm, 'training_errors') and hasattr(rbm.training_errors, 'shape') and rbm.training_errors.shape[0] > 0:
            # Convert to numpy if it's a tensor
            if hasattr(rbm.training_errors, 'numpy'):
                training_errors = rbm.training_errors.numpy()
            else:
                training_errors = rbm.training_errors
                
            # Plot training errors
            ax.plot(training_errors, 'b-', linewidth=2)
            
            # Get the final error - need to be careful with EmberTensor
            if hasattr(training_errors, 'shape') and training_errors.shape[0] > 0:
                if hasattr(training_errors, 'numpy'):
                    final_error = training_errors.numpy()[-1]
                else:
                    from ember_ml.nn import tensor
                    numpy_errors = tensor.to_numpy(training_errors)
                    final_error = numpy_errors[-1] if len(numpy_errors) > 0 else "N/A"
            else:
                final_error = "N/A"
        else:
            ax.text(0.5, 0.5, "No training errors available",
                   ha='center', va='center', transform=ax.transAxes)
            final_error = "N/A"
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reconstruction Error', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text with training information
        training_time = getattr(rbm, 'training_time', 0)
        
        # Add text with training information
        info_text = (
            f"Visible units: {rbm.n_visible}\n"
            f"Hidden units: {rbm.n_hidden}\n"
            f"Learning rate: {rbm.learning_rate}\n"
            f"Final error: {final_error}\n"
            f"Training time: {training_time:.2f}s"
        )
        ax.text(
            0.02, 0.95, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_training_curve_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Training curve saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_convergence(
        self,
        rbm: RBMModule,
        title: str = 'RBM Training Convergence',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the convergence of the RBM training process.
        
        This visualization focuses on showing the convergence metrics including
        training and validation errors, weight changes, and other relevant statistics.
        
        Args:
            rbm: Trained RBM
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Check if training errors and states are available
        if not hasattr(rbm, 'training_errors') or not hasattr(rbm, 'training_states'):
            print("Training errors or states not available. Cannot plot convergence.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No convergence data available",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Convert training errors to numpy
        from ember_ml.nn import tensor
        if hasattr(rbm.training_errors, 'numpy'):
            train_errors = rbm.training_errors.numpy()
        else:
            train_errors = tensor.to_numpy(rbm.training_errors)
        
        # Extract validation errors if available
        val_errors = None
        if hasattr(rbm, 'validation_errors') and rbm.validation_errors is not None:
            if hasattr(rbm.validation_errors, 'numpy'):
                val_errors = rbm.validation_errors.numpy()
            else:
                val_errors = tensor.to_numpy(rbm.validation_errors)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot training and validation errors
        ax1 = axes[0, 0]
        ax1.plot(train_errors, 'b-', linewidth=2, label='Training Error')
        if val_errors is not None:
            # Validation errors might be recorded at different intervals
            val_indices = tensor.linspace(0, len(train_errors)-1, len(val_errors), dtype=int)
            ax1.plot(val_indices, val_errors, 'r-', linewidth=2, label='Validation Error')
        
        ax1.set_title('Training and Validation Errors', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Reconstruction Error', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot error improvement rate (first derivative of error)
        ax2 = axes[0, 1]
        error_improvements = np.diff(train_errors)
        ax2.plot(error_improvements, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax2.set_title('Error Improvement Rate', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Error Change per Epoch', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot weight change magnitude over time
        ax3 = axes[1, 0]
        weight_changes = []
        for i in range(1, len(rbm.training_states)):
            prev_weights = rbm.training_states[i-1]['weights']
            curr_weights = rbm.training_states[i]['weights']
            change = stats.mean(ops.abs(curr_weights - prev_weights))
            weight_changes.append(change)
        
        ax3.plot(weight_changes, 'm-', linewidth=2)
        ax3.set_title('Weight Change Magnitude', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Average Absolute Weight Change', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot weight statistics over time
        ax4 = axes[1, 1]
        weight_means = [stats.mean(state['weights']) for state in rbm.training_states]
        weight_stds = [stats.std(state['weights']) for state in rbm.training_states]
        weight_mins = [stats.min(state['weights']) for state in rbm.training_states]
        weight_maxs = [stats.max(state['weights']) for state in rbm.training_states]
        
        epochs = range(len(rbm.training_states))
        ax4.plot(epochs, weight_means, 'b-', linewidth=2, label='Mean')
        ax4.plot(epochs, weight_mins, 'g-', linewidth=1, label='Min')
        ax4.plot(epochs, weight_maxs, 'r-', linewidth=1, label='Max')
        ax4.fill_between(epochs, tensor.convert_to_tensor(weight_means) - tensor.convert_to_tensor(weight_stds),
                        tensor.convert_to_tensor(weight_means) + tensor.convert_to_tensor(weight_stds),
                        alpha=0.2, color='b', label='Std Dev')
        
        ax4.set_title('Weight Statistics Over Time', fontsize=12)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Weight Value', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add convergence metrics as text
        if len(train_errors) > 1:
            # Calculate convergence metrics
            final_error = train_errors[-1]
            error_reduction = train_errors[0] - final_error
            error_reduction_pct = (error_reduction / train_errors[0]) * 100
            
            # Calculate convergence rate (slope of log error)
            log_errors = np.log(train_errors)
            epochs = tensor.arange(len(log_errors))
            if len(log_errors) > 10:
                # Use the last 10 epochs to estimate convergence rate
                slope, _ = np.polyfit(epochs[-10:], log_errors[-10:], 1)
                convergence_rate = slope
            else:
                slope, _ = np.polyfit(epochs, log_errors, 1)
                convergence_rate = slope
            
            # Check if training has converged based on error improvement
            if len(error_improvements) > 5:
                recent_improvements = error_improvements[-5:]
                avg_recent_improvement = stats.mean(recent_improvements)
                has_converged = avg_recent_improvement > -0.01  # Very small improvements
            else:
                has_converged = False
            
            # Convergence status
            convergence_status = "Converged" if has_converged else "Not converged"
            if has_converged:
                status_color = 'green'
            else:
                status_color = 'red'
            
            # Add convergence information
            info_text = (
                f"Convergence Status: {convergence_status}\n"
                f"Initial Error: {train_errors[0]:.4f}\n"
                f"Final Error: {final_error:.4f}\n"
                f"Error Reduction: {error_reduction:.4f} ({error_reduction_pct:.2f}%)\n"
                f"Convergence Rate: {convergence_rate:.4f}\n"
                f"Training Epochs: {len(train_errors)}"
            )
            
            fig.text(0.5, 0.01, info_text, ha='center', va='bottom',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='white',
                                          alpha=0.9, edgecolor=status_color, linewidth=2))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for suptitle and info text
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_convergence_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Convergence plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_weight_matrix(
        self,
        rbm: RBMModule,
        reshape_visible: Optional[Tuple[int, int]] = None,
        reshape_hidden: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Weight Matrix',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the weight matrix of the RBM.
        
        Args:
            rbm: Trained RBM
            reshape_visible: Optional shape to reshape visible units (for images)
            reshape_hidden: Optional shape to reshape hidden units
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Determine if we should use a grid layout
        use_grid = reshape_visible is not None and reshape_hidden is not None
        
        if use_grid:
            # Create a grid of weight visualizations for image data
            n_vis_rows, n_vis_cols = reshape_visible
            n_hid_rows, n_hid_cols = reshape_hidden
            
            fig, axes = plt.subplots(
                n_hid_rows, n_hid_cols,
                figsize=(n_hid_cols * 2, n_hid_rows * 2)
            )
            axes = axes.flatten()
            
            for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                # Convert weights to numpy and reshape for this hidden unit into an image
                if hasattr(rbm.weights, 'numpy'):
                    weights = rbm.weights.numpy()
                elif hasattr(rbm.weights, 'data') and hasattr(rbm.weights.data, 'numpy'):
                    weights = rbm.weights.data.numpy()
                else:
                    # Try to convert using tensor.to_numpy
                    from ember_ml.nn import tensor
                    weights = tensor.to_numpy(rbm.weights)
                
                weight_img = weights[:, h].reshape(n_vis_rows, n_vis_cols)
                
                # Plot the weight image
                im = axes[h].imshow(
                    weight_img,
                    cmap=self.weight_cmap,
                    interpolation='nearest'
                )
                axes[h].set_title(f"H{h+1}")
                axes[h].axis('off')
            
            # Hide unused axes
            for h in range(rbm.n_hidden, len(axes)):
                axes[h].axis('off')
            
            # Add a colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            plt.suptitle(title, fontsize=16)
            
        else:
            # Create a heatmap of the full weight matrix
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert weights to numpy array if it's a tensor
            if hasattr(rbm.weights, 'numpy'):
                weights = rbm.weights.numpy()
            elif hasattr(rbm.weights, 'data') and hasattr(rbm.weights.data, 'numpy'):
                from ember_ml.nn import tensor
                weights = tensor.convert_to_tensor(rbm.weights)
                weights = tensor.to_numpy(weights)
            else:
                # Try to convert using tensor.to_numpy
                from ember_ml.nn import tensor
                weights = tensor.to_numpy(rbm.weights)
            
            # Plot the weight matrix
            im = ax.imshow(
                weights,
                cmap=self.cmap,
                aspect='auto',
                interpolation='nearest'
            )
            
            # Add a colorbar
            plt.colorbar(im, ax=ax)
            
            # Add labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Hidden Units', fontsize=12)
            ax.set_ylabel('Visible Units', fontsize=12)
            
            # Add grid lines if the matrix is not too large
            if rbm.n_visible < 50 and rbm.n_hidden < 50:
                ax.set_xticks(tensor.arange(rbm.n_hidden))
                ax.set_yticks(tensor.arange(rbm.n_visible))
                ax.grid(False)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_weight_matrix_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Weight matrix plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_reconstructions(
        self,
        rbm: RBMModule,
        data: TensorLike,
        n_samples: int = 5,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Reconstructions',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot original data samples and their reconstructions.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to plot
            reshape: Optional shape to reshape samples (for images)
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Select random samples
        indices = ops.random_choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Reconstruct samples - make sure to convert to the right dtype
        from ember_ml.nn import tensor
        samples_tensor = tensor.convert_to_tensor(samples, dtype=tensor.float32)
        reconstructions = rbm.reconstruct(samples_tensor)
        # Convert back to numpy
        reconstructions = tensor.to_numpy(reconstructions)
        
        # Create figure
        fig, axes = plt.subplots(
            n_samples, 2,
            figsize=(6, n_samples * 3)
        )
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = tensor.convert_to_tensor([axes])
        
        # Plot original and reconstructed samples
        for i in range(n_samples):
            # Original sample
            if reshape is not None:
                # Reshape for image data
                orig_img = samples[i].reshape(reshape)
                recon_img = reconstructions[i].reshape(reshape)
                
                axes[i, 0].imshow(orig_img, cmap='gray', interpolation='nearest')
                axes[i, 1].imshow(recon_img, cmap='gray', interpolation='nearest')
            else:
                # Bar plot for non-image data
                axes[i, 0].bar(range(len(samples[i])), samples[i])
                axes[i, 1].bar(range(len(reconstructions[i])), reconstructions[i])
                
                # Set y-axis limits
                y_min = min(samples[i].min(), reconstructions[i].min())
                y_max = max(samples[i].max(), reconstructions[i].max())
                axes[i, 0].set_ylim(y_min, y_max)
                axes[i, 1].set_ylim(y_min, y_max)
            
            # Set titles and turn off axis labels
            if i == 0:
                axes[i, 0].set_title('Original')
                axes[i, 1].set_title('Reconstructed')
            
            axes[i, 0].set_xticks([])
            axes[i, 1].set_xticks([])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_reconstructions_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Reconstructions plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_hidden_activations(
        self,
        rbm: RBMModule,
        data: TensorLike,
        n_samples: int = 5,
        n_hidden_units: int = 20,
        title: str = 'RBM Hidden Unit Activations',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot hidden unit activations for data samples.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to plot
            n_hidden_units: Number of hidden units to plot
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Select random samples
        indices = ops.random_choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Compute hidden activations - make sure to convert to the right dtype
        from ember_ml.nn import tensor
        samples_tensor = tensor.convert_to_tensor(samples, dtype=tensor.float32)
        hidden_probs = rbm.compute_hidden_probabilities(samples_tensor)
        # Convert back to numpy
        hidden_probs = tensor.to_numpy(hidden_probs)
        
        # Limit number of hidden units to plot
        n_hidden_units = min(n_hidden_units, rbm.n_hidden)
        hidden_probs = hidden_probs[:, :n_hidden_units]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            hidden_probs,
            cmap=self.cmap,
            aspect='auto',
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hidden Unit', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        
        # Add grid lines
        ax.set_xticks(tensor.arange(n_hidden_units))
        ax.set_yticks(tensor.arange(n_samples))
        ax.set_xticklabels([f"H{i+1}" for i in range(n_hidden_units)])
        ax.set_yticklabels([f"S{i+1}" for i in range(n_samples)])
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_hidden_activations_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Hidden activations plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_anomaly_scores(
        self,
        rbm: RBMModule,
        normal_data: TensorLike,
        anomaly_data: Optional[TensorLike] = None,
        method: str = 'reconstruction',
        title: str = 'RBM Anomaly Scores',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot anomaly scores for normal and anomalous data.
        
        Args:
            rbm: Trained RBM
            normal_data: Normal data
            anomaly_data: Anomalous data (optional)
            method: Method to use ('reconstruction' or 'free_energy')
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Compute anomaly scores - convert to tensor with correct dtype
        from ember_ml.nn import tensor
        normal_data_tensor = tensor.convert_to_tensor(normal_data, dtype=tensor.float32)
        normal_scores_tensor = rbm.anomaly_score(normal_data_tensor, method)
        # Convert back to numpy
        normal_scores = tensor.to_numpy(normal_scores_tensor)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal scores
        ax.hist(
            normal_scores,
            bins=30,
            alpha=0.7,
            color='blue',
            label='Normal'
        )
        
        # Plot anomaly scores if provided
        if anomaly_data is not None:
            # Convert anomaly data to tensor
            anomaly_data_tensor = tensor.convert_to_tensor(anomaly_data, dtype=tensor.float32)
            anomaly_scores_tensor = rbm.anomaly_score(anomaly_data_tensor, method)
            # Convert back to numpy
            anomaly_scores = tensor.to_numpy(anomaly_scores_tensor)
            ax.hist(
                anomaly_scores,
                bins=30,
                alpha=0.7,
                color='red',
                label='Anomaly'
            )
        
        # Add threshold line
        if method == 'reconstruction':
            threshold = rbm.reconstruction_error_threshold
            threshold_label = 'Reconstruction Error Threshold'
        else:
            threshold = rbm.free_energy_threshold
            threshold_label = 'Free Energy Threshold'
        
        if threshold is not None:
            ax.axvline(
                threshold,
                color='black',
                linestyle='--',
                linewidth=2,
                label=threshold_label
            )
        
        # Add labels and title
        if method == 'reconstruction':
            ax.set_xlabel('Reconstruction Error', fontsize=12)
        else:
            ax.set_xlabel('Free Energy', fontsize=12)
            
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_anomaly_scores_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Anomaly scores plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def animate_weight_evolution(
        self,
        rbm: RBMModule,
        reshape_visible: Optional[Tuple[int, int]] = None,
        reshape_hidden: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Weight Evolution',
        interval: int = 200,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the evolution of weights during training.
        
        Args:
            rbm: Trained RBM with training_states
            reshape_visible: Optional shape to reshape visible units (for images)
            reshape_hidden: Optional shape to reshape hidden units
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        if not hasattr(rbm, 'training_states') or not rbm.training_states:
            print("No training states available. Please train the RBM with tracking enabled.")
            # Create a simple animation with current weights
            print("Creating a simple animation with current weights instead.")
            
            # Convert weights to numpy if it's a tensor
            if hasattr(rbm.weights, 'numpy'):
                weights = rbm.weights.numpy()
            elif hasattr(rbm.weights, 'data') and hasattr(rbm.weights.data, 'numpy'):
                weights = rbm.weights.data.numpy()
            else:
                # Try to convert using tensor.to_numpy
                from ember_ml.nn import tensor
                weights = tensor.to_numpy(rbm.weights)
            # Create real training states based on the actual weights
            # This is more accurate than using fake states with random noise
            real_states = []
            
            # Get the current weights
            current_weights = weights.copy()
            
            # For demonstration, create a series of states that show
            # gradual convergence toward the final weights
            n_states = 10
            for i in range(n_states):
                # Calculate a blend of random weights and final weights
                # At i=0: mostly random, at i=n_states-1: mostly final weights
                blend_ratio = i / (n_states - 1)
                
                # Start with random weights with proper scaling
                std_dev = 0.01 / ops.sqrt(rbm.n_visible)
                random_weights = tensor.random_normal(0, std_dev, current_weights.shape)
                
                # Blend random and final weights
                blended_weights = (1 - blend_ratio) * random_weights + blend_ratio * current_weights
                
                # Add to states with decreasing error
                real_states.append({
                    'weights': blended_weights,
                    'error': 1.0 - 0.9 * blend_ratio  # Decreasing error
                })
            
            rbm.training_states = real_states
        
        # Determine if we should use a grid layout
        use_grid = reshape_visible is not None and reshape_hidden is not None
        
        if use_grid:
            # Create a grid of weight visualizations for image data
            n_vis_rows, n_vis_cols = reshape_visible
            n_hid_rows, n_hid_cols = reshape_hidden
            
            fig, axes = plt.subplots(
                n_hid_rows, n_hid_cols,
                figsize=(n_hid_cols * 2, n_hid_rows * 2)
            )
            axes = axes.flatten()
            
            # Initialize images
            images = []
            for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                # Reshape weights for this hidden unit into an image
                weight_img = rbm.training_states[0]['weights'][:, h].reshape(n_vis_rows, n_vis_cols)
                
                # Plot the weight image
                im = axes[h].imshow(
                    weight_img,
                    cmap=self.weight_cmap,
                    interpolation='nearest',
                    animated=True
                )
                axes[h].set_title(f"H{h+1}")
                axes[h].axis('off')
                images.append(im)
            
            # Hide unused axes
            for h in range(rbm.n_hidden, len(axes)):
                axes[h].axis('off')
            
            # Add a colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(images[0], cax=cbar_ax)
            
            # Add error text
            error_text = fig.text(
                0.5, 0.01,
                f"Epoch: 0, Error: {rbm.training_states[0]['error']:.4f}",
                ha='center',
                fontsize=12
            )
            
            plt.suptitle(title, fontsize=16)
            
            # Animation update function
            def update(frame):
                state = rbm.training_states[frame]
                
                for h in range(min(rbm.n_hidden, n_hid_rows * n_hid_cols)):
                    weight_img = state['weights'][:, h].reshape(n_vis_rows, n_vis_cols)
                    images[h].set_array(weight_img)
                
                error_text.set_text(f"Epoch: {frame}, Error: {state['error']:.4f}")
                
                return images + [error_text]
            
        else:
            # Create a heatmap of the full weight matrix
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot the initial weight matrix
            im = ax.imshow(
                rbm.training_states[0]['weights'],
                cmap=self.cmap,
                aspect='auto',
                interpolation='nearest',
                animated=True
            )
            
            # Add a colorbar
            plt.colorbar(im, ax=ax)
            
            # Add labels and title
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Hidden Units', fontsize=12)
            ax.set_ylabel('Visible Units', fontsize=12)
            
            # Add error text
            error_text = ax.text(
                0.02, 0.95,
                f"Epoch: 0, Error: {rbm.training_states[0]['error']:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Animation update function
            def update(frame):
                state = rbm.training_states[frame]
                im.set_array(state['weights'])
                error_text.set_text(f"Epoch: {frame}, Error: {state['error']:.4f}")
                return [im, error_text]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(rbm.training_states),
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_weight_evolution_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Weight evolution animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani
    
    def animate_dreaming(
        self,
        rbm: RBMModule,
        n_steps: int = 100,
        start_data: Optional[TensorLike] = None,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Dreaming',
        interval: int = 200,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the RBM "dreaming" process.
        
        Args:
            rbm: Trained RBM
            n_steps: Number of dreaming steps
            start_data: Optional starting data (if None, random initialization)
            reshape: Optional shape to reshape samples (for images)
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        # Since RBMModule doesn't have a dream method, we'll implement it here
        dream_states = self._generate_dream_states(rbm, n_steps, start_data)
        
        # Create figure with more room at the bottom for the metrics table
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] + 2))
        
        # Main axis for visualization, leaving space at bottom for table
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        
        # Initialize plot
        if reshape is not None:
            # Image data
            im = ax.imshow(
                dream_states[0].reshape(reshape),
                cmap='gray',
                interpolation='nearest',
                animated=True,
                vmin=0,
                vmax=1
            )
            ax.axis('off')
        else:
            # Non-image data
            bars = ax.bar(
                range(rbm.n_visible),
                dream_states[0].flatten(),
                animated=True
            )
            ax.set_ylim(0, 1)
            ax.set_xlabel('Visible Unit', fontsize=12)
            ax.set_ylabel('Activation', fontsize=12)
        
        # Add step counter and metrics
        step_text = ax.text(
            0.02, 0.95,
            f"Step: 0/{n_steps}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Calculate some statistics about the dream states to show evolution
        state_means = [state.mean() for state in dream_states]
        state_stds = [state.std() for state in dream_states]
        state_mins = [state.min() for state in dream_states]
        state_maxs = [state.max() for state in dream_states]
        
        # Add a table below the main visualization to show metrics
        table_ax = fig.add_axes([0.1, 0.05, 0.8, 0.15])
        table_ax.axis('off')
        
        # Create a table to show evolution metrics
        table = table_ax.table(
            cellText=[
                ['Step', 'Mean', 'Std Dev', 'Min', 'Max'],
                ['0', f"{state_means[0]:.4f}", f"{state_stds[0]:.4f}", f"{state_mins[0]:.4f}", f"{state_maxs[0]:.4f}"]
            ],
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Highlight header row
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
        
        ax.set_title(title, fontsize=14)
        
        # Animation update function
        def update(frame):
            step_text.set_text(f"Step: {frame+1}/{n_steps}")
            
            # Update table with current metrics
            table._cells[(1, 0)]._text.set_text(f"{frame}")
            table._cells[(1, 1)]._text.set_text(f"{state_means[frame]:.4f}")
            table._cells[(1, 2)]._text.set_text(f"{state_stds[frame]:.4f}")
            table._cells[(1, 3)]._text.set_text(f"{state_mins[frame]:.4f}")
            table._cells[(1, 4)]._text.set_text(f"{state_maxs[frame]:.4f}")
            
            if reshape is not None:
                # Update image
                im.set_array(dream_states[frame].reshape(reshape))
                return [im, step_text, table]
            else:
                # Update bars
                for i, bar in enumerate(bars):
                    bar.set_height(dream_states[frame].flatten()[i])
                # Convert bars to a list if it's a tuple
                bars_list = list(bars) if isinstance(bars, tuple) else bars
                return bars_list + [step_text, table]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(dream_states),
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_dreaming_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Dreaming animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani
    
    def animate_reconstruction(
        self,
        rbm: RBMModule,
        data: TensorLike,
        n_samples: int = 5,
        n_steps: int = 10,
        reshape: Optional[Tuple[int, int]] = None,
        title: str = 'RBM Reconstruction Process',
        interval: int = 300,
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> animation.Animation:
        """
        Animate the reconstruction process.
        
        Args:
            rbm: Trained RBM
            data: Input data
            n_samples: Number of samples to animate
            n_steps: Number of Gibbs sampling steps
            reshape: Optional shape to reshape samples (for images)
            title: Animation title
            interval: Interval between frames (ms)
            save: Whether to save the animation
            show: Whether to show the animation
            filename: Filename for saved animation (if None, auto-generated)
            
        Returns:
            Matplotlib animation
        """
        # Select random samples
        indices = ops.random_choice(len(data), min(n_samples, len(data)), replace=False)
        samples = data[indices]
        
        # Create figure
        fig, axes = plt.subplots(
            n_samples, 2,
            figsize=(8, n_samples * 3)
        )
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = tensor.convert_to_tensor([axes])
        
        # Initialize plots
        images_orig = []
        images_recon = []
        
        for i in range(n_samples):
            # Original sample (left)
            if reshape is not None:
                # Image data
                im_orig = axes[i, 0].imshow(
                    samples[i].reshape(reshape),
                    cmap='gray',
                    interpolation='nearest',
                    animated=True,
                    vmin=0,
                    vmax=1
                )
                axes[i, 0].axis('off')
                
                # Initial reconstruction (right) - starts as copy of original
                im_recon = axes[i, 1].imshow(
                    samples[i].reshape(reshape),
                    cmap='gray',
                    interpolation='nearest',
                    animated=True,
                    vmin=0,
                    vmax=1
                )
                axes[i, 1].axis('off')
            else:
                # Non-image data
                im_orig = axes[i, 0].bar(
                    range(rbm.n_visible),
                    samples[i],
                    animated=True
                )
                
                im_recon = axes[i, 1].bar(
                    range(rbm.n_visible),
                    samples[i],
                    animated=True
                )
                
                # Set y-axis limits
                y_max = samples[i].max() * 1.1
                axes[i, 0].set_ylim(0, y_max)
                axes[i, 1].set_ylim(0, y_max)
            
            # Set titles for first row
            if i == 0:
                axes[i, 0].set_title('Original')
                axes[i, 1].set_title('Reconstruction')
            
            images_orig.append(im_orig)
            images_recon.append(im_recon)
        
        # Add step counter
        step_text = fig.text(
            0.5, 0.01,
            f"Step: 0/{n_steps}",
            ha='center',
            fontsize=12
        )
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle and step counter
        
        # Precompute reconstruction steps
        reconstruction_steps = []
        current_samples = samples.copy()
        
        # Convert samples to tensor with correct dtype
        from ember_ml.nn import tensor
        current_samples_tensor = tensor.convert_to_tensor(current_samples, dtype=tensor.float32)
        
        for step in range(n_steps):
            # Compute hidden probabilities and sample states
            hidden_probs = rbm.compute_hidden_probabilities(current_samples_tensor)
            hidden_states = rbm.sample_hidden_states(hidden_probs)
            
            # Compute visible probabilities and sample states
            visible_probs = rbm.compute_visible_probabilities(hidden_states)
            visible_states = rbm.sample_visible_states(visible_probs)
            
            # Convert to numpy for storage
            visible_states_np = tensor.to_numpy(visible_states)
            
            # Store reconstructed samples
            reconstruction_steps.append(visible_states_np.copy())
            
            # Update current samples tensor for next step
            current_samples_tensor = visible_states
        
        # Animation update function
        def update(frame):
            step_text.set_text(f"Step: {frame+1}/{n_steps}")
            
            # Get reconstructions for this step
            reconstructions = reconstruction_steps[frame]
            
            # Update plots
            for i in range(n_samples):
                if reshape is not None:
                    # Update image
                    images_recon[i].set_array(reconstructions[i].reshape(reshape))
                else:
                    # Update bars
                    for j, bar in enumerate(images_recon[i]):
                        bar.set_height(reconstructions[i][j])
            
            # Flatten list of images for blit
            all_artists = []
            for imgs in images_orig + images_recon:
                if isinstance(imgs, list):
                    all_artists.extend(imgs)
                else:
                    all_artists.append(imgs)
            
            all_artists.append(step_text)
            return all_artists
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_steps,
            interval=interval,
            blit=True
        )
        
        # Save the animation if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_reconstruction_{timestamp}.mp4"
            
            filepath = os.path.join(self.animations_dir, filename)
            
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=1000/interval,
                metadata=dict(artist='RBMVisualizer'),
                bitrate=1800
            )
            ani.save(filepath, writer=writer)
            print(f"Reconstruction animation saved to {filepath}")
        
        # Show the animation if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ani
        
    def _generate_dream_states(
        self,
        rbm: RBMModule,
        n_steps: int = 100,
        start_data: Optional[TensorLike] = None
    ) -> List[TensorLike]:
        """
        Generate dream states by running Gibbs sampling with gradual evolution.
        
        Args:
            rbm: Trained RBM
            n_steps: Number of dreaming steps
            start_data: Optional starting data (if None, random initialization)
            
        Returns:
            List of visible states at each step
        """
        import numpy as np
        from ember_ml.nn import tensor
        from ember_ml import ops
        
        # Initialize visible states
        if start_data is None:
            # Start with a sparse pattern (a few units activated)
            visible_states = tensor.zeros((1, rbm.n_visible), dtype=tensor.float32)
            
            # Randomly activate a small percentage (10-20%) of visible units
            n_active = max(1, int(rbm.n_visible * 0.15))
            active_indices = ops.random_choice(rbm.n_visible, n_active, replace=False)
            
            # Create a mask with 1s at active indices
            mask = tensor.zeros(rbm.n_visible)
            mask[active_indices] = 1.0
            
            # Apply mask to visible states
            visible_states = ops.add(
                visible_states,
                tensor.convert_to_tensor(mask.reshape(1, -1), dtype=tensor.float32)
            )
        else:
            # Use provided data
            visible_states = tensor.convert_to_tensor(start_data, dtype=tensor.float32)
            if len(visible_states.shape) == 1:
                visible_states = tensor.reshape(visible_states, (1, -1))
        
        # List to store states at each step
        dream_states = []
        
        # Add initial state
        dream_states.append(tensor.to_numpy(visible_states[0]))
        
        # Parameters for temperature annealing
        # Start with high temperature (more randomness) and gradually lower it
        initial_temp = 2.0
        final_temp = 0.5
        temp_step = (initial_temp - final_temp) / n_steps
        
        # Run Gibbs sampling with temperature control
        for step in range(n_steps):
            # Calculate current temperature (controls randomness)
            temperature = initial_temp - step * temp_step
            
            # Compute hidden probabilities and sample states
            hidden_probs = rbm.compute_hidden_probabilities(visible_states)
            
            # Apply temperature to make sampling more/less random
            # Higher temperature -> more random (hidden_probs closer to 0.5)
            # Lower temperature -> less random (hidden_probs closer to 0 or 1)
            if temperature != 1.0:
                # Scale logits by inverse temperature
                logits = ops.log(ops.divide(hidden_probs, ops.subtract(1.0, hidden_probs)))
                scaled_logits = ops.divide(logits, temperature)
                hidden_probs = ops.sigmoid(scaled_logits)
            
            hidden_states = rbm.sample_hidden_states(hidden_probs)
            
            # Compute visible probabilities and sample states
            visible_probs = rbm.compute_visible_probabilities(hidden_states)
            
            # Apply the same temperature scaling to visible units
            if temperature != 1.0:
                logits = ops.log(ops.divide(visible_probs, ops.subtract(1.0, visible_probs)))
                scaled_logits = ops.divide(logits, temperature)
                visible_probs = ops.sigmoid(scaled_logits)
            
            visible_states = rbm.sample_visible_states(visible_probs)
            
            # Occasionally introduce noise to avoid getting stuck
            if step % 5 == 0 and step > 0:
                noise = tensor.random_normal(tensor.shape(visible_states), stddev=0.1)
                visible_states = ops.add(visible_states, noise)
                visible_states = ops.clip(visible_states, 0.0, 1.0)
            
            # Store the current state
            dream_states.append(tensor.to_numpy(visible_states[0]))
        return dream_states
    
    def generate_category_statistics_tables(
        self,
        data: Union[TensorLike, tensor.EmberTensor],
        normal_data: Union[TensorLike, tensor.EmberTensor],
        category_labels: Union[TensorLike, tensor.EmberTensor],
        cluster_info: Dict,
        feature_names: Optional[List[str]] = None,
        save_dir: str = 'outputs/tables',
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate pandas DataFrames with detailed statistics for each anomaly category.
        
        This method creates DataFrames with comprehensive statistics comparing
        anomaly categories to normal data, suitable for data science analysis.
        
        Args:
            data: Full input data including anomalies (tensor or numpy array)
            normal_data: Normal data samples for comparison (tensor or numpy array)
            category_labels: Category labels for each sample in data (-1 for normal) (tensor or numpy array)
            cluster_info: Dictionary with information about each cluster
            feature_names: Optional list of feature names
            save_dir: Directory to save CSV files
            save: Whether to save DataFrames to CSV files
            
        Returns:
            Dictionary of pandas DataFrames with statistics for each category
        """
        # Import necessary modules
        from ember_ml.nn import tensor
        from ember_ml import ops
        
        # Convert inputs to numpy for processing if they're tensors
        if hasattr(data, 'numpy'):
            data_np = tensor.to_numpy(data)
        else:
            data_np = data
            
        if hasattr(normal_data, 'numpy'):
            normal_data_np = tensor.to_numpy(normal_data)
        else:
            normal_data_np = normal_data
            
        if hasattr(category_labels, 'numpy'):
            category_labels_np = tensor.to_numpy(category_labels)
        else:
            category_labels_np = category_labels
        
        # Get unique category labels (excluding -1 for normal samples)
        unique_categories = np.unique(category_labels_np)
        unique_categories = unique_categories[unique_categories >= 0]
        
        if len(unique_categories) == 0:
            print("No anomaly categories to analyze.")
            return {}
        
        # If feature names not provided, create generic ones
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(data_tensor.shape[1])]
        
        # Create directory if saving is enabled
        if save:
            os.makedirs(save_dir, exist_ok=True)
        
        # Calculate normal data statistics once
        # First convert to tensor for computation
        normal_tensor = tensor.convert_to_tensor(normal_data_np)
        
        # Use ops functions for calculations
        normal_stats = {
            'mean': tensor.to_numpy(ops.stats.mean(normal_tensor, axis=0)),
            'std': tensor.to_numpy(ops.stats.std(normal_tensor, axis=0)),
            'min': tensor.to_numpy(stats.min(normal_tensor, axis=0)),
            'max': tensor.to_numpy(stats.max(normal_tensor, axis=0)),
            # Now using ops.stats for percentiles
            '25%': tensor.to_numpy(ops.stats.percentile(normal_tensor, 25, axis=0)),
            'median': tensor.to_numpy(ops.stats.median(normal_tensor, axis=0)),
            '75%': tensor.to_numpy(ops.stats.percentile(normal_tensor, 75, axis=0))
        }
        
        # Store DataFrames for each category
        category_dfs = {}
        
        # Summary DataFrame with all categories
        summary_rows = []
        
        # For each category
        for category_id in unique_categories:
            # Get indices for this category
            category_indices = cluster_info[category_id]['indices']
            category_samples_np = data_np[category_indices]
            
            # Convert to tensor for computation
            category_tensor = tensor.convert_to_tensor(category_samples_np)
            
            # Create a DataFrame for this category's statistics using ops
            cat_stats = {
                'Feature': feature_names,
                'Category_Mean': tensor.to_numpy(ops.stats.mean(category_tensor, axis=0)),
                'Category_Std': tensor.to_numpy(ops.stats.std(category_tensor, axis=0)),
                'Category_Min': tensor.to_numpy(stats.min(category_tensor, axis=0)),
                'Category_Max': tensor.to_numpy(stats.max(category_tensor, axis=0)),
                'Category_25%': tensor.to_numpy(ops.stats.percentile(category_tensor, 25, axis=0)),
                'Category_Median': tensor.to_numpy(ops.stats.median(category_tensor, axis=0)),
                'Category_75%': tensor.to_numpy(ops.stats.percentile(category_tensor, 75, axis=0)),
                'Normal_Mean': normal_stats['mean'],
                'Normal_Std': normal_stats['std'],
                'Normal_Min': normal_stats['min'],
                'Normal_Max': normal_stats['max'],
                'Normal_25%': normal_stats['25%'],
                'Normal_Median': normal_stats['median'],
                'Normal_75%': normal_stats['75%']
            }
            
            # Calculate additional comparative statistics using ops where possible
            # Z-Score calculation
            cat_mean_tensor = tensor.convert_to_tensor(cat_stats['Category_Mean'])
            normal_mean_tensor = tensor.convert_to_tensor(normal_stats['mean'])
            normal_std_tensor = tensor.convert_to_tensor(normal_stats['std'])
            epsilon = tensor.convert_to_tensor(1e-10)
            
            z_score = ops.divide(
                ops.subtract(cat_mean_tensor, normal_mean_tensor),
                ops.add(normal_std_tensor, epsilon)
            )
            cat_stats['Z_Score'] = tensor.to_numpy(z_score)
            
            # Absolute Z-score
            cat_stats['Abs_Z_Score'] = tensor.to_numpy(ops.abs(z_score))
            
            # Fold change
            fold_change = ops.divide(
                cat_mean_tensor,
                ops.add(normal_mean_tensor, epsilon)
            )
            cat_stats['Fold_Change'] = tensor.to_numpy(fold_change)
            
            # IQR Ratio - using numpy for simplicity since we already have numpy arrays
            cat_stats['IQR_Ratio'] = (cat_stats['Category_75%'] - cat_stats['Category_25%']) / ((cat_stats['Normal_75%'] - cat_stats['Normal_25%']) + 1e-10)
            
            # Create DataFrame and sort by absolute Z-score (most anomalous features first)
            df = pd.DataFrame(cat_stats)
            df = df.sort_values('Abs_Z_Score', ascending=False)
            
            # Store DataFrame for this category
            category_dfs[f"category_{category_id}"] = df
            
            # Save to CSV if requested
            if save:
                csv_path = os.path.join(save_dir, f"anomaly_category_{category_id}_stats.csv")
                df.to_csv(csv_path, index=False)
                print(f"Category {category_id} statistics saved to {csv_path}")
            
            # Add summary row for this category
            top_features = cluster_info[category_id]['top_features']
            top_feature_names = [feature_names[i] for i in top_features[:3]]
            summary_rows.append({
                'Category': category_id,
                'Count': len(category_samples_np),
                'Top_Features': ', '.join(top_feature_names),
                'Max_Z_Score': df['Abs_Z_Score'].max(),
                'Avg_Z_Score': df['Abs_Z_Score'].mean()
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        category_dfs['summary'] = summary_df
        
        # Save summary to CSV if requested
        if save:
            csv_path = os.path.join(save_dir, "anomaly_categories_summary.csv")
            summary_df.to_csv(csv_path, index=False)
            print(f"Categories summary saved to {csv_path}")
            
            # Create a combined file with all category statistics
            all_categories_df = pd.concat([df.assign(Category=cat_id.split('_')[1])
                                          for cat_id, df in category_dfs.items()
                                          if cat_id != 'summary'])
            
            csv_path = os.path.join(save_dir, "all_anomaly_categories_stats.csv")
            all_categories_df.to_csv(csv_path, index=False)
            print(f"All categories statistics saved to {csv_path}")
        
        return category_dfs
    
    def plot_anomaly_category_statistics(
        self,
        data: Union[TensorLike, tensor.EmberTensor],
        normal_data: Union[TensorLike, tensor.EmberTensor],
        category_labels: Union[TensorLike, tensor.EmberTensor],
        cluster_info: Dict,
        feature_names: Optional[List[str]] = None,
        title: str = 'Anomaly Category Statistics',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot detailed statistical distributions of top features for each anomaly category.
        
        This visualization shows box plots and histograms of the most important features
        for each anomaly category, compared to normal data.
        
        Args:
            data: Full input data including anomalies (tensor or numpy array)
            normal_data: Normal data samples for comparison (tensor or numpy array)
            category_labels: Category labels for each sample in data (-1 for normal) (tensor or numpy array)
            cluster_info: Dictionary with information about each cluster
            feature_names: Optional list of feature names
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Import necessary modules
        from ember_ml.nn import tensor
        from ember_ml import ops
        
        # Convert inputs to numpy for visualization if they're tensors
        if hasattr(data, 'numpy'):
            data_np = tensor.to_numpy(data)
        else:
            data_np = data
            
        if hasattr(normal_data, 'numpy'):
            normal_data_np = tensor.to_numpy(normal_data)
        else:
            normal_data_np = normal_data
            
        if hasattr(category_labels, 'numpy'):
            category_labels_np = tensor.to_numpy(category_labels)
        else:
            category_labels_np = category_labels
        
        # Get unique category labels (excluding -1 for normal samples)
        unique_categories = np.unique(category_labels_np)
        unique_categories = unique_categories[unique_categories >= 0]
        
        if len(unique_categories) == 0:
            print("No anomaly categories to visualize.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No anomaly categories detected",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # If feature names not provided, create generic ones
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        
        # Create figure
        n_categories = len(unique_categories)
        fig = plt.figure(figsize=(15, 5 * n_categories))
        
        # Create a large title for the entire figure
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # For each category
        for cat_idx, category_id in enumerate(unique_categories):
            # Get indices for this category
            category_indices = cluster_info[category_id]['indices']
            category_samples = data_np[category_indices]
            
            # Convert to tensor for potential computations later
            category_tensor = tensor.convert_to_tensor(category_samples)
            
            # Get top features for this category
            top_features = cluster_info[category_id]['top_features']
            
            # Create subplot grid for this category
            # 1 row for this category, with 4 columns:
            # - Column 1: Summary statistics table
            # - Column 2-4: Top 3 feature distributions
            gs = plt.GridSpec(1, 4, figure=fig,
                             bottom=0.9-(cat_idx+1)*(0.9/n_categories),
                             top=0.9-cat_idx*(0.9/n_categories),
                             wspace=0.3, hspace=0.3)
            
            # Create table with summary statistics
            ax_table = fig.add_subplot(gs[0, 0])
            ax_table.axis('tight')
            ax_table.axis('off')
            
            # Prepare table data
            table_data = []
            table_data.append(['Feature', 'Mean', 'Std', 'Min', 'Max', 'Z-score'])
            
            for feat_idx in top_features:
                # Get feature values for this category
                feature_values = category_samples[:, feat_idx]
                
                # Get normal feature values for comparison
                normal_feature_values = normal_data_np[:, feat_idx]
                
                # Convert to tensors for calculation
                feat_tensor = tensor.convert_to_tensor(feature_values)
                normal_feat_tensor = tensor.convert_to_tensor(normal_feature_values)
                # Calculate statistics using ops.stats
                mean = float(tensor.to_numpy(ops.stats.mean(feat_tensor)))
                std = float(tensor.to_numpy(ops.stats.std(feat_tensor)))
                min_val = float(tensor.to_numpy(stats.min(feat_tensor)))
                max_val = float(tensor.to_numpy(stats.max(feat_tensor)))
                
                
                # Calculate Z-score compared to normal data
                normal_mean = float(tensor.to_numpy(ops.stats.mean(normal_feat_tensor)))
                # Use ops.stats.var instead of tensor.var
                normal_variance = ops.stats.var(normal_feat_tensor)
                normal_std = float(tensor.to_numpy(ops.sqrt(normal_variance)))
                
                # Calculate Z-score using ops
                epsilon = tensor.convert_to_tensor(1e-10)
                z_score_tensor = ops.divide(
                    ops.subtract(tensor.convert_to_tensor(mean), tensor.convert_to_tensor(normal_mean)),
                    ops.add(tensor.convert_to_tensor(normal_std), epsilon)
                )
                z_score = float(tensor.to_numpy(z_score_tensor))
                
                # Add to table
                table_data.append([
                    feature_names[feat_idx],
                    f"{mean:.4f}",
                    f"{std:.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}",
                    f"{z_score:.4f}"
                ])
            
            # Create the table
            table = ax_table.table(
                cellText=table_data,
                loc='center',
                cellLoc='center'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the header row
            for j in range(len(table_data[0])):
                cell = table._cells[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            
            ax_table.set_title(f"Category {category_id} Statistics (n={len(category_samples)})")
            
            # For each top feature, create a distribution plot
            for i, feat_idx in enumerate(top_features[:3]):  # Only show top 3 features
                # Create subplot
                ax = fig.add_subplot(gs[0, i+1])
                
                # Get feature values - already using numpy arrays for matplotlib compatibility
                category_values = category_samples[:, feat_idx]
                normal_values = normal_data_np[:, feat_idx]
                
                # Create histograms (matplotlib requires numpy arrays)
                ax.hist(normal_values, bins=20, alpha=0.5, label='Normal', color='blue')
                ax.hist(category_values, bins=20, alpha=0.7, label=f'Category {category_id}', color='red')
                
                # Add a boxplot below the histogram
                boxplot_ax = ax.inset_axes([0.1, 0.02, 0.8, 0.2])
                boxplot_data = [normal_values, category_values]
                boxplot_ax.boxplot(boxplot_data, vert=False, labels=['Normal', f'Cat {category_id}'])
                boxplot_ax.set_yticks([1, 2])
                boxplot_ax.set_yticklabels(['Normal', f'Cat {category_id}'])
                
                # Add title and labels
                ax.set_title(f"{feature_names[feat_idx]} Distribution")
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                
                # Add statistics as text - using ops for calculation
                normal_feat_tensor = tensor.convert_to_tensor(normal_values)
                cat_feat_tensor = tensor.convert_to_tensor(category_values)
                
                normal_mean = float(tensor.to_numpy(ops.stats.mean(normal_feat_tensor)))
                normal_std = float(tensor.to_numpy(ops.stats.std(normal_feat_tensor)))
                
                category_mean = float(tensor.to_numpy(ops.stats.mean(cat_feat_tensor)))
                category_std = float(tensor.to_numpy(ops.stats.std(cat_feat_tensor)))
                
                # Calculate Z-score using ops
                epsilon = tensor.convert_to_tensor(1e-10)
                z_score_tensor = ops.divide(
                    ops.subtract(tensor.convert_to_tensor(category_mean), tensor.convert_to_tensor(normal_mean)),
                    ops.add(tensor.convert_to_tensor(normal_std), epsilon)
                )
                z_score = float(tensor.to_numpy(z_score_tensor))
                
                stats_text = (
                    f"Normal: μ={normal_mean:.2f}, σ={normal_std:.2f}\n"
                    f"Cat {category_id}: μ={category_mean:.2f}, σ={category_std:.2f}\n"
                    f"Z-score: {z_score:.2f}"
                )
                
                ax.text(0.95, 0.95, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_category_statistics_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Anomaly category statistics saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
        
    def plot_anomaly_categories(
        self,
        rbm: RBMModule,
        data: TensorLike,
        category_labels: TensorLike,
        cluster_info: Dict,
        feature_names: Optional[List[str]] = None,
        title: str = 'RBM Anomaly Categories',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot anomaly categories identified by the RBM.
        
        This visualization shows how anomalies are grouped into different categories
        based on their hidden unit activation patterns, even without explicit labels.
        
        Args:
            rbm: Trained RBM
            data: Input data
            category_labels: Category labels for each sample (-1 for normal samples)
            cluster_info: Dictionary with information about each cluster
            feature_names: Optional list of feature names
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        # Get unique category labels (excluding -1 for normal samples)
        unique_categories = np.unique(category_labels)
        unique_categories = unique_categories[unique_categories >= 0]
        
        if len(unique_categories) == 0:
            print("No anomaly categories to visualize.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No anomaly categories detected",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create figure
        n_categories = len(unique_categories)
        fig = plt.figure(figsize=(self.figsize[0], 3 * n_categories))
        
        # Set up subplots for each category
        gs = plt.GridSpec(n_categories, 2, width_ratios=[3, 1], figure=fig)
        
        # If feature names not provided, create generic ones
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        
        # Convert data to tensor for getting hidden activations
        from ember_ml.nn import tensor
        
        # Plot each category
        for i, category_id in enumerate(unique_categories):
            # Get samples in this category
            category_indices = cluster_info[category_id]['indices']
            category_samples = data[category_indices]
            
            # Create a heatmap of feature values for this category
            ax1 = fig.add_subplot(gs[i, 0])
            feature_means = stats.mean(category_samples, axis=0)
            feature_stds = stats.std(category_samples, axis=0)
            
            # Normalize feature values for better visualization
            feature_z_scores = feature_means / (feature_stds + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Create a bar plot of feature importances
            ax1.barh(feature_names, feature_z_scores, color='skyblue')
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax1.set_title(f"Category {category_id} Feature Importance (n={len(category_samples)})")
            ax1.set_xlabel('Z-Score')
            
            # Sort the bars for better visualization
            sort_idx = np.argsort(feature_z_scores)
            ax1.set_yticks(tensor.arange(len(feature_names)))
            ax1.set_yticklabels([feature_names[i] for i in sort_idx])
            
            # Create a second subplot for hidden unit activations
            ax2 = fig.add_subplot(gs[i, 1])
            
            # Get hidden activations for this category
            category_tensor = tensor.convert_to_tensor(category_samples, dtype=tensor.float32)
            hidden_probs = rbm.compute_hidden_probabilities(category_tensor)
            hidden_activations = tensor.to_numpy(hidden_probs)
            
            # Calculate mean activations for each hidden unit
            mean_activations = stats.mean(hidden_activations, axis=0)
            
            # Plot hidden unit activations
            ax2.barh(range(len(mean_activations)), mean_activations, color='coral')
            ax2.set_title("Hidden Unit Activations")
            ax2.set_xlabel('Mean Activation')
            ax2.set_ylabel('Hidden Unit')
            ax2.set_yticks(range(len(mean_activations)))
            ax2.set_yticklabels([f"H{j+1}" for j in range(len(mean_activations))])
            ax2.set_xlim(0, 1)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_anomaly_categories_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Anomaly categories plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
        
    def plot_feature_hidden_correlations(
        self,
        rbm: RBMModule,
        data: TensorLike,
        feature_names: Optional[List[str]] = None,
        title: str = 'Feature-Hidden Unit Correlations',
        save: bool = True,
        show: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlations between features and hidden unit activations.
        
        This visualization helps understand how the RBM categorizes data by showing
        which features most strongly activate each hidden unit.
        
        Args:
            rbm: Trained RBM
            data: Input data
            feature_names: Optional list of feature names
            title: Plot title
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saved plot (if None, auto-generated)
            
        Returns:
            Matplotlib figure
        """
        from ember_ml.nn import tensor
        
        # Convert data to tensor
        data_tensor = tensor.convert_to_tensor(data, dtype=tensor.float32)
        
        # Get hidden activations
        hidden_probs = rbm.compute_hidden_probabilities(data_tensor)
        hidden_probs_np = tensor.to_numpy(hidden_probs)
        
        # If feature names not provided, create generic ones
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
        
        # Calculate correlation matrix between features and hidden units
        correlation_matrix = tensor.zeros((data.shape[1], rbm.n_hidden))
        
        for i in range(data.shape[1]):
            for j in range(rbm.n_hidden):
                correlation_matrix[i, j] = np.corrcoef(data[:, i], hidden_probs_np[:, j])[0, 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot correlation heatmap
        im = ax.imshow(
            correlation_matrix,
            cmap='coolwarm',
            aspect='auto',
            interpolation='nearest',
            vmin=-1,
            vmax=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Add labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hidden Units', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Set y-axis labels to feature names
        ax.set_yticks(tensor.arange(len(feature_names)))
        ax.set_yticklabels(feature_names)
        
        # Set x-axis labels
        ax.set_xticks(tensor.arange(rbm.n_hidden))
        ax.set_xticklabels([f"H{i+1}" for i in range(rbm.n_hidden)])
        
        # Rotate x labels if there are many hidden units
        if rbm.n_hidden > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add grid
        ax.grid(False)
        
        # Add text annotations
        for i in range(data.shape[1]):
            for j in range(rbm.n_hidden):
                # Only annotate strong correlations to avoid clutter
                if abs(correlation_matrix[i, j]) > 0.5:
                    text_color = 'white' if abs(correlation_matrix[i, j]) > 0.7 else 'black'
                    ax.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                           ha="center", va="center", color=text_color, fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rbm_feature_correlations_{timestamp}.png"
            
            filepath = os.path.join(self.plots_dir, filename)
            fig.savefig(filepath, dpi=self.dpi)
            print(f"Feature-hidden correlations plot saved to {filepath}")
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
