"""
Visualization tools for wave memory analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from typing import List, Tuple, Optional

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ..models.multi_sphere import MultiSphereWaveModel
from .metrics import AnalysisMetrics, MetricsCollector

class WaveMemoryAnalyzer:
    """
    Comprehensive visualization and analysis tools for wave memory systems.
    """
    
    def __init__(self):
        """Initialize visualizer with matplotlib settings."""
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def analyze_model(self, 
                     model: MultiSphereWaveModel, 
                     steps: int = 10) -> Tuple[plt.Figure, TensorLike, AnalysisMetrics]:
        """
        Run comprehensive analysis on wave memory model.
        
        Args:
            model: MultiSphereWaveModel instance
            steps: Number of time steps to simulate
            
        Returns:
            Tuple of (figure, history, metrics)
        """
        # Initialize metrics collector
        collector = MetricsCollector()
        
        # Set up initial conditions
        model.set_initial_state(0, fast_vec=[1, 0, 0, 0])
        model.set_initial_state(1, fast_vec=[0.7, 0.7, 0, 0])
        model.set_initial_state(2, fast_vec=[0, 1, 0, 0])
        
        # Create input sequences
        input_waves_seq = []
        gating_seq = []
        for t in range(steps):
            if t < 5:  # Input phase
                wave_0 = tensor.convert_to_tensor([0.0, 0.5, 0.5, 0.0])
                input_waves_seq.append([wave_0, None, None])
                gating_seq.append([True, False, False])
            else:  # Free evolution phase
                input_waves_seq.append([None, None, None])
                gating_seq.append([False, False, False])
        
        # Run simulation
        collector.start_computation()
        history = model.run(steps=steps, 
                          input_waves_seq=input_waves_seq,
                          gating_seq=gating_seq)
        collector.end_computation()
        
        # Create visualization
        fig = self.create_visualization(history)
        
        # Compute metrics
        metrics = collector.compute_metrics()
        
        return fig, history, metrics
    
    def create_visualization(self, history: TensorLike) -> plt.Figure:
        """
        Create comprehensive visualization of wave dynamics.
        
        Args:
            history: Array of shape (steps, num_spheres, 4) containing wave history
            
        Returns:
            Matplotlib figure with multiple subplots
        """
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, height_ratios=[1.2, 1, 1, 1, 1, 0.8], 
                     hspace=0.5, wspace=0.4)
        
        self._plot_component_evolution(fig.add_subplot(gs[0, :]), history)
        self._plot_phase_space(fig.add_subplot(gs[1, 0]), history)
        self._plot_energy_distribution(fig.add_subplot(gs[1, 1]), history)
        self._plot_phase_correlations(fig.add_subplot(gs[1, 2]), history)
        self._plot_state_space(fig.add_subplot(gs[2, 0]), history)
        self._plot_interference_patterns(fig.add_subplot(gs[2, 1]), history)
        self._plot_energy_transfer(fig.add_subplot(gs[2, 2]), history)
        self._plot_combined_analysis(fig.add_subplot(gs[3, :]), history)
        
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9)
        return fig
    
    def _plot_component_evolution(self, ax: plt.Axes, history: TensorLike):
        """Plot evolution of wave components over time."""
        steps = len(history)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['x', 'y', 'z', 'w']
        
        for sphere_id in range(history.shape[1]):
            for dim in range(4):
                ax.plot(range(steps), history[:, sphere_id, dim],
                       color=colors[dim], 
                       label=f'Sphere {sphere_id} - {labels[dim]}')
                
        ax.set_title('Component Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Component Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    def _plot_phase_space(self, ax: plt.Axes, history: TensorLike):
        """Plot phase space trajectories."""
        for sphere_id in range(history.shape[1]):
            phase_angles = np.arctan2(
                ops.linearalg.norm(history[:, sphere_id, 1:], axis=1),
                history[:, sphere_id, 0]
            )
            energies = stats.sum(history[:, sphere_id]**2, axis=1)
            sc = ax.scatter(phase_angles, energies, 
                          c=range(len(phase_angles)),
                          cmap='viridis', 
                          label=f'Sphere {sphere_id}',
                          alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Time Step')
        ax.set_title('Phase Space Trajectories')
        ax.set_xlabel('Phase Angle')
        ax.set_ylabel('Energy')
        ax.legend()
        
    def _plot_energy_distribution(self, ax: plt.Axes, history: TensorLike):
        """Plot energy distribution over time."""
        steps = len(history)
        for sphere_id in range(history.shape[1]):
            energies = [stats.sum(state**2) for state in history[:, sphere_id]]
            ax.plot(range(steps), energies, label=f'Sphere {sphere_id}')
            
        ax.set_title('Energy Distribution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Energy')
        ax.legend()
        
    def _plot_phase_correlations(self, ax: plt.Axes, history: TensorLike):
        """Plot phase correlations between adjacent spheres."""
        steps = len(history)
        for i in range(history.shape[1]-1):
            phase_diff = []
            for t in range(steps):
                p1 = np.arctan2(ops.linearalg.norm(history[t, i, 1:]), 
                               history[t, i, 0])
                p2 = np.arctan2(ops.linearalg.norm(history[t, i+1, 1:]), 
                               history[t, i+1, 0])
                phase_diff.append(p2 - p1)
            ax.plot(range(steps), phase_diff, label=f'Spheres {i}-{i+1}')
            
        ax.set_title('Phase Correlations')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Phase Difference')
        ax.legend()
        
    def _plot_state_space(self, ax: plt.Axes, history: TensorLike):
        """Plot state space projection."""
        markers = ['o', 's', '^']
        for sphere_id in range(history.shape[1]):
            ax.scatter(history[:, sphere_id, 0], 
                      history[:, sphere_id, 1],
                      marker=markers[sphere_id], 
                      label=f'Sphere {sphere_id}', 
                      alpha=0.6)
            ax.plot(history[:, sphere_id, 0], 
                   history[:, sphere_id, 1], 
                   alpha=0.3)
            
        ax.set_title('State Space Projection (x-y)')
        ax.set_xlabel('X Component')
        ax.set_ylabel('Y Component')
        ax.legend()
        
    def _plot_interference_patterns(self, ax: plt.Axes, history: TensorLike):
        """Plot interference pattern heatmap."""
        steps = len(history)
        interference = tensor.zeros((steps, history.shape[1]))
        
        for t in range(steps):
            for i in range(history.shape[1]):
                interference[t, i] = sum(
                    abs(ops.dot(history[t, i], history[t, j]))
                    for j in range(history.shape[1]) if j != i
                )
                
        im = ax.imshow(interference.T, aspect='auto', cmap='RdYlBu_r')
        plt.colorbar(im, ax=ax)
        ax.set_title('Interference Patterns')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sphere')
        ax.set_yticks(range(history.shape[1]))
        ax.set_yticklabels([f'Sphere {i}' for i in range(history.shape[1])])
        
    def _plot_energy_transfer(self, ax: plt.Axes, history: TensorLike):
        """Plot energy transfer between time steps."""
        steps = len(history)
        for sphere_id in range(history.shape[1]):
            energy_transfer = np.diff(
                [stats.sum(state**2) for state in history[:, sphere_id]]
            )
            ax.plot(range(1, steps), energy_transfer, 
                   label=f'Sphere {sphere_id}')
            
        ax.set_title('Energy Transfer')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Energy Transfer')
        ax.legend()
        
    def _plot_combined_analysis(self, ax: plt.Axes, history: TensorLike):
        """Plot combined system analysis metrics."""
        steps = len(history)
        
        # Calculate total energy and phase coherence
        total_energy = [
            stats.sum([stats.sum(state**2) for state in timestep]) / history.shape[1]
            for timestep in history
        ]
        
        phase_coherence = []
        for t in range(steps):
            phases = [
                np.arctan2(ops.linearalg.norm(state[1:]), state[0])
                for state in history[t]
            ]
            diffs = [
                abs(p1 - p2) 
                for i, p1 in enumerate(phases)
                for p2 in phases[i+1:]
            ]
            phase_coherence.append(stats.mean(ops.cos(diffs)))
            
        ax.plot(range(steps), total_energy, 
               label='Normalized Total Energy', linewidth=2)
        ax.plot(range(steps), phase_coherence, 
               label='Phase Coherence', linewidth=2)
        ax.set_title('Combined System Analysis')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        
    def animate_model(self, history: TensorLike) -> animation.FuncAnimation:
        """
        Create animation of wave evolution.
        
        Args:
            history: Wave state history array
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots()
        steps = len(history)
        line, = ax.plot([], [], lw=2)
        
        def init():
            ax.set_xlim(0, history.shape[1])
            ax.set_ylim(-1, 1)
            return line,
            
        def update(frame):
            data = history[frame, :, 0]  # x-component
            line.set_data(range(len(data)), data)
            return line,
            
        ani = animation.FuncAnimation(
            fig, update, frames=steps,
            init_func=init, blit=True
        )
        
        return ani