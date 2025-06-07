"""This module provides backend-agnostic implementations of neural network modules
using the ops abstraction layer.

Available Modules:
    Base Classes:
        RNNCell: Base recurrent cell
        LSTMCell: Long Short-Term Memory cell
        GRUCell: Gated Recurrent Unit cell
        
    Specialized Cells:
        StrideAware: Stride-aware cell base class
        StrideAwareCfC: Stride-aware Closed-form Continuous-time cell
        StrideAwareCell: Stride-aware general cell
        
    Advanced Modules:
        CfC: Closed-form Continuous-time cell
        WiredCfCCell: CfC with wiring capabilities
        LTC: Liquid Time-Constant cell
        LTCCell: LTC cell implementation
        ELTC: Enhanced Liquid Time-Constant cell
        CTGRU: Continuous-Time Gated Recurrent Unit
        CTRNN: Continuous-Time Recurrent Neural Network
        GUCE: Grand Unified Cognitive Equation neuron
        StrideAwareWiredCfCCell: Stride-aware wired cell implementation
        
    Neural Circuit Policies:
        NCP: Neural Circuit Policy
        AutoNCP: Automatic Neural Circuit Policy
        GUCENCP: GUCE Neural Circuit Policy
        AutoGUCENCP: Automatic GUCE Neural Circuit Policy
        
    Wiring Patterns:
        NeuronMap: Base class for neural connectivity structures
        NCPMap: Neural Circuit Policy wiring
        FullyConnectedMap: Fully connected wiring
        RandomMap: Random connectivity wiring
        LanguageWiring: Wiring for language processing tasks
        RoboticsWiring: Wiring for robotics applications
        SignalWiring: Wiring for multi-scale signal processing
        FrequencyWiring: Wiring for frequency analysis
        VisionWiring: Wiring for computer vision tasks
        
    Anomaly Detection:
        LiquidAutoencoder: Autoencoder using liquid neural networks for anomaly detection
        
    Forecasting:
        LiquidForecaster: Forecasting model using liquid neural networks with uncertainty estimation
        
    Trainers:
        MemoryOptimizedTrainer: Memory-optimized trainer for Apple Silicon hardware
        
    Solvers:
        ExpectationSolver: Non-gradient solver using perturbation theory

"""

from ember_ml.nn.modules.base_module import BaseModule as Module, BaseModule, Parameter
# Removed ModuleCell and ModuleWiredCell imports
from ember_ml.nn.modules.ncp import NCP
from ember_ml.nn.modules.auto_ncp import AutoNCP
from ember_ml.nn.modules.guce_ncp import GUCENCP, AutoGUCENCP
from ember_ml.nn.modules.dense import Dense # Import Dense from its new location
# Import NeuronMap classes from the new wiring sub-package
from ember_ml.nn.modules.wiring import (
    NeuronMap, NCPMap, FullyConnectedMap, RandomMap,
    LanguageWiring, RoboticsWiring, SignalWiring, FrequencyWiring, VisionWiring
)
# Import RNN modules (keep existing) - Removed StrideAwareCfC from this line
from ember_ml.nn.modules.rnn import RNN, LSTM, GRU, StrideAware
# Import the separated layer and corrected cell import
from ember_ml.nn.modules.rnn import (
    CfC, LTC, ELTC, ODESolver, CTGRU, CTRNN, GUCE
)
# from ember_ml.nn.modules.rnn.stride_aware_cfc_layer import StrideAwareCfC
# Import activation modules
from ember_ml.nn.modules.activations import ReLU, Tanh, Sigmoid, Softmax, Softplus, LeCunTanh, Dropout
# Import solver modules
from ember_ml.nn.modules.solvers import ExpectationSolver
# Import anomaly detection modules
from ember_ml.nn.modules.anomaly import LiquidAutoencoder
# Import forecasting modules
from ember_ml.nn.modules.forecasting import LiquidForecaster
# Import trainer modules
from ember_ml.nn.modules.trainers import MemoryOptimizedTrainer

__all__ = [
    # Base
    'Module',
    'Parameter',
    'BaseModule',
    # Removed ModuleCell and ModuleWiredCell from exports
    'Dense', # Add Dense export
    'NCP',
    'AutoNCP', # Layer convenience class
    'GUCENCP',
    'AutoGUCENCP',
    
    # NeuronMap exports (imported from .wiring)
    'NeuronMap',
    'NCPMap',
    'FullyConnectedMap',
    'RandomMap',
    'LanguageWiring',
    'RoboticsWiring',
    'SignalWiring',
    'FrequencyWiring',
    'VisionWiring',
    
    # RNN exports (keep existing)
    'RNN',
    'LSTM',
    'GRU',
    
    # Stride-aware
    'StrideAware',
    # 'StrideAwareCfC', # Temporarily commented out
    
    # Advanced modules
    'CfC',
    'LTC',
    'ELTC',
    'ODESolver',
    'CTGRU',
    'CTRNN',
    'GUCE',
    
    # Activations
    'ReLU',
    'Tanh',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'LeCunTanh',
    'Dropout',
    
    # Solvers
    'ExpectationSolver',
    
    # Anomaly Detection
    'LiquidAutoencoder',
    
    # Forecasting
    'LiquidForecaster',
    
    # Trainers
    'MemoryOptimizedTrainer',
]