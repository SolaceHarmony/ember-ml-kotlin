"""
Recurrent Neural Network (RNN) module.

This module provides implementations of various RNN layers,
including LSTM, GRU, RNN, CfC (Closed-form Continuous-time), LTC (Liquid Time-Constant),
ELTC (Enhanced Liquid Time-Constant), CTGRU (Continuous-Time GRU), CTRNN (Continuous-Time RNN),
GUCE (Grand Unified Cognitive Equation), and StrideAware cells for multi-timescale processing.
"""

# Removed cfc_cell, ltc_cell, and wired_cfc_cell imports
from ember_ml.nn.modules.rnn.cfc import CfC
from ember_ml.nn.modules.rnn.ltc import LTC
from ember_ml.nn.modules.rnn.eltc import ELTC, ODESolver
from ember_ml.nn.modules.rnn.ctgru import CTGRU
from ember_ml.nn.modules.rnn.ctrnn import CTRNN
from ember_ml.nn.modules.rnn.guce import GUCE, HolographicCorrector, OscillatoryGating
# Removed lstm_cell import
from ember_ml.nn.modules.rnn.lstm import LSTM
# Removed gru_cell import
from ember_ml.nn.modules.rnn.gru import GRU
# Removed rnn_cell import
from ember_ml.nn.modules.rnn.rnn import RNN
# Removed stride_aware_cell import
from ember_ml.nn.modules.rnn.stride_aware import StrideAware
# Temporarily comment out to avoid circular imports
# from ember_ml.nn.modules.rnn.stride_aware_cfc import StrideAwareWiredCfCCell
# from ember_ml.nn.modules.rnn.stride_aware_cfc_layer import StrideAwareCfC
# Temporarily comment out imports that might cause circular dependencies
# from ember_ml.nn.modules.rnn.blocky import BlockyRoadNeuron, BlockyRoadChain
from ember_ml.nn.modules.rnn.ctrqnet import CTRQNet
# from ember_ml.nn.modules.rnn.geometric import GeometricNeuron
# from ember_ml.nn.modules.rnn.hybrid import HybridNeuron, HybridLNNModel
from ember_ml.nn.modules.rnn.lqnet import LQNet
from ember_ml.nn.modules.rnn.se_cfc import seCfC
# from ember_ml.nn.modules.rnn.spherical_ltc import SphericalLTCConfig, SphericalLTCNeuron, SphericalLTCChain

__all__ = [
    'CfC',
    'LTC',
    'ELTC',
    'ODESolver',
    'CTGRU',
    'CTRNN',
    'GUCE',
    'HolographicCorrector',
    'OscillatoryGating',
    'LSTM',
    'GRU',
    'RNN',
    'StrideAware',
    # 'StrideAwareWiredCfCCell',  # Temporarily commented out
    # 'StrideAwareCfC',  # Temporarily commented out
    # Temporarily commented out classes
    # 'BlockyRoadNeuron',
    # 'BlockyRoadChain',
    'CTRQNet',
    # 'GeometricNeuron',
    # 'HybridNeuron',
    # 'HybridLNNModel',
    'LQNet',
    'seCfC',
    # 'SphericalLTCConfig',
    # 'SphericalLTCNeuron',
    # 'SphericalLTCChain',
]