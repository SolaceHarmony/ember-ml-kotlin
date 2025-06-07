# Liquid Time Constant (LTC) and Closed-form Continuous-time (CfC) Examples

This directory contains example implementations and applications of Liquid Time Constant (LTC) neurons and Closed-form Continuous-time (CfC) cells using the ember_ml framework.

## Files

- `liquid_anomaly_detector.py`: Implementation of an anomaly detector using Liquid Neural Networks
- `liquidtrainer.py`: Training utilities for Liquid Neural Networks
- `spherical_ltc_chain_fix.py`: Fixed implementation of a chain of Spherical LTC neurons
- `stride_aware_cfc.py`: Implementation of stride-aware CfC cells
- `stride_ware_cfc.py`: Alternative implementation of stride-aware CfC cells

## Usage

These examples can be run directly to see LTC and CfC implementations in action:

```bash
# Run the liquid anomaly detector
python examples/ltc/liquid_anomaly_detector.py

# Run the spherical LTC chain example
python examples/ltc/spherical_ltc_chain_fix.py
```

## Background

### Liquid Time Constant (LTC) Neurons

LTC neurons are a type of recurrent neural network unit that can adapt their time constants based on the input. This allows them to capture both fast and slow dynamics in time series data. Key features include:

- Adaptive time constants
- Continuous-time dynamics
- Gradient stability
- Efficient implementation

### Closed-form Continuous-time (CfC) Cells

CfC cells are an extension of LTC neurons that use a closed-form solution to the continuous-time dynamics. This allows for more stable training and better performance on long-range dependencies. Key features include:

- Closed-form solution to ODE
- Stable gradients
- Efficient implementation
- Support for variable stride lengths

The examples in this directory demonstrate these capabilities using the ember_ml framework.