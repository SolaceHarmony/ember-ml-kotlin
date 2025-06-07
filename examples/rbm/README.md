# Restricted Boltzmann Machine (RBM) Examples

This directory contains example implementations and applications of Restricted Boltzmann Machines (RBMs) using the ember_ml framework.

## Files

- `rbm_example.py`: Basic example of RBM implementation and usage
- `rbm_backend_example.py`: Example demonstrating RBM implementation across different backends
- `rbm_anomaly_detection_example.py`: Example of using RBMs for anomaly detection

## Usage

These examples can be run directly to see RBMs in action:

```bash
# Run the basic RBM example
python examples/rbm/rbm_example.py

# Run the backend comparison example
python examples/rbm/rbm_backend_example.py

# Run the anomaly detection example
python examples/rbm/rbm_anomaly_detection_example.py
```

## Background

Restricted Boltzmann Machines (RBMs) are a type of stochastic neural network that can learn a probability distribution over its set of inputs. They are "restricted" in that there are no connections between units within the same layer.

RBMs have been used for:
- Dimensionality reduction
- Feature learning
- Classification
- Anomaly detection
- Collaborative filtering

The examples in this directory demonstrate these capabilities using the ember_ml framework.