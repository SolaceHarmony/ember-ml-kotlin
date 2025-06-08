# Ember ML Kotlin

A pure Kotlin Multiplatform implementation of the Ember ML machine learning framework, focusing on native platforms and actor-based architecture.

## Key Features

- **100% Pure Native/Common Code**: No JVM dependencies, designed for native platforms (macOS, Linux, Windows) and JavaScript.
- **Actor-Based Architecture**: Built on a 100% actor-based model with non-blocking IO and asynchronous communication over Kotlin channels.
- **High-Performance Tensor Operations**: CPU-friendly routines based on bitwise operations and advanced math for tensor manipulation.
- **Float64 Workaround**: Special implementation to handle Float64 limitations in platforms like Apple MLX and Metal.
- **Metal Kernel Integration**: Support for Metal kernels on Apple platforms, with potential for porting to Kotlin Native.

## Overview

This repository contains a Kotlin Multiplatform port of the [Ember ML](https://github.com/ember-ml/ember-ml) Python library. It provides a modern machine learning library that uses cutting-edge neural networks with hardware optimization. Ember ML Kotlin implements various neuron types based on recent research papers and supports multiple backends to run efficiently on different hardware platforms.

The project focuses on several key areas:

1. **Hardware-Optimized Neural Networks**: Implementation of cutting-edge neural network architectures optimized for different hardware platforms
2. **Multi-Backend Support**: Backend-agnostic tensor operations that work with different computational backends
3. **Feature Extraction**: Tools for extracting features for use in neural networks
4. **Liquid Neural Networks**: Design and implementation of liquid neural networks and other advanced architectures

## Architecture

Ember ML Kotlin is built on an actor-based architecture where:
- Each component is an actor
- Actors communicate exclusively through message passing
- All operations are non-blocking
- State is encapsulated within actors
- Concurrency is managed through the actor system

This architecture provides several benefits for machine learning applications:
- **Scalability**: Actors can be distributed across multiple threads, cores, or even machines
- **Resilience**: Supervisor hierarchies provide fault tolerance and recovery
- **Concurrency**: The message-passing model simplifies concurrent programming
- **Modularity**: Actors encapsulate state and behavior, promoting clean design
- **Responsiveness**: Non-blocking operations ensure the system remains responsive

## Tensor Implementation

The tensor implementation in Ember ML Kotlin is based on bitwise operations and advanced math:
- **Bitwise Operations**: Shift operations, bit operations, and wave operations form the foundation for tensor manipulation.
- **MegaBinary and MegaNumber**: High-precision binary and numeric operations for implementing tensor operations.
- **Float64 Workaround**: By implementing high-precision operations using bitwise manipulations, we achieve Float64-like precision even on platforms that don't natively support it.

## Metal Kernel Integration

Ember ML Kotlin includes support for Metal kernels on Apple platforms:
- **Metal Kernel Bindings**: Kotlin Native bindings for Metal kernels.
- **High-Performance Algorithms**: Implementations of key algorithms like SVD using Metal kernels.
- **Cross-Platform Compatibility**: The same API works across all platforms, with Metal acceleration on Apple devices.

## Project Structure

The project follows a standard Kotlin Multiplatform structure with a focus on native and common code:

```
ember-ml-kotlin/
├── build.gradle.kts           # Gradle build script
├── settings.gradle.kts        # Gradle settings
├── src/
│   ├── commonMain/            # Common code for all platforms
│   │   └── kotlin/
│   │       └── ai/
│   │           └── solace/
│   │               └── emberml/
│   │                   ├── actors/             # Actor system
│   │                   ├── tensor/             # Tensor module
│   │                   │   ├── bitwise/        # Bitwise operations
│   │                   │   ├── common/         # Common tensor implementations
│   │                   │   ├── interfaces/     # Tensor interfaces
│   │                   │   └── ops/            # Tensor operations
│   │                   ├── backend/            # Backend abstraction
│   │                   ├── nn/                 # Neural network components
│   │                   ├── ops/                # Core operations
│   │                   ├── training/           # Training utilities
│   │                   └── utils/              # Utility functions
│   └── commonTest/           # Common tests
```

## Getting Started

### Prerequisites

- Kotlin 2.0.20 or higher
- Gradle 8.0 or higher

### Building the Project

```bash
./gradlew build
```

### Running Tests

```bash
./gradlew allTests
```

## Example Usage

```kotlin
// Create a tensor filled with zeros
val zeros = zeros(intArrayOf(2, 3))
println(zeros) // EmberTensor(shape=[2, 3], dtype=float32, device=cpu, requiresGrad=false)

// Create a tensor filled with ones
val ones = ones(intArrayOf(2, 3))
println(ones) // EmberTensor(shape=[2, 3], dtype=float32, device=cpu, requiresGrad=false)

// Create a tensor from a list
val tensor = EmberTensor(listOf(1, 2, 3, 4))
println(tensor) // EmberTensor(shape=[4], dtype=float32, device=cpu, requiresGrad=false)

// Reshape a tensor
val reshaped = tensor.reshape(EmberShape.of(2, 2))
println(reshaped) // EmberTensor(shape=[2, 2], dtype=float32, device=cpu, requiresGrad=false)
```

## Project Status

This project is currently in development. See the [CHECKLIST.md](CHECKLIST.md) file for the current status and planned features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
