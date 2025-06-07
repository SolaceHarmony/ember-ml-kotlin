# Actor-Based Architecture for Ember ML Kotlin

## Overview

Ember ML Kotlin is designed as a 100% actor-based machine learning platform with non-blocking IO and asynchronous communication over Kotlin channels. This architecture provides significant advantages for machine learning workloads, including improved concurrency, better resource utilization, and enhanced fault tolerance.

## Core Principles

1. **Everything is an Actor**: All components in the system are actors that communicate exclusively through message passing
2. **Non-Blocking Operations**: All operations, including IO and computation, are non-blocking
3. **Asynchronous Communication**: Communication between components happens asynchronously over Kotlin channels
4. **State Encapsulation**: Each actor encapsulates its own state, preventing shared mutable state issues
5. **Supervision Hierarchies**: Actors are organized in supervision hierarchies for fault tolerance

## Actor System Architecture

The actor system in Ember ML Kotlin consists of the following components:

### 1. Actor System

The actor system provides the runtime environment for actors:

```kotlin
// Create an actor system
val system = ActorSystem()

// Create an actor
val actor = system.actorOf<TensorActor>()

// Send a message to the actor
actor.send(ComputeMessage(tensor1, tensor2))
```

### 2. Core Actor Types

#### TensorActor

Responsible for tensor operations and state management:

```kotlin
class TensorActor : Actor<TensorMessage> {
    override suspend fun receive(message: TensorMessage) {
        when (message) {
            is ComputeMessage -> {
                val result = compute(message.a, message.b)
                sender.send(ResultMessage(result))
            }
            is TransformMessage -> {
                val result = transform(message.tensor, message.operation)
                sender.send(ResultMessage(result))
            }
            // Other message types
        }
    }

    private suspend fun compute(a: EmberTensor, b: EmberTensor): EmberTensor {
        // Perform computation
        return a + b
    }

    private suspend fun transform(tensor: EmberTensor, operation: Operation): EmberTensor {
        // Apply transformation
        return operation.apply(tensor)
    }
}
```

#### BackendActor

Manages backend operations and state:

```kotlin
class BackendActor : Actor<BackendMessage> {
    private var currentBackend: Backend = CPUBackend()

    override suspend fun receive(message: BackendMessage) {
        when (message) {
            is SetBackendMessage -> {
                val success = setBackend(message.backendType)
                sender.send(SetBackendResponse(success))
            }
            is GetBackendMessage -> {
                sender.send(GetBackendResponse(currentBackend))
            }
            is ExecuteOperationMessage -> {
                val result = executeOperation(message.operation, message.inputs)
                sender.send(ExecuteOperationResponse(result))
            }
            // Other message types
        }
    }

    private fun setBackend(backendType: BackendType): Boolean {
        // Set the backend
        return try {
            currentBackend = when (backendType) {
                BackendType.CPU -> CPUBackend()
                BackendType.METAL -> MetalBackend()
                BackendType.VULKAN -> VulkanBackend()
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    private suspend fun executeOperation(operation: Operation, inputs: List<EmberTensor>): EmberTensor {
        // Execute the operation using the current backend
        return currentBackend.execute(operation, inputs)
    }
}
```

#### ModelActor

Manages model operations and state:

```kotlin
class ModelActor : Actor<ModelMessage> {
    private var model: Model? = null

    override suspend fun receive(message: ModelMessage) {
        when (message) {
            is CreateModelMessage -> {
                model = createModel(message.architecture)
                sender.send(CreateModelResponse(model!!))
            }
            is ForwardMessage -> {
                val output = forward(message.input)
                sender.send(ForwardResponse(output))
            }
            is BackwardMessage -> {
                val gradients = backward(message.gradOutput)
                sender.send(BackwardResponse(gradients))
            }
            is UpdateMessage -> {
                update(message.gradients, message.learningRate)
                sender.send(UpdateResponse())
            }
            // Other message types
        }
    }

    private fun createModel(architecture: Architecture): Model {
        // Create a model based on the architecture
        return Model(architecture)
    }

    private suspend fun forward(input: EmberTensor): EmberTensor {
        // Forward pass
        return model?.forward(input) ?: throw IllegalStateException("Model not initialized")
    }

    private suspend fun backward(gradOutput: EmberTensor): Map<String, EmberTensor> {
        // Backward pass
        return model?.backward(gradOutput) ?: throw IllegalStateException("Model not initialized")
    }

    private suspend fun update(gradients: Map<String, EmberTensor>, learningRate: Float) {
        // Update model parameters
        model?.update(gradients, learningRate)
    }
}
```

#### TrainerActor

Manages training operations:

```kotlin
class TrainerActor : Actor<TrainerMessage> {
    override suspend fun receive(message: TrainerMessage) {
        when (message) {
            is TrainMessage -> {
                val result = train(message.model, message.dataLoader, message.optimizer, message.epochs)
                sender.send(TrainResponse(result))
            }
            is EvaluateMessage -> {
                val metrics = evaluate(message.model, message.dataLoader)
                sender.send(EvaluateResponse(metrics))
            }
            // Other message types
        }
    }

    private suspend fun train(model: ModelActor, dataLoader: DataLoaderActor,
                             optimizer: OptimizerActor, epochs: Int): TrainingResult {
        // Training loop
        for (epoch in 0 until epochs) {
            // Get data batches
            val batches = dataLoader.ask(GetBatchesMessage()).await()

            for (batch in batches) {
                // Forward pass
                val output = model.ask(ForwardMessage(batch.input)).await()

                // Compute loss
                val loss = computeLoss(output, batch.target)

                // Backward pass
                val gradients = model.ask(BackwardMessage(loss)).await()

                // Update parameters
                optimizer.ask(UpdateMessage(gradients)).await()
            }

            // Notify progress
            sender.send(ProgressMessage(epoch, epochs))
        }

        return TrainingResult(/* training metrics */)
    }

    private suspend fun evaluate(model: ModelActor, dataLoader: DataLoaderActor): Map<String, Float> {
        // Evaluation logic
        val metrics = mutableMapOf<String, Float>()

        // Get evaluation data
        val batches = dataLoader.ask(GetBatchesMessage()).await()

        for (batch in batches) {
            // Forward pass
            val output = model.ask(ForwardMessage(batch.input)).await()

            // Compute metrics
            val batchMetrics = computeMetrics(output, batch.target)

            // Accumulate metrics
            batchMetrics.forEach { (key, value) ->
                metrics[key] = (metrics[key] ?: 0f) + value
            }
        }

        // Average metrics
        metrics.forEach { (key, value) ->
            metrics[key] = value / batches.size
        }

        return metrics
    }
}
```

### 3. Message Types

Messages are the primary means of communication between actors:

```kotlin
// Tensor messages
sealed class TensorMessage
data class ComputeMessage(val a: EmberTensor, val b: EmberTensor) : TensorMessage()
data class TransformMessage(val tensor: EmberTensor, val operation: Operation) : TensorMessage()
data class ResultMessage(val result: EmberTensor) : TensorMessage()

// Backend messages
sealed class BackendMessage
data class SetBackendMessage(val backendType: BackendType) : BackendMessage()
data class GetBackendMessage(val unit: Unit = Unit) : BackendMessage()
data class ExecuteOperationMessage(val operation: Operation, val inputs: List<EmberTensor>) : BackendMessage()
data class SetBackendResponse(val success: Boolean) : BackendMessage()
data class GetBackendResponse(val backend: Backend) : BackendMessage()
data class ExecuteOperationResponse(val result: EmberTensor) : BackendMessage()

// Model messages
sealed class ModelMessage
data class CreateModelMessage(val architecture: Architecture) : ModelMessage()
data class ForwardMessage(val input: EmberTensor) : ModelMessage()
data class BackwardMessage(val gradOutput: EmberTensor) : ModelMessage()
data class UpdateMessage(val gradients: Map<String, EmberTensor>, val learningRate: Float) : ModelMessage()
data class CreateModelResponse(val model: Model) : ModelMessage()
data class ForwardResponse(val output: EmberTensor) : ModelMessage()
data class BackwardResponse(val gradients: Map<String, EmberTensor>) : ModelMessage()
data class UpdateResponse(val unit: Unit = Unit) : ModelMessage()

// Trainer messages
sealed class TrainerMessage
data class TrainMessage(val model: ModelActor, val dataLoader: DataLoaderActor,
                       val optimizer: OptimizerActor, val epochs: Int) : TrainerMessage()
data class EvaluateMessage(val model: ModelActor, val dataLoader: DataLoaderActor) : TrainerMessage()
data class ProgressMessage(val currentEpoch: Int, val totalEpochs: Int) : TrainerMessage()
data class TrainResponse(val result: TrainingResult) : TrainerMessage()
data class EvaluateResponse(val metrics: Map<String, Float>) : TrainerMessage()
```

## Benefits of Actor-Based Architecture for ML

### 1. Scalability

The actor model naturally scales across multiple cores and machines:
- Actors can be distributed across available hardware
- Message passing works the same whether local or remote
- No shared state means no complex locking mechanisms

### 2. Fault Tolerance

Actors provide built-in fault tolerance:
- Supervision hierarchies allow for error handling and recovery
- Failed actors can be restarted without affecting the entire system
- Errors are isolated to specific actors

### 3. Concurrency

The actor model simplifies concurrent programming:
- No shared mutable state eliminates race conditions
- Message-passing semantics are easier to reason about than locks
- Each actor processes messages sequentially, eliminating the need for synchronization

### 4. Resource Utilization

Actors enable efficient resource utilization:
- Computation can be distributed across available resources
- Memory usage can be optimized by isolating state
- IO operations can be performed asynchronously

### 5. Responsiveness

The non-blocking nature of actors ensures system responsiveness:
- Long-running operations don't block the entire system
- UI and other interactive components remain responsive
- System can adapt to changing workloads

## Implementation Considerations

### 1. Actor Granularity

Determining the right granularity for actors is crucial:
- Too fine-grained: Message passing overhead becomes significant
- Too coarse-grained: Limits concurrency and fault isolation

For Ember ML Kotlin, we use a mixed approach:
- Coarse-grained actors for high-level components (models, trainers)
- Medium-grained actors for backend operations
- Fine-grained actors for specific tensor operations when beneficial

### 2. Message Design

Well-designed messages are essential for an actor system:
- Messages should be immutable
- Messages should be self-contained
- Messages should be serializable (for potential distribution)

### 3. Supervision Strategy

Proper supervision strategies ensure system resilience:
- Restart actors on failure
- Escalate certain failures to parent actors
- Implement circuit breakers for external dependencies

### 4. Testing

Testing actor-based systems requires specialized approaches:
- Test actors in isolation
- Use test probes to verify message interactions
- Simulate failures to test recovery mechanisms

## Example: Training Loop with Actors

```kotlin
// Create actor system
val system = ActorSystem()

// Create actors
val model = system.actorOf<ModelActor>()
val dataLoader = system.actorOf<DataLoaderActor>()
val optimizer = system.actorOf<OptimizerActor>()
val trainer = system.actorOf<TrainerActor>()

// Configure model
model.ask(CreateModelMessage(Architecture.ResNet50)).await()

// Configure data loader
dataLoader.ask(ConfigureDataLoaderMessage("path/to/data", batchSize = 32)).await()

// Configure optimizer
optimizer.ask(ConfigureOptimizerMessage(OptimizerType.ADAM, learningRate = 0.001)).await()

// Start training
val trainingJob = launch {
    trainer.ask(TrainMessage(model, dataLoader, optimizer, epochs = 10)).await()
}

// Listen for progress updates
launch {
    trainer.messages.filterIsInstance<ProgressMessage>().collect { progress ->
        println("Training progress: ${progress.currentEpoch}/${progress.totalEpochs}")
    }
}

// Wait for training to complete
trainingJob.join()

// Evaluate model
val metrics = trainer.ask(EvaluateMessage(model, dataLoader)).await()
println("Evaluation metrics: $metrics")
```

## Conclusion

The actor-based architecture provides a solid foundation for Ember ML Kotlin, enabling high performance, scalability, and fault tolerance. By designing the system around message passing and non-blocking operations, we can create a machine learning platform that efficiently utilizes available hardware resources while providing a clean, maintainable API for users.
