# Ember ML Architecture Reorganization Plan

## Current Architecture Analysis

The current Ember ML architecture has several components spread across different namespaces:

1. **ember_ml.nn**: Neural network components
2. **ember_ml.ops**: Operations (mostly tensor operations)
3. **ember_ml.backend**: Backend implementations (NumPy, PyTorch, MLX)
4. **ember_ml.attention**: Attention mechanisms
5. **ember_ml.audio**: Audio processing
6. **ember_ml.features**: Feature extraction
7. **ember_ml.models**: Model implementations (RBM, etc.)

This organization has led to some inconsistencies and makes it harder to understand the overall architecture. The frontend/backend separation is not always clear, and related functionality is spread across different namespaces.

## Architectural Principles

1. **Clear Frontend/Backend Separation**: The frontend should never expose backend-specific details to users.
2. **Consistent Namespace Organization**: Related functionality should be grouped together in a logical namespace hierarchy.
3. **Task-Oriented Pipeline Architecture**: Higher-level components should be organized around tasks or workflows.
4. **Asynchronous Processing**: The architecture should support asynchronous processing for better performance and scalability.

## Proposed Architectural Changes

### 1. Namespace Reorganization

Reorganize the codebase into a more consistent namespace hierarchy:

```
ember_ml/
├── nn/                  # Neural network components
│   ├── tensor/          # Tensor operations and classes
│   ├── attention/       # Attention mechanisms
│   ├── layers/          # Basic neural network layers
│   ├── modules/         # Higher-level neural network modules
│   └── models/          # Complete model implementations
├── data/                # Data processing
│   ├── audio/           # Audio processing
│   ├── text/            # Text processing
│   ├── vision/          # Vision processing
│   └── features/        # Feature extraction
├── pipeline/            # Pipeline components
│   ├── wiring/          # Wiring and connectivity
│   ├── tasks/           # Task-specific pipelines
│   └── async/           # Asynchronous processing utilities
└── backend/             # Backend implementations
    ├── numpy/           # NumPy backend
    ├── torch/           # PyTorch backend
    └── mlx/             # MLX backend
```

### 2. EmberTensor Frontend/Backend Separation

Implement the EmberTensor frontend/backend separation as outlined in the [EmberTensor Frontend/Backend Separation](ember_tensor_frontend_backend_separation.md) plan:

- Create a clear separation between frontend and backend tensor operations
- Ensure all frontend functions return EmberTensor objects
- Implement Python protocol methods for EmberTensor to make it behave like native tensors

### 3. Pipeline Architecture

Develop a new pipeline architecture for task-oriented workflows:

```python
class Pipeline:
    """Base class for all pipelines."""
    
    def __init__(self, name=None):
        self.name = name
        self.stages = []
    
    def add_stage(self, stage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def run(self, input_data):
        """Run the pipeline on the input data."""
        result = input_data
        for stage in self.stages:
            result = stage(result)
        return result
```

Example usage:

```python
# Create a feature extraction pipeline
pipeline = Pipeline("feature_extraction")
pipeline.add_stage(Preprocessor())
pipeline.add_stage(FeatureExtractor())
pipeline.add_stage(Normalizer())

# Run the pipeline
features = pipeline.run(raw_data)
```

### 4. Asynchronous Processing

Implement asynchronous processing using Python's asyncio or a similar framework:

```python
class AsyncPipeline(Pipeline):
    """Asynchronous pipeline."""
    
    async def run(self, input_data):
        """Run the pipeline asynchronously."""
        result = input_data
        for stage in self.stages:
            if asyncio.iscoroutinefunction(stage.__call__):
                result = await stage(result)
            else:
                result = stage(result)
        return result
```

Example usage:

```python
# Create an asynchronous feature extraction pipeline
pipeline = AsyncPipeline("feature_extraction")
pipeline.add_stage(AsyncPreprocessor())
pipeline.add_stage(AsyncFeatureExtractor())
pipeline.add_stage(AsyncNormalizer())

# Run the pipeline asynchronously
features = await pipeline.run(raw_data)
```

## Actor-Based Concurrency Model

Inspired by Kotlin's actor model, we can implement an actor-based concurrency model for Ember ML:

```python
class Actor:
    """Base class for all actors."""
    
    def __init__(self):
        self.mailbox = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Start the actor."""
        self.running = True
        while self.running:
            message = await self.mailbox.get()
            await self.process_message(message)
            self.mailbox.task_done()
    
    async def send(self, message):
        """Send a message to the actor."""
        await self.mailbox.put(message)
    
    async def process_message(self, message):
        """Process a message."""
        raise NotImplementedError("Subclasses must implement process_message")
    
    async def stop(self):
        """Stop the actor."""
        self.running = False
```

Example usage:

```python
class FeatureExtractorActor(Actor):
    """Actor for feature extraction."""
    
    async def process_message(self, message):
        """Process a message."""
        if message["type"] == "extract_features":
            data = message["data"]
            features = extract_features(data)
            await message["reply_to"].send({
                "type": "features_extracted",
                "features": features
            })

# Create and start the actor
extractor = FeatureExtractorActor()
asyncio.create_task(extractor.start())

# Send a message to the actor
await extractor.send({
    "type": "extract_features",
    "data": raw_data,
    "reply_to": result_collector
})
```

## Implementation Strategy

1. **Phase 1**: Implement the EmberTensor frontend/backend separation
2. **Phase 2**: Reorganize the codebase into the new namespace hierarchy
3. **Phase 3**: Develop the pipeline architecture
4. **Phase 4**: Implement asynchronous processing and the actor-based concurrency model

## Migration Path

To minimize disruption, we'll implement these changes incrementally:

1. First, create the new namespace structure and move code gradually
2. Implement the EmberTensor frontend/backend separation
3. Develop the pipeline architecture and actor model
4. Update existing code to use the new architecture

## Backward Compatibility Considerations

To maintain backward compatibility during the transition:

1. Keep the original namespaces and functions available but mark them as deprecated
2. Add warnings when deprecated code is used
3. Provide clear migration guides for users

## Timeline and Milestones

1. **Phase 1 (1-2 weeks)**: Implement the EmberTensor frontend/backend separation
2. **Phase 2 (2-3 weeks)**: Reorganize the codebase into the new namespace hierarchy
3. **Phase 3 (2-3 weeks)**: Develop the pipeline architecture
4. **Phase 4 (2-3 weeks)**: Implement asynchronous processing and the actor-based concurrency model

## Conclusion

This architectural reorganization will significantly improve the consistency, usability, and performance of the Ember ML library. By implementing a clear frontend/backend separation, a consistent namespace hierarchy, a task-oriented pipeline architecture, and asynchronous processing, we'll make the library more intuitive, flexible, and scalable.