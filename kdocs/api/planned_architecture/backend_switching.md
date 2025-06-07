# Backend Selection in Kotlin Native Without Shared Libraries

## Overview

Ember ML Kotlin requires a flexible backend selection mechanism that works across all platforms, including those with limited dynamic loading capabilities. This document outlines the architecture for a configuration-based backend selection system that doesn't rely on shared libraries.

## Design Principles

1. **Universal Compatibility**: The solution must work on all platforms targeted by Kotlin Native
2. **Runtime Flexibility**: Allow changing backends at runtime without restarting
3. **Graceful Fallback**: Always provide a CPU backend as fallback
4. **Configuration Persistence**: Save user preferences between runs
5. **Hot Reloading**: Detect configuration changes while running

## Architecture

The backend selection system consists of the following components:

### 1. Backend Registry

The `BackendRegistry` is responsible for managing available backends and the currently active backend:

```kotlin
object BackendRegistry {
    private var currentBackendType: BackendType = BackendType.CPU
    private val backendRegistry = mutableMapOf<BackendType, Backend>()

    // Initialize all possible backends at startup
    init {
        // All backends are compiled into the binary
        backendRegistry[BackendType.CPU] = CPUBackend()

        // Conditionally register platform-specific backends
        if (Platform.isMacOS) {
            backendRegistry[BackendType.METAL] = MetalBackend()
        }

        if (hasVulkanSupport()) {
            backendRegistry[BackendType.VULKAN] = VulkanBackend()
        }

        // Load initial backend from config
        loadConfiguredBackend()
    }

    // Get current backend
    fun getCurrentBackend(): Backend = backendRegistry[currentBackendType]
        ?: backendRegistry[BackendType.CPU]!! // Fallback to CPU

    // Change backend at runtime
    fun setBackend(type: BackendType): Boolean {
        if (backendRegistry.containsKey(type)) {
            currentBackendType = type
            saveBackendConfig(type)
            return true
        }
        return false
    }

    // Load backend from configuration
    private fun loadConfiguredBackend() {
        val configFile = File("emberml_config.json")
        if (configFile.exists()) {
            try {
                val config = Json.decodeFromString<EmberConfig>(configFile.readText())
                val requestedType = config.backendType
                if (backendRegistry.containsKey(requestedType)) {
                    currentBackendType = requestedType
                }
            } catch (e: Exception) {
                // Log error and use default
            }
        }
    }

    // Save backend configuration
    private fun saveBackendConfig(type: BackendType) {
        try {
            val config = EmberConfig(backendType = type)
            File("emberml_config.json").writeText(Json.encodeToString(config))
        } catch (e: Exception) {
            // Log error
        }
    }

    // Check for configuration changes periodically
    fun startConfigWatcher() {
        GlobalScope.launch {
            while (true) {
                delay(5000) // Check every 5 seconds
                loadConfiguredBackend()
            }
        }
    }
}
```

### 2. Backend Interface

All backends implement a common interface that defines the operations they must support:

```kotlin
interface Backend {
    fun name(): String
    fun isAvailable(): Boolean
    fun priority(): Int  // Higher priority backends are preferred in auto-selection

    // Tensor operations
    fun matmul(a: EmberTensor, b: EmberTensor): EmberTensor
    fun add(a: EmberTensor, b: EmberTensor): EmberTensor
    // ... other operations
}
```

### 3. Actor-Based Backend Manager

Since Ember ML Kotlin uses an actor-based architecture, the backend manager is implemented as an actor:

```kotlin
class BackendManagerActor : Actor<BackendMessage> {
    private val backendRegistry = mutableMapOf<BackendType, Backend>()
    private var currentBackend: Backend = CPUBackend()

    override suspend fun receive(message: BackendMessage) {
        when (message) {
            is GetCurrentBackend -> sender.send(CurrentBackendResponse(currentBackend))
            is SetBackend -> {
                val success = setBackend(message.type)
                sender.send(SetBackendResponse(success))
            }
            is AutoSelectBackend -> {
                val selected = autoSelectBackend()
                sender.send(AutoSelectBackendResponse(selected))
            }
        }
    }

    private fun setBackend(type: BackendType): Boolean {
        backendRegistry[type]?.let {
            if (it.isAvailable()) {
                currentBackend = it
                saveConfig(type)
                return true
            }
        }
        return false
    }

    private fun autoSelectBackend(): BackendType {
        // Select highest priority available backend
        return backendRegistry.entries
            .filter { it.value.isAvailable() }
            .maxByOrNull { it.value.priority() }
            ?.key ?: BackendType.CPU
    }
}
```

### 4. Platform-Specific Backend Implementations

For platform-specific implementations, Kotlin's `expect/actual` mechanism is used:

```kotlin
// In common code
expect object PlatformBackend {
    fun createNativeBackend(): Backend
}

// In macOS-specific code
actual object PlatformBackend {
    actual fun createNativeBackend(): Backend = MetalBackend()
}

// In Linux-specific code
actual object PlatformBackend {
    actual fun createNativeBackend(): Backend = VulkanBackend()
}
```

## Usage Examples

### Basic Backend Selection

```kotlin
// Get the current backend
val backend = BackendRegistry.getCurrentBackend()

// Set a specific backend
BackendRegistry.setBackend(BackendType.METAL)

// Auto-select the best backend
val selectedBackend = BackendRegistry.autoSelectBackend()
```

### Actor-Based Backend Selection

```kotlin
// Create a backend manager actor
val backendManager = actorSystem.spawn<BackendManagerActor>()

// Get the current backend
val currentBackend = backendManager.ask(GetCurrentBackend()).await()

// Set a specific backend
val success = backendManager.ask(SetBackend(BackendType.METAL)).await()

// Auto-select the best backend
val selectedBackend = backendManager.ask(AutoSelectBackend()).await()
```

### Configuration-Based Selection

The configuration file (emberml_config.json) would look like this:

```json
{
    "backendType": "METAL",
    "tensorCacheSize": 1000,
    "otherSettings": {
        "useThreads": 4,
        "enableLogging": false
    }
}
```

## Benefits of This Approach

1. **Works Everywhere**: No reliance on dynamic loading or shared libraries
2. **Runtime Flexibility**: Can change backends without restarting
3. **User Preference**: Remembers the user's preferred backend
4. **Automatic Selection**: Can choose the best backend based on hardware
5. **Actor Integration**: Fits seamlessly with the actor-based architecture

## Limitations and Considerations

1. **Binary Size**: All backend implementations are included in the binary
2. **Memory Usage**: All backends are loaded at startup
3. **Configuration File**: Requires file system access for persistence
4. **Hot Reloading**: Periodic checking may introduce overhead

## Future Enhancements

1. **Backend Plugins**: Explore ways to load backends as plugins on platforms that support it
2. **Remote Backends**: Support for remote computation backends over network
3. **Backend Composition**: Allow using multiple backends simultaneously for different operations
4. **Dynamic Compilation**: JIT compilation of specialized kernels for specific operations
