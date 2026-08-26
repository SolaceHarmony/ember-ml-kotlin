package ai.solace.ember.backend

import ai.solace.ember.backend.metal.MetalBackend

/**
 * Registers the platform-shipping backends: default CPU and Metal backend.
 */
actual fun initializePlatformBackends() {
    BackendRegistry.registerBackend("cpu", DefaultCpuBackend())
    BackendRegistry.registerBackend("metal", MetalBackend())
    if (BackendRegistry.getBackend("cpu") != null) {
        BackendRegistry.setBackend("cpu")
    }
}

/**
 * Automatically selects the best available backend.
 */
actual fun autoSelectBackend(): String {
    BackendRegistry.ensureBackendsInitialized()
    val metalBackend = BackendRegistry.getBackend("metal") as? MetalBackend
    if (metalBackend?.isAvailable() == true) {
        BackendRegistry.setBackend("metal")
        return "metal"
    }
    if (BackendRegistry.getBackend("cpu") != null) {
        BackendRegistry.setBackend("cpu")
        return "cpu"
    }
    return BackendRegistry.getAvailableBackends().firstOrNull()
        ?: throw IllegalStateException("No backends are registered for this target")
}
