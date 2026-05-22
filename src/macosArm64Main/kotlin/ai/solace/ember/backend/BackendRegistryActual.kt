package ai.solace.ember.backend

import ai.solace.ember.backend.metal.MetalBackend

/**
 * Registers the macOS-shipping backends: the default CPU backend (using
 * native Kotlin storage for every supported DType) and Metal (GPU).
 * The CPU backend is selected as the default when no backend has been
 * explicitly set.
 */
actual fun initializePlatformBackends() {
    BackendRegistry.registerBackend("cpu", DefaultCpuBackend())
    BackendRegistry.registerBackend("metal", MetalBackend())
    if (BackendRegistry.getBackend("cpu") != null) {
        BackendRegistry.setBackend("cpu")
    }
}

/**
 * Selects the highest-performance backend available on Apple hardware:
 * Metal when reachable, otherwise the native CPU backend.
 */
actual fun autoSelectBackend(): String {
    BackendRegistry.ensureBackendsInitialized()
    val metalBackend = BackendRegistry.getBackend("metal") as? MetalBackend
    if (metalBackend?.isAvailable() == true) {
        BackendRegistry.setBackend("metal")
        return "metal"
    }
    BackendRegistry.setBackend("cpu")
    return "cpu"
}
