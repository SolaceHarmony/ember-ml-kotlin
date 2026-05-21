package ai.solace.ember.backend

import ai.solace.ember.backend.metal.MetalBackend

/**
 * Registers the macOS-shipping backends: MegaTensor (CPU) and Metal (GPU).
 * MegaTensor is selected as the default when no backend has been explicitly set.
 */
actual fun initializePlatformBackends() {
    BackendRegistry.registerBackend("mega", MegaTensorBackend())
    BackendRegistry.registerBackend("metal", MetalBackend())
    if (BackendRegistry.getBackend("mega") != null) {
        BackendRegistry.setBackend("mega")
    }
}

/**
 * Selects the highest-performance backend available on Apple hardware:
 * Metal when reachable, otherwise MegaTensor.
 */
actual fun autoSelectBackend(): String {
    BackendRegistry.ensureBackendsInitialized()
    val metalBackend = BackendRegistry.getBackend("metal") as? MetalBackend
    if (metalBackend?.isAvailable() == true) {
        BackendRegistry.setBackend("metal")
        return "metal"
    }
    BackendRegistry.setBackend("mega")
    return "mega"
}
