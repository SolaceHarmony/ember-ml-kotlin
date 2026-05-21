package ai.solace.ember.backend

/**
 * No backends ship for this target. Callers that need a concrete backend
 * must register one explicitly before reading from the registry.
 */
actual fun initializePlatformBackends() {
}

/**
 * No native backends ship for this target. Falls back to the first
 * explicitly registered backend, or throws if none have been registered.
 */
actual fun autoSelectBackend(): String {
    return BackendRegistry.getAvailableBackends().firstOrNull()
        ?: throw IllegalStateException("No backends are registered for this target")
}
