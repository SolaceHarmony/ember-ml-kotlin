package ai.solace.emberml.testing

import kotlin.js.Date

/**
 * JavaScript implementation of cross-platform time measurement
 */
actual fun getCurrentTimeMs(): Long = Date.now().toLong()