package ai.solace.emberml.tensor.common

import ai.solace.emberml.tensor.interfaces.TensorInterface
import ai.solace.emberml.backend.BackendRegistry

/**
 * The main tensor class that users interact with.
 * This is a backend-agnostic tensor implementation that delegates operations to the current backend.
 *
 * @property shape The shape of the tensor.
 * @property dtype The data type of the tensor.
 * @property device The device where the tensor is stored.
 * @property requiresGrad Whether the tensor requires gradients.
 * @property backendTensor The backend-specific tensor implementation.
 */
class EmberTensor(
    override val shape: EmberShape,
    override val dtype: EmberDType,
    override val device: String = "cpu",
    override val requiresGrad: Boolean = false,
    private val backendTensor: Any
) : TensorInterface {

    /**
     * Creates a tensor from a list of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: List<*>,
        dtype: EmberDType = float32,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        shape = inferShape(data),
        dtype = dtype,
        device = device,
        requiresGrad = requiresGrad,
        backendTensor = createBackendTensor(data, dtype, device, requiresGrad)
    )

    /**
     * Creates a tensor from an array of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: Array<*>,
        dtype: EmberDType = float32,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        data.toList(),
        dtype,
        device,
        requiresGrad
    )

    /**
     * Creates a tensor from a primitive array of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: IntArray,
        dtype: EmberDType = int32,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        data.toList(),
        dtype,
        device,
        requiresGrad
    )

    /**
     * Creates a tensor from a primitive array of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: FloatArray,
        dtype: EmberDType = float32,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        data.toList(),
        dtype,
        device,
        requiresGrad
    )

    /**
     * Creates a tensor from a primitive array of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: DoubleArray,
        dtype: EmberDType = float64,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        data.toList(),
        dtype,
        device,
        requiresGrad
    )

    /**
     * Creates a tensor from a primitive array of values.
     *
     * @param data The data to create the tensor from.
     * @param dtype The data type of the tensor.
     * @param device The device where the tensor is stored.
     * @param requiresGrad Whether the tensor requires gradients.
     */
    constructor(
        data: BooleanArray,
        dtype: EmberDType = bool,
        device: String = "cpu",
        requiresGrad: Boolean = false
    ) : this(
        data.toList(),
        dtype,
        device,
        requiresGrad
    )

    /**
     * Casts the tensor to a different data type.
     *
     * @param dtype The target data type.
     * @return A new tensor with the same data but different data type.
     */
    override fun cast(dtype: EmberDType): TensorInterface {
        // Delegate to the backend implementation
        val backend = BackendRegistry.getCurrentBackend()
        val newBackendTensor = backend.cast(this.backendTensor, dtype)

        return EmberTensor(
            shape = this.shape,
            dtype = dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = newBackendTensor
        )
    }

    /**
     * Reshapes the tensor to a new shape.
     *
     * @param newShape The new shape.
     * @return A new tensor with the same data but different shape.
     */
    override fun reshape(newShape: EmberShape): TensorInterface {
        // Delegate to the backend implementation
        val backend = BackendRegistry.getCurrentBackend()
        val newBackendTensor = backend.reshape(this.backendTensor, newShape.dimensions)

        return EmberTensor(
            shape = newShape,
            dtype = this.dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = newBackendTensor
        )
    }

    /**
     * Transposes the tensor.
     *
     * @param axes The permutation of the dimensions. If null, reverses the dimensions.
     * @return A new tensor with the dimensions permuted.
     */
    override fun transpose(axes: IntArray?): TensorInterface {
        // Delegate to the backend implementation
        val backend = BackendRegistry.getCurrentBackend()
        val newBackendTensor = backend.transpose(this.backendTensor, axes)

        // Get the new shape from the backend
        val newShapeDimensions = backend.getTensorShape(newBackendTensor)
        val newShape = EmberShape(newShapeDimensions)

        return EmberTensor(
            shape = newShape,
            dtype = this.dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = newBackendTensor
        )
    }

    /**
     * Converts the tensor to a string representation.
     *
     * @return A string representation of the tensor.
     */
    override fun toString(): String {
        return "EmberTensor(shape=$shape, dtype=$dtype, device=$device, requiresGrad=$requiresGrad)"
    }

    /**
     * Adds another tensor to this tensor.
     *
     * @param other The tensor to add.
     * @return The result of the addition.
     */
    operator fun plus(other: EmberTensor): EmberTensor {
        val backend = BackendRegistry.getCurrentBackend()
        val resultTensor = backend.add(this.backendTensor, other.backendTensor)
        val resultShape = EmberShape(backend.getTensorShape(resultTensor))
        val resultDType = backend.getTensorDType(resultTensor)
        val resultDevice = backend.getTensorDevice(resultTensor)

        return EmberTensor(
            shape = resultShape,
            dtype = resultDType,
            device = resultDevice,
            requiresGrad = this.requiresGrad || other.requiresGrad,
            backendTensor = resultTensor
        )
    }

    /**
     * Subtracts another tensor from this tensor.
     *
     * @param other The tensor to subtract.
     * @return The result of the subtraction.
     */
    operator fun minus(other: EmberTensor): EmberTensor {
        val backend = BackendRegistry.getCurrentBackend()
        val resultTensor = backend.subtract(this.backendTensor, other.backendTensor)
        val resultShape = EmberShape(backend.getTensorShape(resultTensor))
        val resultDType = backend.getTensorDType(resultTensor)
        val resultDevice = backend.getTensorDevice(resultTensor)

        return EmberTensor(
            shape = resultShape,
            dtype = resultDType,
            device = resultDevice,
            requiresGrad = this.requiresGrad || other.requiresGrad,
            backendTensor = resultTensor
        )
    }

    /**
     * Multiplies this tensor by another tensor.
     *
     * @param other The tensor to multiply by.
     * @return The result of the multiplication.
     */
    operator fun times(other: EmberTensor): EmberTensor {
        val backend = BackendRegistry.getCurrentBackend()
        val resultTensor = backend.multiply(this.backendTensor, other.backendTensor)
        val resultShape = EmberShape(backend.getTensorShape(resultTensor))
        val resultDType = backend.getTensorDType(resultTensor)
        val resultDevice = backend.getTensorDevice(resultTensor)

        return EmberTensor(
            shape = resultShape,
            dtype = resultDType,
            device = resultDevice,
            requiresGrad = this.requiresGrad || other.requiresGrad,
            backendTensor = resultTensor
        )
    }

    /**
     * Divides this tensor by another tensor.
     *
     * @param other The tensor to divide by.
     * @return The result of the division.
     */
    operator fun div(other: EmberTensor): EmberTensor {
        val backend = BackendRegistry.getCurrentBackend()
        val resultTensor = backend.divide(this.backendTensor, other.backendTensor)
        val resultShape = EmberShape(backend.getTensorShape(resultTensor))
        val resultDType = backend.getTensorDType(resultTensor)
        val resultDevice = backend.getTensorDevice(resultTensor)

        return EmberTensor(
            shape = resultShape,
            dtype = resultDType,
            device = resultDevice,
            requiresGrad = this.requiresGrad || other.requiresGrad,
            backendTensor = resultTensor
        )
    }

    /**
     * Performs matrix multiplication of this tensor with another tensor.
     *
     * @param other The tensor to multiply with.
     * @return The result of the matrix multiplication.
     */
    fun matmul(other: EmberTensor): EmberTensor {
        val backend = BackendRegistry.getCurrentBackend()
        val resultTensor = backend.matmul(this.backendTensor, other.backendTensor)
        val resultShape = EmberShape(backend.getTensorShape(resultTensor))
        val resultDType = backend.getTensorDType(resultTensor)
        val resultDevice = backend.getTensorDevice(resultTensor)

        return EmberTensor(
            shape = resultShape,
            dtype = resultDType,
            device = resultDevice,
            requiresGrad = this.requiresGrad || other.requiresGrad,
            backendTensor = resultTensor
        )
    }

    companion object {
        /**
         * Infers the shape of a tensor from a list of values.
         *
         * @param data The data to infer the shape from.
         * @return The inferred shape.
         */
        private fun inferShape(data: List<*>): EmberShape {
            val dimensions = mutableListOf<Int>()
            var current: Any? = data

            while (current is List<*> && current.isNotEmpty()) {
                dimensions.add(current.size)
                current = current.firstOrNull()
            }

            return EmberShape(dimensions.toIntArray())
        }

        /**
         * Creates a backend-specific tensor from a list of values.
         *
         * @param data The data to create the tensor from.
         * @param dtype The data type of the tensor.
         * @param device The device where the tensor is stored.
         * @param requiresGrad Whether the tensor requires gradients.
         * @return The backend-specific tensor.
         */
        private fun createBackendTensor(
            data: List<*>,
            dtype: EmberDType,
            device: String,
            requiresGrad: Boolean
        ): Any {
            // Delegate to the current backend's tensor creation function
            val backend = BackendRegistry.getCurrentBackend()
            val shape = inferShape(data).dimensions
            return backend.createTensor(data, shape, dtype)
        }
    }
}
