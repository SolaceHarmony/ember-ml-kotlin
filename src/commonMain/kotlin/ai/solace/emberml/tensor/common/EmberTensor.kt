package ai.solace.emberml.tensor.common

import ai.solace.emberml.tensor.interfaces.TensorInterface

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
        // This would delegate to the backend implementation
        // For now, we'll just return a copy of this tensor with the new dtype
        return EmberTensor(
            shape = this.shape,
            dtype = dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = this.backendTensor
        )
    }

    /**
     * Reshapes the tensor to a new shape.
     *
     * @param newShape The new shape.
     * @return A new tensor with the same data but different shape.
     */
    override fun reshape(newShape: EmberShape): TensorInterface {
        // This would delegate to the backend implementation
        // For now, we'll just return a copy of this tensor with the new shape
        return EmberTensor(
            shape = newShape,
            dtype = this.dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = this.backendTensor
        )
    }

    /**
     * Transposes the tensor.
     *
     * @param axes The permutation of the dimensions. If null, reverses the dimensions.
     * @return A new tensor with the dimensions permuted.
     */
    override fun transpose(axes: IntArray?): TensorInterface {
        // This would delegate to the backend implementation
        // For now, we'll just return a copy of this tensor
        return EmberTensor(
            shape = this.shape,
            dtype = this.dtype,
            device = this.device,
            requiresGrad = this.requiresGrad,
            backendTensor = this.backendTensor
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
            // This would delegate to the current backend's tensor creation function
            // For now, we'll just return the data as is
            return data
        }
    }
}
