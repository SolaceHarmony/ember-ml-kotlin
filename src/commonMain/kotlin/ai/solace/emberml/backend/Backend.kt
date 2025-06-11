package ai.solace.emberml.backend

import ai.solace.emberml.tensor.common.EmberDType

/**
 * Interface for all backend implementations.
 * This is the core interface that all backends must implement.
 */
interface Backend {
    /**
     * Creates a tensor from the given data.
     *
     * @param data The data to create the tensor from.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     * @return The backend-specific tensor.
     */
    fun createTensor(data: Any, shape: IntArray, dtype: EmberDType): Any

    /**
     * Gets the shape of a tensor.
     *
     * @param tensor The backend-specific tensor.
     * @return The shape of the tensor as an IntArray.
     */
    fun getTensorShape(tensor: Any): IntArray

    /**
     * Gets the data type of a tensor.
     *
     * @param tensor The backend-specific tensor.
     * @return The data type of the tensor.
     */
    fun getTensorDType(tensor: Any): EmberDType

    /**
     * Gets the device where a tensor is stored.
     *
     * @param tensor The backend-specific tensor.
     * @return The device where the tensor is stored.
     */
    fun getTensorDevice(tensor: Any): String

    /**
     * Adds two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of the addition.
     */
    fun add(a: Any, b: Any): Any

    /**
     * Subtracts one tensor from another.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of the subtraction.
     */
    fun subtract(a: Any, b: Any): Any

    /**
     * Multiplies two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of the multiplication.
     */
    fun multiply(a: Any, b: Any): Any

    /**
     * Divides one tensor by another.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of the division.
     */
    fun divide(a: Any, b: Any): Any

    /**
     * Performs matrix multiplication of two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of the matrix multiplication.
     */
    fun matmul(a: Any, b: Any): Any

    /**
     * Casts a tensor to a different data type.
     *
     * @param tensor The tensor to cast.
     * @param dtype The target data type.
     * @return The tensor with the new data type.
     */
    fun cast(tensor: Any, dtype: EmberDType): Any

    /**
     * Reshapes a tensor to a new shape.
     *
     * @param tensor The tensor to reshape.
     * @param newShape The new shape.
     * @return The reshaped tensor.
     */
    fun reshape(tensor: Any, newShape: IntArray): Any

    /**
     * Transposes a tensor.
     *
     * @param tensor The tensor to transpose.
     * @param axes The permutation of the dimensions. If null, reverses the dimensions.
     * @return The transposed tensor.
     */
    fun transpose(tensor: Any, axes: IntArray? = null): Any

    /**
     * Moves a tensor to a different device.
     *
     * @param tensor The tensor to move.
     * @param device The target device.
     * @return The tensor on the new device.
     */
    fun toDevice(tensor: Any, device: String): Any

    /**
     * Gets a list of available devices.
     *
     * @return A list of available devices.
     */
    fun getAvailableDevices(): List<String>

    /**
     * Sets the default device for tensor operations.
     *
     * @param device The device to use as the default.
     */
    fun setDefaultDevice(device: String)

    /**
     * Gets the default device for tensor operations.
     *
     * @return The default device.
     */
    fun getDefaultDevice(): String
}