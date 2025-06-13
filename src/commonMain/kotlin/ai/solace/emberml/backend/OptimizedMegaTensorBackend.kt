package ai.solace.emberml.backend

import ai.solace.emberml.tensor.common.EmberDType
import ai.solace.emberml.tensor.bitwise.MegaNumber
import ai.solace.emberml.tensor.bitwise.MegaFloat
import ai.solace.emberml.tensor.bitwise.MegaInteger
import ai.solace.emberml.backend.storage.TensorStorage

/**
 * An optimized backend implementation that uses hybrid storage for tensor operations.
 * 
 * This backend addresses the critical 32-bit limb inefficiency by:
 * - Using efficient native storage for common data types (Boolean, UINT8, INT32, etc.)
 * - Falling back to MegaNumber storage only when arbitrary precision is needed
 * - Providing significant memory reductions: 256x for booleans, 32x for UINT8, etc.
 * 
 * This replaces the previous MegaTensorBackend which forced all data types into
 * expensive MegaNumber storage regardless of their natural size.
 */
class OptimizedMegaTensorBackend : Backend {
    // The default device for tensor operations
    private var defaultDevice: String = "cpu"

    /**
     * An optimized tensor implementation using hybrid storage.
     * This class wraps a TensorStorage object that uses the most efficient
     * storage strategy for the given data type.
     *
     * @property storage The hybrid storage implementation
     * @property shape The shape of the tensor
     * @property device The device where the tensor is stored
     */
    data class OptimizedMegaTensor(
        val storage: TensorStorage,
        val shape: IntArray,
        val device: String
    ) {
        val dtype: EmberDType get() = storage.dtype
        val size: Int get() = storage.size
        
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is OptimizedMegaTensor) return false

            if (storage != other.storage) return false
            if (!shape.contentEquals(other.shape)) return false
            if (device != other.device) return false

            return true
        }

        override fun hashCode(): Int {
            var result = storage.hashCode()
            result = 31 * result + shape.contentHashCode()
            result = 31 * result + device.hashCode()
            return result
        }
    }

    /**
     * Creates a tensor from the given data using optimal storage.
     *
     * @param data The data to create the tensor from.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     * @return The backend-specific tensor with optimized storage.
     */
    override fun createTensor(data: Any, shape: IntArray, dtype: EmberDType): Any {
        val totalSize = shape.fold(1) { acc, dim -> acc * dim }
        val storage = TensorStorage.createOptimalStorage(dtype, totalSize)
        
        // Fill the storage with data
        when (data) {
            is List<*> -> fillStorageFromList(storage, data, dtype)
            is Array<*> -> fillStorageFromArray(storage, data, dtype)
            is IntArray -> fillStorageFromIntArray(storage, data, dtype)
            is FloatArray -> fillStorageFromFloatArray(storage, data, dtype)
            is DoubleArray -> fillStorageFromDoubleArray(storage, data, dtype)
            is BooleanArray -> fillStorageFromBooleanArray(storage, data, dtype)
            else -> throw IllegalArgumentException("Unsupported data type: ${data::class.simpleName}")
        }

        return OptimizedMegaTensor(storage, shape, defaultDevice)
    }

    /**
     * Gets the shape of a tensor.
     */
    override fun getTensorShape(tensor: Any): IntArray {
        if (tensor !is OptimizedMegaTensor) {
            throw IllegalArgumentException("Expected OptimizedMegaTensor, got ${tensor::class.simpleName}")
        }
        return tensor.shape
    }

    /**
     * Gets the data type of a tensor.
     */
    override fun getTensorDType(tensor: Any): EmberDType {
        if (tensor !is OptimizedMegaTensor) {
            throw IllegalArgumentException("Expected OptimizedMegaTensor, got ${tensor::class.simpleName}")
        }
        return tensor.dtype
    }

    /**
     * Gets the device of a tensor.
     */
    override fun getTensorDevice(tensor: Any): String {
        if (tensor !is OptimizedMegaTensor) {
            throw IllegalArgumentException("Expected OptimizedMegaTensor, got ${tensor::class.simpleName}")
        }
        return tensor.device
    }

    /**
     * Adds two tensors element-wise.
     */
    override fun add(tensor1: Any, tensor2: Any): Any {
        val t1 = tensor1 as OptimizedMegaTensor
        val t2 = tensor2 as OptimizedMegaTensor
        
        if (!t1.shape.contentEquals(t2.shape)) {
            throw IllegalArgumentException("Shape mismatch: ${t1.shape.contentToString()} vs ${t2.shape.contentToString()}")
        }
        
        if (t1.dtype != t2.dtype) {
            throw IllegalArgumentException("Data type mismatch: ${t1.dtype} vs ${t2.dtype}")
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(t1.dtype, t1.size)
        performElementWiseOperation(t1.storage, t2.storage, resultStorage) { a, b ->
            when (a) {
                is Boolean -> (a || b as Boolean) // For boolean, OR operation
                is UByte -> ((a.toInt() + (b as UByte).toInt()).toUByte())
                is Int -> (a + b as Int)
                is Long -> (a + b as Long)
                is Float -> (a + b as Float)
                is Double -> (a + b as Double)
                else -> throw IllegalArgumentException("Unsupported type for addition: ${a::class.simpleName}")
            }
        }
        
        return OptimizedMegaTensor(resultStorage, t1.shape, t1.device)
    }

    /**
     * Subtracts tensor2 from tensor1 element-wise.
     */
    override fun subtract(tensor1: Any, tensor2: Any): Any {
        val t1 = tensor1 as OptimizedMegaTensor
        val t2 = tensor2 as OptimizedMegaTensor
        
        if (!t1.shape.contentEquals(t2.shape)) {
            throw IllegalArgumentException("Shape mismatch: ${t1.shape.contentToString()} vs ${t2.shape.contentToString()}")
        }
        
        if (t1.dtype != t2.dtype) {
            throw IllegalArgumentException("Data type mismatch: ${t1.dtype} vs ${t2.dtype}")
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(t1.dtype, t1.size)
        performElementWiseOperation(t1.storage, t2.storage, resultStorage) { a, b ->
            when (a) {
                is Boolean -> (a && !(b as Boolean)) // For boolean, AND NOT operation
                is UByte -> ((a.toInt() - (b as UByte).toInt()).coerceAtLeast(0).toUByte())
                is Int -> (a - b as Int)
                is Long -> (a - b as Long)
                is Float -> (a - b as Float)
                is Double -> (a - b as Double)
                else -> throw IllegalArgumentException("Unsupported type for subtraction: ${a::class.simpleName}")
            }
        }
        
        return OptimizedMegaTensor(resultStorage, t1.shape, t1.device)
    }

    /**
     * Multiplies two tensors element-wise.
     */
    override fun multiply(tensor1: Any, tensor2: Any): Any {
        val t1 = tensor1 as OptimizedMegaTensor
        val t2 = tensor2 as OptimizedMegaTensor
        
        if (!t1.shape.contentEquals(t2.shape)) {
            throw IllegalArgumentException("Shape mismatch: ${t1.shape.contentToString()} vs ${t2.shape.contentToString()}")
        }
        
        if (t1.dtype != t2.dtype) {
            throw IllegalArgumentException("Data type mismatch: ${t1.dtype} vs ${t2.dtype}")
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(t1.dtype, t1.size)
        performElementWiseOperation(t1.storage, t2.storage, resultStorage) { a, b ->
            when (a) {
                is Boolean -> (a && b as Boolean) // For boolean, AND operation
                is UByte -> ((a.toInt() * (b as UByte).toInt()).coerceAtMost(255).toUByte())
                is Int -> (a * b as Int)
                is Long -> (a * b as Long)
                is Float -> (a * b as Float)
                is Double -> (a * b as Double)
                else -> throw IllegalArgumentException("Unsupported type for multiplication: ${a::class.simpleName}")
            }
        }
        
        return OptimizedMegaTensor(resultStorage, t1.shape, t1.device)
    }

    /**
     * Divides tensor1 by tensor2 element-wise.
     */
    override fun divide(tensor1: Any, tensor2: Any): Any {
        val t1 = tensor1 as OptimizedMegaTensor
        val t2 = tensor2 as OptimizedMegaTensor
        
        if (!t1.shape.contentEquals(t2.shape)) {
            throw IllegalArgumentException("Shape mismatch: ${t1.shape.contentToString()} vs ${t2.shape.contentToString()}")
        }
        
        if (t1.dtype != t2.dtype) {
            throw IllegalArgumentException("Data type mismatch: ${t1.dtype} vs ${t2.dtype}")
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(t1.dtype, t1.size)
        performElementWiseOperation(t1.storage, t2.storage, resultStorage) { a, b ->
            when (a) {
                is Boolean -> a // For boolean, just return first value
                is UByte -> {
                    val bVal = (b as UByte).toInt()
                    if (bVal == 0) throw ArithmeticException("Division by zero")
                    (a.toInt() / bVal).toUByte()
                }
                is Int -> {
                    val bVal = b as Int
                    if (bVal == 0) throw ArithmeticException("Division by zero")
                    a / bVal
                }
                is Long -> {
                    val bVal = b as Long
                    if (bVal == 0L) throw ArithmeticException("Division by zero")
                    a / bVal
                }
                is Float -> {
                    val bVal = b as Float
                    if (bVal == 0f) throw ArithmeticException("Division by zero")
                    a / bVal
                }
                is Double -> {
                    val bVal = b as Double
                    if (bVal == 0.0) throw ArithmeticException("Division by zero")
                    a / bVal
                }
                else -> throw IllegalArgumentException("Unsupported type for division: ${a::class.simpleName}")
            }
        }
        
        return OptimizedMegaTensor(resultStorage, t1.shape, t1.device)
    }

    // ===== ADDITIONAL TENSOR OPERATIONS =====
    
    /**
     * Sums all elements in the tensor, returning a scalar tensor.
     */
    fun sum(tensor: Any): Any {
        val t = tensor as OptimizedMegaTensor
        
        when (t.storage) {
            is TensorStorage.PackedBooleanStorage -> {
                var count = 0
                for (i in 0 until t.size) {
                    if (t.storage.get(i)) count++
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT32, 1)
                (resultStorage as TensorStorage.NativeIntStorage).set(0, count)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeUByteStorage -> {
                var sum = 0
                for (i in 0 until t.size) {
                    sum += t.storage.get(i).toInt()
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT32, 1)
                (resultStorage as TensorStorage.NativeIntStorage).set(0, sum)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeIntStorage -> {
                var sum = 0L
                for (i in 0 until t.size) {
                    sum += t.storage.get(i)
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT64, 1)
                (resultStorage as TensorStorage.NativeLongStorage).set(0, sum)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeLongStorage -> {
                var sum = 0L
                for (i in 0 until t.size) {
                    sum += t.storage.get(i)
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT64, 1)
                (resultStorage as TensorStorage.NativeLongStorage).set(0, sum)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeFloatStorage -> {
                var sum = 0.0
                for (i in 0 until t.size) {
                    sum += t.storage.get(i)
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT64, 1)
                (resultStorage as TensorStorage.NativeDoubleStorage).set(0, sum)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeDoubleStorage -> {
                var sum = 0.0
                for (i in 0 until t.size) {
                    sum += t.storage.get(i)
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT64, 1)
                (resultStorage as TensorStorage.NativeDoubleStorage).set(0, sum)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            else -> throw IllegalArgumentException("Unsupported storage type for sum operation")
        }
    }
    
    /**
     * Computes the mean of all elements in the tensor.
     */
    fun mean(tensor: Any): Any {
        val t = tensor as OptimizedMegaTensor
        val sumTensor = sum(tensor) as OptimizedMegaTensor
        
        // Create a tensor with the size value to divide by
        val sizeStorage = TensorStorage.createOptimalStorage(sumTensor.dtype, 1)
        when (sizeStorage) {
            is TensorStorage.NativeIntStorage -> sizeStorage.set(0, t.size)
            is TensorStorage.NativeLongStorage -> sizeStorage.set(0, t.size.toLong())
            is TensorStorage.NativeDoubleStorage -> sizeStorage.set(0, t.size.toDouble())
            else -> throw IllegalArgumentException("Unsupported storage type for mean operation")
        }
        val sizeTensor = OptimizedMegaTensor(sizeStorage, intArrayOf(1), t.device)
        
        return divide(sumTensor, sizeTensor)
    }
    
    /**
     * Finds the minimum value in the tensor.
     */
    fun min(tensor: Any): Any {
        val t = tensor as OptimizedMegaTensor
        
        when (t.storage) {
            is TensorStorage.PackedBooleanStorage -> {
                var min = true
                for (i in 0 until t.size) {
                    val value = t.storage.get(i)
                    if (!value) {
                        min = false
                        break
                    }
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.BOOL, 1)
                (resultStorage as TensorStorage.PackedBooleanStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeUByteStorage -> {
                var min = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value < min) min = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.UINT8, 1)
                (resultStorage as TensorStorage.NativeUByteStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeIntStorage -> {
                var min = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value < min) min = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT32, 1)
                (resultStorage as TensorStorage.NativeIntStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeLongStorage -> {
                var min = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value < min) min = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT64, 1)
                (resultStorage as TensorStorage.NativeLongStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeFloatStorage -> {
                var min = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value < min) min = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT32, 1)
                (resultStorage as TensorStorage.NativeFloatStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeDoubleStorage -> {
                var min = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value < min) min = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT64, 1)
                (resultStorage as TensorStorage.NativeDoubleStorage).set(0, min)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            else -> throw IllegalArgumentException("Unsupported storage type for min operation")
        }
    }
    
    /**
     * Finds the maximum value in the tensor.
     */
    fun max(tensor: Any): Any {
        val t = tensor as OptimizedMegaTensor
        
        when (t.storage) {
            is TensorStorage.PackedBooleanStorage -> {
                var max = false
                for (i in 0 until t.size) {
                    val value = t.storage.get(i)
                    if (value) {
                        max = true
                        break
                    }
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.BOOL, 1)
                (resultStorage as TensorStorage.PackedBooleanStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeUByteStorage -> {
                var max = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value > max) max = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.UINT8, 1)
                (resultStorage as TensorStorage.NativeUByteStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeIntStorage -> {
                var max = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value > max) max = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT32, 1)
                (resultStorage as TensorStorage.NativeIntStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeLongStorage -> {
                var max = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value > max) max = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.INT64, 1)
                (resultStorage as TensorStorage.NativeLongStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeFloatStorage -> {
                var max = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value > max) max = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT32, 1)
                (resultStorage as TensorStorage.NativeFloatStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            is TensorStorage.NativeDoubleStorage -> {
                var max = t.storage.get(0)
                for (i in 1 until t.size) {
                    val value = t.storage.get(i)
                    if (value > max) max = value
                }
                val resultStorage = TensorStorage.createOptimalStorage(EmberDType.FLOAT64, 1)
                (resultStorage as TensorStorage.NativeDoubleStorage).set(0, max)
                return OptimizedMegaTensor(resultStorage, intArrayOf(1), t.device)
            }
            else -> throw IllegalArgumentException("Unsupported storage type for max operation")
        }
    }
    
    /**
     * Gets a specific element from the tensor at the given flat index.
     */
    fun getElement(tensor: Any, index: Int): Any {
        val t = tensor as OptimizedMegaTensor
        
        if (index < 0 || index >= t.size) {
            throw IndexOutOfBoundsException("Index $index out of bounds for tensor of size ${t.size}")
        }
        
        return getStorageValue(t.storage, index)
    }
    
    /**
     * Sets a specific element in the tensor at the given flat index.
     */
    fun setElement(tensor: Any, index: Int, value: Any): Any {
        val t = tensor as OptimizedMegaTensor
        
        if (index < 0 || index >= t.size) {
            throw IndexOutOfBoundsException("Index $index out of bounds for tensor of size ${t.size}")
        }
        
        // Create a new tensor with the updated value (immutable approach)
        val newStorage = when (t.storage) {
            is TensorStorage.PackedBooleanStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.PackedBooleanStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.NativeUByteStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.NativeUByteStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.NativeIntStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.NativeIntStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.NativeLongStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.NativeLongStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.NativeFloatStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.NativeFloatStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.NativeDoubleStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.NativeDoubleStorage(newData, t.size, t.dtype)
            }
            is TensorStorage.MegaNumberStorage -> {
                val newData = t.storage.data.copyOf()
                TensorStorage.MegaNumberStorage(newData, t.size, t.dtype)
            }
        }
        
        setStorageValue(newStorage, index, value, t.dtype)
        return OptimizedMegaTensor(newStorage, t.shape, t.device)
    }

    // Helper functions

    private fun fillStorageFromList(storage: TensorStorage, data: List<*>, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun fillStorageFromArray(storage: TensorStorage, data: Array<*>, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun fillStorageFromIntArray(storage: TensorStorage, data: IntArray, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun fillStorageFromFloatArray(storage: TensorStorage, data: FloatArray, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun fillStorageFromDoubleArray(storage: TensorStorage, data: DoubleArray, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun fillStorageFromBooleanArray(storage: TensorStorage, data: BooleanArray, dtype: EmberDType) {
        for (i in data.indices) {
            val value = data[i]
            setStorageValue(storage, i, value, dtype)
        }
    }

    private fun setStorageValue(storage: TensorStorage, index: Int, value: Any?, dtype: EmberDType) {
        when (storage) {
            is TensorStorage.PackedBooleanStorage -> {
                storage.set(index, convertToBoolean(value))
            }
            is TensorStorage.NativeUByteStorage -> {
                storage.set(index, convertToUByte(value))
            }
            is TensorStorage.NativeIntStorage -> {
                storage.set(index, convertToInt(value))
            }
            is TensorStorage.NativeLongStorage -> {
                storage.set(index, convertToLong(value))
            }
            is TensorStorage.NativeFloatStorage -> {
                storage.set(index, convertToFloat(value))
            }
            is TensorStorage.NativeDoubleStorage -> {
                storage.set(index, convertToDouble(value))
            }
            is TensorStorage.MegaNumberStorage -> {
                storage.set(index, convertToMegaNumber(value, dtype))
            }
        }
    }

    private fun getStorageValue(storage: TensorStorage, index: Int): Any {
        return when (storage) {
            is TensorStorage.PackedBooleanStorage -> storage.get(index)
            is TensorStorage.NativeUByteStorage -> storage.get(index)
            is TensorStorage.NativeIntStorage -> storage.get(index)
            is TensorStorage.NativeLongStorage -> storage.get(index)
            is TensorStorage.NativeFloatStorage -> storage.get(index)
            is TensorStorage.NativeDoubleStorage -> storage.get(index)
            is TensorStorage.MegaNumberStorage -> storage.get(index)
        }
    }

    private fun performElementWiseOperation(
        storage1: TensorStorage,
        storage2: TensorStorage,
        resultStorage: TensorStorage,
        operation: (Any, Any) -> Any
    ) {
        for (i in 0 until storage1.size) {
            val value1 = getStorageValue(storage1, i)
            val value2 = getStorageValue(storage2, i)
            val result = operation(value1, value2)
            setStorageValue(resultStorage, i, result, resultStorage.dtype)
        }
    }

    // Conversion helpers

    private fun convertToBoolean(value: Any?): Boolean {
        return when (value) {
            is Boolean -> value
            is Number -> value.toDouble() != 0.0
            is String -> value.toBoolean()
            else -> false
        }
    }

    private fun convertToUByte(value: Any?): UByte {
        return when (value) {
            is Number -> value.toInt().coerceIn(0, 255).toUByte()
            is Boolean -> if (value) 1u else 0u
            is String -> value.toInt().coerceIn(0, 255).toUByte()
            else -> 0u
        }
    }

    private fun convertToInt(value: Any?): Int {
        return when (value) {
            is Number -> value.toInt()
            is Boolean -> if (value) 1 else 0
            is String -> value.toInt()
            else -> 0
        }
    }

    private fun convertToLong(value: Any?): Long {
        return when (value) {
            is Number -> value.toLong()
            is Boolean -> if (value) 1L else 0L
            is String -> value.toLong()
            else -> 0L
        }
    }

    private fun convertToFloat(value: Any?): Float {
        return when (value) {
            is Number -> value.toFloat()
            is Boolean -> if (value) 1f else 0f
            is String -> value.toFloat()
            else -> 0f
        }
    }

    private fun convertToDouble(value: Any?): Double {
        return when (value) {
            is Number -> value.toDouble()
            is Boolean -> if (value) 1.0 else 0.0
            is String -> value.toDouble()
            else -> 0.0
        }
    }

    private fun convertToMegaNumber(value: Any?, dtype: EmberDType): MegaNumber {
        return when (value) {
            is Int -> when (dtype) {
                EmberDType.FLOAT32, EmberDType.FLOAT64 -> MegaFloat.fromValue(value.toDouble())
                else -> MegaInteger.fromValue(value)
            }
            is Long -> when (dtype) {
                EmberDType.FLOAT32, EmberDType.FLOAT64 -> MegaFloat.fromValue(value.toDouble())
                else -> MegaInteger.fromValue(value.toInt())
            }
            is Float -> MegaFloat.fromValue(value.toDouble())
            is Double -> MegaFloat.fromValue(value)
            is Boolean -> MegaInteger.fromValue(if (value) 1 else 0)
            is String -> when (dtype) {
                EmberDType.FLOAT32, EmberDType.FLOAT64 -> MegaFloat.fromValue(value.toDouble())
                else -> MegaInteger.fromValue(value.toInt())
            }
            else -> MegaInteger.fromValue(0)
        }
    }
}