package ai.solace.emberml.backend

import ai.solace.emberml.tensor.common.EmberDType
import ai.solace.emberml.backend.storage.TensorStorage
import kotlin.math.*

/**
 * Mathematical functions for the OptimizedMegaTensorBackend.
 * 
 * This class provides element-wise mathematical operations that were missing
 * in the original implementation, including trigonometric, exponential, 
 * and other mathematical functions.
 */
class MathematicalOperations(private val backend: OptimizedMegaTensorBackend) {
    
    /**
     * Applies sine function element-wise.
     */
    fun sin(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> sin(value)
                is Double -> sin(value)
                is Int -> sin(value.toDouble())
                is Long -> sin(value.toDouble())
                is UByte -> sin(value.toDouble())
                else -> throw IllegalArgumentException("Sin operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies cosine function element-wise.
     */
    fun cos(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> cos(value)
                is Double -> cos(value)
                is Int -> cos(value.toDouble())
                is Long -> cos(value.toDouble())
                is UByte -> cos(value.toDouble())
                else -> throw IllegalArgumentException("Cos operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies tangent function element-wise.
     */
    fun tan(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> tan(value)
                is Double -> tan(value)
                is Int -> tan(value.toDouble())
                is Long -> tan(value.toDouble())
                is UByte -> tan(value.toDouble())
                else -> throw IllegalArgumentException("Tan operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies exponential function element-wise.
     */
    fun exp(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> exp(value)
                is Double -> exp(value)
                is Int -> exp(value.toDouble())
                is Long -> exp(value.toDouble())
                is UByte -> exp(value.toDouble())
                else -> throw IllegalArgumentException("Exp operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies natural logarithm function element-wise.
     */
    fun log(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> {
                    if (value <= 0f) throw ArithmeticException("Log of non-positive number: $value")
                    ln(value)
                }
                is Double -> {
                    if (value <= 0.0) throw ArithmeticException("Log of non-positive number: $value")
                    ln(value)
                }
                is Int -> {
                    if (value <= 0) throw ArithmeticException("Log of non-positive number: $value")
                    ln(value.toDouble())
                }
                is Long -> {
                    if (value <= 0L) throw ArithmeticException("Log of non-positive number: $value")
                    ln(value.toDouble())
                }
                is UByte -> {
                    if (value.toInt() == 0) throw ArithmeticException("Log of zero")
                    ln(value.toDouble())
                }
                else -> throw IllegalArgumentException("Log operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies square root function element-wise.
     */
    fun sqrt(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> {
                    if (value < 0f) throw ArithmeticException("Square root of negative number: $value")
                    sqrt(value)
                }
                is Double -> {
                    if (value < 0.0) throw ArithmeticException("Square root of negative number: $value")
                    sqrt(value)
                }
                is Int -> {
                    if (value < 0) throw ArithmeticException("Square root of negative number: $value")
                    sqrt(value.toDouble())
                }
                is Long -> {
                    if (value < 0L) throw ArithmeticException("Square root of negative number: $value")
                    sqrt(value.toDouble())
                }
                is UByte -> sqrt(value.toDouble())
                else -> throw IllegalArgumentException("Sqrt operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies power function element-wise.
     */
    fun pow(tensor: Any, exponent: Double): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> value.pow(exponent.toFloat())
                is Double -> value.pow(exponent)
                is Int -> value.toDouble().pow(exponent)
                is Long -> value.toDouble().pow(exponent)
                is UByte -> value.toDouble().pow(exponent)
                else -> throw IllegalArgumentException("Pow operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Applies absolute value function element-wise.
     */
    fun abs(tensor: Any): Any {
        return applyUnaryMathFunction(tensor) { value ->
            when (value) {
                is Float -> abs(value)
                is Double -> abs(value)
                is Int -> abs(value)
                is Long -> abs(value)
                is UByte -> value // UByte is always positive
                is Boolean -> if (value) 1 else 0
                else -> throw IllegalArgumentException("Abs operation not supported for type: ${value::class.simpleName}")
            }
        }
    }
    
    /**
     * Element-wise greater than comparison.
     */
    fun greaterThan(tensor1: Any, tensor2: Any): Any {
        return applyBinaryComparisonFunction(tensor1, tensor2) { a, b ->
            when {
                a is Float && b is Float -> a > b
                a is Double && b is Double -> a > b
                a is Int && b is Int -> a > b
                a is Long && b is Long -> a > b
                a is UByte && b is UByte -> a > b
                a is Boolean && b is Boolean -> a && !b
                else -> {
                    // Convert to comparable values
                    val aVal = convertToDouble(a)
                    val bVal = convertToDouble(b)
                    aVal > bVal
                }
            }
        }
    }
    
    /**
     * Element-wise less than comparison.
     */
    fun lessThan(tensor1: Any, tensor2: Any): Any {
        return applyBinaryComparisonFunction(tensor1, tensor2) { a, b ->
            when {
                a is Float && b is Float -> a < b
                a is Double && b is Double -> a < b
                a is Int && b is Int -> a < b
                a is Long && b is Long -> a < b
                a is UByte && b is UByte -> a < b
                a is Boolean && b is Boolean -> !a && b
                else -> {
                    // Convert to comparable values
                    val aVal = convertToDouble(a)
                    val bVal = convertToDouble(b)
                    aVal < bVal
                }
            }
        }
    }
    
    /**
     * Element-wise equality comparison.
     */
    fun equal(tensor1: Any, tensor2: Any): Any {
        return applyBinaryComparisonFunction(tensor1, tensor2) { a, b ->
            when {
                a is Float && b is Float -> a == b
                a is Double && b is Double -> a == b
                a is Int && b is Int -> a == b
                a is Long && b is Long -> a == b
                a is UByte && b is UByte -> a == b
                a is Boolean && b is Boolean -> a == b
                else -> {
                    // Convert to comparable values
                    val aVal = convertToDouble(a)
                    val bVal = convertToDouble(b)
                    aVal == bVal
                }
            }
        }
    }
    
    // Helper functions
    
    private fun applyUnaryMathFunction(tensor: Any, operation: (Any) -> Any): Any {
        val t = tensor as OptimizedMegaTensorBackend.OptimizedMegaTensor
        
        // Determine output data type (usually promote to float/double for math operations)
        val outputDType = when (t.dtype) {
            EmberDType.BOOL, EmberDType.UINT8, EmberDType.INT32 -> EmberDType.FLOAT64
            EmberDType.INT64 -> EmberDType.FLOAT64
            EmberDType.FLOAT32 -> EmberDType.FLOAT32
            EmberDType.FLOAT64 -> EmberDType.FLOAT64
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(outputDType, t.size)
        
        for (i in 0 until t.size) {
            val inputValue = getStorageValue(t.storage, i)
            val outputValue = operation(inputValue)
            setStorageValue(resultStorage, i, outputValue, outputDType)
        }
        
        return OptimizedMegaTensorBackend.OptimizedMegaTensor(resultStorage, t.shape, t.device)
    }
    
    private fun applyBinaryComparisonFunction(tensor1: Any, tensor2: Any, operation: (Any, Any) -> Boolean): Any {
        val t1 = tensor1 as OptimizedMegaTensorBackend.OptimizedMegaTensor
        val t2 = tensor2 as OptimizedMegaTensorBackend.OptimizedMegaTensor
        
        if (!t1.shape.contentEquals(t2.shape)) {
            throw IllegalArgumentException("Shape mismatch: ${t1.shape.contentToString()} vs ${t2.shape.contentToString()}")
        }
        
        val resultStorage = TensorStorage.createOptimalStorage(EmberDType.BOOL, t1.size)
        
        for (i in 0 until t1.size) {
            val value1 = getStorageValue(t1.storage, i)
            val value2 = getStorageValue(t2.storage, i)
            val result = operation(value1, value2)
            setStorageValue(resultStorage, i, result, EmberDType.BOOL)
        }
        
        return OptimizedMegaTensorBackend.OptimizedMegaTensor(resultStorage, t1.shape, t1.device)
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
    
    private fun setStorageValue(storage: TensorStorage, index: Int, value: Any, dtype: EmberDType) {
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
                // For MegaNumber storage, we would need to convert appropriately
                throw UnsupportedOperationException("MegaNumber storage not yet implemented for math operations")
            }
        }
    }
    
    private fun convertToBoolean(value: Any): Boolean {
        return when (value) {
            is Boolean -> value
            is Number -> value.toDouble() != 0.0
            else -> false
        }
    }
    
    private fun convertToUByte(value: Any): UByte {
        return when (value) {
            is Number -> value.toInt().coerceIn(0, 255).toUByte()
            is Boolean -> if (value) 1u else 0u
            else -> 0u
        }
    }
    
    private fun convertToInt(value: Any): Int {
        return when (value) {
            is Number -> value.toInt()
            is Boolean -> if (value) 1 else 0
            else -> 0
        }
    }
    
    private fun convertToLong(value: Any): Long {
        return when (value) {
            is Number -> value.toLong()
            is Boolean -> if (value) 1L else 0L
            else -> 0L
        }
    }
    
    private fun convertToFloat(value: Any): Float {
        return when (value) {
            is Number -> value.toFloat()
            is Boolean -> if (value) 1f else 0f
            else -> 0f
        }
    }
    
    private fun convertToDouble(value: Any): Double {
        return when (value) {
            is Number -> value.toDouble()
            is Boolean -> if (value) 1.0 else 0.0
            else -> 0.0
        }
    }
}