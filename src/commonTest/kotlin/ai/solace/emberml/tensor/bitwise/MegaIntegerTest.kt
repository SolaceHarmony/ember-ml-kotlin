package ai.solace.emberml.tensor.bitwise

import kotlin.test.*

class MegaIntegerTest {

    @Test
    fun testDecimalStringConstructor() {
        // Test the constructor that takes a decimal string
        val intFromString = MegaInteger("12345")

        // Verify that the mantissa is correct
        assertEquals("12345", intFromString.toDecimalString())

        // Verify that the sign is correct
        assertFalse(intFromString.negative)

        // Test with a negative number
        val negativeInt = MegaInteger("-54321")
        assertEquals("-54321", negativeInt.toDecimalString())
        assertTrue(negativeInt.negative)

        // Test with zero
        val zero = MegaInteger("0")
        assertEquals("0", zero.toDecimalString())
        assertFalse(zero.negative)
    }

    @Test
    fun testExponentIsAlwaysZero() {
        // Create a MegaInteger with non-zero exponent
        val integer = MegaInteger(
            mantissa = longArrayOf(123),
            exponent = longArrayOf(456), // This should be ignored
            negative = false,
            isFloat = true, // This should be ignored
            exponentNegative = true // This should be ignored
        )

        // Verify that exponent is always zero
        assertEquals(1, integer.exponent.size)
        assertEquals(0, integer.exponent[0].toLong())

        // Verify that isFloat is always false
        assertFalse(integer.isFloat)

        // Verify that exponentNegative is always false
        assertFalse(integer.exponentNegative)
    }

    @Test
    fun testArithmeticOperations() {
        val a = MegaInteger("100")
        val b = MegaInteger("50")

        // Test addition
        val sum = a.add(b) as MegaInteger
        val sumStr = sum.toDecimalString()
        println("[DEBUG_LOG] Sum decimal string: $sumStr")
        assertEquals("150", sumStr)

        // Test subtraction
        val diff = a.sub(b) as MegaInteger
        val diffStr = diff.toDecimalString()
        println("[DEBUG_LOG] Difference decimal string: $diffStr")
        assertEquals("50", diffStr)

        // Test multiplication
        val product = a.mul(b) as MegaInteger
        val productStr = product.toDecimalString()
        println("[DEBUG_LOG] Product decimal string: $productStr")
        assertEquals("5000", productStr)

        // Test division
        val quotient = a.divide(b) as MegaInteger
        val quotientStr = quotient.toDecimalString()
        println("[DEBUG_LOG] Quotient decimal string: $quotientStr")
        assertEquals("2", quotientStr)
    }

    @Test
    fun testConversionToMegaFloat() {
        val integer = MegaInteger("12345")

        // Convert to MegaFloat
        val float = MegaFloat(integer)

        // Verify that the mantissa is preserved
        assertEquals(integer.mantissa.size, float.mantissa.size)
        for (i in integer.mantissa.indices) {
            assertEquals(integer.mantissa[i], float.mantissa[i])
        }

        // Verify that the sign is preserved
        assertEquals(integer.negative, float.negative)

        // Verify that isFloat is true for MegaFloat
        assertTrue(float.isFloat)

        // Verify that exponent is zero
        assertEquals(1, float.exponent.size)
        assertEquals(0, float.exponent[0].toLong())

        // Verify that exponentNegative is false
        assertFalse(float.exponentNegative)
    }

    @Test
    fun testFloatOperationsThrowException() {
        val integer = MegaInteger("100")
        val float = MegaFloat("50.5")

        // Test that addFloat throws an exception
        assertFailsWith<IllegalStateException> {
            integer.addFloat(float)
        }

        // Test that mulFloat throws an exception
        assertFailsWith<IllegalStateException> {
            integer.mulFloat(float)
        }

        // Test that divFloat throws an exception
        assertFailsWith<IllegalStateException> {
            integer.divFloat(float)
        }
    }
}
