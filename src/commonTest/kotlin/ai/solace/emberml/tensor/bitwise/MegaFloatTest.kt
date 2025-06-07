package ai.solace.emberml.tensor.bitwise

import kotlin.test.*

class MegaFloatTest {

    @Test
    fun testDecimalStringConstructor() {
        // Test the constructor that takes a decimal string
        val floatFromString = MegaFloat("123.45")

        // Verify that the mantissa and exponent are set correctly
        // The exact representation might vary, but we can check the decimal string
        assertEquals("123.45", floatFromString.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Verify that isFloat is true
        assertTrue(floatFromString.isFloat)

        // Test with a negative number
        val negativeFloat = MegaFloat("-54.321")
        assertTrue(negativeFloat.toDecimalString().startsWith("-"))
        assertTrue(negativeFloat.negative)

        // Test with zero
        val zero = MegaFloat("0")
        assertEquals("0", zero.toDecimalString())
        assertFalse(zero.negative)
    }

    @Test
    fun testExponentIsUsed() {
        // Create a MegaFloat with non-zero exponent
        val float = MegaFloat(
            mantissa = longArrayOf(123),
            exponent = longArrayOf(456),
            negative = false,
            isFloat = true,
            exponentNegative = true
        )

        // Verify that exponent is preserved
        assertEquals(1, float.exponent.size)
        assertEquals(456L, float.exponent[0])

        // Verify that isFloat is always true
        assertTrue(float.isFloat)

        // Verify that exponentNegative is preserved
        assertTrue(float.exponentNegative)
    }

    @Test
    fun testArithmeticOperationsUseExponents() {
        val a = MegaFloat("10.0")
        val b = MegaFloat("2.0")

        // Test addition
        val sum = a.add(b) as MegaFloat
        assertEquals("12.0", sum.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test multiplication
        val product = a.mul(b) as MegaFloat
        assertEquals("20.0", product.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test division
        val quotient = a.divide(b) as MegaFloat
        assertEquals("5.0", quotient.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test with fractional numbers
        val c = MegaFloat("1.5")
        val d = MegaFloat("0.5")

        // Test multiplication with fractions
        val fracProduct = c.mul(d) as MegaFloat
        assertEquals("0.75", fracProduct.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test division with fractions
        val fracQuotient = c.divide(d) as MegaFloat
        assertEquals("3.0", fracQuotient.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))
    }

    @Test
    fun testConversionFromMegaInteger() {
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
        assertEquals(0L, float.exponent[0])

        // Verify that exponentNegative is false
        assertFalse(float.exponentNegative)
    }

    @Test
    fun testMixedOperations() {
        val float = MegaFloat("10.5")
        val integer = MegaInteger("5")

        // Test addition of float and integer
        val sum = float.add(integer) as MegaFloat
        assertEquals("15.5", sum.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test multiplication of float and integer
        val product = float.mul(integer) as MegaFloat
        assertEquals("52.5", product.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))

        // Test division of float by integer
        val quotient = float.divide(integer) as MegaFloat
        assertEquals("2.1", quotient.toDecimalString().replace(" * 2^(", "e").replace(" * 16)", ""))
    }
}
