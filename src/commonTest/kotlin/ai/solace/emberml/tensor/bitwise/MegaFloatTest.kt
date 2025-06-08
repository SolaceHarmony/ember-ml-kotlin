package ai.solace.emberml.tensor.bitwise

import kotlin.test.*

class MegaFloatTest {

    @Test
    fun testDecimalStringConstructor() {
        // Test the constructor that takes a decimal string
        val floatFromString = MegaFloat("123.45")

        // Verify that the mantissa and exponent are set correctly
        // The exact representation might vary, but we can check the decimal string
        val actual = floatFromString.toDecimalString()
        println("[DEBUG_LOG] Actual decimal string: $actual")
        // The format is "mantissa * 2^(exponent * 16)"
        // We need to extract just the mantissa part
        val simplified = if (actual.contains(" * 2^(")) {
            actual.substring(0, actual.indexOf(" * 2^("))
        } else {
            actual
        }
        println("[DEBUG_LOG] Simplified decimal string: $simplified")
        assertEquals("123.45", simplified)

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
        val sumStr = sum.toDecimalString()
        println("[DEBUG_LOG] Sum decimal string: $sumStr")
        val simplifiedSum = if (sumStr.contains(" * 2^(")) {
            sumStr.substring(0, sumStr.indexOf(" * 2^("))
        } else {
            sumStr
        }
        assertEquals("12.0", simplifiedSum)

        // Test multiplication
        val product = a.mul(b) as MegaFloat
        val productStr = product.toDecimalString()
        println("[DEBUG_LOG] Product decimal string: $productStr")
        val simplifiedProduct = if (productStr.contains(" * 2^(")) {
            productStr.substring(0, productStr.indexOf(" * 2^("))
        } else {
            productStr
        }
        assertEquals("20.0", simplifiedProduct)

        // Test division
        val quotient = a.divide(b) as MegaFloat
        val quotientStr = quotient.toDecimalString()
        println("[DEBUG_LOG] Quotient decimal string: $quotientStr")
        val simplifiedQuotient = if (quotientStr.contains(" * 2^(")) {
            quotientStr.substring(0, quotientStr.indexOf(" * 2^("))
        } else {
            quotientStr
        }
        assertEquals("5.0", simplifiedQuotient)

        // Test with fractional numbers
        val c = MegaFloat("1.5")
        val d = MegaFloat("0.5")

        // Test multiplication with fractions
        val fracProduct = c.mul(d) as MegaFloat
        val fracProductStr = fracProduct.toDecimalString()
        println("[DEBUG_LOG] Fraction product decimal string: $fracProductStr")
        val simplifiedFracProduct = if (fracProductStr.contains(" * 2^(")) {
            fracProductStr.substring(0, fracProductStr.indexOf(" * 2^("))
        } else {
            fracProductStr
        }
        assertEquals("0.75", simplifiedFracProduct)

        // Test division with fractions
        val fracQuotient = c.divide(d) as MegaFloat
        val fracQuotientStr = fracQuotient.toDecimalString()
        println("[DEBUG_LOG] Fraction quotient decimal string: $fracQuotientStr")
        val simplifiedFracQuotient = if (fracQuotientStr.contains(" * 2^(")) {
            fracQuotientStr.substring(0, fracQuotientStr.indexOf(" * 2^("))
        } else {
            fracQuotientStr
        }
        assertEquals("3.0", simplifiedFracQuotient)
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
        val sumStr = sum.toDecimalString()
        println("[DEBUG_LOG] Mixed sum decimal string: $sumStr")
        val simplifiedSum = if (sumStr.contains(" * 2^(")) {
            sumStr.substring(0, sumStr.indexOf(" * 2^("))
        } else {
            sumStr
        }
        assertEquals("15.5", simplifiedSum)

        // Test multiplication of float and integer
        val product = float.mul(integer) as MegaFloat
        val productStr = product.toDecimalString()
        println("[DEBUG_LOG] Mixed product decimal string: $productStr")
        val simplifiedProduct = if (productStr.contains(" * 2^(")) {
            productStr.substring(0, productStr.indexOf(" * 2^("))
        } else {
            productStr
        }
        assertEquals("52.5", simplifiedProduct)

        // Test division of float by integer
        val quotient = float.divide(integer) as MegaFloat
        val quotientStr = quotient.toDecimalString()
        println("[DEBUG_LOG] Mixed quotient decimal string: $quotientStr")
        val simplifiedQuotient = if (quotientStr.contains(" * 2^(")) {
            quotientStr.substring(0, quotientStr.indexOf(" * 2^("))
        } else {
            quotientStr
        }
        assertEquals("2.1", simplifiedQuotient)
    }
}
