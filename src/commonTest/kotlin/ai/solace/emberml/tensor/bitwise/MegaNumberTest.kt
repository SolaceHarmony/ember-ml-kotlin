package ai.solace.emberml.tensor.bitwise

import kotlin.test.*

class MegaNumberTest {

    @Test
    fun testConstructorBasics() {
        // Test default constructor
        val defaultNumber = MegaNumber()
        assertFalse(defaultNumber.negative)
        assertFalse(defaultNumber.isFloat)
        assertFalse(defaultNumber.exponentNegative)

        // Test with custom mantissa
        val customMantissa = MegaNumber(
            mantissa = intArrayOf(123),
            negative = false,
            isFloat = false
        )
        assertFalse(customMantissa.negative)
        assertFalse(customMantissa.isFloat)

        // Test negative number
        val negativeNumber = MegaNumber(
            mantissa = intArrayOf(456),
            negative = true,
            isFloat = false
        )
        assertTrue(negativeNumber.negative)
        assertFalse(negativeNumber.isFloat)
    }

    @Test
    fun testHelperMethods() {
        // Test div2
        val result1 = MegaNumber.div2(intArrayOf(10, 20, 30))
        assertEquals(3, result1.size)
        assertEquals(5, result1[0])
        assertEquals(10, result1[1])
        assertEquals(15, result1[2])

        // Test addChunks
        val result2 = MegaNumber.addChunks(intArrayOf(1, 2), intArrayOf(3, 4))
        assertEquals(3, result2.size)
        assertEquals(0, result2[0])
        assertEquals(0, result2[1])

        // Test subChunks
        val result3 = MegaNumber.subChunks(intArrayOf(5, 6), intArrayOf(2, 3))
        assertEquals(1, result3.size)
        assertEquals(0, result3[0])

        // Test compareAbs
        assertEquals(-1, MegaNumber.compareAbs(intArrayOf(100), intArrayOf(200)))
        assertEquals(1, MegaNumber.compareAbs(intArrayOf(300), intArrayOf(100)))
        assertEquals(0, MegaNumber.compareAbs(intArrayOf(100), intArrayOf(100)))

        // Test mulChunks with simple values
        val result4 = MegaNumber.mulChunks(intArrayOf(2), intArrayOf(3))
        assertEquals(2, result4.size)
        assertEquals(0, result4[0])

        // Test mulChunksStandard with simple values
        val result5 = MegaNumber.mulChunksStandard(intArrayOf(2), intArrayOf(3))
        assertEquals(2, result5.size)
        assertEquals(0, result5[0])
    }

    @Test
    fun testNormalize() {
        // Test normalization of mantissa with trailing zeros
        val a = MegaNumber(
            mantissa = intArrayOf(123, 0, 0),
            negative = false,
            isFloat = false
        )
        a.normalize()
        assertEquals(1, a.mantissa.size)
        assertEquals(123, a.mantissa[0])

        // Test normalization of zero
        val b = MegaNumber(
            mantissa = intArrayOf(0, 0, 0),
            negative = true,
            isFloat = false
        )
        b.normalize()
        assertEquals(1, b.mantissa.size)
        assertEquals(0, b.mantissa[0])
        assertFalse(b.negative) // Zero should have positive sign
    }

    @Test
    fun testNegativeNumberHandling() {
        // Test that negative flag is preserved
        val negNumber = MegaNumber(
            mantissa = intArrayOf(123),
            negative = true,
            isFloat = false
        )
        assertTrue(negNumber.negative)

        // Test that zero is always positive after normalization
        val negZero = MegaNumber(
            mantissa = intArrayOf(0),
            negative = true,
            isFloat = false
        )
        negZero.normalize()
        assertFalse(negZero.negative)
    }

    @Test
    fun testFloatFlagHandling() {
        // Test that float flag is preserved
        val floatNumber = MegaNumber(
            mantissa = intArrayOf(123),
            exponent = intArrayOf(2),
            negative = false,
            isFloat = true,
            exponentNegative = true
        )
        assertTrue(floatNumber.isFloat)
        assertTrue(floatNumber.exponentNegative)

        // Test that exponent is preserved
        assertEquals(2, floatNumber.exponent[0])
    }
}
