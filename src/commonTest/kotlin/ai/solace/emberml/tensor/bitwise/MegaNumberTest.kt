package ai.solace.emberml.tensor.bitwise

import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.addChunks
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.compareAbs
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.decimalStringToChunks
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.div2
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.fromDecimalString
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.intToChunks
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.intToIntArray
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.karatsubaMulChunks
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.mulChunksStandard
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.shiftLeft
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.subChunks
import ai.solace.emberml.tensor.bitwise.MegaNumber.Companion.zeroIntArray
import kotlin.test.*

class MegaNumberTest {
    // Helper function to convert MegaNumber to Int for testing purposes
    private fun expAsInt(mn: MegaNumber): Int {
        val str = mn.toDecimalString()
        // If it's a simple integer, parse it directly
        if (!str.contains("*")) {
            return str.toInt()
        }
        // Otherwise, it's in the format "mantissa * 2^(exponent * chunkBits)"
        // For our test cases, we just need the exponent value
        return 0 // Default value for complex cases, adjust as needed for specific tests
    }
    @Test
    fun testBasicMultiplication() {
        val a = intArrayOf(2)
        val b = intArrayOf(3)
        val result = mulChunksStandard(a, b)
        assertEquals(1, result.size)
        assertEquals(6, result[0])
    }

    @Test
    fun testLargeAddition() {
        // Test that causes carry
        val a = intArrayOf(0xFFFFFFFF.toInt())
        val b = intArrayOf(1)
        val result = addChunks(a, b)
        assertEquals(2, result.size)
        assertEquals(0, result[0])
        assertEquals(1, result[1])
    }

    @Test
    fun testSubChunks() {
        // Test basic subtraction
        val a = intArrayOf(10)
        val b = intArrayOf(3)
        val result = subChunks(a, b)
        assertEquals(1, result.size)
        assertEquals(7, result[0])

        // Test subtraction with borrow
        val c = intArrayOf(0, 1) // Represents 2^32
        val d = intArrayOf(1)
        val result2 = subChunks(c, d)
        assertEquals(2, result2.size)
        assertEquals(0xFFFFFFFF.toInt(), result2[0])
        assertEquals(0, result2[1])

        // Test subtraction resulting in zero
        val e = intArrayOf(5)
        val f = intArrayOf(5)
        val result3 = subChunks(e, f)
        assertEquals(1, result3.size)
        assertEquals(0, result3[0])
    }

    @Test
    fun testCompareAbs() {
        // Test equal values
        val a = intArrayOf(10)
        val b = intArrayOf(10)
        assertEquals(0, compareAbs(a, b))

        // Test first greater
        val c = intArrayOf(20)
        val d = intArrayOf(10)
        assertEquals(1, compareAbs(c, d))

        // Test second greater
        val e = intArrayOf(5)
        val f = intArrayOf(10)
        assertEquals(-1, compareAbs(e, f))

        // Test different lengths
        val g = intArrayOf(1, 1) // Represents 1 + 2^32
        val h = intArrayOf(1)
        assertEquals(1, compareAbs(g, h))

        // Test different lengths, opposite direction
        val i = intArrayOf(1)
        val j = intArrayOf(1, 1)
        assertEquals(-1, compareAbs(i, j))
    }

    @Test
    fun testKaratsubaMulChunks() {
        // Test basic multiplication
        val a = intArrayOf(123, 456)
        val b = intArrayOf(789, 101)

        // Compare with standard multiplication
        val standardResult = mulChunksStandard(a, b)
        val karatsubaResult = karatsubaMulChunks(a, b)

        assertEquals(standardResult.size, karatsubaResult.size)
        for (i in standardResult.indices) {
            assertEquals(standardResult[i], karatsubaResult[i])
        }

        // Test with larger numbers
        val c = IntArray(40) { 0xFFFFFFFF.toInt() }
        val d = IntArray(40) { 0xFFFFFFFF.toInt() }

        val largeResult = karatsubaMulChunks(c, d)
        // The result should be (2^32 - 1)^2 * sum(i=0 to 39, j=0 to 39, 2^(32*(i+j)))
        // We can at least check that it's not all zeros
        assertNotEquals(0, largeResult[largeResult.size - 1])
    }

    @Test
    fun testShiftLeft() {
        // Test basic shift
        val a = intArrayOf(1, 2, 3)
        val result = shiftLeft(a, 2)
        assertEquals(5, result.size)
        assertEquals(0, result[0])
        assertEquals(0, result[1])
        assertEquals(1, result[2])
        assertEquals(2, result[3])
        assertEquals(3, result[4])

        // Test zero shift
        val b = intArrayOf(4, 5, 6)
        val result2 = shiftLeft(b, 0)
        assertEquals(3, result2.size)
        assertEquals(4, result2[0])
        assertEquals(5, result2[1])
        assertEquals(6, result2[2])
    }

    @Test
    fun testIntToChunks() {
        // Test zero
        val result = intToChunks(0)
        assertEquals(1, result.size)
        assertEquals(0, result[0])

        // Test positive number
        val result2 = intToChunks(123456)
        assertEquals(1, result2.size)
        assertEquals(123456, result2[0])

        // Test with chunk size
        val result3 = intToChunks(15, 4) // 15 in binary is 1111, with chunk size 4 it's just one chunk
        assertEquals(1, result3.size)
        assertEquals(15, result3[0])

        // Test with chunk size that splits the number
        val result4 = intToChunks(0b10101010, 4) // 10101010 in binary, with chunk size 4 it's 1010 and 1010
        assertEquals(2, result4.size)
        assertEquals(10, result4[0]) // 1010 in binary is 10
        assertEquals(10, result4[1]) // 1010 in binary is 10
    }

    @Test
    fun testIntToIntArray() {
        val result = intToIntArray(42)
        assertEquals(1, result.size)
        assertEquals(42, result[0])
    }

    @Test
    fun testZeroIntArray() {
        val result = zeroIntArray()
        assertEquals(1, result.size)
        assertEquals(0, result[0])
    }

    @Test
    fun testDecimalStringToChunks() {
        // Test zero
        val result = decimalStringToChunks("0")
        assertEquals(1, result.size)
        assertEquals(0, result[0])

        // Test simple number
        val result2 = decimalStringToChunks("123")
        assertEquals(1, result2.size)
        assertEquals(123, result2[0])

        // Test large number
        val result3 = decimalStringToChunks("12345678901234567890")
        assertTrue(result3.size > 1) // Should require multiple chunks

        // Test invalid input
        assertFailsWith<IllegalArgumentException> {
            decimalStringToChunks("123a456")
        }
    }

    @Test
    fun testDiv2() {
        // Test simple division
        val a = intArrayOf(10)
        val result = div2(a)
        assertEquals(1, result.size)
        assertEquals(5, result[0])

        // Test odd number
        val b = intArrayOf(11)
        val result2 = div2(b)
        assertEquals(1, result2.size)
        assertEquals(5, result2[0]) // Integer division, so 11/2 = 5

        // Test with carry
        val c = intArrayOf(1, 1) // Represents 1 + 2^32
        val result3 = div2(c)
        // Leading zero chunk is trimmed, so result length is 1
        assertEquals(1, result3.size)
        assertEquals(0x80000000.toInt(), result3[0]) // 2^31
    }

    @Test
    fun testFromDecimalString() {
        // Test integer
        val num = fromDecimalString("12345")
        assertEquals("12345", num.toDecimalString())
        assertFalse(num.negative)
        assertFalse(num.isFloat)

        // Test negative integer
        val negNum = fromDecimalString("-54321")
        assertEquals("-54321", negNum.toDecimalString())
        assertTrue(negNum.negative)
        assertFalse(negNum.isFloat)

        // Test float
        val floatNum = fromDecimalString("123.45")
        assertTrue(floatNum.isFloat)

        // Test negative float
        val negFloatNum = fromDecimalString("-67.89")
        assertTrue(negFloatNum.negative)
        assertTrue(negFloatNum.isFloat)
    }

    @Test
    fun testNormalize() {
        // Test normalization of zero
        val zero = MegaNumber(intArrayOf(0), intArrayOf(1), negative = true, isFloat = true, exponentNegative = true)
        zero.normalize()
        assertFalse(zero.negative)
        assertFalse(zero.exponentNegative)
        assertEquals(1, zero.exponent.size)
        assertEquals(0, zero.exponent[0])

        // Test trimming of mantissa
        val num = MegaNumber(intArrayOf(1, 0, 0), intArrayOf(0))
        num.normalize()
        assertEquals(1, num.mantissa.size)
        assertEquals(1, num.mantissa[0])

        // Test with keepLeadingZeros
        val numWithZeros = MegaNumber(intArrayOf(0, 0, 1), intArrayOf(0),
            negative = false,
            isFloat = false,
            exponentNegative = false,
            keepLeadingZeros = true
        )
        numWithZeros.normalize()
        assertEquals(3, numWithZeros.mantissa.size)
        assertEquals(0, numWithZeros.mantissa[0])
        assertEquals(0, numWithZeros.mantissa[1])
        assertEquals(1, numWithZeros.mantissa[2])
    }

    @Test
    fun testShiftRight() {
        val num = MegaNumber(intArrayOf(0xFFFFFFFF.toInt(), 0xFFFFFFFF.toInt()))

        // Test shifting by 32 bits (one chunk)
        val result = num.shiftRight(num.mantissa, 32)
        assertEquals(1, result.size)
        assertEquals(0xFFFFFFFF.toInt(), result[0])

        // Test shifting by 16 bits (half chunk)
        val result2 = num.shiftRight(num.mantissa, 16)
        assertEquals(2, result2.size)
        assertEquals(0xFFFF0000.toInt(), result2[0])
        assertEquals(0x0000FFFF, result2[1])

        // Test shifting by more bits than available
        val result3 = num.shiftRight(num.mantissa, 64)
        assertEquals(1, result3.size)
        assertEquals(0, result3[0])
    }

    @Test
    fun testAdd() {
        // Test integer addition
        val a = MegaNumber(intArrayOf(10))
        val b = MegaNumber(intArrayOf(20))
        val sum = a.add(b)
        assertEquals("30", sum.toDecimalString())

        // Test addition with different signs
        val c = MegaNumber(intArrayOf(30), negative = true)
        val d = MegaNumber(intArrayOf(20))
        val diff = c.add(d)
        assertEquals("-10", diff.toDecimalString())

        // Test addition resulting in zero
        val e = MegaNumber(intArrayOf(15))
        val f = MegaNumber(intArrayOf(15), negative = true)
        val zero = e.add(f)
        assertEquals("0", zero.toDecimalString())
    }

    @Test
    fun testAddFloat() {
        // Test float addition
        val a = MegaNumber(intArrayOf(10), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val b = MegaNumber(intArrayOf(20), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val sum = a.addFloat(b)
        assertTrue(sum.isFloat)

        // Test addition with different exponents
        val c = MegaNumber(intArrayOf(10), intArrayOf(2), negative = false, isFloat = true, exponentNegative = false)
        val d = MegaNumber(intArrayOf(10), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val result = c.addFloat(d)
        assertTrue(result.isFloat)

        // Test addition with different signs
        val e = MegaNumber(intArrayOf(30), intArrayOf(1), negative = true, isFloat = true, exponentNegative = false)
        val f = MegaNumber(intArrayOf(20), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val diff = e.addFloat(f)
        assertTrue(diff.negative)
        assertTrue(diff.isFloat)
    }

    @Test
    fun testSub() {
        // Test integer subtraction
        val a = MegaNumber(intArrayOf(30))
        val b = MegaNumber(intArrayOf(10))
        val diff = a.sub(b)
        assertEquals("20", diff.toDecimalString())

        // Test subtraction resulting in negative
        val c = MegaNumber(intArrayOf(10))
        val d = MegaNumber(intArrayOf(20))
        val negDiff = c.sub(d)
        assertEquals("-10", negDiff.toDecimalString())

        // Test subtraction of negative number
        val e = MegaNumber(intArrayOf(15))
        val f = MegaNumber(intArrayOf(10), negative = true)
        val sum = e.sub(f)
        assertEquals("25", sum.toDecimalString())
    }

    @Test
    fun testMul() {
        // Test integer multiplication
        val a = MegaNumber(intArrayOf(10))
        val b = MegaNumber(intArrayOf(20))
        val product = a.mul(b)
        assertEquals("200", product.toDecimalString())

        // Test multiplication with different signs
        val c = MegaNumber(intArrayOf(30), negative = true)
        val d = MegaNumber(intArrayOf(20))
        val negProduct = c.mul(d)
        assertEquals("-600", negProduct.toDecimalString())

        // Test multiplication by zero
        val e = MegaNumber(intArrayOf(15))
        val f = MegaNumber(intArrayOf(0))
        val zero = e.mul(f)
        assertEquals("0", zero.toDecimalString())
    }

    @Test
    fun testMulFloat() {
        // Test float multiplication
        val a = MegaNumber(intArrayOf(10), intArrayOf(1), false, isFloat = true, exponentNegative = false)
        val b = MegaNumber(intArrayOf(20), intArrayOf(1), false, isFloat = true, exponentNegative = false)
        val product = a.mulFloat(b)
        assertTrue(product.isFloat)

        // Test multiplication with different exponents
        val c = MegaNumber(intArrayOf(10), intArrayOf(2), negative = false, isFloat = true, exponentNegative = false)
        val d = MegaNumber(intArrayOf(10), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val result = c.mulFloat(d)
        assertTrue(result.isFloat)
        assertEquals(expected = 3, actual = expAsInt(result.exponentValue())) // 2 + 1 = 3

        // Test multiplication with different signs
        val e = MegaNumber(intArrayOf(30), intArrayOf(1), negative = true, isFloat = true, exponentNegative = false)
        val f = MegaNumber(intArrayOf(20), intArrayOf(1), false, isFloat = true, exponentNegative = false)
        val negProduct = e.mulFloat(f)
        assertTrue(negProduct.negative)
        assertTrue(negProduct.isFloat)
    }

    @Test
    fun testDivide() {
        // Test integer division
        val a = MegaNumber(intArrayOf(100))
        val b = MegaNumber(intArrayOf(20))
        val quotient = a.divide(b)
        assertEquals("5", quotient.toDecimalString())

        // Test division with different signs
        val c = MegaNumber(intArrayOf(30), negative = true)
        val d = MegaNumber(intArrayOf(10))
        val negQuotient = c.divide(d)
        assertEquals("-3", negQuotient.toDecimalString())

        // Test division by larger number
        val e = MegaNumber(intArrayOf(10))
        val f = MegaNumber(intArrayOf(20))
        val zeroQuotient = e.divide(f)
        assertEquals("0", zeroQuotient.toDecimalString())

        // Test division by zero
        val g = MegaNumber(intArrayOf(15))
        val h = MegaNumber(intArrayOf(0))
        assertFailsWith<IllegalArgumentException> {
            g.divide(h)
        }
    }

    @Test
    fun testFloatDivide() {
        // Test float division
        val a = MegaNumber(intArrayOf(100), intArrayOf(1), false, isFloat = true, exponentNegative = false)
        val b = MegaNumber(intArrayOf(20), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val quotient = a.divide(b)
        assertTrue(quotient.isFloat)

        // Test division with different exponents
        val c = MegaNumber(intArrayOf(10), intArrayOf(3), negative = false, isFloat = true, exponentNegative = false)
        val d = MegaNumber(intArrayOf(10), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val result = c.divide(d)
        assertTrue(result.isFloat)
        assertEquals(expected = 2, actual = expAsInt(result.exponentValue())) // 3 - 1 = 2

        // Test division with different signs
        val e = MegaNumber(intArrayOf(30), intArrayOf(1), negative = true, isFloat = true, exponentNegative = false)
        val f = MegaNumber(intArrayOf(10), intArrayOf(1), negative = false, isFloat = true, exponentNegative = false)
        val negQuotient = e.divide(f)
        assertTrue(negQuotient.negative)
        assertTrue(negQuotient.isFloat)
    }

    @Test
    fun testSqrt() {
        // Test integer square root
        val a = MegaNumber(intArrayOf(100))
        val sqrt = a.sqrt()
        assertEquals("10", sqrt.toDecimalString())

        // Test square root of zero
        val zero = MegaNumber(intArrayOf(0))
        val zeroSqrt = zero.sqrt()
        assertEquals("0", zeroSqrt.toDecimalString())

        // Test square root of negative number
        val neg = MegaNumber(intArrayOf(25), negative = true)
        assertFailsWith<IllegalArgumentException> {
            neg.sqrt()
        }

        // Test square root of non-perfect square
        val nonPerfect = MegaNumber(intArrayOf(12))
        val nonPerfectSqrt = nonPerfect.sqrt()
        assertEquals("3", nonPerfectSqrt.toDecimalString()) // Integer sqrt of 12 is 3
    }

    @Test
    fun testFloatSqrt() {
        // Test float square root
        val a = MegaNumber(intArrayOf(100), intArrayOf(0), negative = false, isFloat = true, exponentNegative = false)
        val sqrt = a.sqrt()
        assertTrue(sqrt.isFloat)

        // Test with odd exponent
        val b = MegaNumber(intArrayOf(100), intArrayOf(3), negative = false, isFloat = true, exponentNegative = false)
        val oddExpSqrt = b.sqrt()
        assertTrue(oddExpSqrt.isFloat)
        assertEquals(1, expAsInt(oddExpSqrt.exponentValue())) // 3/2 = 1.5, but integer division gives 1
    }

    // Note: divideBy2ToThePower and multiplyBy2ToThePower are protected methods
    // and cannot be tested directly from outside the class
}
