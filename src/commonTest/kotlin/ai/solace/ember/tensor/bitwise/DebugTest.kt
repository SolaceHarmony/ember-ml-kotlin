package ai.solace.ember.tensor.bitwise

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DebugTest {
    @Test
    fun decimalStringRoundTripUsesCurrentConstructors() {
        val integer = MegaInteger.fromValue("100")
        assertEquals("100", integer.toDecimalString())

        val sum = MegaInteger.fromValue(100).add(MegaInteger.fromValue(50))
        assertEquals("150", sum.toDecimalString())

        val product = MegaInteger.fromValue(100).mul(MegaInteger.fromValue(50))
        assertEquals("5000", product.toDecimalString())

        val float = MegaFloat("123.45")
        assertTrue(float.toDecimalString().startsWith("123.45"))

        val floatSum = MegaFloat("10.0").add(MegaFloat("2.0"))
        assertTrue(floatSum.toDecimalString().startsWith("12"))
    }
}
