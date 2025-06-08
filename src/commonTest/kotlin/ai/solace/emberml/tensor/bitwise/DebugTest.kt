package ai.solace.emberml.tensor.bitwise

import kotlin.test.*

class DebugTest {
    
    @Test
    fun testToDecimalString() {
        // Test MegaInteger
        val integer = MegaInteger("100")
        println("[DEBUG_LOG] MegaInteger(100).toDecimalString(): ${integer.toDecimalString()}")
        
        // Test MegaFloat
        val float = MegaFloat("123.45")
        println("[DEBUG_LOG] MegaFloat(123.45).toDecimalString(): ${float.toDecimalString()}")
        
        // Test arithmetic operations
        val a = MegaInteger("100")
        val b = MegaInteger("50")
        val sum = a.add(b) as MegaInteger
        println("[DEBUG_LOG] MegaInteger(100).add(MegaInteger(50)).toDecimalString(): ${sum.toDecimalString()}")
        
        val product = a.mul(b) as MegaInteger
        println("[DEBUG_LOG] MegaInteger(100).mul(MegaInteger(50)).toDecimalString(): ${product.toDecimalString()}")
        
        // Test MegaFloat arithmetic
        val c = MegaFloat("10.0")
        val d = MegaFloat("2.0")
        val floatSum = c.add(d) as MegaFloat
        println("[DEBUG_LOG] MegaFloat(10.0).add(MegaFloat(2.0)).toDecimalString(): ${floatSum.toDecimalString()}")
        
        val floatProduct = c.mul(d) as MegaFloat
        println("[DEBUG_LOG] MegaFloat(10.0).mul(MegaFloat(2.0)).toDecimalString(): ${floatProduct.toDecimalString()}")
    }
}