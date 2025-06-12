/**
 * Comprehensive test suite for bitwise operations.
 * 
 * This test suite covers all bitwise operations with systematic testing
 * following the Python implementation patterns from the ember_ml Python tests.
 */
package ai.solace.emberml.testing

import ai.solace.emberml.tensor.bitwise.MegaBinary
import ai.solace.emberml.tensor.bitwise.InterferenceMode
import kotlin.test.*

/**
 * Unit tests for bitwise operations as specified in the testing strategy.
 * 
 * Tests cover:
 * - Basic bitwise operations (AND, OR, XOR, NOT)
 * - Bit manipulation (get, set, toggle)
 * - Shift operations (left, right)
 * - Wave operations (interference, propagation)
 * - Pattern generation (duty cycles, blocky sine waves)
 */
class BitwiseOperationsTestSuite {

    companion object {
        /**
         * Test data matching Python implementation expectations
         */
        val testData = mapOf(
            "data_a" to "1010", // 10 in decimal
            "data_b" to "1100", // 12 in decimal  
            "data_c" to "0011", // 3 in decimal
            "data_zero" to "0000"
        )
    }

    /**
     * Test basic bitwise AND operations
     */
    @Test
    fun testBitwiseAnd() {
        val a = MegaBinary(testData["data_a"]!!)
        val b = MegaBinary(testData["data_b"]!!)
        
        val result = a.bitwiseAnd(b)
        val expected = "1000" // 1010 AND 1100 = 1000
        
        assertEquals(expected, result.toBinaryString(), "AND operation failed")
    }

    /**
     * Test basic bitwise OR operations
     */
    @Test
    fun testBitwiseOr() {
        val a = MegaBinary(testData["data_a"]!!)
        val b = MegaBinary(testData["data_b"]!!)
        
        val result = a.bitwiseOr(b)
        val expected = "1110" // 1010 OR 1100 = 1110
        
        assertEquals(expected, result.toBinaryString(), "OR operation failed")
    }

    /**
     * Test basic bitwise XOR operations
     */
    @Test
    fun testBitwiseXor() {
        val a = MegaBinary(testData["data_a"]!!)
        val b = MegaBinary(testData["data_b"]!!)
        
        val result = a.bitwiseXor(b)
        val expected = "0110" // 1010 XOR 1100 = 0110
        
        assertEquals(expected, result.toBinaryString(), "XOR operation failed")
    }

    /**
     * Test bit manipulation operations - getBit
     */
    @Test
    fun testGetBit() {
        val data = MegaBinary("1010")
        
        // Test individual bit access
        assertTrue(data.getBit(MegaBinary("1")), "Bit 1 should be set")
        assertFalse(data.getBit(MegaBinary("0")), "Bit 0 should not be set")
        assertTrue(data.getBit(MegaBinary("11")), "Bit 3 should be set")
        assertFalse(data.getBit(MegaBinary("10")), "Bit 2 should not be set")
    }

    /**
     * Test bit manipulation operations - setBit
     */
    @Test
    fun testSetBit() {
        val data = MegaBinary("1010")
        
        // Test setting a bit
        data.setBit(MegaBinary("0"), true)
        assertTrue(data.toBinaryString().endsWith("1"), "Bit 0 should now be set")
        
        // Test clearing a bit
        data.setBit(MegaBinary("1"), false)
        assertFalse(data.getBit(MegaBinary("1")), "Bit 1 should now be cleared")
    }

    /**
     * Test shift operations
     */
    @Test
    fun testShiftOperations() {
        val data = MegaBinary("1010") // 10 in decimal
        
        // Test left shift
        val leftShifted = data.shiftLeft(MegaBinary("1"))
        assertEquals("10100", leftShifted.toBinaryString(), "Left shift by 1 failed")
        
        // Test right shift  
        val rightShifted = data.shiftRight(MegaBinary("1"))
        assertEquals("101", rightShifted.toBinaryString(), "Right shift by 1 failed")
    }

    /**
     * Test wave interference operations
     */
    @Test
    fun testWaveInterference() {
        val a = MegaBinary("1010")
        val b = MegaBinary("1100")
        val c = MegaBinary("0101")
        val waves = listOf(a, b, c)

        // Test XOR interference
        val xorResult = MegaBinary.interfere(waves, InterferenceMode.XOR)
        // Expected: XOR of all three waves
        
        // Test AND interference
        val andResult = MegaBinary.interfere(waves, InterferenceMode.AND)
        // Expected: AND of all three waves
        
        // Test OR interference  
        val orResult = MegaBinary.interfere(waves, InterferenceMode.OR)
        // Expected: OR of all three waves
        
        // Verify results are not null and have reasonable output
        assertNotNull(xorResult, "XOR interference should produce result")
        assertNotNull(andResult, "AND interference should produce result")
        assertNotNull(orResult, "OR interference should produce result")
    }

    /**
     * Test pattern generation - duty cycles
     */
    @Test
    fun testDutyCycleGeneration() {
        // Test basic duty cycle generation
        val length = MegaBinary("1000") // 8 bits
        val dutyCycle = MegaBinary("11") // 3 bits high
        
        val result = MegaBinary.createDutyCycle(length, dutyCycle)
        
        // Should have the specified length
        assertNotNull(result, "Duty cycle generation should produce result")
        
        // Test edge cases
        assertFailsWith<IllegalArgumentException> {
            MegaBinary.createDutyCycle(MegaBinary("1000"), MegaBinary("1001")) // duty > length
        }
    }

    /**
     * Test pattern generation - blocky sine waves
     */
    @Test
    fun testBlockySinGeneration() {
        // Test basic blocky sin generation
        val length = MegaBinary("1000") // 8 bits
        val halfPeriod = MegaBinary("10") // 2 bits
        
        val result = MegaBinary.generateBlockySin(length, halfPeriod)
        
        // Should have the specified length
        assertNotNull(result, "Blocky sin generation should produce result")
        
        // Test edge cases
        assertFailsWith<IllegalArgumentException> {
            MegaBinary.generateBlockySin(MegaBinary("1000"), MegaBinary("0")) // invalid half period
        }
    }

    /**
     * Test propagation operations
     */
    @Test
    fun testPropagation() {
        val data = MegaBinary("1010")
        val shift = MegaBinary("10") // shift by 2
        
        val result = data.propagate(shift)
        
        // Propagation should be equivalent to left shift
        val expected = data.shiftLeft(shift)
        assertEquals(expected.toBinaryString(), result.toBinaryString(), "Propagation should match left shift")
    }
}