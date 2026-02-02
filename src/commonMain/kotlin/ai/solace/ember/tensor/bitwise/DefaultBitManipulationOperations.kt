package ai.solace.ember.tensor.bitwise

/**
 * Default implementation of BitManipulationOperations interface.
 * This class provides implementations for the methods defined in the BitManipulationOperations interface.
 */
object DefaultBitManipulationOperations : BitManipulationOperations {
    /**
     * Shift bits left
     * 
     * @param limbs The array of limbs to shift
     * @param shift The number of bits to shift
     * @return The shifted array
     */
    fun shiftLeft(limbs: IntArray, shift: Int): IntArray {
        if (shift < 0) {
            throw IllegalArgumentException("shift must be non-negative")
        }
        if (shift == 0) {
            return limbs.copyOf()
        }

        val chunkShift = shift / MegaNumberConstants.GLOBAL_CHUNK_SIZE
        val bitShift = shift % MegaNumberConstants.GLOBAL_CHUNK_SIZE

        val result = IntArray(limbs.size + chunkShift + (if (bitShift > 0) 1 else 0))

        if (bitShift == 0) {
            // Just shift chunks
            limbs.copyInto(result, chunkShift)
        } else {
            // Need to handle bit shifting (treat limbs as unsigned 32-bit)
            var carry = 0L
            val mask = MegaNumberConstants.MASK
            for (i in limbs.indices) {
                val value = limbs[i].toLong() and mask
                val shifted = (value shl bitShift)
                result[i + chunkShift] = ((carry or shifted) and mask).toInt()
                carry = value ushr (MegaNumberConstants.GLOBAL_CHUNK_SIZE - bitShift)
            }
            if (carry != 0L) {
                result[limbs.size + chunkShift] = carry.toInt()
            }
        }

        // Trim trailing zeros
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0) {
            lastNonZero--
        }

        return if (lastNonZero < result.size - 1) result.copyOf(lastNonZero + 1) else result
    }

    /**
     * Shift bits right
     * 
     * @param limbs The array of limbs to shift
     * @param shiftBits The number of bits to shift
     * @return The shifted array
     */
    fun shiftRight(limbs: IntArray, shiftBits: Int): IntArray {
        // Shift count must be non-negative.
        if (shiftBits < 0) {
            throw IllegalArgumentException("shiftBits must be non-negative")
        }
        if (shiftBits == 0) {
            return limbs.copyOf()
        }

        val chunkShift = shiftBits / MegaNumberConstants.GLOBAL_CHUNK_SIZE
        val bitShift = shiftBits % MegaNumberConstants.GLOBAL_CHUNK_SIZE

        // If we're shifting more than the total bits, result is 0
        if (chunkShift >= limbs.size) {
            return intArrayOf(0)
        }

        val resultSize = limbs.size - chunkShift
        val result = IntArray(resultSize)

        if (bitShift == 0) {
            // Just shift chunks
            limbs.copyInto(result, 0, chunkShift)
        } else {
            // Need to handle bit shifting (treat limbs as unsigned 32-bit)
            val mask = MegaNumberConstants.MASK
            for (i in 0 until resultSize - 1) {
                val cur = limbs[i + chunkShift].toLong() and mask
                val next = limbs[i + chunkShift + 1].toLong() and mask
                val highBits = cur ushr bitShift
                val lowBits = next shl (MegaNumberConstants.GLOBAL_CHUNK_SIZE - bitShift)
                result[i] = ((highBits or lowBits) and mask).toInt()
            }
            // Handle the last chunk
            val lastVal = limbs[limbs.size - 1].toLong() and mask
            result[resultSize - 1] = (lastVal ushr bitShift).toInt()
        }

        // Trim trailing zeros
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0) {
            lastNonZero--
        }

        return if (lastNonZero < result.size - 1) result.copyOf(lastNonZero + 1) else result
    }

    /**
     * Multiply by a power of 2
     * 
     * @param chunks The array of chunks to multiply
     * @param bits The power of 2 to multiply by
     * @return The result array
     */
    fun multiplyBy2ToThePower(chunks: IntArray, bits: Int): IntArray {
        return shiftLeft(chunks, bits)
    }

    /**
     * Divide by a power of 2
     * 
     * @param chunks The array of chunks to divide
     * @param bits The power of 2 to divide by
     * @return A pair containing the quotient and remainder
     */
    fun divideBy2ToThePower(chunks: IntArray, bits: Int): Pair<IntArray, IntArray> {
        val quotient = shiftRight(chunks, bits)
        
        // Calculate remainder: original - (quotient << bits)
        val quotientShifted = shiftLeft(quotient, bits)
        val remainder = if (DefaultChunkOperations.compareAbs(chunks, quotientShifted) >= 0) {
            DefaultChunkOperations.subChunks(chunks, quotientShifted)
        } else {
            intArrayOf(0)
        }
        
        return Pair(quotient, remainder)
    }
}
