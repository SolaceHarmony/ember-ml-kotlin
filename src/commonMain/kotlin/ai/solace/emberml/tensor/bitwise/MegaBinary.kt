/**
 * Kotlin Native implementation of MegaBinary, inheriting from MegaNumber.
 *
 * This class provides a binary data representation with operations for
 * binary wave and bitwise operations.
 */
package ai.solace.emberml.tensor.bitwise

/**
 * Interference modes for binary wave operations.
 */
enum class InterferenceMode {
    XOR,
    AND,
    OR
}

/**
 * Binary data class, storing bits in IntArray with 32-bit values.
 * Includes wave generation, duty-cycle patterns, interference, and
 * optional leading-zero preservation. Inherits from MegaNumber.
 *
 * @property byteData ByteArray representation of the binary data
 * @property bitLength Length of the binary representation in bits
 */
class MegaBinary : MegaNumber {
    var byteData: ByteArray
    private var bitLength: Int = 0

    /**
     * Initialize a MegaBinary object.
     *
     * @param value Initial value, can be:
     *              - String of binary digits (e.g., "1010" or "0b1010")
     *              - Default "0" => IntArray of just [0]
     * @param keepLeadingZeros Whether to keep leading zeros (default: true)
     */
    constructor(
        value: String = "0",
        keepLeadingZeros: Boolean = true
    ) : super(
        mantissa = intArrayOf(0),
        exponent = intArrayOf(0),
        negative = false,
        isFloat = false,
        exponentNegative = false,
        keepLeadingZeros = keepLeadingZeros
    ) {
        // Auto-detect and convert input
        var binStr = value
        if (binStr.startsWith("0b")) {
            binStr = binStr.substring(2)
        }
        if (binStr.isEmpty()) {
            binStr = "0"
        }

        // Build byteData from binary string
        // We'll chunk every 8 bits => int => byte
        byteData = ByteArray((binStr.length + 7) / 8)
        // Pad with leading zeros if necessary to make length a multiple of 8
        val paddedBinStr = binStr.padStart((binStr.length + 7) / 8 * 8, '0')
        for (i in paddedBinStr.indices step 8) {
            val chunk = paddedBinStr.substring(i, minOf(i + 8, paddedBinStr.length))
            byteData[i / 8] = chunk.toInt(2).toByte()
        }

        // Parse binary string into mantissa
        parseBinaryString(binStr)

        // Normalize
        normalize()

        // Store bit length
        bitLength = binStr.length
    }

    /**
     * Initialize a MegaBinary object from another MegaBinary.
     *
     * @param other Another MegaBinary object to copy
     */
    constructor(other: MegaBinary) : super(
        mantissa = other.mantissa.copyOf(),
        exponent = other.exponent.copyOf(),
        negative = other.negative,
        isFloat = other.isFloat,
        exponentNegative = other.exponentNegative,
        keepLeadingZeros = other.keepLeadingZeros
    ) {
        this.byteData = other.byteData.copyOf()
        this.bitLength = other.bitLength
    }

    /**
     * Initialize a MegaBinary object from a ByteArray.
     *
     * @param bytes ByteArray representation
     * @param keepLeadingZeros Whether to keep leading zeros (default: true)
     */
    constructor(
        bytes: ByteArray,
        keepLeadingZeros: Boolean = true
    ) : super(
        mantissa = intArrayOf(0),
        exponent = intArrayOf(0),
        negative = false,
        isFloat = false,
        exponentNegative = false,
        keepLeadingZeros = keepLeadingZeros
    ) {
        // Store original bytes
        byteData = bytes.copyOf()

        // Convert them to a binary string
        val binStr = bytes.joinToString("") { byte ->
            byte.toUByte().toString(2).padStart(8, '0')
        }

        // Parse binary string into mantissa
        parseBinaryString(binStr)

        // Normalize
        normalize()

        // Store bit length
        bitLength = binStr.length
    }

    /**
     * Initialize a MegaBinary object with a specific mantissa.
     *
     * @param mantissa IntArray of limbs
     * @param keepLeadingZeros Whether to keep leading zeros (default: true)
     */
    constructor(
        mantissa: IntArray,
        keepLeadingZeros: Boolean = true
    ) : super(
        mantissa = mantissa,
        exponent = intArrayOf(0),
        negative = false,
        isFloat = false,
        exponentNegative = false,
        keepLeadingZeros = keepLeadingZeros
    ) {
        // Convert mantissa to binary string
        val binStr = toBinaryString()

        // Build byteData from binary string
        byteData = ByteArray((binStr.length + 7) / 8)
        val paddedBinStr = binStr.padStart((binStr.length + 7) / 8 * 8, '0')
        for (i in paddedBinStr.indices step 8) {
            val chunk = paddedBinStr.substring(i, minOf(i + 8, paddedBinStr.length))
            byteData[i / 8] = chunk.toInt(2).toByte()
        }

        // Store bit length
        bitLength = binStr.length
    }

    /**
     * Convert binary string to mantissa.
     *
     * @param binStr Binary string (e.g., "1010")
     */
    private fun parseBinaryString(binStr: String) {
        if (binStr.isEmpty()) {
            mantissa = intArrayOf(0)
            return
        }

        // Store bit length
        bitLength = binStr.length

        // Convert to integer
        val value : Int = binStr.toIntOrNull(2) ?: throw IllegalArgumentException("Invalid binary string: $binStr")

        // Convert to limbs
        mantissa = intToChunks(value, MegaNumberConstants.GLOBAL_CHUNK_SIZE)
        exponent = intArrayOf(0)
        isFloat = false
        negative = false
        exponentNegative = false
    }

    /**
     * Perform bitwise AND operation.
     *
     * @param other Another MegaBinary object
     * @return Result of bitwise AND operation
     */
    fun bitwiseAnd(other: MegaBinary): MegaBinary {
        // Get maximum length
        val maxLen = maxOf(mantissa.size, other.mantissa.size)

        // Pad arrays to the same length
        val selfArr = if (mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            mantissa.copyInto(padded)
            padded
        } else {
            mantissa
        }
        val otherArr = if (other.mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            other.mantissa.copyInto(padded)
            padded
        } else {
            other.mantissa
        }

        // Perform bitwise AND
        val resultArr = IntArray(maxLen) { i ->
            selfArr[i] and otherArr[i]
        }

        // Create result
        val result = MegaBinary(mantissa = resultArr, keepLeadingZeros = keepLeadingZeros)
        result.normalize()

        return result
    }

    /**
     * Perform bitwise OR operation.
     *
     * @param other Another MegaBinary object
     * @return Result of bitwise OR operation
     */
    fun bitwiseOr(other: MegaBinary): MegaBinary {
        // Get maximum length
        val maxLen = maxOf(mantissa.size, other.mantissa.size)

        // Pad arrays to the same length
        val selfArr = if (mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            mantissa.copyInto(padded)
            padded
        } else {
            mantissa
        }
        val otherArr = if (other.mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            other.mantissa.copyInto(padded)
            padded
        } else {
            other.mantissa
        }

        // Perform bitwise OR
        val resultArr = IntArray(maxLen) { i ->
            selfArr[i] or otherArr[i]
        }

        // Create result
        val result = MegaBinary(mantissa = resultArr, keepLeadingZeros = keepLeadingZeros)
        result.normalize()

        return result
    }

    /**
     * Perform bitwise XOR operation.
     *
     * @param other Another MegaBinary object
     * @return Result of bitwise XOR operation
     */
    fun bitwiseXor(other: MegaBinary): MegaBinary {
        // Get maximum length
        val maxLen = maxOf(mantissa.size, other.mantissa.size)

        // Pad arrays to the same length
        val selfArr = if (mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            mantissa.copyInto(padded)
            padded
        } else {
            mantissa
        }
        val otherArr = if (other.mantissa.size < maxLen) {
            val padded = IntArray(maxLen)
            other.mantissa.copyInto(padded)
            padded
        } else {
            other.mantissa
        }

        // Perform bitwise XOR
        val resultArr = IntArray(maxLen) { i -> 
            selfArr[i] xor otherArr[i] 
        }

        // Create result
        val result = MegaBinary(mantissa = resultArr, keepLeadingZeros = keepLeadingZeros)
        result.normalize()

        return result
    }

    /**
     * Perform bitwise NOT operation.
     *
     * @return Result of bitwise NOT operation
     */
    fun bitwiseNot(): MegaBinary {
        // Perform bitwise NOT on existing limbs
        val resultArr = IntArray(mantissa.size) { i -> 
            mantissa[i].inv() and MegaNumberConstants.MASK
        }

        // Create result
        val result = MegaBinary(mantissa = resultArr, keepLeadingZeros = keepLeadingZeros)
        result.normalize()

        return result
    }

    /**
     * Add two MegaBinary objects (treating them as unsigned integers).
     *
     * @param other Another MegaBinary object
     * @return Sum as MegaBinary
     */
    fun add(other: MegaBinary): MegaBinary {
        // Use base class integer addition logic
        val baseResult = super.add(other)

        // Create new MegaBinary from the result mantissa
        val resultBin = MegaBinary(mantissa = baseResult.mantissa, keepLeadingZeros = keepLeadingZeros)
        resultBin.normalize()
        return resultBin
    }

    /**
     * Subtract other from self (treating them as unsigned integers).
     * Result is undefined if other > self.
     *
     * @param other Another MegaBinary object
     * @return Difference as MegaBinary
     */
    fun sub(other: MegaBinary): MegaBinary {
        // Use base class integer subtraction logic
        val baseResult = super.sub(other)

        // Check for negative result which is invalid for binary representation
        if (baseResult.negative) {
            throw IllegalArgumentException("Subtraction resulted in a negative value, invalid for MegaBinary")
        }

        // Create new MegaBinary from the result mantissa
        val resultBin = MegaBinary(mantissa = baseResult.mantissa, keepLeadingZeros = keepLeadingZeros)
        resultBin.normalize()
        return resultBin
    }

    /**
     * Multiply two MegaBinary objects (treating them as unsigned integers).
     *
     * @param other Another MegaBinary object
     * @return Product as MegaBinary
     */
    fun mul(other: MegaBinary): MegaBinary {
        // Use base class integer multiplication logic
        val baseResult = super.mul(other)

        // Create new MegaBinary from the result mantissa
        val resultBin = MegaBinary(mantissa = baseResult.mantissa, keepLeadingZeros = keepLeadingZeros)
        resultBin.normalize()
        return resultBin
    }

    /**
     * Divide self by other (integer division).
     *
     * @param other Another MegaBinary object
     * @return Quotient as MegaBinary
     */
    fun div(other: MegaBinary): MegaBinary {
        if (other.isZero()) {
            throw ArithmeticException("Division by zero")
        }

        // Use base class integer division logic
        val baseResult = super.divide(other)

        // Create new MegaBinary from the result mantissa
        val resultBin = MegaBinary(mantissa = baseResult.mantissa, keepLeadingZeros = keepLeadingZeros)
        resultBin.normalize()
        return resultBin
    }

    /**
     * Shift left by bits.
     *
     * @param bits Number of bits to shift (as MegaBinary)
     * @return Shifted MegaBinary
     */
    fun shiftLeft(bits: MegaBinary): MegaBinary {
        // Convert bits to integer
        val shiftVal : Int = chunklistToInt(bits.mantissa)

        // Convert self to integer
        val selfVal : Int = chunklistToInt(mantissa)

        // Perform left shift
        val shiftedVal: Int = (selfVal shl shiftVal)

        // Convert back to limbs
        val resultLimbs = intToChunks(shiftedVal, MegaNumberConstants.GLOBAL_CHUNK_SIZE)

        // Create result
        val result = MegaBinary(mantissa = resultLimbs, keepLeadingZeros = keepLeadingZeros)
        result.normalize()
        return result
    }

    /**
     * Shift right by bits.
     *
     * @param bits Number of bits to shift (as MegaBinary)
     * @return Shifted MegaBinary
     */
    fun shiftRight(bits: MegaBinary): MegaBinary {
        // Convert bits to integer
        val shiftVal = chunklistToInt(bits.mantissa)

        // Convert self to integer
        val selfVal = chunklistToInt(mantissa)

        // Perform right shift
        val shiftedVal: Int = (selfVal shr shiftVal)

        // Convert back to limbs
        val resultLimbs = intToChunks(shiftedVal, MegaNumberConstants.GLOBAL_CHUNK_SIZE)

        // Create result
        val result = MegaBinary(mantissa = resultLimbs, keepLeadingZeros = keepLeadingZeros)
        result.normalize()
        return result
    }

    /**
     * Get the bit at the specified position.
     *
     * @param position Bit position (0-based, from least significant bit)
     * @return Bit value (true or false)
     */
    fun getBit(position: MegaBinary): Boolean {
        // Convert position to integer
        val posVal = chunklistToInt(position.mantissa)

        // Convert self to integer
        val selfVal = chunklistToInt(mantissa)

        // Create mask
        val mask = 1 shl posVal

        // Check the bit
        return (selfVal and mask) != 0
    }

    /**
     * Set the bit at the specified position. Modifies the object in place.
     *
     * @param position Bit position (0-based, from least significant bit)
     * @param value Bit value (true or false)
     */
    fun setBit(position: MegaBinary, value: Boolean) {
        // Convert position to integer
        val posVal = chunklistToInt(position.mantissa)

        // Convert self to integer
        val selfVal = chunklistToInt(mantissa)

        // Create mask
        val mask = 1 shl posVal

        val newVal = if (value) {
            // Set bit using OR
            selfVal or mask
        } else {
            // Clear bit using AND with NOT mask
            selfVal and mask.inv()
        }

        // Convert back to limbs and update mantissa
        mantissa = intToChunks(newVal, MegaNumberConstants.GLOBAL_CHUNK_SIZE)
        normalize()
    }

    /**
     * Propagate the wave by shifting it left.
     *
     * @param shift Number of bits to shift (as MegaBinary)
     * @return Propagated wave
     */
    fun propagate(shift: MegaBinary): MegaBinary {
        // Propagation is typically a left shift in wave contexts
        return shiftLeft(shift)
    }

    /**
     * Convert to list of bits (LSB first).
     *
     * @return List of bits (0 or 1)
     */
    fun toBits(): List<Int> {
        val binStr = toBinaryString()
        // Pad with leading zeros if keepLeadingZeros is true and bitLength is set
        val paddedBinStr = if (keepLeadingZeros && bitLength > 0) {
            binStr.padStart(bitLength, '0')
        } else {
            binStr
        }
        return paddedBinStr.reversed().map { it.toString().toInt() }
    }

    /**
     * Convert to list of bits (MSB first).
     *
     * @return List of bits (0 or 1)
     */
    fun toBitsBigEndian(): List<Int> {
        val binStr = toBinaryString()
        val paddedBinStr = if (keepLeadingZeros && bitLength > 0) {
            binStr.padStart(bitLength, '0')
        } else {
            binStr
        }
        return paddedBinStr.map { it.toString().toInt() }
    }

    /**
     * Convert to binary string (MSB first).
     *
     * @return Binary string representation
     */
    fun toBinaryString(): String {
        if (mantissa.size == 1 && mantissa[0] == 0) {
            return "0"
        }

        // Convert limbs to integer
        val value = chunklistToInt(mantissa)

        // Convert to binary string, remove "0b" prefix
        val binStr = value.toString(2)

        // Handle potential padding if keepLeadingZeros is true
        return if (keepLeadingZeros && bitLength > 0) {
            binStr.padStart(bitLength, '0')
        } else {
            binStr
        }
    }

    /**
     * Convert to binary string (MSB first). Alias for toBinaryString.
     *
     * @return Binary string representation (MSB first)
     */
    fun toStringBigEndian(): String {
        return toBinaryString()
    }

    /**
     * Check if the value is zero.
     *
     * @return True if the value is zero, False otherwise
     */
    fun isZero(): Boolean {
        // Check if mantissa represents zero after normalization
        normalize()
        return mantissa.size == 1 && mantissa[0] == 0
    }

    /**
     * Convert to bytes (big-endian).
     *
     * @return Byte representation
     */
    fun toBytes(): ByteArray {
        val binStr = toBinaryString()
        // Pad with leading zeros to make length a multiple of 8
        val paddedBinStr = binStr.padStart((binStr.length + 7) / 8 * 8, '0')
        val byteArr = ByteArray(paddedBinStr.length / 8)
        for (i in paddedBinStr.indices step 8) {
            val chunk = paddedBinStr.substring(i, minOf(i + 8, paddedBinStr.length))
            byteArr[i / 8] = chunk.toInt(2).toByte()
        }
        return byteArr
    }

    /**
     * Create a copy of this MegaBinary.
     *
     * @return Copy of this MegaBinary
     */
    fun copy(): MegaBinary {
        return MegaBinary(this)
    }

    /**
     * String representation.
     *
     * @return String representation
     */
    override fun toString(): String {
        return "<MegaBinary ${toBinaryString()}>"
    }

    companion object {
        /**
         * Combine multiple waves bitwise (XOR, AND, OR).
         *
         * @param waves List of MegaBinary objects
         * @param mode Interference mode (XOR, AND, OR)
         * @return Interference pattern
         */
        fun interfere(waves: List<MegaBinary>, mode: InterferenceMode): MegaBinary {
            if (waves.isEmpty()) {
                throw IllegalArgumentException("Need at least one wave for interference")
            }

            // Find max length among all wave mantissas
            var maxLen = 0
            for (wave in waves) {
                maxLen = maxOf(maxLen, wave.mantissa.size)
            }

            // Pad all mantissas to maxLen and perform operation
            var resultArr = if (waves[0].mantissa.size < maxLen) {
                val padded = IntArray(maxLen)
                waves[0].mantissa.copyInto(padded)
                padded
            } else {
                waves[0].mantissa.copyOf()
            }

            for (wave in waves.subList(1, waves.size)) {
                val paddedWaveArr = if (wave.mantissa.size < maxLen) {
                    val padded = IntArray(maxLen)
                    wave.mantissa.copyInto(padded)
                    padded
                } else {
                    wave.mantissa.copyOf()
                }

                resultArr = when (mode) {
                    InterferenceMode.XOR -> IntArray(maxLen) { i -> resultArr[i] xor paddedWaveArr[i] }
                    InterferenceMode.AND -> IntArray(maxLen) { i -> resultArr[i] and paddedWaveArr[i] }
                    InterferenceMode.OR -> IntArray(maxLen) { i -> resultArr[i] or paddedWaveArr[i] }
                }
            }

            // Create result
            // Determine keepLeadingZeros based on the first wave
            val keepZeros = waves[0].keepLeadingZeros
            val result = MegaBinary(mantissa = resultArr, keepLeadingZeros = keepZeros)
            result.normalize()
            return result
        }

        /**
         * Create a blocky sine wave pattern.
         *
         * @param length Length of the pattern in bits (as MegaBinary)
         * @param halfPeriod Half the period of the wave in bits (as MegaBinary)
         * @return Blocky sine wave pattern
         */
        fun generateBlockySin(length: MegaBinary, halfPeriod: MegaBinary): MegaBinary {
            // Convert inputs to integers
            val lenInt = length.chunklistToInt(length.mantissa)
            val hpInt = halfPeriod.chunklistToInt(halfPeriod.mantissa)

            if (hpInt <= 0) {
                throw IllegalArgumentException("Half period must be positive")
            }
            if (lenInt <= 0) {
                return MegaBinary("0")
            }

            // Generate pattern
            val binStr = buildString {
                for (i in 0 until lenInt) {
                    if ((i / hpInt) % 2 == 0) {
                        append('1')
                    } else {
                        append('0')
                    }
                }
            }

            return MegaBinary(binStr, keepLeadingZeros = length.keepLeadingZeros)
        }

        /**
         * Create a binary pattern with the specified duty cycle.
         *
         * @param length Length of the pattern in bits (as MegaBinary)
         * @param dutyCycleVal Number of '1' bits (as MegaBinary)
         * @return Binary pattern with the specified duty cycle
         */
        fun createDutyCycle(length: MegaBinary, dutyCycleVal: MegaBinary): MegaBinary {
            val lenInt = length.chunklistToInt(length.mantissa)
            val numOnes = dutyCycleVal.chunklistToInt(dutyCycleVal.mantissa)

            if (numOnes < 0 || numOnes > lenInt) {
                throw IllegalArgumentException("Number of ones must be between 0 and length")
            }
            if (lenInt <= 0) {
                return MegaBinary("0")
            }

            // Create pattern string
            val binStr = "1".repeat(numOnes) + "0".repeat(lenInt - numOnes)

            return MegaBinary(binStr, keepLeadingZeros = length.keepLeadingZeros)
        }
    }
}