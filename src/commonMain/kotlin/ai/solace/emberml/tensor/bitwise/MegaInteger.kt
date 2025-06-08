/**
 * Kotlin Native implementation of MegaInteger, inheriting from MegaNumber.
 *
 * This class provides an arbitrary-precision integer with operations specific to integer arithmetic.
 */
package ai.solace.emberml.tensor.bitwise

/**
 * Represents an arbitrary-precision integer.
 * All operations maintain the integer nature of the result.
 */
class MegaInteger : MegaNumber {

    /**
     * Initializes a new MegaInteger with specified parameters.
     *
     * @param mantissa The mantissa in chunk-limbs form
     * @param exponent The exponent in chunk-limbs form (should be [0] for integers)
     * @param negative Indicates if the number is negative
     * @param isFloat Indicates if the number is a floating-point number (should be false for integers)
     * @param exponentNegative Indicates if the exponent is negative
     * @param keepLeadingZeros Whether to keep leading zeros
     */
    constructor(
        mantissa: LongArray = longArrayOf(0),
        exponent: LongArray = longArrayOf(0),
        negative: Boolean = false,
        isFloat: Boolean = false, // Always integer
        exponentNegative: Boolean = false,
        keepLeadingZeros: Boolean = false
    ) : super(
        mantissa = mantissa,
        exponent = longArrayOf(0), // Exponent is zero for integers
        negative = negative,
        isFloat = false,  // Always integer
        exponentNegative = false,
        keepLeadingZeros = keepLeadingZeros
    )

    /**
     * Convenience constructor to create a MegaInteger from a decimal string.
     *
     * @param decimalStr The decimal string representation of the integer (e.g., "123456789")
     */
    constructor(decimalStr: String) : this() {
        try {
            val tmp = fromDecimalString(decimalStr)
            this.mantissa = tmp.mantissa
            this.negative = tmp.negative
        } catch (e: Exception) {
            // If parsing fails, default to zero
            this.mantissa = longArrayOf(0)
            this.negative = false
        }
    }

    companion object {
        /**
         * Creates a MegaInteger instance from a decimal string specifically as an integer.
         *
         * @param s The decimal string representation of the integer
         * @return A new MegaInteger instance
         */
        fun fromDecimalString(s: String): MegaInteger {
            val baseNum = MegaNumber.fromDecimalString(s)
            // Force exponent to zero
            return MegaInteger(
                mantissa = baseNum.mantissa,
                exponent = longArrayOf(0),
                negative = baseNum.negative,
                isFloat = false,
                exponentNegative = false
            )
        }

        /**
         * Creates a MegaInteger instance from an Int.
         *
         * @param val_ The integer value to initialize with
         * @return A new MegaInteger instance
         */
        fun fromInt(val_: Int): MegaInteger {
            val limbs = intToChunks(value, val_)
            val negative = val_ < 0
            return MegaInteger(
                mantissa = limbs,
                exponent = longArrayOf(0),
                negative = negative,
                isFloat = false,
                exponentNegative = false
            )
        }
    }

    /**
     * Overrides the base class's addFloat to prevent floating-point addition in integer context.
     *
     * @param other The MegaNumber to add
     * @throws IllegalStateException Always throws this exception
     */
    override fun addFloat(other: MegaNumber): MegaNumber {
        throw IllegalStateException("MegaInteger cannot perform floating-point addition.")
    }

    /**
     * Overrides the base class's mulFloat to prevent floating-point multiplication in integer context.
     *
     * @param other The MegaNumber to multiply with
     * @throws IllegalStateException Always throws this exception
     */
    override fun mulFloat(other: MegaNumber): MegaNumber {
        throw IllegalStateException("MegaInteger cannot perform floating-point multiplication.")
    }

    /**
     * Overrides the base class's divFloat to prevent floating-point division in integer context.
     *
     * @param other The MegaNumber to divide by
     * @throws IllegalStateException Always throws this exception
     */
    override fun divFloat(other: MegaNumber): MegaNumber {
        throw IllegalStateException("MegaInteger cannot perform floating-point division.")
    }

    /**
     * Adds another MegaNumber to this MegaInteger. Only allows addition with another MegaInteger.
     *
     * @param other The MegaNumber to add
     * @return A new MegaInteger representing the sum
     * @throws IllegalArgumentException if other is not a MegaInteger
     */
    override fun add(other: MegaNumber): MegaNumber {
        if (other !is MegaInteger) {
            throw IllegalArgumentException("MegaInteger can only be added to another MegaInteger.")
        }
        val sumMant = addChunks(this.mantissa, other.mantissa)
        val result = MegaInteger(
            mantissa = sumMant,
            exponent = longArrayOf(0),
            negative = this.negative,
            isFloat = false,
            exponentNegative = false
        )
        result.normalize()
        return result
    }

    /**
     * Subtracts another MegaNumber from this MegaInteger. Only allows subtraction with another MegaInteger.
     *
     * @param other The MegaNumber to subtract
     * @return A new MegaInteger representing the difference
     * @throws IllegalArgumentException if other is not a MegaInteger
     */
    override fun sub(other: MegaNumber): MegaNumber {
        if (other !is MegaInteger) {
            throw IllegalArgumentException("MegaInteger can only subtract another MegaInteger.")
        }
        val diffMant = subChunks(this.mantissa, other.mantissa)
        val result = MegaInteger(
            mantissa = diffMant,
            exponent = longArrayOf(0),
            negative = this.negative,
            isFloat = false,
            exponentNegative = false
        )
        result.normalize()
        return result
    }

    /**
     * Multiplies this MegaInteger with another MegaNumber. Only allows multiplication with another MegaInteger.
     *
     * @param other The MegaNumber to multiply with
     * @return A new MegaInteger representing the product
     * @throws IllegalArgumentException if other is not a MegaInteger
     */
    override fun mul(other: MegaNumber): MegaNumber {
        if (other !is MegaInteger) {
            throw IllegalArgumentException("MegaInteger can only multiply with another MegaInteger.")
        }
        val productMant = mulChunks(this.mantissa, other.mantissa)
        val sign = (this.negative != other.negative)
        val result = MegaInteger(
            mantissa = productMant,
            exponent = longArrayOf(0),
            negative = sign,
            isFloat = false,
            exponentNegative = false
        )
        result.normalize()
        return result
    }

    /**
     * Divides this MegaInteger by another MegaNumber. Only allows division with another MegaInteger.
     *
     * @param other The MegaNumber to divide by
     * @return A new MegaInteger representing the quotient
     * @throws IllegalArgumentException if other is not a MegaInteger
     * @throws ArithmeticException if other is zero
     */
    override fun divide(other: MegaNumber): MegaNumber {
        if (other !is MegaInteger) {
            throw IllegalArgumentException("MegaInteger can only divide by another MegaInteger.")
        }
        // Check for division by zero
        if (other.mantissa.size == 1 && other.mantissa[0] == 0L) {
            throw ArithmeticException("Division by zero is not allowed.")
        }
        val sign = (this.negative != other.negative)
        val comparison = compareAbs(this.mantissa, other.mantissa)
        if (comparison < 0) {
            // Self < Other => quotient is 0
            return MegaInteger(
                mantissa = longArrayOf(0),
                exponent = longArrayOf(0),
                negative = false,
                isFloat = false,
                exponentNegative = false
            )
        } else if (comparison == 0) {
            // Self == Other => quotient is 1 or -1 based on sign
            return MegaInteger(
                mantissa = longArrayOf(1),
                exponent = longArrayOf(0),
                negative = sign,
                isFloat = false,
                exponentNegative = false
            )
        } else {
            val (quotient, _) = divChunks(this.mantissa, other.mantissa)
            val result = MegaInteger(
                mantissa = quotient,
                exponent = longArrayOf(0),
                negative = sign,
                isFloat = false,
                exponentNegative = false
            )
            result.normalize()
            return result
        }
    }

    /**
     * Negates the integer.
     *
     * @return A new MegaInteger representing the negated value
     */
    fun negateValue(): MegaInteger {
        return MegaInteger(
            mantissa = mantissa.copyOf(),
            exponent = longArrayOf(0),
            negative = !this.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Computes the modulo of this integer with another.
     *
     * @param other The MegaInteger to modulo with
     * @return A new MegaInteger representing the result
     * @throws ArithmeticException if other is zero
     */
    fun mod(other: MegaInteger): MegaInteger {
        val (_, remainder) = divChunks(this.mantissa, other.mantissa)
        return MegaInteger(
            mantissa = remainder,
            exponent = longArrayOf(0),
            negative = this.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Creates a copy of the current MegaInteger instance.
     *
     * @return A new MegaInteger instance with the same properties
     */
    override fun copy(): MegaNumber {
        return MegaInteger(
            mantissa = this.mantissa.copyOf(),
            exponent = longArrayOf(0), // Exponent is always zero for integers
            negative = this.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Computes the exponentiation of this integer to a non-negative power.
     *
     * @param power The non-negative power to raise the integer to
     * @return A new MegaInteger representing the result
     */
    fun pow(power: MegaInteger): MegaInteger {
        var result = fromInt(1)
        var base = this.copy() as MegaInteger
        var exp = power.copy() as MegaInteger

        while (!isZero(exp)) {
            if (isOdd(exp)) {
                result = result.mul(base) as MegaInteger
            }
            base = base.mul(base) as MegaInteger
            exp = divideByTwo(exp)
        }

        return result
    }

    /**
     * Performs a bitwise AND operation with another MegaInteger.
     *
     * @param other The MegaInteger to perform the AND with
     * @return A new MegaInteger representing the result
     */
    fun and(other: MegaInteger): MegaInteger {
        val minCount = minOf(this.mantissa.size, other.mantissa.size)
        val resultMantissa = LongArray(minCount)

        for (i in 0 until minCount) {
            resultMantissa[i] = this.mantissa[i] and other.mantissa[i]
        }

        return MegaInteger(
            mantissa = resultMantissa,
            exponent = longArrayOf(0),
            negative = this.negative && other.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Performs a bitwise OR operation with another MegaInteger.
     *
     * @param other The MegaInteger to perform the OR with
     * @return A new MegaInteger representing the result
     */
    fun or(other: MegaInteger): MegaInteger {
        val maxCount = maxOf(this.mantissa.size, other.mantissa.size)
        val resultMantissa = LongArray(maxCount)

        for (i in 0 until maxCount) {
            val a = if (i < this.mantissa.size) this.mantissa[i] else 0L
            val b = if (i < other.mantissa.size) other.mantissa[i] else 0L
            resultMantissa[i] = a or b
        }
        return MegaInteger(
            mantissa = resultMantissa,
            exponent = longArrayOf(0),
            negative = this.negative || other.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Performs a bitwise XOR operation with another MegaInteger.
     *
     * @param other The MegaInteger to perform the XOR with
     * @return A new MegaInteger representing the result
     */
    fun xor(other: MegaInteger): MegaInteger {
        val maxCount = maxOf(this.mantissa.size, other.mantissa.size)
        val resultMantissa = LongArray(maxCount)

        for (i in 0 until maxCount) {
            val a = if (i < this.mantissa.size) this.mantissa[i] else 0L
            val b = if (i < other.mantissa.size) other.mantissa[i] else 0L
            resultMantissa[i] = a xor b
        }
        return MegaInteger(
            mantissa = resultMantissa,
            exponent = longArrayOf(0),
            negative = this.negative != other.negative,  // XOR sign
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Performs a bitwise NOT operation on the integer.
     *
     * @return A new MegaInteger representing the bitwise NOT of the original integer
     */
    fun bitwiseNot(): MegaInteger {
        val resultMantissa = LongArray(mantissa.size)
        for (i in mantissa.indices) {
            resultMantissa[i] = mantissa[i].inv() and MegaNumberConstants.mask
        }
        return MegaInteger(
            mantissa = resultMantissa,
            exponent = longArrayOf(0),
            negative = !this.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Performs a right shift by a specified number of bits.
     *
     * @param bits The number of bits to shift
     * @return A new MegaInteger representing the shifted value
     */
    fun rightShift(bits: Int): MegaInteger {
        val chunkShift = bits / MegaNumberConstants.globalChunkSize
        val bitShift = bits % MegaNumberConstants.globalChunkSize

        // Handle whole chunk shifts
        var shifted = if (chunkShift > 0) {
            if (chunkShift >= mantissa.size) {
                return MegaInteger(longArrayOf(0))
            }
            mantissa.copyOfRange(chunkShift, mantissa.size)
        } else {
            mantissa.copyOf()
        }

        // Handle bit shifts within chunks
        if (bitShift > 0) {
            val result = LongArray(shifted.size)
            var carry = 0L
            for (i in shifted.indices.reversed()) {
                val newVal = (shifted[i] ushr bitShift) or (carry shl (MegaNumberConstants.globalChunkSize - bitShift))
                carry = shifted[i] and ((1L shl bitShift) - 1L)
                result[i] = newVal and MegaNumberConstants.mask
            }

            // Trim trailing zeros
            var lastNonZero = result.size - 1
            while (lastNonZero > 0 && result[lastNonZero] == 0L) {
                lastNonZero--
            }
            shifted = result.copyOf(lastNonZero + 1)
        }

        return MegaInteger(
            mantissa = shifted,
            exponent = longArrayOf(0),
            negative = this.negative,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Computes the greatest common divisor (GCD) of this integer and another.
     *
     * @param other The MegaInteger to compute the GCD with
     * @return A new MegaInteger representing the GCD
     */
    fun gcd(other: MegaInteger): MegaInteger {
        var a = this.copy() as MegaInteger
        var b = other.copy() as MegaInteger
        while (!(b.mantissa.size == 1 && b.mantissa[0] == 0L)) {
            val r = a.mod(b)
            a = b
            b = r
        }
        return a
    }

    /**
     * Computes the least common multiple (LCM) of this integer and another.
     *
     * @param other The MegaInteger to compute the LCM with
     * @return A new MegaInteger representing the LCM
     */
    fun lcm(other: MegaInteger): MegaInteger {
        val gcdValue = this.gcd(other)
        return (this.mul(other)).divide(gcdValue) as MegaInteger
    }

    /**
     * Computes the binary logarithm (log2) of the MegaInteger.
     *
     * @return A new MegaInteger representing floor(log2(N))
     * @throws IllegalArgumentException if the number is zero or negative
     */
    override fun log2(): MegaNumber {
        // Handle zero and negative numbers
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            throw IllegalArgumentException("Cannot compute log2 of zero.")
        }
        if (this.negative) {
            throw IllegalArgumentException("Cannot compute log2 of a negative number.")
        }

        // Compute log2(mantissa)
        val log2Mantissa = log2Integer()

        // Since MegaInteger represents integers only, exponent is zero
        // Total log2(N) = log2(mantissa)
        return log2Mantissa
    }

    /**
     * Computes the integer part of log2(mantissa).
     *
     * @return A new MegaInteger representing floor(log2(mantissa))
     */
    private fun log2Integer(): MegaInteger {
        // Initialize log2 to zero
        var log2 = fromInt(0)

        // Iterate from the most significant chunk to the least
        for (chunkIndex in mantissa.indices.reversed()) {
            val chunk = mantissa[chunkIndex]
            if (chunk == 0L) {
                continue
            }
            // Find the highest set bit in this chunk
            val highestBit = highestSetBit(chunk)
            // Calculate log2 contribution from this chunk
            val chunkLog2 = fromInt(chunkIndex * MegaNumberConstants.globalChunkSize + highestBit)
            log2 = chunkLog2
            break
        }

        return log2
    }

    /**
     * Finds the position of the highest set bit in a Long.
     *
     * @param x The Long to inspect
     * @return The zero-based position of the highest set bit
     */
    private fun highestSetBit(x: Long): Int {
        var value = x
        var pos = 0
        while (value > 1) {
            value = value ushr 1
            pos += 1
        }
        return pos
    }

    /**
     * Checks if the MegaInteger is zero.
     *
     * @param value The MegaInteger to check
     * @return True if the value is zero, false otherwise
     */
    private fun isZero(value: MegaInteger): Boolean {
        return value.mantissa.size == 1 && value.mantissa[0] == 0L
    }

    /**
     * Checks if the MegaInteger is odd.
     *
     * @param value The MegaInteger to check
     * @return True if the value is odd, false otherwise
     */
    private fun isOdd(value: MegaInteger): Boolean {
        return value.mantissa.size > 0 && (value.mantissa[0] and 1L) == 1L
    }

    /**
     * Divides the MegaInteger by two.
     *
     * @param value The MegaInteger to divide
     * @return A new MegaInteger representing the result
     */
    private fun divideByTwo(value: MegaInteger): MegaInteger {
        return value.rightShift(1)
    }

    /**
     * Provides a textual representation of the MegaInteger.
     *
     * @return A string representation of the MegaInteger
     */
    override fun toString(): String {
        return "<MegaInteger ${toDecimalString()}>"
    }
}
