/**
 * Kotlin Native implementation of MegaFloat, inheriting from MegaNumber.
 *
 * This class provides an arbitrary-precision floating-point number.
 */
package ai.solace.emberml.tensor.bitwise

/**
 * Represents an arbitrary-precision floating-point number.
 */
class MegaFloat : MegaNumber {

    /**
     * Initializes a new MegaFloat with specified parameters.
     *
     * @param mantissa The mantissa in chunk-limbs form
     * @param exponent The exponent in chunk-limbs form
     * @param negative Indicates if the number is negative
     * @param isFloat Indicates if the number is a floating-point number (should be true)
     * @param exponentNegative Indicates if the exponent is negative
     * @param keepLeadingZeros Whether to keep leading zeros
     */
    constructor(
        mantissa: IntArray = IntArray(1) { 0 },
        exponent: IntArray = IntArray(1) { 0 },
        negative: Boolean = false,
        isFloat: Boolean = true,
        exponentNegative: Boolean = false,
        keepLeadingZeros: Boolean = false
    ) : super(
        mantissa = mantissa,
        exponent = exponent,
        negative = negative,
        isFloat = true, // Always float
        exponentNegative = exponentNegative,
        keepLeadingZeros = keepLeadingZeros
    )

    /**
     * Convenience constructor to create a MegaFloat from a decimal string.
     *
     * @param decimalStr The decimal string representation of the number (e.g., "123.456")
     */
    constructor(decimalStr: String) : this() {
        try {
            val tmp = fromDecimalString(decimalStr)
            this.mantissa = tmp.mantissa
            this.exponent = tmp.exponent
            this.negative = tmp.negative
            this.exponentNegative = tmp.exponentNegative
        } catch (e: Exception) {
            // If parsing fails, default to zero
            this.mantissa = IntArray(1) { 0 }
            this.exponent = IntArray(1) { 0 }
            this.negative = false
            this.exponentNegative = false
        }
    }

    /**
     * Convenience constructor to create a MegaFloat from a base MegaNumber.
     * Copies its mantissa, exponent, sign, etc., but forces isFloat=true.
     *
     * @param source The source MegaNumber
     */
    constructor(source: MegaNumber) : this(
        mantissa = source.mantissa,
        exponent = source.exponent,
        negative = source.negative,
        isFloat = true,
        exponentNegative = source.exponentNegative,
        keepLeadingZeros = source.keepLeadingZeros
    )

    companion object {
        /**
         * Creates a MegaFloat instance from a decimal string specifically as a float.
         *
         * @param s The decimal string representation of the number
         * @return A new MegaFloat instance
         */
        fun fromDecimalString(s: String): MegaFloat {
            val baseNum = MegaNumber.fromDecimalString(s)
            return MegaFloat(baseNum)
        }
    }

    /**
     * Adds another MegaNumber to this MegaFloat. Returns MegaFloat.
     *
     * @param other The MegaNumber to add
     * @return A new MegaFloat representing the sum
     */
    override fun addFloat(other: MegaNumber): MegaNumber {
        val baseResult = super.addFloat(other) // returns MegaNumber
        return MegaFloat(baseResult)
    }

    /**
     * Multiplies this MegaFloat with another MegaNumber. Returns MegaFloat.
     *
     * @param other The MegaNumber to multiply with
     * @return A new MegaFloat representing the product
     */
    override fun mulFloat(other: MegaNumber): MegaNumber {
        val baseResult = super.mulFloat(other)
        return MegaFloat(baseResult)
    }

    /**
     * Divides this MegaFloat by another MegaNumber. Returns MegaFloat.
     *
     * @param other The MegaNumber to divide by
     * @return A new MegaFloat representing the quotient
     */
    override fun divFloat(other: MegaNumber): MegaNumber {
        val baseResult = super.divFloat(other)
        return MegaFloat(baseResult)
    }

    /**
     * Ensures all operations return MegaFloat:
     * Adds another MegaNumber to this MegaFloat.
     *
     * @param other The MegaNumber to add
     * @return A new MegaFloat representing the sum
     */
    override fun add(other: MegaNumber): MegaNumber {
        val baseResult = super.add(other)
        return MegaFloat(baseResult)
    }

    /**
     * Subtracts another MegaNumber from this MegaFloat.
     *
     * @param other The MegaNumber to subtract
     * @return A new MegaFloat representing the difference
     */
    override fun sub(other: MegaNumber): MegaNumber {
        val baseResult = super.sub(other)
        return MegaFloat(baseResult)
    }

    /**
     * Multiplies this MegaFloat with another MegaNumber.
     *
     * @param other The MegaNumber to multiply with
     * @return A new MegaFloat representing the product
     */
    override fun mul(other: MegaNumber): MegaNumber {
        val baseResult = super.mul(other)
        return MegaFloat(baseResult)
    }

    /**
     * Divides this MegaFloat by another MegaNumber.
     *
     * @param other The MegaNumber to divide by
     * @return A new MegaFloat representing the quotient
     */
    override fun divide(other: MegaNumber): MegaNumber {
        val baseResult = super.divide(other)
        return MegaFloat(baseResult)
    }

    /**
     * Creates a copy of the current MegaFloat instance.
     *
     * @return A new MegaFloat instance with the same properties
     */
    fun copy(): MegaNumber {
        return MegaFloat(
            mantissa = this.mantissa.copyOf(),
            exponent = this.exponent.copyOf(),
            negative = this.negative,
            isFloat = true,
            exponentNegative = this.exponentNegative,
            keepLeadingZeros = this.keepLeadingZeros
        )
    }

    /**
     * Override toDecimalString to properly convert floating-point numbers using mantissa * 2^exponent.
     * This implements the correct floating-point to decimal conversion.
     *
     * @return A human-readable decimal string representation
     */
    override fun toDecimalString(): String {
        // If zero, return "0.0"
        if (mantissa.size == 1 && mantissa[0] == 0) {
            return "0.0"
        }

        // Get the sign prefix
        val signStr = if (negative) "-" else ""

        // If not float, convert as integer
        if (!isFloat) {
            return signStr + chunkToDecimal(mantissa)
        }

        // For float: need to compute mantissa * 2^exponent or mantissa / 2^exponent
        val expInt = exponentValue() // Use existing method that properly interprets exponent chunks

        if (exponentNegative) {
            // mantissa / 2^exponent - split into integer and fractional parts
            val (integerPart, remainder) = divideBy2ToThePower(mantissa, expInt)
            val integerStr = if (integerPart.size == 1 && integerPart[0] == 0) "0" else chunkToDecimal(integerPart)

            // If no remainder, just return integer part with .0
            if (remainder.size == 1 && remainder[0] == 0) {
                return signStr + integerStr + ".0"
            }

            // Build fractional digits by repeatedly multiplying remainder by 10 and dividing by 2^exponent
            val fracDigits = mutableListOf<String>()
            var currentRemainder = remainder.copyOf()
            val ten = intArrayOf(10)
            val maxFracDigits = 50 // Limit fractional precision

            for (i in 0 until maxFracDigits) {
                // Multiply remainder by 10
                currentRemainder = mulChunks(currentRemainder, ten)

                // Divide by 2^exponent to get next digit
                val (quotient, newRemainder) = divideBy2ToThePower(currentRemainder, expInt)
                val digitValue = chunksToInt(quotient)
                fracDigits.add(digitValue.toString())

                currentRemainder = newRemainder
                if (currentRemainder.size == 1 && currentRemainder[0] == 0) {
                    break
                }
            }

            var result = integerStr + "." + fracDigits.joinToString("")

            // Remove trailing zeros but keep at least one decimal place
            while (result.endsWith("0") && result.length > result.indexOf('.') + 2) {
                result = result.substring(0, result.length - 1)
            }

            return signStr + result
        } else {
            // mantissa * 2^exponent - multiply mantissa by 2^exponent
            val shifted = multiplyBy2ToThePower(mantissa, expInt)
            return signStr + chunkToDecimal(shifted) + ".0"
        }
    }







    /**
     * Provides a textual representation of the MegaFloat.
     *
     * @return A string representation of the MegaFloat
     */
    override fun toString(): String {
        return "<MegaFloat ${toDecimalString()}>"
    }
}
