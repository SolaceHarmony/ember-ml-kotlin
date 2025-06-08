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
        mantissa: LongArray = longArrayOf(0),
        exponent: LongArray = longArrayOf(0),
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
            this.mantissa = longArrayOf(0)
            this.exponent = longArrayOf(0)
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
    override fun copy(): MegaNumber {
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
     * Override toDecimalString to format floating-point numbers with a decimal point.
     * Inserts a decimal point 3 places from the right in the mantissa string.
     *
     * @return A human-readable decimal string representation
     */
    override fun toDecimalString(): String {
        // If zero, return "0.0"
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            return "0.0"
        }

        // Get the sign prefix
        val signStr = if (negative) "-" else ""

        // Convert mantissa to decimal string
        val numStr = chunkToDecimal(mantissa)

        // If not float, just return integer form (though we forced isFloat = True)
        if (!isFloat) {
            return signStr + numStr
        }

        // Insert decimal point 3 places from the right
        // If the string is shorter than 3 characters, pad with leading zeros
        val paddedStr = if (numStr.length <= 3) {
            "0".repeat(4 - numStr.length) + numStr
        } else {
            numStr
        }

        val decimalPos = paddedStr.length - 3
        var result = paddedStr.substring(0, decimalPos) + "." + paddedStr.substring(decimalPos)

        // Remove trailing zeros
        while (result.endsWith("0")) {
            result = result.substring(0, result.length - 1)
        }

        // Ensure at least one decimal place
        if (result.endsWith(".")) {
            result += "0"
        }

        return signStr + result
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
