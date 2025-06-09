package ai.solace.emberml.tensor.bitwise

/**
 * Interface for conversion operations
 */
interface ConversionOperations {
    /**
     * Convert to a decimal string representation
     * 
     * @return The decimal string representation
     */
    fun toDecimalString(): String

    companion object {
        /**
         * Convert from a decimal string
         * 
         * @param s The decimal string
         * @return The resulting number
         */
        fun fromDecimalString(s: String): MegaNumber = throw NotImplementedError("Implemented by concrete classes")
    }
}
