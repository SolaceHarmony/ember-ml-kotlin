package ai.solace.emberml.tensor.common

/**
 * Represents the shape of a tensor.
 * A shape is a list of dimensions that define the size of the tensor along each axis.
 *
 * @property dimensions The dimensions of the tensor.
 */
class EmberShape(val dimensions: IntArray) {
    /**
     * The number of dimensions in the shape.
     */
    val size: Int
        get() = dimensions.size

    /**
     * Gets the dimension at the specified index.
     *
     * @param index The index of the dimension to get.
     * @return The dimension at the specified index.
     */
    operator fun get(index: Int): Int = dimensions[index]

    /**
     * Returns a string representation of the shape.
     *
     * @return A string representation of the shape.
     */
    override fun toString(): String = dimensions.contentToString()

    /**
     * Checks if this shape is equal to another object.
     *
     * @param other The object to compare with.
     * @return True if the shapes are equal, false otherwise.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EmberShape) return false
        return dimensions.contentEquals(other.dimensions)
    }

    /**
     * Returns a hash code for this shape.
     *
     * @return A hash code for this shape.
     */
    override fun hashCode(): Int = dimensions.contentHashCode()

    companion object {
        /**
         * Creates a shape from a list of dimensions.
         *
         * @param dims The dimensions of the tensor.
         * @return A new shape with the specified dimensions.
         */
        fun of(vararg dims: Int): EmberShape = EmberShape(dims)
    }
}
