/**
 * Kotlin Native implementation of MegaNumber, the foundation for arbitrary precision arithmetic.
 *
 * This class provides a chunk-based big integer (or float) with arbitrary precision arithmetic,
 * using Long arrays with 64-bit values.
 */
package ai.solace.emberml.tensor.bitwise

/**
 * Constants used by MegaNumber implementation
 */
object MegaNumberConstants {
    const val globalChunkSize: Int = 16 // Optimum size for O(n) complexity drop in certain operations
    val base: Long = 1L shl globalChunkSize // 2^16, fits in a JavaScript number
    val mask: Long = (1L shl globalChunkSize) - 1L // 2^16 - 1, all bits set in a 16-bit chunk

    // Thresholds for picking naive vs. Karatsuba vs. Toom-3
    const val MUL_THRESHOLD_KARATSUBA = 64 // Increased to use standard multiplication for smaller numbers
    const val MUL_THRESHOLD_TOOM = 128 // Increased to use Karatsuba for medium-sized numbers

    // Maximum precision in bits (limited to avoid excessive memory usage)
    var maxPrecisionBits: Int? = 1024 // 1024 bits should be enough for most use cases
}

/**
 * A chunk-based big integer (or float) with arbitrary precision arithmetic,
 * using LongArray with 16-bit values.
 *
 * @property mantissa LongArray of limbs (16-bit chunks)
 * @property exponent LongArray of limbs (16-bit chunks)
 * @property negative Sign flag
 * @property isFloat Float flag
 * @property exponentNegative Exponent sign flag
 * @property keepLeadingZeros Whether to keep leading zeros
 */
open class MegaNumber(
    open var mantissa: LongArray = longArrayOf(0),
    open var exponent: LongArray = longArrayOf(0),
    open var negative: Boolean = false,
    open var isFloat: Boolean = false,
    open var exponentNegative: Boolean = false,
    open val keepLeadingZeros: Boolean = false
) {
    companion object {
        /**
         * Add two chunk-limb arrays => sum-limb array
         */
        internal fun addChunks(a: LongArray, b: LongArray): LongArray {
            val maxLen = maxOf(a.size, b.size)
            val out = LongArray(maxLen + 1)
            var carry = 0L

            for (i in 0 until maxLen) {
                val av = if (i < a.size) a[i] else 0L
                val bv = if (i < b.size) b[i] else 0L
                val s = av + bv + carry
                // Check for overflow
                carry = if ((s xor av) and (s xor bv) < 0) 1L else 0L
                out[i] = s and MegaNumberConstants.mask
            }
            if (carry != 0L) out[maxLen] = carry

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0L) {
                lastNonZero--
            }
            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Subtract B from A (assuming A >= B), returning chunk-limb array
         */
        internal fun subChunks(a: LongArray, b: LongArray): LongArray {
            // A>=B must hold externally; we do a standard chunk-based subtraction with borrow
            val out = LongArray(a.size)
            var carry = 0L

            for (i in 0 until a.size) {
                val av = a[i]
                val bv = if (i < b.size) b[i] else 0L
                var diff = av - bv - carry
                if (diff < 0) {
                    diff += MegaNumberConstants.base
                    carry = 1
                } else {
                    carry = 0
                }
                out[i] = diff and MegaNumberConstants.mask
            }

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0L) {
                lastNonZero--
            }
            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Compare absolute magnitude of A vs. B => -1 if A<B, 0 if ==, 1 if A>B
         */
        internal fun compareAbs(a: LongArray, b: LongArray): Int {
            if (a.size > b.size) return 1
            if (a.size < b.size) return -1

            for (i in a.indices.reversed()) {
                if (a[i] > b[i]) return 1
                if (a[i] < b[i]) return -1
            }
            return 0
        }

        /**
         * Multiply two chunk-limb arrays => product-limb array using standard algorithm
         */
        internal fun mulChunksStandard(a: LongArray, b: LongArray): LongArray {
            if (a.size == 1 && a[0] == 0L) return longArrayOf(0)
            if (b.size == 1 && b[0] == 0L) return longArrayOf(0)

            val la = a.size
            val lb = b.size
            val out = LongArray(la + lb)

            for (i in 0 until la) {
                var carry = 0L
                val av = a[i]
                for (j in 0 until lb) {
                    val prod = av * b[j] + out[i + j] + carry
                    val lo = prod and MegaNumberConstants.mask
                    carry = prod ushr MegaNumberConstants.globalChunkSize
                    out[i + j] = lo
                }
                if (carry != 0L) {
                    out[i + lb] = out[i + lb] + carry
                }
            }

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0L) {
                lastNonZero--
            }
            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Multiply two chunk-limb arrays => product-limb array
         * Dispatches to naive, Karatsuba, or Toom-3 based on thresholds
         */
        internal fun mulChunks(a: LongArray, b: LongArray): LongArray {
            val n = maxOf(a.size, b.size)
            if (n < MegaNumberConstants.MUL_THRESHOLD_KARATSUBA) {
                return mulChunksStandard(a, b)
            } else if (n < MegaNumberConstants.MUL_THRESHOLD_TOOM) {
                return karatsubaMulChunks(a, b)
            } else {
                // For now, we'll use Karatsuba for Toom-3 threshold too
                // TODO: Implement Toom-3 algorithm
                return karatsubaMulChunks(a, b)
            }
        }

        /**
         * Implements Karatsuba multiplication for large numbers.
         */
        internal fun karatsubaMulChunks(a: LongArray, b: LongArray): LongArray {
            val n = maxOf(a.size, b.size)
            if (n <= 32) {
                return mulChunksStandard(a, b) // Use standard multiplication for small sizes
            }

            val m = n / 2

            val aLow = a.copyOf(minOf(m, a.size))
            val aHigh = if (m < a.size) a.copyOfRange(m, a.size) else longArrayOf(0)
            val bLow = b.copyOf(minOf(m, b.size))
            val bHigh = if (m < b.size) b.copyOfRange(m, b.size) else longArrayOf(0)

            val z0 = karatsubaMulChunks(aLow, bLow)
            val z2 = karatsubaMulChunks(aHigh, bHigh)

            // Perform (A_low + A_high) * (B_low + B_high)
            val aSum = addChunks(aLow, aHigh)
            val bSum = addChunks(bLow, bHigh)
            val z1Full = karatsubaMulChunks(aSum, bSum)

            // Compute z1 = z1_full - z0 - z2
            val z1Intermediate = subChunks(z1Full, z0)
            val z1 = subChunks(z1Intermediate, z2)

            // Combine results: z2 * BASE^(2*m) + z1 * BASE^m + z0
            val z2Shifted = shiftLeft(z2, 2 * m)
            val z1Shifted = shiftLeft(z1, m)
            val intermediate = addChunks(z2Shifted, z1Shifted)
            return addChunks(intermediate, z0)
        }

        /**
         * Shifts the limbs left by `shift` chunks (equivalent to multiplying by BASE^shift).
         */
        internal fun shiftLeft(limbs: LongArray, shift: Int): LongArray {
            if (shift <= 0) return limbs.copyOf()
            val result = LongArray(limbs.size + shift)
            limbs.copyInto(result, shift)
            return result
        }

        /**
         * Convert an Int into chunk-limbs. A zero value => [0].
         */
        internal fun intToChunks(val_: Int): LongArray {
            if (val_ == 0) return longArrayOf(0)
            var x = val_.toLong()
            val out = mutableListOf<Long>()
            while (x != 0L) {
                out.add(x and MegaNumberConstants.mask)
                x = x ushr MegaNumberConstants.globalChunkSize
            }
            return out.toLongArray()
        }

        /**
         * Combine chunk-limbs => a single Int. (May overflow if large.)
         */
        internal fun chunksToInt(limbs: LongArray): Int {
            var val_ = 0L
            var shift = 0
            for (limb in limbs) {
                val part = limb shl shift
                val_ += part  // May overflow
                shift += MegaNumberConstants.globalChunkSize
            }
            return val_.toInt()
        }

        /**
         * Convert decimal string => chunk-limb array
         */
        internal fun decimalStringToChunks(dec: String): LongArray {
            if (dec.isEmpty()) return longArrayOf(0)
            if (dec == "0") return longArrayOf(0)

            var limbs = longArrayOf(0)
            for (ch in dec) {
                if (ch < '0' || ch > '9') {
                    throw IllegalArgumentException("Invalid decimal digit in $dec")
                }
                val digit = ch - '0'
                // Multiply limbs by 10, then add digit
                limbs = addChunks(mulChunks(limbs, longArrayOf(10)), longArrayOf(digit.toLong()))
            }
            return limbs
        }

        /**
         * Create from decimal string, e.g. "123.456"
         */
        fun fromDecimalString(s: String): MegaNumber {
            // Basic parse
            var negative = false
            var raw = s.trim()
            if (raw.startsWith("-")) {
                negative = true
                raw = raw.substring(1).trim()
            }
            if (raw.isEmpty()) return MegaNumber()

            // Check float or int
            val parts = raw.split(".")
            if (parts.size == 1) {
                // Integer
                val mant = decimalStringToChunks(parts[0])
                return MegaNumber(
                    mantissa = mant,
                    exponent = longArrayOf(0),
                    negative = negative,
                    isFloat = false,
                    exponentNegative = false
                )
            } else {
                // Float
                val intPart = parts[0]
                val fracPart = parts[1]

                // Combine them as integer => do repeated multiply/add
                val fullNumStr = intPart + fracPart
                val mant = decimalStringToChunks(fullNumStr)
                // Approximate exponent using length of fraction => treat fraction as 2^some shift
                // E.g., log2(10) * fracLen
                val fracLen = fracPart.length
                val shiftBits = kotlin.math.ceil(fracLen * kotlin.math.log2(10.0)).toInt()
                val expChunks = intToChunks(shiftBits)

                return MegaNumber(
                    mantissa = mant,
                    exponent = expChunks,
                    negative = negative,
                    isFloat = true,
                    exponentNegative = true
                )
            }
        }

        /**
         * Create from binary string, e.g. "1010"
         */
        fun fromBinaryString(s: String): MegaNumber {
            val binStr = s.trim()
            if (binStr.isEmpty()) return MegaNumber()

            // Convert binary string to integer
            val intVal = binStr.toLongOrNull(2) ?: throw IllegalArgumentException("Invalid binary string: $binStr")

            // Convert to limbs
            val mant = if (intVal == 0L) {
                longArrayOf(0)
            } else {
                val chunks = mutableListOf<Long>()
                var value = intVal
                while (value != 0L) {
                    chunks.add(value and MegaNumberConstants.mask)
                    value = value ushr MegaNumberConstants.globalChunkSize
                }
                chunks.toLongArray()
            }

            return MegaNumber(
                mantissa = mant,
                exponent = longArrayOf(0),
                negative = false,
                isFloat = false,
                exponentNegative = false
            )
        }

        /**
         * Dynamically determine a reasonable precision limit based on performance benchmarks.
         * 
         * @param operation The operation to benchmark ('mul' by default)
         * @param thresholdSeconds The time threshold in seconds (2.0 by default)
         * @param hardLimit The hard limit in seconds (6.0 by default)
         * @return The determined maximum precision in bits
         */
        fun dynamicPrecisionTest(
            operation: String = "mul",
            thresholdSeconds: Double = 2.0,
            hardLimit: Double = 6.0
        ): Int {
            // If already set, return the existing value
            MegaNumberConstants.maxPrecisionBits?.let { return it }

            // For simplicity, we'll use a fixed value for now
            // In a real implementation, this would benchmark operations and determine a suitable limit
            val maxBits = 999999
            MegaNumberConstants.maxPrecisionBits = maxBits
            return maxBits
        }

        /**
         * Load cached precision limit from a file.
         * 
         * @param cacheFile The file path to load from
         */
        fun loadCachedPrecision(cacheFile: String = "precision.dat") {
            // In a real implementation, this would load the precision limit from a file
            // For now, we'll just set a default value if not already set
            if (MegaNumberConstants.maxPrecisionBits == null) {
                MegaNumberConstants.maxPrecisionBits = 999999
            }
        }

        /**
         * Save current precision limit to a file.
         * 
         * @param cacheFile The file path to save to
         */
        fun saveCachedPrecision(cacheFile: String = "precision.dat") {
            // In a real implementation, this would save the precision limit to a file
            // For now, this is a no-op
        }
    }

    /**
     * Initialize with specified parameters
     */
    init {
        normalize()
    }

    /**
     * Check if a MegaNumber exceeds the maximum precision limit.
     * Throws an exception if the limit is exceeded.
     *
     * @param num The MegaNumber to check
     * @throws IllegalStateException if the precision limit is exceeded
     */
    private fun checkPrecisionLimit(num: MegaNumber) {
        val maxBits = MegaNumberConstants.maxPrecisionBits
        if (maxBits != null) {
            val totalBits = num.mantissa.size * MegaNumberConstants.globalChunkSize
            if (totalBits > maxBits) {
                throw IllegalStateException("Precision limit exceeded: $totalBits bits > $maxBits bits")
            }
        }
    }

    /**
     * Convert an integer to a chunk list (for compatibility with MegaBinary)
     */
    internal fun intToChunklist(value: Int, chunkSize: Int): LongArray {
        return intToChunks(value)
    }

    /**
     * Convert a chunk list to an integer (for compatibility with MegaBinary)
     */
    internal fun chunklistToInt(limbs: LongArray): Int {
        return chunksToInt(limbs)
    }

    /**
     * Remove trailing zeros, handle zero sign, etc.
     */
    open fun normalize() {
        // Trim mantissa
        var lastNonZero = mantissa.size - 1
        while (lastNonZero > 0 && mantissa[lastNonZero] == 0L) {
            lastNonZero--
        }
        if (lastNonZero < mantissa.size - 1) {
            mantissa = mantissa.copyOf(lastNonZero + 1)
        }

        // Trim exponent if float
        if (isFloat) {
            lastNonZero = exponent.size - 1
            while (lastNonZero > 0 && exponent[lastNonZero] == 0L) {
                lastNonZero--
            }
            if (lastNonZero < exponent.size - 1) {
                exponent = exponent.copyOf(lastNonZero + 1)
            }
        }

        // If zero => unify sign
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            negative = false
            exponent = longArrayOf(0)
            exponentNegative = false
        }
    }

    /**
     * Implement chunk-based right shift
     */
    internal fun shiftRight(limbs: LongArray, shiftBits: Int): LongArray {
        // If shift <= 0, do nothing.
        if (shiftBits <= 0) return limbs.copyOf()

        // Number of whole chunks to drop.
        val chunkShift = shiftBits / MegaNumberConstants.globalChunkSize
        // Bits within one chunk to shift.
        val bitShift = shiftBits % MegaNumberConstants.globalChunkSize

        // If chunkShift >= limbs.count => result is 0.
        if (chunkShift >= limbs.size) {
            return longArrayOf(0)
        }

        // Remove the lowest 'chunkShift' limbs (since each is 2^GLOBAL_CHUNK_SIZE).
        var shifted = limbs.copyOfRange(chunkShift, limbs.size)

        // If there's no partial bit shift, we're done.
        if (bitShift == 0) {
            return shifted
        }

        // Otherwise, shift each limb to the right by bitShift bits,
        // carrying bits from the next limb.
        val result = LongArray(shifted.size)
        var carry = 0L
        for (i in shifted.indices.reversed()) {
            val val_ = shifted[i]
            // Right-shift this limb
            val newVal = (val_ ushr bitShift) or (carry shl (MegaNumberConstants.globalChunkSize - bitShift))
            // The "carry" (i.e. bits that fall off) comes from the lower part of val
            carry = val_ and ((1L shl bitShift) - 1L)

            result[i] = newVal and MegaNumberConstants.mask  // ensure we stay within chunk size
        }

        // Trim trailing zeroes
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0L) {
            lastNonZero--
        }
        return result.copyOf(lastNonZero + 1)
    }

    /**
     * Divide chunk-limb arrays => (quotient, remainder), integer division
     */
    internal fun divChunks(a: LongArray, b: LongArray): Pair<LongArray, LongArray> {
        // B must not be zero
        if (b.size == 1 && b[0] == 0L) {
            throw ArithmeticException("Division by zero")
        }
        val c = compareAbs(a, b)
        if (c < 0) return Pair(longArrayOf(0), a.copyOf()) // A<B => Q=0, R=A
        if (c == 0) return Pair(longArrayOf(1), longArrayOf(0)) // A=B => Q=1, R=0

        val q = LongArray(a.size)
        var r = longArrayOf(0)

        // We do a standard chunk-based long division
        for (i in a.indices.reversed()) {
            // shift R left by one chunk
            r = LongArray(r.size + 1).also { 
                r.copyInto(it, 1)
                it[0] = a[i]
            }

            // binary search in [0..BASE-1] for the best q
            var low = 0L
            var high = Long.MAX_VALUE // Approximation for BASE-1
            var guess = 0L

            while (low <= high) {
                val mid = (low + high) ushr 1
                val mm = mulChunks(b, longArrayOf(mid))
                val cmpv = compareAbs(mm, r)
                if (cmpv <= 0) {
                    guess = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }
            if (guess != 0L) {
                val mm = mulChunks(b, longArrayOf(guess))
                r = subChunks(r, mm)
            }
            q[i] = guess
        }

        // Trim q
        var lastNonZero = q.size - 1
        while (lastNonZero > 0 && q[lastNonZero] == 0L) {
            lastNonZero--
        }
        return Pair(q.copyOf(lastNonZero + 1), r)
    }

    /**
     * Divmod by small_val <= BASE
     */
    internal fun divmodSmall(a: LongArray, smallVal: Long): Pair<LongArray, Long> {
        var remainder = 0L
        val out = LongArray(a.size)

        for (i in a.indices.reversed()) {
            // Shift the remainder left by GLOBAL_CHUNK_SIZE bits and add the current limb
            val cur = (remainder shl MegaNumberConstants.globalChunkSize) + a[i]

            // Compute the quotient digit and the new remainder
            val qd = cur / smallVal
            remainder = cur % smallVal

            // Assign the quotient digit to the output array, ensuring it fits within the chunk mask
            out[i] = qd and MegaNumberConstants.mask
        }

        // Trim any unnecessary trailing zeros from the output array
        var lastNonZero = out.size - 1
        while (lastNonZero > 0 && out[lastNonZero] == 0L) {
            lastNonZero--
        }

        return Pair(out.copyOf(lastNonZero + 1), remainder)
    }

    /**
     * Convert chunk-limbs to decimal string
     */
    internal fun chunkToDecimal(limbs: LongArray): String {
        // quick check for zero
        if (limbs.size == 1 && limbs[0] == 0L) {
            return "0"
        }
        var temp = limbs.copyOf()
        val digits = mutableListOf<Char>()
        while (!(temp.size == 1 && temp[0] == 0L)) {
            val (q, r) = divmodSmall(temp, 10)
            temp = q
            digits.add('0' + r.toInt())
        }
        return digits.reversed().joinToString("")
    }

    /**
     * Compute exponent value as signed Int
     */
    internal fun exponentValue(): Int {
        val raw = chunksToInt(exponent)
        return if (exponentNegative) -raw else raw
    }

    /**
     * Return a decimal-string representation. (Integer-only if exponent=0.)
     */
    open fun toDecimalString(): String {
        // If zero
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            return "0"
        }

        // If exponent is zero or we are integer => treat as integer
        val expNonZero = (exponent.size > 1 || exponent[0] != 0L)
        if (!expNonZero) {
            // purely integer
            val s = chunkToDecimal(mantissa)
            return (if (negative) "-" else "") + s
        } else {
            // float => represent as "mantissa * 2^(exponent * chunkBits)" for simplicity
            var eVal = chunksToInt(exponent)
            if (exponentNegative) eVal = -eVal
            val mantString = chunkToDecimal(mantissa)
            val signStr = if (negative) "-" else ""
            // This is a simplistic representation.
            return "$signStr$mantString * 2^($eVal * ${MegaNumberConstants.globalChunkSize})"
        }
    }

    /**
     * Add two MegaNumbers. If either is float, handle float addition
     */
    open fun add(other: MegaNumber): MegaNumber {
        // If either is float, handle float addition
        if (this.isFloat || other.isFloat) {
            return addFloat(other)
        }

        // Integer addition
        if (this.negative == other.negative) {
            // Same sign => add magnitudes
            val sumMant = addChunks(this.mantissa, other.mantissa)
            val sign = this.negative
            return MegaNumber(
                mantissa = sumMant,
                exponent = longArrayOf(0),
                negative = sign,
                isFloat = false,
                exponentNegative = false
            )
        } else {
            // Different signs => subtract magnitudes
            val cmp = compareAbs(this.mantissa, other.mantissa)
            if (cmp == 0) {
                // Result is zero
                return MegaNumber(
                    mantissa = longArrayOf(0),
                    exponent = longArrayOf(0),
                    negative = false,
                    isFloat = false,
                    exponentNegative = false
                )
            } else if (cmp > 0) {
                // self > other in magnitude
                val diff = subChunks(this.mantissa, other.mantissa)
                val sign = this.negative
                return MegaNumber(
                    mantissa = diff,
                    exponent = longArrayOf(0),
                    negative = sign,
                    isFloat = false,
                    exponentNegative = false
                )
            } else {
                // other > self in magnitude
                val diff = subChunks(other.mantissa, this.mantissa)
                val sign = other.negative
                return MegaNumber(
                    mantissa = diff,
                    exponent = longArrayOf(0),
                    negative = sign,
                    isFloat = false,
                    exponentNegative = false
                )
            }
        }
    }

    /**
     * Float addition using chunk-based arithmetic
     */
    open fun addFloat(other: MegaNumber): MegaNumber {
        val expA = exponentValue()
        val expB = other.exponentValue()

        val diff = expA - expB
        var adjustedMantA = this.mantissa.copyOf()
        var adjustedMantB = other.mantissa.copyOf()
        var finalExp = expA

        // Align exponents by shifting mantissas
        if (diff > 0) {
            adjustedMantB = shiftRight(adjustedMantB, diff)
            finalExp = expA
        } else if (diff < 0) {
            adjustedMantA = shiftRight(adjustedMantA, -diff)
            finalExp = expB
        }

        // Add or subtract mantissas
        val sameSign = (this.negative == other.negative)
        val resultMant: LongArray
        val resultSign: Boolean

        if (sameSign) {
            // Same sign => add
            resultMant = addChunks(adjustedMantA, adjustedMantB)
            resultSign = this.negative
        } else {
            // Different sign => subtract
            val cmp = compareAbs(adjustedMantA, adjustedMantB)
            if (cmp == 0) {
                // Zero
                return MegaNumber(
                    mantissa = longArrayOf(0),
                    exponent = longArrayOf(0),
                    negative = false,
                    isFloat = false,
                    exponentNegative = false
                )
            } else if (cmp > 0) {
                resultMant = subChunks(adjustedMantA, adjustedMantB)
                resultSign = this.negative
            } else {
                resultMant = subChunks(adjustedMantB, adjustedMantA)
                resultSign = other.negative
            }
        }

        // Build a new MegaNumber (float) with that mantissa and `finalExp`.
        val newExponent = intToChunks(finalExp)
        val out = MegaNumber(
            mantissa = resultMant,
            exponent = newExponent,
            negative = resultSign,
            isFloat = true,
            exponentNegative = (finalExp < 0)
        )
        out.normalize()
        return out
    }

    /**
     * Subtract two MegaNumbers. a - b = a + (-b)
     */
    open fun sub(other: MegaNumber): MegaNumber {
        val negOther = MegaNumber(
            mantissa = other.mantissa.copyOf(),
            exponent = other.exponent.copyOf(),
            negative = !other.negative,
            isFloat = other.isFloat,
            exponentNegative = other.exponentNegative
        )
        return this.add(negOther)
    }

    /**
     * Multiply two MegaNumbers. If either is float, delegate to float multiply
     */
    open fun mul(other: MegaNumber): MegaNumber {
        if (this.isFloat || other.isFloat) {
            return mulFloat(other)
        }

        // Integer multiply
        val sign = (this.negative != other.negative)
        val product = mulChunks(this.mantissa, other.mantissa)
        return MegaNumber(
            mantissa = product,
            exponent = longArrayOf(0),
            negative = sign,
            isFloat = false,
            exponentNegative = false
        )
    }

    /**
     * Float multiplication using chunk-based arithmetic
     */
    open fun mulFloat(other: MegaNumber): MegaNumber {
        // Multiply mantissas
        val productMant = mulChunks(this.mantissa, other.mantissa)
        // Add exponents
        val expA = exponentValue()
        val expB = other.exponentValue()
        val sumExp = expA + expB
        val newExponent = intToChunks(sumExp)
        // Determine sign
        val newNegative = (this.negative != other.negative)
        // Create result
        val out = MegaNumber(
            mantissa = productMant,
            exponent = newExponent,
            negative = newNegative,
            isFloat = true,
            exponentNegative = (sumExp < 0)
        )
        out.normalize()
        return out
    }

    /**
     * Divide two MegaNumbers. If either is float, delegate to float division
     */
    open fun divide(other: MegaNumber): MegaNumber {
        // if other=0 => error
        if (other.mantissa.size == 1 && other.mantissa[0] == 0L) {
            throw ArithmeticException("Division by zero")
        }
        // If float => delegate
        if (this.isFloat || other.isFloat) {
            return divFloat(other)
        }
        // integer division
        val sign = (this.negative != other.negative)
        val c = compareAbs(this.mantissa, other.mantissa)
        if (c < 0) {
            return MegaNumber(
                mantissa = longArrayOf(0),
                exponent = longArrayOf(0),
                negative = false,
                isFloat = false,
                exponentNegative = false
            )
        } else if (c == 0) {
            return MegaNumber(
                mantissa = longArrayOf(1),
                exponent = longArrayOf(0),
                negative = sign,
                isFloat = false,
                exponentNegative = false
            )
        } else {
            val (q, _) = divChunks(this.mantissa, other.mantissa)
            return MegaNumber(
                mantissa = q,
                exponent = longArrayOf(0),
                negative = sign,
                isFloat = false,
                exponentNegative = false
            )
        }
    }

    /**
     * Float division using chunk-based arithmetic
     */
    open fun divFloat(other: MegaNumber): MegaNumber {
        // Divide mantissas
        val (quotientMant, _) = divChunks(this.mantissa, other.mantissa)
        // Subtract exponents
        val expA = exponentValue()
        val expB = other.exponentValue()
        val diffExp = expA - expB
        val newExponent = intToChunks(diffExp)
        // Determine sign
        val newNegative = (this.negative != other.negative)
        // Create result
        val out = MegaNumber(
            mantissa = quotientMant,
            exponent = newExponent,
            negative = newNegative,
            isFloat = true,
            exponentNegative = (diffExp < 0)
        )
        out.normalize()
        checkPrecisionLimit(out)
        return out
    }

    /**
     * Compute the square root of this MegaNumber.
     * For integer values, returns the integer square root.
     * For float values, returns a float approximation.
     *
     * @return The square root as a MegaNumber
     * @throws IllegalArgumentException if this MegaNumber is negative
     */
    open fun sqrt(): MegaNumber {
        if (negative) {
            throw IllegalArgumentException("Cannot compute square root of a negative number")
        }

        // If zero, return zero
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            return MegaNumber(
                mantissa = longArrayOf(0),
                exponent = longArrayOf(0),
                negative = false,
                isFloat = isFloat,
                exponentNegative = false
            )
        }

        // For integer values
        if (!isFloat) {
            // Use binary search to find the integer square root
            val a = mantissa.copyOf()
            var low = longArrayOf(0)
            var high = a.copyOf()

            while (true) {
                // mid = (low + high) / 2
                val sumLH = addChunks(low, high)
                val mid = div2(sumLH)

                // Check if we've converged
                val cLo = compareAbs(mid, low)
                val cHi = compareAbs(mid, high)
                if (cLo == 0 || cHi == 0) {
                    return MegaNumber(mid, longArrayOf(0), false)
                }

                // mid^2
                val midSqr = mulChunks(mid, mid)

                // Compare mid^2 with a
                val cCmp = compareAbs(midSqr, a)
                if (cCmp == 0) {
                    return MegaNumber(mid, longArrayOf(0), false)
                } else if (cCmp < 0) {
                    low = mid
                } else {
                    high = mid
                }
            }
        } else {
            // For float values, use floatSqrt
            return floatSqrt()
        }
    }

    /**
     * Compute the square root for float values.
     * 
     * @return The square root as a MegaNumber with float representation
     */
    private fun floatSqrt(): MegaNumber {
        // Get the exponent as an integer
        val totalExp = exponentValue()

        // Check if exponent is odd
        val remainder = totalExp and 1

        // Make a working copy of mantissa
        var workMantissa = mantissa.copyOf()
        var adjustedExp = totalExp

        // If exponent is odd, adjust mantissa and exponent
        if (remainder != 0) {
            if (totalExp > 0) {
                // Double the mantissa (shift left by 1 bit)
                var carry = 0L
                val result = LongArray(workMantissa.size + 1)
                for (i in workMantissa.indices) {
                    val doubled = (workMantissa[i] shl 1) + carry
                    result[i] = doubled and MegaNumberConstants.mask
                    carry = doubled ushr MegaNumberConstants.globalChunkSize
                }
                if (carry != 0L) {
                    result[workMantissa.size] = carry
                }
                workMantissa = result
                adjustedExp -= 1
            } else {
                // Halve the mantissa (shift right by 1 bit)
                val result = LongArray(workMantissa.size)
                var carry = 0L
                for (i in workMantissa.indices.reversed()) {
                    val value = workMantissa[i]
                    result[i] = (value ushr 1) or (carry shl (MegaNumberConstants.globalChunkSize - 1))
                    carry = value and 1L
                }
                workMantissa = result
                adjustedExp += 1
            }
        }

        // Half of exponent
        val halfExp = adjustedExp / 2

        // Do integer sqrt on workMantissa
        var low = longArrayOf(0)
        var high = workMantissa.copyOf()
        var sqrtMantissa: LongArray

        while (true) {
            // mid = (low + high) / 2
            val sumLH = addChunks(low, high)
            val mid = div2(sumLH)

            // Check if we've converged
            val cLo = compareAbs(mid, low)
            val cHi = compareAbs(mid, high)
            if (cLo == 0 || cHi == 0) {
                sqrtMantissa = mid
                break
            }

            // mid^2
            val midSqr = mulChunks(mid, mid)

            // Compare mid^2 with workMantissa
            val cCmp = compareAbs(midSqr, workMantissa)
            if (cCmp == 0) {
                sqrtMantissa = mid
                break
            } else if (cCmp < 0) {
                low = mid
            } else {
                high = mid
            }
        }

        // Create the result with half the exponent
        val expNeg = halfExp < 0
        val halfExpAbs = kotlin.math.abs(halfExp)
        val newExponent = if (halfExpAbs != 0) intToChunks(halfExpAbs) else longArrayOf(0)

        val out = MegaNumber(
            mantissa = sqrtMantissa,
            exponent = newExponent,
            negative = false,
            isFloat = true,
            exponentNegative = expNeg
        )
        out.normalize()
        checkPrecisionLimit(out)
        return out
    }

    /**
     * Divide a number by 2 (right shift by 1 bit).
     * 
     * @param limbs The limbs to divide
     * @return The result of dividing by 2
     */
    private fun div2(limbs: LongArray): LongArray {
        val result = LongArray(limbs.size)
        var carry = 0L

        for (i in limbs.indices.reversed()) {
            val value = limbs[i]
            result[i] = (value ushr 1) or (carry shl (MegaNumberConstants.globalChunkSize - 1))
            carry = value and 1L
        }

        // Trim trailing zeros
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0L) {
            lastNonZero--
        }

        return result.copyOf(lastNonZero + 1)
    }

    /**
     * Compute the binary logarithm (log2) of this MegaNumber.
     * 
     * @return The log2 value as a MegaNumber
     * @throws IllegalArgumentException if this MegaNumber is negative or zero
     */
    open fun log2(): MegaNumber {
        if (negative) {
            throw IllegalArgumentException("Cannot compute log2 of a negative number")
        }
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            throw IllegalArgumentException("Cannot compute log2 of zero")
        }

        // For simplicity, we'll use a more direct approach for log2
        // Find the highest set bit in the mantissa
        val highestBit = findHighestSetBit()

        // Adjust for the exponent
        val exponentVal = exponentValue()
        val log2Result = highestBit + exponentVal * MegaNumberConstants.globalChunkSize

        // Convert to MegaNumber
        val resultMantissa = intToChunks(log2Result)

        val out = MegaNumber(
            mantissa = resultMantissa,
            exponent = longArrayOf(0),
            negative = false,
            isFloat = true,
            exponentNegative = false
        )
        out.normalize()
        checkPrecisionLimit(out)
        return out
    }

    /**
     * Find the position of the highest set bit in the mantissa.
     * 
     * @return The position of the highest set bit
     */
    private fun findHighestSetBit(): Int {
        // Start from the highest limb
        for (i in mantissa.indices.reversed()) {
            val limb = mantissa[i]
            if (limb != 0L) {
                // Find the highest bit in this limb
                var bitPos = MegaNumberConstants.globalChunkSize - 1
                var mask = 1L shl bitPos
                while (bitPos >= 0) {
                    if ((limb and mask) != 0L) {
                        return i * MegaNumberConstants.globalChunkSize + bitPos
                    }
                    bitPos--
                    mask = mask ushr 1
                }
            }
        }
        return 0 // Should not reach here if number is non-zero
    }

    /**
     * Compute 2 raised to the power of this MegaNumber.
     * 
     * @return 2^this as a MegaNumber
     * @throws IllegalArgumentException if this MegaNumber is negative
     */
    open fun exp2(): MegaNumber {
        if (negative) {
            throw IllegalArgumentException("Cannot compute exp2 of a negative number")
        }

        // If zero, return 1
        if (mantissa.size == 1 && mantissa[0] == 0L) {
            return MegaNumber(longArrayOf(1), longArrayOf(0), false)
        }

        // For small pure integer
        if (!isFloat && exponent.size == 1 && exponent[0] == 0L) {
            val valInt = chunksToInt(mantissa)
            if (valInt >= 0 && valInt < 63) {  // Limit to avoid overflow
                val result = longArrayOf(1L shl valInt)
                return MegaNumber(result, longArrayOf(0), false)
            }
        }

        // For simplicity, we'll use a more direct approach for exp2
        // This is a simplified implementation for large values

        // For integer values, we can use bit shifting
        if (!isFloat) {
            val valInt = chunksToInt(mantissa)

            // For small values, use direct bit shifting
            if (valInt >= 0 && valInt < 30) {  // Limit to avoid overflow
                val result = longArrayOf(1L shl valInt)
                return MegaNumber(result, longArrayOf(0), false)
            }

            // For larger values, split into chunks
            val chunkShift = valInt / MegaNumberConstants.globalChunkSize
            val bitShift = valInt % MegaNumberConstants.globalChunkSize

            // Create a result with 1 shifted by bitShift
            val shiftedOne = longArrayOf(1L shl bitShift)

            // Create a result with appropriate exponent for the chunk shift
            return MegaNumber(
                mantissa = shiftedOne,
                exponent = longArrayOf(chunkShift.toLong()),
                negative = false,
                isFloat = true,
                exponentNegative = false
            )
        }

        // For float values, use a simplified approach
        // Get the integer and fractional parts
        val intPart = chunksToInt(mantissa)

        // 2^intPart
        val intPartResult = if (intPart < 30) {
            longArrayOf(1L shl intPart)
        } else {
            // For larger values, use exponent
            val chunkShift = intPart / MegaNumberConstants.globalChunkSize
            val bitShift = intPart % MegaNumberConstants.globalChunkSize
            val shiftedOne = longArrayOf(1L shl bitShift)

            return MegaNumber(
                mantissa = shiftedOne,
                exponent = longArrayOf(chunkShift.toLong()),
                negative = false,
                isFloat = true,
                exponentNegative = false
            )
        }

        // For fractional part, use Taylor series approximation
        val exponentVal = exponentValue()
        if (exponentVal != 0) {
            // If there's a non-zero exponent, use it
            val fracPartResult = exp2Frac(0.5) // Simplified approximation

            // Combine results
            val resultMantissa = mulChunks(intPartResult, fracPartResult)

            val out = MegaNumber(
                mantissa = resultMantissa,
                exponent = longArrayOf(exponentVal.toLong()),
                negative = false,
                isFloat = true,
                exponentNegative = exponentNegative
            )
            out.normalize()
            checkPrecisionLimit(out)
            return out
        }

        // If no fractional part, return the integer part
        val out = MegaNumber(
            mantissa = intPartResult,
            exponent = longArrayOf(0),
            negative = false,
            isFloat = false,
            exponentNegative = false
        )
        out.normalize()
        checkPrecisionLimit(out)
        return out
    }

    /**
     * Compute 2^x for 0 <= x < 1 using Taylor series approximation.
     * 
     * @param x The fractional part of the exponent
     * @return Approximation of 2^x as a LongArray
     */
    private fun exp2Frac(x: Double): LongArray {
        val terms = 10
        var result = longArrayOf(1)  // First term
        var factorial = 1.0
        var power = 1.0
        val ln2 = 0.693147

        for (i in 1 until terms) {
            power *= x * ln2
            factorial *= i
            // Scale factor
            val termVal = (power / factorial * (1L shl MegaNumberConstants.globalChunkSize)).toLong()
            // Add in chunk-based
            result = addChunks(result, longArrayOf(termVal))
        }

        return result
    }

    /**
     * Return the exponent part of this MegaNumber as a new MegaNumber.
     * 
     * @return The exponent as a MegaNumber
     */
    open fun exp(): MegaNumber {
        if (exponent.size == 1 && exponent[0] == 0L) {
            return MegaNumber(longArrayOf(0), longArrayOf(0), false)
        }

        val out = MegaNumber(
            mantissa = exponent.copyOf(),
            exponent = longArrayOf(0),
            negative = exponentNegative,
            isFloat = false,
            exponentNegative = false
        )
        out.normalize()
        return out
    }

    /**
     * Negate the MegaNumber (for subtraction)
     */
    fun negate(): MegaNumber {
        return MegaNumber(
            mantissa = mantissa.copyOf(),
            exponent = exponent.copyOf(),
            negative = !this.negative,
            isFloat = this.isFloat,
            exponentNegative = this.exponentNegative
        )
    }

    /**
     * Create a copy of this MegaNumber
     */
    open fun copy(): MegaNumber {
        return MegaNumber(
            mantissa = mantissa.copyOf(),
            exponent = exponent.copyOf(),
            negative = negative,
            isFloat = isFloat,
            exponentNegative = exponentNegative,
            keepLeadingZeros = keepLeadingZeros
        )
    }

    /**
     * String representation
     */
    override fun toString(): String {
        return "<MegaNumber ${toDecimalString()}>"
    }

    // Operator overloads
    operator fun plus(other: MegaNumber): MegaNumber = add(other)
    operator fun minus(other: MegaNumber): MegaNumber = sub(other)
    operator fun times(other: MegaNumber): MegaNumber = mul(other)
    operator fun div(other: MegaNumber): MegaNumber = divide(other)
}
