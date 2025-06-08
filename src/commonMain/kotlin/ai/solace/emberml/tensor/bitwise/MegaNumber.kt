/**
 * Kotlin Native implementation of MegaNumber, the foundation for arbitrary precision arithmetic.
 *
 * This class provides a chunk-based big integer (or float) with arbitrary precision arithmetic,
 * using Int arrays with 32-bit values.
 */
package ai.solace.emberml.tensor.bitwise

/**
 * Constants used by MegaNumber implementation
 */
object MegaNumberConstants {
    const val GLOBAL_CHUNK_SIZE: Int = 32 // Optimum size for O(n) complexity drop in certain operations
    const val base: Int = 1 shl GLOBAL_CHUNK_SIZE // 2^32, used for carry/borrow
    const val mask: Long = 0xFFFFFFFFL // Use Long for proper unsigned handling

    // Thresholds for picking naive vs. Karatsuba vs. Toom-3
    const val MUL_THRESHOLD_KARATSUBA = 64 // Increased to use standard multiplication for smaller numbers
    const val MUL_THRESHOLD_TOOM = 128 // Increased to use Karatsuba for medium-sized numbers
    // Maximum precision in bits (limited to avoid excessive memory usage)
    var maxPrecisionBits: Int? = 1024 // 1024 bits should be enough for most use cases
}

/**
 * A chunk-based big integer (or float) with arbitrary precision arithmetic,
 * using IntArray with 32-bit values.
 *
 * @property mantissa     IntArray of limbs (32-bit chunks)
 * @property exponent     `MegaNumber` representing the binary‑exponent; its `negative`
 *                        flag encodes whether the overall exponent is positive
 *                        or negative, and its `mantissa` holds the magnitude in
 *                        32‑bit limbs.
 * @property negative     Sign flag
 * @property isFloat      Float flag
 * @property keepLeadingZeros Whether to keep leading zeros
 */
open class MegaNumber(
    var mantissa: IntArray = intArrayOf(0),
    var exponent: MegaNumber = MegaNumber(intArrayOf(0)),
    var negative: Boolean = false,
    var isFloat: Boolean = false,
    val keepLeadingZeros: Boolean = false
) {
    companion object {
        /**
         * Right shift chunk-limbs by 1 bit => integer //2.
         */
        internal fun div2(limbs: IntArray): IntArray {
            val out = IntArray(limbs.size)
            var carry = 0
            val csize = MegaNumberConstants.GLOBAL_CHUNK_SIZE

            for (i in limbs.indices.reversed()) {
                val value = (carry shl csize) + limbs[i]
                val q = value shr 1
                carry = value and 1
                out[i] = q
            }

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0) {
                lastNonZero--
            }

            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Add two chunk-limb arrays => sum-limb array
         */
        internal fun addChunks(a: IntArray, b: IntArray): IntArray {
            val maxLen = maxOf(a.size, b.size)
            val out = IntArray(maxLen + 1)
            var carry = 0L

            for (i in 0 until maxLen) {
                val av = if (i < a.size) a[i].toLong() and 0xFFFFFFFFL else 0L
                val bv = if (i < b.size) b[i].toLong() and 0xFFFFFFFFL else 0L
                val s = av + bv + carry
                out[i] = (s and 0xFFFFFFFFL).toInt()
                carry = s ushr 32
            }

            if (carry != 0L) {
                out[maxLen] = carry.toInt()
                return out
            }

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0) {
                lastNonZero--
            }

            return if (lastNonZero == out.size - 1) out else out.copyOf(lastNonZero + 1)
        }

        /**
         * Subtract B from A (assuming A >= B), returning chunk-limb array
         */
        internal fun subChunks(a: IntArray, b: IntArray, preserveSize: Boolean = true): IntArray {
            // Same implementation as above until the trimming part
            val out = IntArray(a.size)
            var borrow = 0L

            for (i in 0 until a.size) {
                val av = a[i].toLong() and 0xFFFFFFFFL
                val bv = if (i < b.size) b[i].toLong() and 0xFFFFFFFFL else 0L
                var diff = av - bv - borrow

                if (diff < 0) {
                    diff += 0x100000000L
                    borrow = 1L
                } else {
                    borrow = 0L
                }

                out[i] = (diff and 0xFFFFFFFFL).toInt()
            }

            if (preserveSize) {
                return out
            }

            // Find last non-zero element
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0) {
                lastNonZero--
            }

            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Compare absolute magnitude of A vs. B => -1 if A<B, 0 if ==, 1 if A>B
         */
        internal fun compareAbs(a: IntArray, b: IntArray): Int {
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
        internal fun mulChunksStandard(a: IntArray, b: IntArray): IntArray {
            if (a.size == 1 && a[0] == 0) return intArrayOf(0)
            if (b.size == 1 && b[0] == 0) return intArrayOf(0)

            val la = a.size
            val lb = b.size
            val out = IntArray(la + lb)

            for (i in 0 until la) {
                val av = a[i].toLong() and 0xFFFFFFFFL
                var carry = 0L

                for (j in 0 until lb) {
                    val bv = b[j].toLong() and 0xFFFFFFFFL
                    val existing = out[i + j].toLong() and 0xFFFFFFFFL
                    val prod = av * bv + existing + carry
                    out[i + j] = (prod and 0xFFFFFFFFL).toInt()
                    carry = prod ushr 32
                }

                if (carry != 0L) {
                    out[i + lb] = carry.toInt()
                }
            }

            // Trim trailing zeros
            var lastNonZero = out.size - 1
            while (lastNonZero > 0 && out[lastNonZero] == 0) {
                lastNonZero--
            }

            return out.copyOf(lastNonZero + 1)
        }

        /**
         * Multiply two chunk-limb arrays => product-limb array
         * Dispatches to naive, Karatsuba, or Toom-3 based on thresholds
         */
        internal fun mulChunks(a: IntArray, b: IntArray): IntArray {
            val n = maxOf(a.size, b.size)
            return if (n < MegaNumberConstants.MUL_THRESHOLD_KARATSUBA) {
                mulChunksStandard(a, b)
            } else if (n < MegaNumberConstants.MUL_THRESHOLD_TOOM) {
                karatsubaMulChunks(a, b)
            } else {
                // For now, we'll use Karatsuba for Toom-3 threshold too
                // TODO: Implement Toom-3 algorithm
                karatsubaMulChunks(a, b)
            }
        }

        /**
         * Implements Karatsuba multiplication for large numbers.
         */
        internal fun karatsubaMulChunks(a: IntArray, b: IntArray): IntArray {
            val n = maxOf(a.size, b.size)
            if (n <= 32) {
                return mulChunksStandard(a, b) // Use standard multiplication for small sizes
            }

            val m = n / 2

            val aLow = a.copyOf(minOf(m, a.size))
            val aHigh = if (m < a.size) a.copyOfRange(m, a.size) else intArrayOf(0)
            val bLow = b.copyOf(minOf(m, b.size))
            val bHigh = if (m < b.size) b.copyOfRange(m, b.size) else intArrayOf(0)

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
        internal fun shiftLeft(limbs: IntArray, shift: Int): IntArray {
            if (shift <= 0) return limbs.copyOf()
            val result = IntArray(limbs.size + shift)
            limbs.copyInto(result, shift)
            return result
        }

        /**
         * Convert an Int into chunk-limbs. A zero value => [0].
         */
        internal fun intToChunks(val_: Int): IntArray {
        if (val_ == 0) return intArrayOf(0)

        // If the chunk size is the full 32 bits we can store the value in a single limb
        if (MegaNumberConstants.GLOBAL_CHUNK_SIZE >= 32) {
            return intArrayOf(val_)
        }

        var x = val_
        val chunkMask = (1 shl MegaNumberConstants.GLOBAL_CHUNK_SIZE) - 1
        val out = mutableListOf<Int>()

        while (x != 0) {
            out.add(x and chunkMask)
            // Use **unsigned** right‑shift so the loop terminates even for negative numbers
            x = x ushr MegaNumberConstants.GLOBAL_CHUNK_SIZE
        }
        return out.toIntArray()
        }

        /**
         * Convert an Int into chunk-limbs with specified chunk size. A zero value => [0].
         * This overload is provided for compatibility with MegaBinary.
         */
        internal fun intToChunks(val_: Int, chunkSize: Int): IntArray {
        if (val_ == 0) return intArrayOf(0)

        // If the requested chunk size is >= 32 bits we only need one limb
        if (chunkSize >= 32) {
            return intArrayOf(val_)
        }

        var x = val_
        val chunkMask = (1 shl chunkSize) - 1
        val out = mutableListOf<Int>()

        while (x != 0) {
            out.add(x and chunkMask)
            x = x ushr chunkSize   // unsigned shift to avoid sign extension loops
        }
        return out.toIntArray()
        }

        /**
         * Convert an Int into an IntArray with a single element.
         */
        internal fun intToIntArray(val_: Int): IntArray {
            return IntArray(1) { val_ }
        }


        /**
         * Combine chunk-limbs => a single Int. (May overflow if large.)
         */
        internal fun chunksToInt(limbs: IntArray): Int {
            /* Convert chunk-limbs to a single Int value.
             * This is a simple conversion, but may overflow if the value is too large.
             * This is needed for exponent calculations, where we assume the exponent fits in an Int.
             */
            var val_ = 0
            var shift = 0
            for (limb in limbs) {
                val part = limb shl shift
                val_ += part  // May overflow
                shift += MegaNumberConstants.GLOBAL_CHUNK_SIZE
            }
            return val_ // Convert to Int, may overflow
        }


        /**
         * Convert decimal string => chunk-limb array
         */
        internal fun decimalStringToChunks(dec: String): IntArray {
            if (dec.isEmpty()) return intArrayOf(0)
            if (dec == "0") return intArrayOf(0)

            var limbs = intArrayOf(0)
            for (ch in dec) {
                if (ch < '0' || ch > '9') {
                    throw IllegalArgumentException("Invalid decimal digit in $dec")
                }
                val digit : Int = (ch - '0')
                // Multiply limbs by 10, then add digit
                limbs = addChunks(mulChunks(limbs, intArrayOf(10)), intArrayOf(digit))
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
                    exponent = MegaNumber(intArrayOf(0)),
                    negative = negative,
                    isFloat = false
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
                val expChunks = intToIntArray(shiftBits)

                return MegaNumber(
                    mantissa = mant,
                    exponent = MegaNumber(expChunks, negative = true),
                    negative = negative,
                    isFloat = true
                )
            }
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
            val totalBits = num.mantissa.size * MegaNumberConstants.GLOBAL_CHUNK_SIZE
            if (totalBits > maxBits) {
                throw IllegalStateException("Precision limit exceeded: $totalBits bits > $maxBits bits")
            }
        }
    }


    /**
     * Convert a chunk list to an integer (for compatibility with MegaBinary)
     */
    internal fun compare(a: MegaNumber, b: MegaNumber): Int {
        if (a.negative != b.negative) {
            return if (a.negative) -1 else 1
        }

        val absCompare = compareAbs(a.mantissa, b.mantissa)
        return if (a.negative) -absCompare else absCompare
    }

    /**
     * Remove trailing zeros, handle zero sign, etc.
     */
    open fun normalize() {
        if (!keepLeadingZeros) {
            // Trim mantissa
            var lastNonZero = mantissa.size - 1
            while (lastNonZero > 0 && mantissa[lastNonZero] == 0) {
                lastNonZero--
            }
            if (lastNonZero < mantissa.size - 1) {
                mantissa = mantissa.copyOf(lastNonZero + 1)
            }

            // Trim exponent mantissa if this number is a float
            if (isFloat) {
                lastNonZero = exponent.mantissa.size - 1
                while (lastNonZero > 0 && exponent.mantissa[lastNonZero] == 0) {
                    lastNonZero--
                }
                if (lastNonZero < exponent.mantissa.size - 1) {
                    exponent.mantissa = exponent.mantissa.copyOf(lastNonZero + 1)
                }
            }

            // If zero => unify sign
            if (mantissa.size == 1 && mantissa[0] == 0) {
                negative = false
                exponent = MegaNumber(intArrayOf(0))
            }
        } else {
            // keepLeadingZeros = true: do not trim mantissa/exponent, but unify sign if all zero
            if (mantissa.all { it == 0 }) {
                negative = false
            }
        }
    }

    /**
     * Implement chunk-based right shift.
     *
     * @param shiftBits Number of bits to shift; must be >= 0.
     */
    internal fun shiftRight(limbs: IntArray, shiftBits: Int): IntArray {
        // Shift count must be non-negative.
        if (shiftBits < 0) {
            throw IllegalArgumentException("shiftBits must be non-negative")
        }
        if (shiftBits == 0) {
            // No shift – return a copy so callers can freely mutate the result
            return limbs.copyOf()
        }

        val chunkShift = shiftBits / 32
        val bitShift   = shiftBits % 32

        if (chunkShift >= limbs.size) {
            return intArrayOf(0)
        }

        // Create result array
        val resultSize = limbs.size - chunkShift
        val result     = IntArray(resultSize)

        if (bitShift == 0) {
            // Just copy the chunks
            for (i in 0 until resultSize) {
                result[i] = limbs[i + chunkShift]
            }
        } else {
            // Shift with carry
            for (i in 0 until resultSize) {
                val currentChunk = limbs[i + chunkShift].toLong() and 0xFFFFFFFFL
                val nextChunk    = if (i + chunkShift + 1 < limbs.size) {
                    limbs[i + chunkShift + 1].toLong() and 0xFFFFFFFFL
                } else {
                    0L
                }

                // Take upper bits from next chunk and lower bits from current
                result[i] = (((nextChunk shl (32 - bitShift)) or
                        (currentChunk ushr bitShift)) and 0xFFFFFFFFL).toInt()
            }
        }

        // Zero-fill: discard the least-significant `bitShift` bits that were shifted out.
        if (bitShift != 0) {
            val mask = -1 shl bitShift      // e.g. bitShift=16 → 0xFFFF0000
            result[0] = result[0] and mask  // clear those lower bits
        }

        // Trim trailing zeros
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0) {
            lastNonZero--
        }

        return if (lastNonZero < result.size - 1) {
            result.copyOf(lastNonZero + 1)
        } else {
            result
        }
    }

    /**
     * Divide chunk-limb arrays => (quotient, remainder), integer division
     */
    private fun chunkDivide(a: IntArray, b: IntArray): Pair<IntArray, IntArray> {
        // B must not be zero
        if (b.size == 1 && b[0] == 0) {
            throw ArithmeticException("Division by zero")
        }
        val c = compareAbs(a, b)
        if (c < 0) return Pair(intArrayOf(0), a.copyOf()) // A<B => Q=0, R=A
        if (c == 0) return Pair(intArrayOf(1), intArrayOf(0)) // A=B => Q=1, R=0

        val q = IntArray(a.size)
        var r = intArrayOf(0)

        // We do a standard chunk-based short division
        for (i in a.indices.reversed()) {
            // shift R left by one chunk
            r = IntArray(r.size + 1).also {
                r.copyInto(it, 1)
                it[0] = a[i]
            }

            // binary search in [0..BASE-1] for the best q
            var low = 0
            var high = Int.MAX_VALUE // Approximation for BASE-1
            var guess = 0

            while (low <= high) {
                val mid = (low + high) shr 1
                val mm = mulChunks(b, intArrayOf(mid))
                val cmpv = compareAbs(mm, r)
                if (cmpv <= 0) {
                    guess = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }
            if (guess != 0) {
                val mm = mulChunks(b, intArrayOf(guess))
                r = subChunks(r, mm)
            }
            q[i] = guess
        }

        // Trim q
        var lastNonZero = q.size - 1
        while (lastNonZero > 0 && q[lastNonZero] == 0) {
            lastNonZero--
        }
        return Pair(q.copyOf(lastNonZero + 1), r)
    }

    /**
     * Divmod by small_val <= BASE
     */
    private fun divMod10(a: IntArray, smallVal: Int): Pair<IntArray, Int> {
        var remainder = 0
        val out = IntArray(a.size)

        for (i in a.indices.reversed()) {
            // Shift the remainder left by GLOBAL_CHUNK_SIZE bits and add the current limb
            val cur = (remainder shl MegaNumberConstants.GLOBAL_CHUNK_SIZE) + a[i]

            // Compute the quotient digit and the new remainder
            val qd = cur / smallVal
            remainder = cur % smallVal

            // Assign the quotient digit to the output array, ensuring it fits within the chunk mask
            out[i] = qd and MegaNumberConstants.mask.toInt()
        }

        // Trim any unnecessary trailing zeros from the output array
        var lastNonZero = out.size - 1
        while (lastNonZero > 0 && out[lastNonZero] == 0) {
            lastNonZero--
        }

        return Pair(out.copyOf(lastNonZero + 1), remainder)
    }

    /**
     * Convert chunk-limbs to decimal string
     */
    internal fun chunkToDecimal(limbs: IntArray): String {
        // quick check for zero
        if (limbs.size == 1 && limbs[0] == 0) {
            return "0"
        }
        var temp = limbs.copyOf()
        val digits = mutableListOf<Char>()
        while (!(temp.size == 1 && temp[0] == 0)) {
            val (q, r) = divMod10(temp, 10)
            temp = q
            digits.add('0' + r)
        }
        return digits.reversed().joinToString("")
    }

    /** Treat `this` (a MegaNumber exponent) as a signed Int */
    private fun MegaNumber.expAsInt(): Int {
        val absVal = chunksToInt(this.mantissa)
        return if (this.negative) -absVal else absVal
    }

    /**
     * Return a decimal-string representation. (Integer-only if exponent=0.)
     */
    open fun toDecimalString(): String {
        // If zero
        if (mantissa.size == 1 && mantissa[0] == 0) {
            return "0"
        }

        // If exponent is zero or we are integer => treat as integer
        val expNonZero = !(exponent.mantissa.size == 1 && exponent.mantissa[0] == 0)
        if (!expNonZero) {
            // purely integer
            val s = chunkToDecimal(mantissa)
            return (if (negative) "-" else "") + s
        } else {
            // float => represent as "mantissa * 2^(exponent * chunkBits)" for simplicity
            val eVal = if (exponent.negative) {
                -chunksToInt(exponent.mantissa)
            } else {
                chunksToInt(exponent.mantissa)
            }
            val mantString = chunkToDecimal(mantissa)
            val signStr = if (negative) "-" else ""
            // This is a simplistic representation.
            return "$signStr$mantString * 2^($eVal * ${MegaNumberConstants.GLOBAL_CHUNK_SIZE})"
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
                exponent = MegaNumber(intArrayOf(0)),
                negative = sign,
                isFloat = false
            )
        } else {
            // Different signs => subtract magnitudes
            val cmp = compareAbs(this.mantissa, other.mantissa)
            if (cmp == 0) {
                // Result is zero
                return MegaNumber(
                    mantissa = intArrayOf(0),
                    exponent = MegaNumber(intArrayOf(0)),
                    negative = false,
                    isFloat = false
                )
            } else if (cmp > 0) {
                // self > other in magnitude
                val diff = subChunks(this.mantissa, other.mantissa)
                val sign = this.negative
                return MegaNumber(
                    mantissa = diff,
                    exponent = MegaNumber(intArrayOf(0)),
                    negative = sign,
                    isFloat = false
                )
            } else {
                // other > self in magnitude
                val diff = subChunks(other.mantissa, this.mantissa)
                val sign = other.negative
                return MegaNumber(
                    mantissa = diff,
                    exponent = MegaNumber(intArrayOf(0)),
                    negative = sign,
                    isFloat = false
                )
            }
        }
    }

    /**
     * Float addition using chunk-based arithmetic
     */
    open fun addFloat(other: MegaNumber): MegaNumber {
        // Signed exponents as Int
        val expA = this.exponent.expAsInt()
        val expB = other.exponent.expAsInt()

        // Align mantissas
        var mantA = this.mantissa.copyOf()
        var mantB = other.mantissa.copyOf()
        val finalExp: MegaNumber

        if (expA > expB) {
            mantB = shiftRight(mantB, expA - expB)
            finalExp = this.exponent
        } else if (expB > expA) {
            mantA = shiftRight(mantA, expB - expA)
            finalExp = other.exponent
        } else {
            finalExp = this.exponent          // equal exponents
        }

        // Combine mantissas
        val sameSign = (this.negative == other.negative)
        val resultMant: IntArray
        val resultNeg: Boolean

        if (sameSign) {
            resultMant = addChunks(mantA, mantB)
            resultNeg  = this.negative
        } else {
            val cmp = compareAbs(mantA, mantB)
            when {
                cmp == 0 -> return MegaNumber(intArrayOf(0))   // exact zero
                cmp > 0  -> {
                    resultMant = subChunks(mantA, mantB)
                    resultNeg  = this.negative
                }
                else     -> {
                    resultMant = subChunks(mantB, mantA)
                    resultNeg  = other.negative
                }
            }
        }

        val out = MegaNumber(
            mantissa = resultMant,
            exponent = finalExp,
            negative = resultNeg,
            isFloat  = true
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
            exponent = MegaNumber(other.exponent.mantissa.copyOf(), negative = other.exponent.negative),
            negative = !other.negative,
            isFloat  = other.isFloat
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
            exponent = MegaNumber(intArrayOf(0)),
            negative = sign,
            isFloat = false
        )
    }

    /**
     * Float multiplication using chunk-based arithmetic
     */
    open fun mulFloat(other: MegaNumber): MegaNumber {
        // Multiply mantissas
        val productMant = mulChunks(this.mantissa, other.mantissa)

        // Add exponents (signed)
        val sumExp = this.exponent.expAsInt() + other.exponent.expAsInt()
        val newExponent = MegaNumber(intArrayOf(kotlin.math.abs(sumExp)), negative = sumExp < 0)

        // Determine sign
        val newNegative = (this.negative != other.negative)

        // Create result
        val out = MegaNumber(
            mantissa = productMant,
            exponent = newExponent,
            negative = newNegative,
            isFloat = true
        )
        out.normalize()
        return out
    }

    /** Integer division branch, used when *both* numbers are plain integers */
    private fun divideInteger(other: MegaNumber): MegaNumber {
        /* --- fast path for single‑chunk integers --------------------------- */
        if (!this.isFloat && !other.isFloat &&
            this.mantissa.size == 1 && other.mantissa.size == 1 &&
            this.exponent.mantissa.size == 1 && this.exponent.mantissa[0] == 0 &&
            other.exponent.mantissa.size == 1 && other.exponent.mantissa[0] == 0
        ) {
            val lhs = this.mantissa[0].toUInt()
            val rhs = other.mantissa[0].toUInt()
            require(rhs != 0u) { "Division by zero" }
            val neg = this.negative xor other.negative
            val q = (lhs / rhs).toInt()
            return MegaNumber(intArrayOf(q), MegaNumber(intArrayOf(0)), neg, false)
        }
        /* --- long‑form integer division ------------------------------------ */
        if (other.mantissa.size == 1 && other.mantissa[0] == 0) {
            throw ArithmeticException("Division by zero")
        }
        val sign = (this.negative != other.negative)
        val cmp = compareAbs(this.mantissa, other.mantissa)
        return when {
            cmp < 0 -> MegaNumber(intArrayOf(0))
            cmp == 0 -> MegaNumber(intArrayOf(1), MegaNumber(intArrayOf(0)), sign, false)
            else -> {
                val (q, _) = chunkDivide(this.mantissa, other.mantissa)
                MegaNumber(q, MegaNumber(intArrayOf(0)), sign, false)
            }
        }
    }

    /** Float division branch used when either operand is float */
    private fun divideFloat(other: MegaNumber): MegaNumber {
        // Divide mantissas
        val (quotientMant, _) = chunkDivide(this.mantissa, other.mantissa)
        // Subtract exponents
        val diffExp = this.exponent.expAsInt() - other.exponent.expAsInt()
        val newExponent = MegaNumber(intArrayOf(kotlin.math.abs(diffExp)), negative = diffExp < 0)
        // Determine sign
        val newNegative = (this.negative != other.negative)
        val out = MegaNumber(
            mantissa = quotientMant,
            exponent = newExponent,
            negative = newNegative,
            isFloat = true
        )
        out.normalize()
        return out
    }

    /**
     * Divide two MegaNumbers. If either is float, delegate to float division
     */
    open fun divide(other: MegaNumber): MegaNumber {
        // Unified public entry‑point – dispatches to integer or float path
        return if (this.isFloat || other.isFloat) {
            divideFloat(other)
        } else {
            divideInteger(other)
        }
    }

    // (divFloat removed; logic now in divideFloat)

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
        if (mantissa.size == 1 && mantissa[0] == 0) {
            return MegaNumber(
                mantissa = intArrayOf(0),
                exponent = MegaNumber(intArrayOf(0)),
                negative = false,
                isFloat = isFloat
            )
        }

        // For integer values
        if (!isFloat) {
            // Use binary search to find the integer square root
            val a = mantissa.copyOf()
            var low = intArrayOf(0)
            var high = a.copyOf()

            while (true) {
                // mid = (low + high) / 2
                val sumLH = addChunks(low, high)
                val mid = div2(sumLH)

                // Check if we've converged
                val cLo = compareAbs(mid, low)
                val cHi = compareAbs(mid, high)
                if (cLo == 0 || cHi == 0) {
                    return MegaNumber(mid, MegaNumber(intArrayOf(0)), false)
                }

                // mid^2
                val midSqr = mulChunks(mid, mid)

                // Compare mid^2 with a
                val cCmp = compareAbs(midSqr, a)
                if (cCmp == 0) {
                    return MegaNumber(mid, MegaNumber(intArrayOf(0)), false)
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
        val totalExp = this.exponent.expAsInt()

        // Check if exponent is odd
        val remainder = totalExp and 1          // 1 = odd, 0 = even

        // Make a working copy of mantissa
        var workMantissa = mantissa.copyOf()
        var adjustedExp = totalExp

        // If exponent is odd, adjust mantissa and exponent
        if (remainder != 0) {
            if (totalExp > 0) {
                // Double the mantissa (shift left by 1 bit)
                var carry = 0
                val result = IntArray(workMantissa.size + 1)
                for (i in workMantissa.indices) {
                    val doubled = (workMantissa[i] shl 1) + carry
                    result[i] = doubled and MegaNumberConstants.mask.toInt()
                    carry = doubled shr MegaNumberConstants.GLOBAL_CHUNK_SIZE
                }
                if (carry != 0) {
                    result[workMantissa.size] = carry
                }
                workMantissa = result
                adjustedExp = adjustedExp - 1
            } else {
                // Halve the mantissa (shift right by 1 bit)
                val result = IntArray(workMantissa.size)
                var carry = 0
                for (i in workMantissa.indices.reversed()) {
                    val value = workMantissa[i]
                    result[i] = (value shr 1) or (carry shl (MegaNumberConstants.GLOBAL_CHUNK_SIZE - 1))
                    carry = value and 1
                }
                workMantissa = result
                adjustedExp = adjustedExp + 1
            }
        }

        // Half of exponent
        val halfExp = adjustedExp / 2

        // Do integer sqrt on workMantissa
        var low = intArrayOf(0)
        var high = workMantissa.copyOf()
        var sqrtMantissa: IntArray

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
        val newExponent = MegaNumber(intArrayOf(kotlin.math.abs(halfExp)), negative = halfExp < 0)

        val out = MegaNumber(
            mantissa = sqrtMantissa,
            exponent = newExponent,
            negative = false,
            isFloat = true
        )
        out.normalize()
        checkPrecisionLimit(out)
        return out
    }

    /**
     * Divide chunks by 2^bits, returning quotient and remainder.
     * This uses IntArray for proper 32-bit chunk operations.
     *
     * @param chunks The chunks to divide
     * @param bits The power of 2 to divide by
     * @return Pair of (quotient, remainder)
     */
    protected fun divideBy2ToThePower(chunks: IntArray, bits: Int): Pair<IntArray, IntArray> {
        if (bits <= 0) {
            return Pair(chunks.copyOf(), intArrayOf(0))
        }

        // Use the chunks directly since they are already IntArray
        val intChunks = chunks.copyOf()

        // Calculate whole chunk shifts and bit shifts within chunks
        val chunkShift = bits / MegaNumberConstants.GLOBAL_CHUNK_SIZE
        val bitShift = bits % MegaNumberConstants.GLOBAL_CHUNK_SIZE

        // Handle chunk-level right shift
        val quotientInt = if (chunkShift >= intChunks.size) {
            IntArray(1) { 0 }
        } else {
            IntArray(intChunks.size - chunkShift) { i ->
                if (i + chunkShift < intChunks.size) {
                    intChunks[i + chunkShift]
                } else {
                    0
                }
            }
        }

        // Handle bit-level right shift within chunks
        val finalQuotient = if (bitShift > 0) {
            IntArray(quotientInt.size) { i ->
                val current = quotientInt[i]
                val carry = if (i + 1 < quotientInt.size) {
                    (quotientInt[i + 1] shl (MegaNumberConstants.GLOBAL_CHUNK_SIZE - bitShift)) and MegaNumberConstants.mask.toInt()
                } else {
                    0
                }
                ((current shr bitShift) or carry)
            }
        } else {
            quotientInt
        }

        // Convert back to IntArray
        val quotient = IntArray(finalQuotient.size) { finalQuotient[it] }

        // Calculate remainder: original - (quotient << bits)
        val quotientShifted = multiplyBy2ToThePower(quotient, bits)
        val remainder = if (compareAbs(chunks, quotientShifted) >= 0) {
            subChunks(chunks, quotientShifted)
        } else {
            intArrayOf(0)
        }

        return Pair(quotient, remainder)
    }

    /**
     * Multiply chunks by 2^bits using IntArray for proper 32-bit operations.
     */
    protected fun multiplyBy2ToThePower(chunks: IntArray, bits: Int): IntArray {
        if (bits <= 0) {
            return chunks.copyOf()
        }

        // Use the chunks directly since they are already IntArray
        val intChunks = chunks.copyOf()

        // Calculate whole chunk shifts and bit shifts within chunks
        val chunkShift = bits / MegaNumberConstants.GLOBAL_CHUNK_SIZE
        val bitShift = bits % MegaNumberConstants.GLOBAL_CHUNK_SIZE

        // Handle chunk-level left shift (add zero chunks at the beginning)
        val expandedSize = intChunks.size + chunkShift + if (bitShift > 0) 1 else 0
        val shiftedInt = IntArray(expandedSize) { i ->
            if (i < chunkShift) {
                0
            } else if (i - chunkShift < intChunks.size) {
                intChunks[i - chunkShift]
            } else {
                0
            }
        }

        // Handle bit-level left shift within chunks
        val finalResult = if (bitShift > 0) {
            var carry = 0
            IntArray(shiftedInt.size) { i ->
                val current = shiftedInt[i]
                val shifted = (current shl bitShift) or carry
                carry = (shifted shr MegaNumberConstants.GLOBAL_CHUNK_SIZE) and MegaNumberConstants.mask.toInt()
                (shifted and MegaNumberConstants.mask.toInt())
            }
        } else {
            shiftedInt
        }

        // Convert back to IntArray and trim trailing zeros
        val result = IntArray(finalResult.size) { finalResult[it] }
        var lastNonZero = result.size - 1
        while (lastNonZero > 0 && result[lastNonZero] == 0) {
            lastNonZero--
        }
        return result.copyOf(lastNonZero + 1)
    }

}
