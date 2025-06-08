# MegaNumber and MegaBinary: Arbitrary Precision Arithmetic in Kotlin Multiplatform

## Overview

The MegaNumber and MegaBinary classes provide arbitrary precision arithmetic capabilities
for the Ember ML Kotlin Multiplatform library. These classes enable precise mathematical
operations beyond the limits of standard numeric types, with specialized support for
binary operations.

## Class Hierarchy

- **MegaNumber**: Base class providing arbitrary precision arithmetic for both integers and floats
- **MegaBinary**: Specialized subclass for binary data representation and bitwise operations

## Internal Representation

### MegaNumber

MegaNumber uses a chunk-based representation with the following key properties:

- **mantissa**: IntArray of 32-bit chunks (limbs) storing the significant digits
- **exponent**: Another MegaNumber representing the binary exponent
- **negative**: Boolean flag indicating the sign
- **isFloat**: Boolean flag indicating whether the number is a float
- **keepLeadingZeros**: Boolean flag controlling normalization behavior

The number is conceptually represented as: mantissa * 2^exponent, with the sign determined
by the negative flag.

### MegaBinary

MegaBinary extends MegaNumber with additional properties:

- **byteData**: ByteArray representation of the binary data
- **bitLength**: Length of the binary representation in bits

## Key Algorithms

### Multiplication Algorithms

MegaNumber implements three multiplication algorithms with different performance characteristics:

1. **Standard Multiplication**: O(n²) algorithm used for small numbers
2. **Karatsuba Multiplication**: O(n^log₂3) ≈ O(n^1.585) algorithm for medium-sized numbers
3. **Toom-3 Multiplication**: O(n^log₃5) ≈ O(n^1.465) algorithm for large numbers

The appropriate algorithm is selected automatically based on the size of the operands.

### Division

Division is implemented using chunk-based short division with binary search for the quotient.
Special optimizations are included for division by powers of two.

### Square Root

Square root is implemented using binary search for both integer and float values.
For float values, the exponent is handled separately to ensure correct results.

### Bitwise Operations

MegaBinary provides comprehensive bitwise operations:

- **AND, OR, XOR**: Implemented by aligning operands and applying the operation to each chunk
- **NOT**: Implemented by inverting each bit in the mantissa
- **Shift**: Implemented with both chunk-level and bit-level shifting

## Pattern Generation

MegaBinary includes methods for generating binary patterns:

- **generateBlockySin**: Creates a blocky sine wave pattern with specified length and half-period
- **createDutyCycle**: Creates a binary pattern with specified duty cycle
- **interfere**: Combines multiple binary patterns using specified interference mode (XOR, AND, OR)

## Usage Examples

### Basic Arithmetic

```kotlin
// Create MegaNumber instances
val a = MegaNumber.fromDecimalString("123456789")
val b = MegaNumber.fromDecimalString("987654321")

// Perform arithmetic operations
val sum = a.add(b)
val product = a.mul(b)
val quotient = a.divide(b)
val root = a.sqrt()

// Convert to string
println(sum.toDecimalString())
```

### Binary Operations

```kotlin
// Create MegaBinary instances
val a = MegaBinary("1010")
val b = MegaBinary("1100")

// Perform bitwise operations
val and = a.bitwiseAnd(b)
val or = a.bitwiseOr(b)
val xor = a.bitwiseXor(b)
val not = a.bitwiseNot()

// Shift operations
val leftShift = a.shiftLeft(MegaBinary("2"))
val rightShift = a.shiftRight(MegaBinary("1"))

// Generate patterns
val sineWave = MegaBinary.generateBlockySin(MegaBinary("16"), MegaBinary("4"))
val dutyCycle = MegaBinary.createDutyCycle(MegaBinary("8"), MegaBinary("3"))

// Combine patterns
val combined = MegaBinary.interfere(listOf(a, b), InterferenceMode.XOR)
```

## Implementation Details

### Normalization

Both classes implement normalization to ensure consistent representation:

- Removing trailing zeros from the mantissa
- Handling zero sign (always positive)
- Maintaining leading zeros when requested

### Precision Control

MegaNumber includes precision control to prevent excessive memory usage:

- Maximum precision in bits can be configured via MegaNumberConstants.maxPrecisionBits
- Operations that would exceed this limit throw an IllegalStateException

### Conversion Methods

Both classes provide methods for converting between different representations:

- Binary string (MSB first)
- Decimal string
- Byte array
- Bit list (LSB first or MSB first)

## Performance Considerations

- For multiplication of large numbers, the library automatically selects the most efficient algorithm
- Division by powers of two is optimized
- Normalization is performed after operations to maintain efficiency
- The chunk-based representation allows efficient bitwise operations