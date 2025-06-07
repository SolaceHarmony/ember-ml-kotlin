# MegaNumber, MegaInteger, and MegaFloat

This package provides arbitrary-precision arithmetic for both integers and floating-point numbers.

## Class Structure

- `MegaNumber`: Base class that provides common functionality for arbitrary-precision arithmetic.
- `MegaInteger`: Subclass of MegaNumber that represents arbitrary-precision integers.
- `MegaFloat`: Subclass of MegaNumber that represents arbitrary-precision floating-point numbers.

## Exponents in MegaNumber

The `MegaNumber` class uses exponents to represent floating-point numbers. The exponent is stored as a `LongArray` and can be negative, indicated by the `exponentNegative` flag.

### Exponents in MegaInteger

In `MegaInteger`, exponents are always zero because integers don't have fractional parts. The constructor parameters `exponent` and `exponentNegative` are accepted for compatibility with the base class, but they are ignored and always set to `longArrayOf(0)` and `false` respectively.

This is by design, as integers don't need exponents. The `isFloat` parameter is also ignored and always set to `false`.

### Exponents in MegaFloat

In `MegaFloat`, exponents are used to represent the fractional part of floating-point numbers. The constructor parameters `exponent` and `exponentNegative` are used to set the exponent value and sign.

## String Constructors

Both `MegaInteger` and `MegaFloat` provide constructors that take a decimal string:

```kotlin
val integer = MegaInteger("12345")
val float = MegaFloat("123.45")
```

These constructors are useful for creating numbers from string representations, especially when the numbers are too large to be represented by built-in types.

## Arithmetic Operations

Arithmetic operations (`add`, `sub`, `mul`, `divide`) are implemented in the base class and overridden in the subclasses to ensure type safety and proper handling of exponents.

- In `MegaInteger`, these operations only work with other `MegaInteger` instances and always return `MegaInteger` results.
- In `MegaFloat`, these operations work with both `MegaFloat` and `MegaInteger` instances and always return `MegaFloat` results.

## Float Operations

The base class provides special methods for floating-point operations (`addFloat`, `mulFloat`, `divFloat`). These methods are overridden in `MegaInteger` to throw exceptions, as integers don't support floating-point operations.

## Conversion Between Types

You can convert between `MegaInteger` and `MegaFloat` using the appropriate constructors:

```kotlin
val integer = MegaInteger("12345")
val float = MegaFloat(integer)  // Convert to MegaFloat
```

This preserves the mantissa and sign, but sets `isFloat` to `true` and allows the use of exponents.