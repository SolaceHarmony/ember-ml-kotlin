#!/bin/bash

# Test runner utility for the Ember ML Kotlin testing strategy

echo "Ember ML Kotlin Testing Strategy Runner"
echo "========================================"

# Function to run specific test categories
run_tests() {
    local category=$1
    local pattern=$2
    
    echo ""
    echo "Running $category tests..."
    echo "Pattern: $pattern"
    echo "----------------------------------------"
    
    ./gradlew jvmTest --tests "$pattern" || {
        echo "❌ $category tests failed"
        return 1
    }
    
    echo "✅ $category tests completed successfully"
    return 0
}

# Check if specific test category is requested
case "$1" in
    "bitwise")
        run_tests "Bitwise Operations" "*BitwiseOperationsTestSuite*"
        ;;
    "integration") 
        run_tests "Integration" "*TensorOperationsIntegrationTestSuite*"
        ;;
    "performance")
        run_tests "Performance Benchmarks" "*PerformanceBenchmarkSuite*"
        ;;
    "correctness")
        run_tests "Correctness" "*CorrectnessTestSuite*" 
        ;;
    "all-strategy")
        echo "Running all testing strategy tests..."
        run_tests "Bitwise Operations" "*BitwiseOperationsTestSuite*" &&
        run_tests "Integration" "*TensorOperationsIntegrationTestSuite*" &&
        run_tests "Performance Benchmarks" "*PerformanceBenchmarkSuite*" &&
        run_tests "Correctness" "*CorrectnessTestSuite*"
        ;;
    "all")
        echo "Running all tests in the project..."
        ./gradlew jvmTest
        ;;
    *)
        echo "Usage: $0 {bitwise|integration|performance|correctness|all-strategy|all}"
        echo ""
        echo "Test Categories:"
        echo "  bitwise      - Unit tests for bitwise operations"
        echo "  integration  - Integration tests for tensor operations" 
        echo "  performance  - Performance benchmarks"
        echo "  correctness  - Correctness tests against reference implementation"
        echo "  all-strategy - All testing strategy tests"
        echo "  all          - All tests in the project"
        exit 1
        ;;
esac