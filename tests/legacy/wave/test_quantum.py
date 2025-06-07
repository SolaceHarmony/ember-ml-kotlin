"""
Tests for quantum wave processing components.
"""

import pytest
import torch
import torch.nn as nn
import math
import cmath
from ember_ml.wave.quantum import (
    WaveFunction,
    QuantumState,
    QuantumWave
)

@pytest.fixture
def device():
    """Fixture providing computation device."""
    return torch.device('cpu')

@pytest.fixture
def num_qubits():
    """Fixture providing number of qubits."""
    return 3

@pytest.fixture
def hidden_size():
    """Fixture providing hidden dimension."""
    return 16

@pytest.fixture
def wave_function():
    """Fixture providing test wave function."""
    amplitudes = torch.tensor([0.8, 0.6])
    phases = torch.tensor([0.0, math.pi/2])
    return WaveFunction(amplitudes, phases)

@pytest.fixture
def quantum_state(num_qubits, device):
    """Fixture providing quantum state."""
    return QuantumState(num_qubits, device)

@pytest.fixture
def quantum_wave(num_qubits, hidden_size):
    """Fixture providing quantum wave processor."""
    return QuantumWave(num_qubits, hidden_size)

class TestWaveFunction:
    """Test suite for WaveFunction."""

    def test_initialization(self, wave_function):
        """Test wave function initialization."""
        assert torch.is_tensor(wave_function.amplitudes)
        assert torch.is_tensor(wave_function.phases)
        assert wave_function.amplitudes.shape == (2,)
        assert wave_function.phases.shape == (2,)

    def test_complex_conversion(self, wave_function):
        """Test conversion to complex representation."""
        complex_form = wave_function.to_complex()
        
        # Check complex values
        assert torch.is_complex(complex_form)
        assert complex_form.shape == wave_function.amplitudes.shape
        
        # Verify conversion
        expected = wave_function.amplitudes * torch.exp(1j * wave_function.phases)
        assert torch.allclose(complex_form, expected)

    def test_probability_density(self, wave_function):
        """Test probability density computation."""
        density = wave_function.probability_density()
        
        # Check properties
        assert torch.all(density >= 0)
        assert torch.allclose(torch.sum(density), torch.tensor(1.0), atol=1e-6)

    def test_normalization(self, wave_function):
        """Test wave function normalization."""
        normalized = wave_function.normalize()
        
        # Check normalization
        prob_sum = torch.sum(normalized.probability_density())
        assert torch.allclose(prob_sum, torch.tensor(1.0), atol=1e-6)

    def test_evolution(self, wave_function):
        """Test time evolution."""
        # Create test Hamiltonian
        hamiltonian = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
        dt = 0.1
        
        evolved = wave_function.evolve(hamiltonian, dt)
        
        # Check properties
        assert isinstance(evolved, WaveFunction)
        assert evolved.amplitudes.shape == wave_function.amplitudes.shape
        assert evolved.phases.shape == wave_function.phases.shape

class TestQuantumState:
    """Test suite for QuantumState."""

    def test_initialization(self, quantum_state):
        """Test quantum state initialization."""
        # Check initial state |0...0⟩
        assert torch.allclose(torch.abs(quantum_state.amplitudes[0]), torch.tensor(1.0))
        assert torch.sum(torch.abs(quantum_state.amplitudes[1:])) < 1e-6

    def test_gate_application(self, quantum_state):
        """Test quantum gate application."""
        # Apply Hadamard-like gate
        gate = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64) / math.sqrt(2)
        quantum_state.apply_gate(gate, [0])
        
        # Check superposition
        amplitudes = torch.abs(quantum_state.amplitudes[:2])
        expected = torch.ones(2) / math.sqrt(2)
        assert torch.allclose(amplitudes, expected, atol=1e-6)

    def test_measurement(self, quantum_state):
        """Test qubit measurement."""
        # Prepare superposition state
        gate = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64) / math.sqrt(2)
        quantum_state.apply_gate(gate, [0])
        
        # Measure
        result, prob = quantum_state.measure(0)
        
        # Check measurement
        assert result in [0, 1]
        assert 0.0 <= prob <= 1.0

    def test_get_probabilities(self, quantum_state):
        """Test probability distribution."""
        probs = quantum_state.get_probabilities()
        
        # Check probability properties
        assert torch.all(probs >= 0)
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))

class TestQuantumWave:
    """Test suite for QuantumWave."""

    def test_initialization(self, quantum_wave):
        """Test quantum wave initialization."""
        assert quantum_wave.num_qubits > 0
        assert quantum_wave.hidden_size > 0
        assert isinstance(quantum_wave.single_qubit_gates, nn.Parameter)
        assert isinstance(quantum_wave.entangling_gates, nn.Parameter)

    def test_unitary_conversion(self, quantum_wave):
        """Test unitary matrix conversion."""
        matrix = torch.randn(2, 2, dtype=torch.complex64)
        unitary = quantum_wave._make_unitary(matrix)
        
        # Check unitary properties
        identity = torch.eye(2, dtype=torch.complex64)
        product = torch.matmul(unitary, unitary.conj().transpose(-2, -1))
        assert torch.allclose(product, identity, atol=1e-6)

    def test_quantum_layer(self, quantum_wave):
        """Test quantum circuit layer."""
        state = QuantumState(quantum_wave.num_qubits)
        processed_state = quantum_wave.quantum_layer(state)
        
        # Check state properties
        assert isinstance(processed_state, QuantumState)
        probs = processed_state.get_probabilities()
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))

    def test_forward_pass(self, quantum_wave):
        """Test forward processing."""
        batch_size = 4
        x = torch.randn(batch_size, quantum_wave.hidden_size)
        output = quantum_wave(x)
        
        # Check output properties
        assert output.shape == (batch_size, quantum_wave.hidden_size)
        assert not torch.allclose(output, x)  # Should be transformed

    def test_get_quantum_state(self, quantum_wave):
        """Test quantum state extraction."""
        x = torch.randn(quantum_wave.hidden_size)
        state = quantum_wave.get_quantum_state(x)
        
        # Check state properties
        assert isinstance(state, QuantumState)
        assert state.num_qubits == quantum_wave.num_qubits
        
        # Verify normalization
        probs = state.get_probabilities()
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))

if __name__ == '__main__':
    pytest.main([__file__])
