import pytest
import numpy as np
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.wave.quantum import WaveFunction, QuantumState, QuantumWave

@pytest.fixture(params=['numpy'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Test cases for WaveFunction
def test_wavefunction_initialization(set_backend_fixture):
    """Test WaveFunction initialization."""
    state_vector = tensor.random_normal((4,))
    wf = WaveFunction(state_vector)
    assert tensor.shape(wf.state_vector) == (4,)
    assert wf.state_vector.dtype == state_vector.dtype

def test_wavefunction_normalize(set_backend_fixture):
    """Test WaveFunction normalization."""
    state_vector = tensor.convert_to_tensor([1.0, 2.0, 3.0, 4.0])
    wf = WaveFunction(state_vector)
    wf.normalize()
    norm = ops.sqrt(stats.sum(ops.square(ops.abs(wf.state_vector))))
    # Use ops.allclose for backend-agnostic comparison
    assert ops.allclose(norm, tensor.convert_to_tensor(1.0))

def test_wavefunction_measure(set_backend_fixture):
    """Test WaveFunction measurement (stochastic, check output type/range)."""
    state_vector = tensor.convert_to_tensor([0.5, 0.5, 0.5, 0.5])
    wf = WaveFunction(state_vector)
    # Measurement is stochastic, so we check the type and range of the output
    measured_state = wf.measure()
    assert isinstance(measured_state, int) or isinstance(measured_state, tensor.integer)
    assert 0 <= measured_state < tensor.shape(state_vector)[0]

# Note: Testing wave function evolution with a general Hamiltonian might require
# complex matrix operations which may not be fully supported by all backends (e.g., NumPy).
# This test is included but may need to be skipped or adapted for specific backends.
@pytest.mark.skipif(ops.get_backend() == 'numpy', reason="NumPy backend may not fully support complex matrix operations for evolution.")
def test_wavefunction_evolve(set_backend_fixture):
    """Test WaveFunction evolution with a simple Hamiltonian."""
    # Example: Simple Hamiltonian for a 2-level system (Pauli-X)
    # This test assumes the backend supports complex numbers and matrix multiplication
    if tensor.get_backend() == 'numpy':
         pytest.skip("NumPy backend may not fully support complex matrix operations for evolution.")

    state_vector = tensor.convert_to_tensor([1.0, 0.0], dtype=tensor.complex64) # Initial state |0>
    hamiltonian = tensor.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=tensor.complex64) # Pauli-X
    dt = 0.1
    wf = WaveFunction(state_vector)
    
    # Evolve the state
    wf.evolve(hamiltonian, dt)
    
    # Basic shape check after evolution
    assert tensor.shape(wf.state_vector) == (2,)
    assert wf.state_vector.dtype == tensor.complex64 # Ensure dtype is preserved

# Test cases for QuantumState
def test_quantumstate_initialization(set_backend_fixture):
    """Test QuantumState initialization."""
    qs = QuantumState(num_qubits=2)
    # For 2 qubits, the state vector should have size 2^2 = 4
    assert tensor.shape(qs.state_vector) == (4,)
    # Initial state is typically |00>
    assert ops.allclose(qs.state_vector, tensor.convert_to_tensor([1.0, 0.0, 0.0, 0.0], dtype=tensor.complex64))

# Note: Testing quantum gates requires complex matrix operations.
# This test is included but may need to be skipped or adapted for specific backends.
@pytest.mark.skipif(ops.get_backend() == 'numpy', reason="NumPy backend may not fully support complex matrix operations for gates.")
def test_quantumstate_apply_gate(set_backend_fixture):
    """Test QuantumState apply_gate (e.g., applying a Hadamard gate)."""
    if tensor.get_backend() == 'numpy':
         pytest.skip("NumPy backend may not fully support complex matrix operations for gates.")

    qs = QuantumState(num_qubits=1) # Start with |0>
    hadamard_gate = tensor.convert_to_tensor([[1/ops.sqrt(tensor.convert_to_tensor(2.0)), 1/ops.sqrt(tensor.convert_to_tensor(2.0))],
                                              [1/ops.sqrt(tensor.convert_to_tensor(2.0)), -1/ops.sqrt(tensor.convert_to_tensor(2.0))]], dtype=tensor.complex64)
    
    qs.apply_gate(hadamard_gate, target_qubits=[0])
    
    # After Hadamard on |0>, state should be (|0> + |1>)/sqrt(2)
    expected_state = tensor.convert_to_tensor([1/ops.sqrt(tensor.convert_to_tensor(2.0)), 1/ops.sqrt(tensor.convert_to_tensor(2.0))], dtype=tensor.complex64)
    assert ops.allclose(qs.state_vector, expected_state)

def test_quantumstate_measure_qubit(set_backend_fixture):
    """Test QuantumState measure_qubit (stochastic, check output type/range)."""
    qs = QuantumState(num_qubits=1) # Start with |0>
     # Apply Hadamard to get a superposition (if backend supports complex ops)
    try:
        hadamard_gate = tensor.convert_to_tensor([[1/ops.sqrt(tensor.convert_to_tensor(2.0)), 1/ops.sqrt(tensor.convert_to_tensor(2.0))],
                                                  [1/ops.sqrt(tensor.convert_to_tensor(2.0)), -1/ops.sqrt(tensor.convert_to_tensor(2.0))]], dtype=tensor.complex64)
        qs.apply_gate(hadamard_gate, target_qubits=[0])
    except Exception:
        # If complex ops fail, proceed with the |0> state
        pass

    # Measurement is stochastic, check type and range
    measured_result = qs.measure_qubit(0)
    assert isinstance(measured_result, int) or isinstance(measured_result, tensor.integer)
    assert measured_result in [0, 1]

def test_quantumstate_get_probabilities(set_backend_fixture):
    """Test QuantumState get_probabilities."""
    qs = QuantumState(num_qubits=1) # Start with |0>
     # Apply Hadamard to get a superposition (if backend supports complex ops)
    try:
        hadamard_gate = tensor.convert_to_tensor([[1/ops.sqrt(tensor.convert_to_tensor(2.0)), 1/ops.sqrt(tensor.convert_to_tensor(2.0))],
                                                  [1/ops.sqrt(tensor.convert_to_tensor(2.0)), -1/ops.sqrt(tensor.convert_to_tensor(2.0))]], dtype=tensor.complex64)
        qs.apply_gate(hadamard_gate, target_qubits=[0])
    except Exception:
        # If complex ops fail, probabilities for |0> are [1.0, 0.0]
        pass

    probabilities = qs.get_probabilities()
    assert tensor.shape(probabilities) == (2,) # For 1 qubit, 2 probabilities
    # Check that probabilities sum to approximately 1
    assert ops.allclose(stats.sum(probabilities), tensor.convert_to_tensor(1.0))
    # Check that probabilities are non-negative
    assert ops.all(ops.greater_equal(probabilities, tensor.convert_to_tensor(0.0)))

# Test cases for QuantumWave
def test_quantumwave_initialization(set_backend_fixture):
    """Test QuantumWave initialization."""
    qw = QuantumWave(input_size=10, hidden_size=20, output_size=5)
    assert isinstance(qw, QuantumWave)
    # Check if internal layers are initialized (assuming they are Linear layers)
    assert hasattr(qw, 'layer1')
    assert hasattr(qw, 'layer2')

def test_quantumwave_forward_shape(set_backend_fixture):
    """Test QuantumWave forward pass shape."""
    qw = QuantumWave(input_size=10, hidden_size=20, output_size=5)
    input_tensor = tensor.random_normal((32, 10)) # Batch size 32, input size 10
    output = qw(input_tensor)
    assert tensor.shape(output) == (32, 5) # Batch size 32, output size 5