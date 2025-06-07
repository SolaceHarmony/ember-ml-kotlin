"""
Tests for Liquid Time Constant (LTC) neuron implementations.
"""

import pytest
import numpy as np
from ember_ml.core.ltc import LTCNeuron, LTCChain, create_ltc_chain

@pytest.fixture
def basic_neuron():
    """Fixture for basic LTC neuron with default parameters."""
    return LTCNeuron(neuron_id=1, tau=1.0, dt=0.01, gleak=0.5, cm=1.0)

@pytest.fixture
def basic_chain():
    """Fixture for basic LTC chain with 3 neurons."""
    return create_ltc_chain(num_neurons=3, base_tau=1.0, dt=0.01, gleak=0.5, cm=1.0)

class TestLTCNeuron:
    """Test suite for LTCNeuron class."""

    def test_initialization(self, basic_neuron):
        """Test proper neuron initialization."""
        assert basic_neuron.neuron_id == 1
        assert basic_neuron.tau == 1.0
        assert basic_neuron.dt == 0.01
        assert basic_neuron.gleak == 0.5
        assert basic_neuron.cm == 1.0
        assert basic_neuron.state == 0.0
        assert basic_neuron.last_prediction == 0.0
        assert len(basic_neuron.history) == 0

    def test_update_single_step(self, basic_neuron):
        """Test single update step with constant input."""
        input_signal = 1.0
        state = basic_neuron.update(input_signal)
        
        # Verify state update calculation
        expected_dh = (1.0 / basic_neuron.tau) * (input_signal - 0.0) - basic_neuron.gleak * 0.0
        expected_state = basic_neuron.dt * expected_dh / basic_neuron.cm
        
        assert np.isclose(state, expected_state)
        assert len(basic_neuron.history) == 1
        assert basic_neuron.last_prediction == state

    def test_multiple_updates(self, basic_neuron):
        """Test multiple update steps."""
        input_signals = [1.0, 0.5, 0.0, -0.5]
        states = []
        
        for signal in input_signals:
            state = basic_neuron.update(signal)
            states.append(state)
        
        assert len(states) == len(input_signals)
        assert len(basic_neuron.history) == len(input_signals)
        assert basic_neuron.last_prediction == states[-1]

    def test_save_load_state(self, basic_neuron):
        """Test state saving and loading."""
        # Update neuron a few times
        for signal in [1.0, 0.5, 0.0]:
            basic_neuron.update(signal)
        
        # Save state
        state_dict = basic_neuron.save_state()
        
        # Create new neuron and load state
        new_neuron = LTCNeuron(neuron_id=1, tau=1.0)
        new_neuron.load_state(state_dict)
        
        # Verify state was properly loaded
        assert new_neuron.gleak == basic_neuron.gleak
        assert new_neuron.cm == basic_neuron.cm
        assert new_neuron.last_prediction == basic_neuron.last_prediction
        assert new_neuron.state == basic_neuron.state

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        with pytest.raises(ValueError):
            LTCNeuron(neuron_id=1, tau=-1.0)  # Negative tau
            
        with pytest.raises(ValueError):
            LTCNeuron(neuron_id=1, dt=0)  # Zero time step
            
        with pytest.raises(ValueError):
            LTCNeuron(neuron_id=1, gleak=-0.5)  # Negative leak conductance

class TestLTCChain:
    """Test suite for LTCChain class."""

    def test_chain_initialization(self, basic_chain):
        """Test proper chain initialization."""
        assert basic_chain.num_neurons == 3
        assert len(basic_chain.neurons) == 3
        assert len(basic_chain.weights) == 3
        assert all(isinstance(n, LTCNeuron) for n in basic_chain.neurons)
        assert len(basic_chain.state_history) == 0

    def test_chain_update(self, basic_chain):
        """Test chain update with input signals."""
        input_signals = np.array([1.0, 0.0, 0.0])
        states = basic_chain.update(input_signals)
        
        assert len(states) == 3
        assert len(basic_chain.state_history) == 1
        
        # First neuron should process external input
        assert np.isclose(states[0], basic_chain.neurons[0].state)
        
        # Subsequent neurons should process chain inputs
        for i in range(1, 3):
            expected_input = basic_chain.weights[i-1] * states[i-1]
            assert np.isclose(states[i], basic_chain.neurons[i].state)

    def test_chain_save_load(self, basic_chain):
        """Test chain state saving and loading."""
        # Update chain
        input_signals = np.array([1.0, 0.0, 0.0])
        basic_chain.update(input_signals)
        
        # Save state
        state_dict = basic_chain.save_state()
        
        # Create new chain and load state
        new_chain = create_ltc_chain(num_neurons=3)
        new_chain.load_state(state_dict)
        
        # Verify state was properly loaded
        assert np.array_equal(new_chain.weights, basic_chain.weights)
        assert len(new_chain.neurons) == len(basic_chain.neurons)
        
        for new_neuron, old_neuron in zip(new_chain.neurons, basic_chain.neurons):
            assert new_neuron.state == old_neuron.state
            assert new_neuron.last_prediction == old_neuron.last_prediction

    def test_progressive_time_constants(self):
        """Test progressive time constants in chain."""
        chain = create_ltc_chain(num_neurons=3, base_tau=1.0)
        
        # Verify progressive time constants
        taus = [neuron.tau for neuron in chain.neurons]
        assert taus[0] == 1.0  # Base tau
        assert all(taus[i] > taus[i-1] for i in range(1, len(taus)))  # Progressive increase

if __name__ == '__main__':
    pytest.main([__file__])