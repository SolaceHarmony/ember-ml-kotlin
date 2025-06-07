"""
Tests for base neural components.
"""

import pytest
import torch
import numpy as np
from ember_ml.core.base import BaseNeuron, BaseChain

class TestNeuron(BaseNeuron):
    """Test implementation of BaseNeuron."""
    
    def __init__(self, neuron_id: int, tau: float = 1.0, dt: float = 0.01):
        """Initialize test neuron."""
        super().__init__(neuron_id, tau, dt)
        
    def _initialize_state(self) -> float:
        """Initialize neuron state."""
        return 0.0
        
    def update(self, input_signal: float, **kwargs) -> float:
        """Update neuron state."""
        self.state += self.dt * (input_signal - self.state) / self.tau
        self.history.append(self.state)
        return self.state

class TestChain(BaseChain):
    """Test implementation of BaseChain."""
    
    def __init__(self,
                 num_neurons: int,
                 base_tau: float = 1.0,
                 dt: float = 0.01):
        """Initialize test chain."""
        super().__init__(
            num_neurons=num_neurons,
            neuron_class=TestNeuron,
            base_tau=base_tau,
            dt=dt
        )
        
    def update(self, input_signals: np.ndarray) -> np.ndarray:
        """Update chain state."""
        states = np.zeros(self.num_neurons)
        
        # Update first neuron with external input
        states[0] = self.neurons[0].update(input_signals[0])
        
        # Update subsequent neurons using chain connections
        for i in range(1, self.num_neurons):
            states[i] = self.neurons[i].update(states[i-1])
            
        # Store chain state history
        self.state_history.append(states.copy())
        
        return states

class TestBaseNeuron:
    """Test suite for BaseNeuron."""
    
    def test_initialization(self):
        """Test neuron initialization."""
        neuron = TestNeuron(1, tau=2.0, dt=0.1)
        assert neuron.neuron_id == 1
        assert neuron.tau == 2.0
        assert neuron.dt == 0.1
        assert neuron.state == 0.0
        assert len(neuron.history) == 0
        
    def test_state_update(self):
        """Test neuron state update."""
        neuron = TestNeuron(1)
        state = neuron.update(1.0)
        assert state > 0.0
        assert len(neuron.history) == 1
        assert neuron.history[0] == state
        
    def test_reset(self):
        """Test neuron reset."""
        neuron = TestNeuron(1)
        neuron.update(1.0)
        neuron.reset()
        assert neuron.state == 0.0
        assert len(neuron.history) == 0
        
    def test_save_load_state(self):
        """Test state saving and loading."""
        neuron = TestNeuron(1)
        state = neuron.update(1.0)
        
        state_dict = neuron.save_state()
        assert 'neuron_id' in state_dict
        assert 'tau' in state_dict
        assert 'dt' in state_dict
        assert 'state' in state_dict
        assert 'history' in state_dict
        assert state_dict['history'] == [state]
        
        new_neuron = TestNeuron(2)
        new_neuron.load_state(state_dict)
        assert new_neuron.state == neuron.state
        assert new_neuron.history == neuron.history

class TestBaseChain:
    """Test suite for BaseChain."""
    
    def test_initialization(self):
        """Test chain initialization."""
        chain = TestChain(3, base_tau=2.0, dt=0.1)
        assert chain.num_neurons == 3
        assert len(chain.neurons) == 3
        assert len(chain.state_history) == 0
        
    def test_progressive_time_constants(self):
        """Test progressive time constant scaling."""
        chain = TestChain(3, base_tau=1.0)
        taus = [n.tau for n in chain.neurons]
        assert taus[0] == 1.0
        assert taus[1] > taus[0]
        assert taus[2] > taus[1]
        
    def test_parameter_validation_num_neurons(self):
        """Test parameter validation for num_neurons."""
        with pytest.raises(ValueError):
            TestChain(0)  # Invalid number of neurons
            
    def test_parameter_validation_tau(self):
        """Test parameter validation for base_tau."""
        with pytest.raises(ValueError):
            TestChain(3, base_tau=-1.0)  # Invalid time constant
            
    def test_parameter_validation_dt(self):
        """Test parameter validation for dt."""
        with pytest.raises(ValueError):
            TestChain(3, dt=-0.1)  # Invalid time step
            
    def test_reset_chain(self):
        """Test chain reset."""
        chain = TestChain(3)
        chain.update(np.array([1.0, 0.0, 0.0]))
        chain.reset()
        
        assert len(chain.state_history) == 0
        for neuron in chain.neurons:
            assert neuron.state == 0.0
            assert len(neuron.history) == 0
            
    def test_save_load_chain_state(self):
        """Test chain state saving and loading."""
        chain = TestChain(3)
        chain.update(np.array([1.0, 0.0, 0.0]))
        
        state_dict = chain.save_state()
        assert 'num_neurons' in state_dict
        assert 'base_tau' in state_dict
        assert 'dt' in state_dict
        assert 'neuron_states' in state_dict
        assert 'state_history' in state_dict
        
        new_chain = TestChain(3)  # Same size as original
        new_chain.load_state(state_dict)
        assert new_chain.num_neurons == chain.num_neurons
        assert len(new_chain.neurons) == len(chain.neurons)
        assert len(new_chain.state_history) == len(chain.state_history)