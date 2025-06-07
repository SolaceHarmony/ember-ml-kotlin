class SphericalLTCChain(BaseChain):
    """Chain of LTC neurons operating on unit sphere."""
    
    def __init__(self,
                 num_neurons: int,
                 base_tau_or_config=None,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 dim: int = 3):
        """
        Initialize spherical LTC chain.

        Args:
            num_neurons: Number of neurons in chain
            base_tau_or_config: Base time constant or SphericalLTCConfig object
            dt: Time step
            gleak: Leak conductance
            dim: Dimension of sphere
        """
        self.dim = dim
        
        # Handle config object
        if isinstance(base_tau_or_config, SphericalLTCConfig):
            config = base_tau_or_config
            base_tau = config.tau
            dt = config.dt
            gleak = config.gleak
        else:
            base_tau = base_tau_or_config
        
        super().__init__(
            num_neurons=num_neurons,
            neuron_class=lambda nid, tau, dt: SphericalLTCNeuron(
                neuron_id=nid,
                tau=tau,
                dt=dt,
                gleak=gleak,
                dim=dim
            ),
            base_tau=base_tau,
            dt=dt
        )
