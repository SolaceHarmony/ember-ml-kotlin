# phonological_loop/models/s4_layer.py
""" Standalone S4 Layer """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import opt_einsum as oe # Import opt_einsum

# Assuming s4_kernel.py is in the same directory
from .s4_kernel import SSKernel

contract = oe.contract # Define contract
import logging # Import logging
# Basic logger setup
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Helper Activation/Linear/Dropout Layers (adapted from s4.py)
def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """
    # Construct core module
    linear_cls = nn.Conv1d if transposed else nn.Linear
    # Adjust for Conv1d which needs kernel_size
    if transposed:
        # Check if kernel_size is already provided, otherwise default to 1
        kwargs.setdefault('kernel_size', 1)
        linear = linear_cls(d_input, d_output, bias=bias, **kwargs)
    else:
        # Remove kernel_size if present for nn.Linear
        kwargs.pop('kernel_size', None)
        linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activation == 'glu': d_output *= 2 # This logic might be incorrect if applied before layer creation

    # Re-create layer if GLU adjusted output dim - safer approach
    if activation == 'glu':
        if transposed:
            kwargs.setdefault('kernel_size', 1)
            linear = linear_cls(d_input, d_output, bias=bias, **kwargs)
        else:
            kwargs.pop('kernel_size', None)
            linear = linear_cls(d_input, d_output, bias=bias, **kwargs)


    if activate and activation is not None:
        activation_module = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation_module)
    return linear

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        # self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # Not used in forward

    def forward(self, X):
        """ X: (batch, dim, lengths...) if transposed else (batch, lengths..., dim) """
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...') # Make it (B, D, L)
            # Determine mask shape based on tying
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # Generate mask using torch.rand and apply scaling
            mask = torch.rand(*mask_shape, device=X.device) < (1.0 - self.p)
            X = X * mask * (1.0 / (1.0 - self.p)) # Apply mask and scale
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d') # Transpose back if needed
            return X
        return X

# --- S4 Layer ---
class S4(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            activation='gelu',
            postact='glu', # GLU seems common in S4 context
            # hyper_act=None, # Removed dt_hypernet related params
            dropout=0.0, tie_dropout=False,
            bottleneck=None,
            gate=None,
            transposed=True, # Assume input is (B, H, L)
            verbose=False,
            # dt_hypernet_size: int = 32, # Removed dt_hypernet related params
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        Standalone S4 layer implementation (Standard version without dt_hypernet).

        Args:
            d_model (int): Dimension of input and output features (H).
            d_state (int): Dimension of the state space (N). Default: 64.
            l_max (int, optional): Maximum sequence length for kernel construction. If None, uses input sequence length. Default: None.
            channels (int): Number of independent SSM heads. Default: 1.
            bidirectional (bool): If True, uses a two-sided convolution kernel. Default: False.
            activation (str): Activation function between SSM and output linear layer. Default: 'gelu'.
            postact (str, optional): Activation function after output linear layer. GLU is used if 'glu'. Default: 'glu'.
            dropout (float): Dropout rate. Default: 0.0.
            tie_dropout (bool): If True, ties dropout mask across the sequence length. Default: False.
            bottleneck (int, optional): If set, reduces the input dimension to the SSM by this factor. Default: None.
            gate (int, optional): If set, applies a gated activation unit (GSS-style). Dimension is multiplied by this factor. Default: None.
            transposed (bool): If True, expects input shape (B, H, L) and returns (B, H, L). If False, expects (B, L, H). Default: True.
            verbose (bool): If True, prints verbose logs. Default: False.
            **kernel_args: Additional arguments passed to the SSKernel constructor (e.g., measure, rank, mode, dt_min, dt_max, lr).
        """

        super().__init__()
        # Removed verbose logging related to dt_hypernet
        # if verbose:
        #     log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.verbose = verbose # Keep verbose flag if SSKernel uses it

        self.gate = gate
        self.bottleneck = bottleneck

        # Input projection/bottleneck layer
        if bottleneck is not None:
            self.H_bottleneck = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H_bottleneck,
                transposed=self.transposed,
                activation=activation, # Activation before SSM
                activate=True,
            )
            ssm_input_dim = self.H_bottleneck
        else:
            self.H_bottleneck = self.H
            self.input_linear = nn.Identity()
            ssm_input_dim = self.d_model # H

        # Gate layers (optional, GSS-style)
        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate, # Expands dimension
                transposed=self.transposed,
                activation=activation, # Uses main activation
                activate=True,
            )
            # Output gate reduces dimension back
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=self.transposed,
                activation=None, # No activation before final multiplication
                activate=False,
            )

        # Skip connection parameter D
        self.D = nn.Parameter(torch.randn(self.channels, ssm_input_dim))

        # Bidirectional processing doubles the effective channels for the kernel
        kernel_channels = self.channels * 2 if self.bidirectional else self.channels

        # SSM Kernel
        self.kernel = SSKernel(
            H=ssm_input_dim, # Dimension fed into the kernel
            N=self.N,
            L=self.L,
            channels=kernel_channels,
            verbose=verbose, # Pass verbose flag
            **kernel_args # Pass kernel args (e.g., dt_min, dt_max for fixed dt)
        )

        # Removed dt_hypernet
        # self.dt_hypernet = ...

        # Activation and Dropout after SSM
        self.activation = Activation(activation)
        # Corrected dropout instantiation
        if dropout > 0.0:
             dropout_fn = DropoutNd if tie_dropout else nn.Dropout
             # Pass p for nn.Dropout, handle tie/transposed for DropoutNd
             if tie_dropout:
                 self.dropout = dropout_fn(p=dropout, tie=tie_dropout, transposed=self.transposed)
             else:
                 self.dropout = dropout_fn(p=dropout)
        else:
             self.dropout = nn.Identity()


        # Output linear layer
        output_linear_input_dim = ssm_input_dim * self.channels # H_bottleneck * C
        output_linear_output_dim = self.d_model * (1 if self.gate is None else self.gate) # Expand if gating
        self.output_linear = LinearActivation(
            output_linear_input_dim,
            output_linear_output_dim,
            transposed=self.transposed,
            activation=postact, # Activation after output linear (e.g., GLU)
            activate=True,
        )

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        """
        Input u: shape (B, H, L) if transposed else (B, L, H)
        state: optional initial state (B, H, N)
        rate: sampling rate factor for kernel
        lengths: optional tensor of sequence lengths (B,) for masking

        Returns: output tensor of same shape as u, and next_state (if state is provided)
        """
        if not self.transposed: u = u.transpose(-1, -2) # B H L
        L_input = u.size(-1)

        # Apply length masking
        if isinstance(lengths, int):
            if lengths != L_input:
                lengths = tensor.convert_to_tensor(lengths, dtype=torch.long, device=u.device)
            else: lengths = None # Avoid unnecessary masking

        if lengths is not None:
            assert isinstance(lengths, tensor.convert_to_tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.arange(L_input, device=lengths.device) < lengths.unsqueeze(-1) # (B, L) or (1, L)
            u = u * mask.unsqueeze(1) # (B, H, L)

        # Optional Gating mechanism (applied before bottleneck)
        v = None
        if self.gate is not None:
            v = self.input_gate(u) # (B, H*gate, L)

        # Optional Bottleneck layer
        u = self.input_linear(u) # (B, H_bottleneck, L)

        # Removed Dynamic dt Calculation
        # log_dt = self.dt_hypernet(...)
        # log_dt_kernel = log_dt

        # --- Compute SS Kernel (Standard S4 - uses fixed dt from kernel_args) ---
        L_kernel = L_input if self.L is None else min(L_input, round(self.L / rate))
        # Call kernel without log_dt argument
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)
        # k: (kernel_channels, ssm_input_dim, L_kernel)
        # k_state: (B, C, ssm_input_dim, L_kernel) or None

        # --- Convolution ---
        if self.bidirectional:
            # Fold channels and bidirectional dim, then split
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2, c=self.channels)
            # Causal convolution (k0) + Anticausal convolution (k1 flipped)
            k = F.pad(k0, (0, L_input)) + F.pad(k1.flip(-1), (L_input, 0)) # (C, H_bottleneck, L_kernel + L_input)
        else:
            k = rearrange(k, 'c h l -> c h l') # Ensure shape (C, H_bottleneck, L_kernel)

        # Compute FFTs
        k_f = torch.fft.rfft(k, n=L_kernel + L_input) # (C, H_bottleneck, L_fft)
        u_f = torch.fft.rfft(u, n=L_kernel + L_input) # (B, H_bottleneck, L_fft)

        # Convolution theorem: Y(f) = K(f) * U(f)
        # Einsum for (B H L), (C H L) -> (B C H L)
        y_f = contract('b h l, c h l -> b c h l', u_f, k_f)
        # Inverse FFT
        y = torch.fft.irfft(y_f, n=L_kernel + L_input)[..., :L_input] # (B, C, H_bottleneck, L)

        # Compute D term (skip connection)
        y = y + contract('b h l, c h -> b c h l', u, self.D) # (B, C, H_bottleneck, L)

        # Compute state update if initial state was provided
        next_state = None
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            if hasattr(self.kernel, 'forward_state'):
                 next_state = self.kernel.forward_state(u, state)
            else:
                 # Use the defined log object if available, otherwise print
                 if 'log' in globals():
                     log.warning("SSKernel does not have forward_state, state will not be updated.")
                 else:
                     print("Warning: SSKernel does not have forward_state, state will not be updated.")
                 next_state = state # Pass through state if no update method

        # Reshape to flatten channels
        y = rearrange(y, 'b c h l -> b (c h) l') # (B, C*H_bottleneck, L)

        # Apply activation and dropout
        y = self.dropout(self.activation(y))

        # Transpose back if necessary
        if not self.transposed: y = y.transpose(-1, -2) # (B, L, C*H_bottleneck)

        # Output linear layer
        y = self.output_linear(y) # (B, L, H*gate or H) or (B, H*gate or H, L)

        # Optional Output Gating
        if self.gate is not None:
            if not self.transposed: v = v.transpose(-1, -2) # Ensure v is (B, L, H*gate)
            y = self.output_gate(y * v) # Apply gate, reduces dim back to H

        return y, next_state

    def setup_step(self, **kwargs):
        """ Set up for recurrent stepping. """
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """ Recurrent step. Assumes setup_step() has been called.

        Args:
            u (tensor.convert_to_tensor): Input tensor, shape (B, H)
            state (tensor.convert_to_tensor): Previous state tensor, shape (B, H, N) or (B, H, N/2)

        Returns:
            Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]: Output tensor (B, H) and next state tensor.
        """
        assert not self.training
        assert not self.bidirectional

        # Optional Bottleneck
        if self.bottleneck is not None:
            u_ssm = self.input_linear(u) # (B, H_bottleneck)
        else:
            u_ssm = u

        # SSM step (kernel step does not use dt_hypernet)
        y, next_state = self.kernel.step(u_ssm, state) # y: (B, C, H_bottleneck), next_state: (B, H_bottleneck, N)

        # Skip connection D
        y = y + u_ssm.unsqueeze(1) * self.D # (B, C, H_bottleneck)
        y = rearrange(y, 'b c h -> b (c h)') # (B, C*H_bottleneck)

        # Activation and Dropout (applied after SSM step)
        y = self.dropout(self.activation(y))

        # Output linear layer
        # Need to handle transposed logic carefully if output_linear is Conv1d
        # Check the first module in the potential Sequential wrapper
        output_layer_module = self.output_linear[0] if isinstance(self.output_linear, nn.Sequential) else self.output_linear
        if isinstance(output_layer_module, nn.Conv1d): # Check if it's Conv1d due to transposed=True
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1) # (B, H*gate or H)
        else:
            y = self.output_linear(y) # (B, H*gate or H)

        # Optional Output Gating
        if self.gate is not None:
            # Gate needs original input u, not bottlenecked u_ssm
            v = self.input_gate(u) # (B, H*gate)
            if isinstance(self.output_gate, nn.Conv1d):
                 y = self.output_gate( (y * v).unsqueeze(-1) ).squeeze(-1) # (B, H)
            else:
                 y = self.output_gate(y * v) # (B, H)

        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape, device=device)

    @property
    def d_output(self):
        return self.d_model