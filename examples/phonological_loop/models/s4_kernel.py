# phonological_loop/models/s4_kernel.py
""" Standalone version of Structured (Sequence) State Space (S4) model. """

import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning.utilities import rank_zero_only # Removed PL dependency
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract
# contract_expression = oe.contract_expression # Removed PL dependency

# Define rank_zero_only decorator locally if needed, or remove its usage
# For simplicity, removing rank_zero_only usage for now
# def rank_zero_only(fn):
#     return fn

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Removed rank_zero_only decorator
    # for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
log = get_logger(__name__)

""" Cauchy and Vandermonde kernels """

try: # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
except:
    log.warning(
        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try: # Try pykeops
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    log.info("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def cauchy_conj(v, z, w):
        """ Pykeops version """
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return 2*_r2c(r).real

    def log_vandermonde_transpose(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)

except ImportError:
    has_pykeops = False
    if not has_cauchy_extension:
        log.warning(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency."
        )
        def cauchy_naive(v, z, w):
            """
            v, w: (..., N)
            z: (..., L)
            returns: (..., L)
            """
            cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
            return stats.sum(cauchy_matrix, dim=-2)

    # Vandermonde functions
    log.warning( # Changed from error to warning
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )
    def log_vandermonde(v, x, L):
        """
        v: (..., N)
        x: (..., N)
        returns: (..., L) \sum v x^l
        """
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
        vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
        return 2*vandermonde_prod.real

    def log_vandermonde_transpose(u, v, x, L):
        vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
        # Einsum equivalent: torch.einsum('...l, ...n, ...nl -> ...n', u.to(x), v.to(x), vandermonde_matrix)
        vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... N) # Corrected einsum
        return vandermonde_prod

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()



""" Simple nn.Module components """

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
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        # self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # Removed binomial

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...') # Changed order
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d') # Changed order
            return X
        return X

""" Misc functional utilities """

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    if k > 0: # Check if padding is needed
        v_ = powers.pop() @ v[..., l:]
        v = v[..., :l]
        v[..., :k] = v[..., :k] + v_
    elif k < 0: # If L was smaller than the smallest power of 2
         v = powers.pop() @ v # Apply the correct power

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" HiPPO utilities """

def transition(measure, N):
    """ A, B transition matrices for different measures """
    # Legendre (translated)
    if measure == 'legt':
        Q = tensor.arange(N, dtype=tensor.float64)
        R = (2*Q + 1) ** .5
        j, i = tensor.meshgrid(Q, Q)
        A = R[:, None] * ops.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == 'legs':
        q = tensor.arange(N, dtype=tensor.float64)
        col, row = tensor.meshgrid(q, q)
        r = 2 * q + 1
        M = -(ops.where(row >= col, r, 0) - np.diag(q))
        T = ops.sqrt(np.diag(2 * q + 1))
        A = T @ M @ ops.linearalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'legsd':
        # Essentially equivalent to S4D-LegS
        q = tensor.arange(N, dtype=tensor.float64)
        col, row = tensor.meshgrid(q, q)
        r = 2 * q + 1
        M = -(ops.where(row >= col, r, 0) - np.diag(q))
        T = ops.sqrt(np.diag(2 * q + 1))
        A = T @ M @ ops.linearalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
        A += .5 * B*B[None, :, 0]
        B = B / 2.0
    elif measure in ['fourier_diag', 'foud']:
        # Essentially equivalent to S4D-Lin
        freqs = tensor.arange(N//2)
        d = tensor.stack([freqs, tensor.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*ops.pi*(-np.diag(d, 1) + np.diag(d, -1))
        A = A - .5 * ops.eye(N)
        B = tensor.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1
        B = B[:, None]
    elif measure in ['fourier', 'fout']:
        freqs = tensor.arange(N//2)
        d = tensor.stack([tensor.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = ops.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = tensor.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N, dtype=dtype) # Corrected dtype
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    elif measure in ['fourier_diag', 'foud', 'legsd']:
        P = torch.zeros(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + stats.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    # Check if _A is close to a scalar multiple of identity
    diag_mean = torch.mean(torch.diagonal(_A))
    is_identity_multiple = torch.allclose(_A, diag_mean * torch.eye(N, dtype=_A.dtype, device=_A.device), atol=1e-5)
    if not is_identity_multiple:
        err = stats.sum((_A - diag_mean * torch.eye(N, dtype=_A.dtype, device=_A.device))**2) / N
        log.warning(f"HiPPO matrix correction AP not skew symmetric ({err=})")


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # Ensure AP is Hermitian for eigh (it should be close to skew-symmetric + identity multiple)
    AP_H = (AP - AP.conj().transpose(-1, -2)) / 2j # Skew-Hermitian part
    w_im, V = torch.linalg.eigh(AP_H) # Eigenvalues are real, V is unitary
    if diagonalize_precision: w_im, V = w_im.to(cdtype), V.to(cdtype)
    # Combine with real part (which is diagonal)
    w = w_re + 1j * w_im # Eigenvalues

    # Check diagonalization: V @ diag(w) @ V^* should be approx AP
    diag_w = torch.diag_embed(w.squeeze(-1)) # Ensure w is 1D for diag_embed
    AP_recon = V @ diag_w @ V.conj().transpose(-1, -2)
    err_diag = torch.dist(AP_recon, AP.to(V)) / N
    if err_diag > 1e-5:
         log.warning(f"Diagonalization error {err_diag:.2e}")


    # Only keep half of each conjugate pair
    _, idx = torch.sort(w.imag.squeeze(-1)) # Squeeze w.imag
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # Handle edge case for zero eigenvalues (common in Fourier)
    zero_threshold = 1e-5
    zero_eigenvalues = w_sorted.abs() < zero_threshold
    if stats.sum(zero_eigenvalues) > 1:
         log.warning(f"Multiple zero eigenvalues found ({stats.sum(zero_eigenvalues)}), handling might be approximate.")
         # Keep only one zero eigenvalue if multiple exist (heuristic)
         first_zero_idx = torch.where(zero_eigenvalues.squeeze(-1))[0][0]
         non_zero_mask = torch.ones_like(zero_eigenvalues, dtype=torch.bool)
         non_zero_mask[first_zero_idx+1:] = ~zero_eigenvalues[first_zero_idx+1:] # Keep first zero, remove others
         w_sorted = w_sorted[non_zero_mask]
         V_sorted = V_sorted[:, non_zero_mask.squeeze(-1)]


    # Keep positive imaginary part (and potentially one zero)
    pos_imag_mask = w_sorted.imag >= 0
    # Include the zero eigenvalue if it exists and was kept
    if torch.any(w_sorted.abs() < zero_threshold):
         zero_idx = torch.where(w_sorted.abs() < zero_threshold)[0][0]
         pos_imag_mask[zero_idx] = True # Ensure the single zero eigenvalue is kept

    V = V_sorted[:, pos_imag_mask.squeeze(-1)]
    w = w_sorted[pos_imag_mask]

    # Adjust V for the zero eigenvalue case (Fourier specific hack from original code)
    if measure in ['fourier', 'fout'] and torch.any(w.abs() < zero_threshold):
         zero_col_idx = torch.where(w.abs() < zero_threshold)[0][0]
         V[:, zero_col_idx] = 0.
         V[0, zero_col_idx] = 2**-0.5
         V[1, zero_col_idx] = 2**-0.5 * 1j


    # Final check on dimensions if N is odd (might need adjustment)
    if N % 2 != 0:
         log.warning("NPLR for odd N might require adjustments in selecting eigenvalues/vectors.")
         # Simple truncation for now, might miss the DC component if N is odd
         if V.shape[1] > N // 2: V = V[:, :N//2]
         if w.shape[0] > N // 2: w = w[:N//2]


    V_inv = V.conj().transpose(-1, -2)

    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    return w.squeeze(-1), P, B, V # Return w as 1D

def dplr(scaling, N, rank=1, H=1, dtype=torch.float, real_scale=1.0, imag_scale=1.0, random_real=False, random_imag=False, normalize=False, diagonal=True, random_B=False):
    assert dtype == torch.float or torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble # Use cdtype

    pi = tensor.convert_to_tensor(math.pi)
    if random_real:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    if random_imag:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    real_part = real_scale * real_part
    if scaling == 'random':
        imag_part = torch.randn(H, N//2)
    elif scaling == 'real':
        imag_part = 0 * imag_part
        real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif scaling in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif scaling in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif scaling in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif scaling in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif scaling in ['legs', 'hippo']:
        # Need to handle potential shape mismatch if N is odd
        hippo_N = N if N % 2 == 0 else N + 1 # Use even N for HiPPO eigenvalues
        w_hippo, _, _, _ = nplr('legsd', hippo_N)
        imag_part = repeat(w_hippo.imag, 'n -> h n', h=H)
        # Truncate if N is odd
        if N % 2 != 0: imag_part = imag_part[:, :N//2]

    else: raise NotImplementedError
    imag_part = imag_scale * imag_part
    w = -real_part + 1j * imag_part # Shape (H, N/2)

    # Initialize B
    if random_B:
        B = torch.randn(H, N//2, dtype=cdtype) # Use cdtype
    else:
        B = torch.ones(H, N//2, dtype=cdtype) # Use cdtype

    if normalize:
        norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*stats.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    P = torch.randn(rank, H, N//2, dtype=cdtype) # Use cdtype
    if diagonal: P = P * 0.0
    # V is not used in diagonal mode, return dummy
    V = torch.eye(N, dtype=cdtype)[:, :N//2] if N%2==0 else torch.eye(N, dtype=cdtype)[:, :(N+1)//2] # Adjust slice for odd N
    V = repeat(V, 'n m -> h n m', h=H)


    return w, P, B, V

def ssm(measure, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if measure == "dplr":
        # Assuming dplr handles diagonal=False internally if needed
        w, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    elif measure.startswith("diag"):
        args = measure.split("-")
        assert args[0] == "diag" and len(args) > 1
        scaling = args[1]
        # Pass diagonal=True via ssm_args, remove explicit here
        w, P, B, V = dplr(scaling=scaling, N=N, rank=R, H=H, **ssm_args)
    else: # NPLR
        # Adjust N for nplr if odd, as it expects even N for conjugate pairs
        nplr_N = N if N % 2 == 0 else N + 1
        w, P, B, V = nplr(measure, nplr_N, R, **ssm_args)
        # Truncate outputs if N was odd
        if N % 2 != 0:
             w = w[:N//2]
             P = P[...,:N//2]
             B = B[:N//2]
             V = V[:N, :N//2] # Adjust V slicing carefully

        # Repeat H times
        w = repeat(w, 'n -> s n', s=H)
        P = repeat(P, 'r n -> r s n', s=H)
        B = repeat(B, 'n -> s n', s=H)
        V = repeat(V, 'n m -> s n m', s=H)
    return w, P, B, V

combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(measures, N, R, S, **ssm_args):
    if isinstance(measures, str):
        measures = combinations[measures] if measures in combinations else [measures]

    assert S % len(measures) == 0, f"{S} independent trainable SSM copies must be multiple of {len(measures)} different measures"
    H = S // len(measures) # Number of copies per measure
    w, P, B, V = zip(
        *[ssm(measure, N, R, H, **ssm_args) for measure in measures]
    )
    w = torch.cat(w, dim=0) # (S, N/2)
    P = torch.cat(P, dim=1) # (R, S, N/2)
    B = torch.cat(B, dim=0) # (S, N/2)
    V = torch.cat(V, dim=0) # (S, N, N/2)
    return w, P, B, V


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0): # Added wd
        """Register a tensor with a configurable learning rate and weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {} # Initialize optim dict
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd # Use provided wd
            # Only set _optim if lr or wd is specified
            if optim:
                 setattr(getattr(self, name), "_optim", optim)

class SSKernelNPLR(OptimModule):
    """ Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)
    """

    @torch.no_grad()
    def _setup_C(self, L):
        """ Construct C~ from C

        Two modes are supported: go directly to length L if self.L is 0, or length is doubled
        """

        if self.L.item() == 0:
            if self.verbose: log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
            # Initialize dA_power needed later
            dA, _ = self._setup_state()
            self.dA_power = torch.eye(dA.shape[-1], dtype=dA.dtype, device=dA.device).unsqueeze(0).repeat(dA.shape[0], 1, 1) # Store I
        elif L > self.L.item(): # 2*int(self.L) == L:
            if self.verbose: log.info(f"S4: Doubling length from L = {self.L.item()} to {2*self.L.item()}")
            double_length = True
            L = self.L.item() # Convenience for the math below
            # Need dA for power calculation if doubling
            dA, _ = self._setup_state()
        else: return # No change needed if L <= self.L

        C = _r2c(self.C) # (C, H, N)

        # Calculate dA^L
        if double_length:
             dA_L = self.dA_power # dA^(L_prev)
             self.dA_power = dA_L @ dA_L # dA^(2*L_prev)
        else: # Initial setup
             dA_L, _ = power(L, dA) # Calculate dA^L
             self.dA_power = dA_L # Store dA^L

        # Multiply C by I - dA_L (or I + dA_L if doubling)
        C_ = _conj(C) # (C H 2N)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod # Equivalent to C * (I + dA_L)
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        # Update internal length L
        new_L = 2*self.L if double_length else self.L + L
        self.L.fill_(new_L.item()) # Use fill_ with item()


    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform
        This should be called everytime the internal length self.L changes """

        # Use cached if available
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L//2+1:
            return self.omega, self.z

        omega = tensor.convert_to_tensor(
            ops.exp(-2j * ops.pi / (L)), dtype=dtype, device=device
        )  # \omega_{L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega) # Bilinear transform

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(
        self,
        w, P, B, C, log_dt,
        L=None, # starting/maximum length of kernel
        lr=None,
        verbose=False,
        keops=False,
        real_type='exp', # ['none' | 'exp' | 'relu' | sigmoid']
        real_tolerance=1e-3,
        bandlimit=None,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        A is represented by diag(w) - PP^*
        w: (S, N) diagonal part
        P: (R, S, N) low-rank part

        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature
        lr: [dict | float | None] hook to set lr of special parameters (A, B, dt)

        Dimensions:
        N (or d_state): state size (actually N/2 for complex)
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim map;

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.real_tolerance = real_tolerance

        # Rank of low-rank correction
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) # N/2
        # C has shape (C, H, N/2)
        assert w.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1) # This is N/2

        # Check different SSM inits
        assert w.size(-2) == P.size(-2) == B.size(-2) # n_ssm
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.broadcast = self.H // w.size(0)  # Each trainable SSM needs to be duplicated this many times

        # Broadcast C shape if necessary (this was missing)
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)

        B = B.unsqueeze(0) # (1, n_ssm, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C))) # (C, H, N, 2) -> (C, H, N/2, 2) ? No, C is already N/2 complex
        # C should be (channels, H, N/2) complex. _conj makes it (channels, H, N) complex. _c2r makes it (channels, H, N, 2) real.
        # Let's adjust C input shape assumption if needed. Assume C is (channels, H, N/2) complex.
        self.C = nn.Parameter(_c2r(C)) # (channels, H, N/2, 2)

        if lr is None or isinstance(lr, float): lr_dict = {}
        else: lr_dict, lr = lr, None
        self.register("log_dt", log_dt, lr=lr_dict.get('dt', lr))
        self.register("B", _c2r(B), lr=lr_dict.get('B', lr)) # (1, n_ssm, N/2, 2)
        self.register("P", _c2r(P), lr=lr_dict.get('A', lr)) # (R, n_ssm, N/2, 2)
        # Register w
        self.register("inv_w_real", self._w_init(w.real), lr=lr_dict.get('A', lr)) # (n_ssm, N/2)
        self.register("w_imag", w.imag, lr=lr_dict.get('A', lr)) # (n_ssm, N/2)

        self.l_max = L
        self.register_buffer('L', tensor.convert_to_tensor(0, dtype=torch.long)) # Internal length, ensure Long


    def _w_init(self, w_real):
        w_real = torch.clamp(w_real, max=-self.real_tolerance)
        if self.real_type == 'none':
            return -w_real
        elif self.real_type == 'exp':
            return torch.log(-w_real + 1e-10) # Add epsilon for stability if w_real is 0
        elif self.real_type == 'relu':
            return -w_real # Should be positive due to clamp? Needs review
        elif self.real_type == 'sigmoid':
            # Inverse sigmoid (logit) needs input in (0, 1)
            # If -w_real is in (0, 1), then w_real is in (-1, 0)
            # Clamp -w_real to avoid log(0)
            w_real_sig = torch.clamp(-w_real, 1e-7, 1-1e-7)
            return torch.logit(w_real_sig)
        elif self.real_type == 'softplus':
            # Inverse softplus: log(exp(y)-1) = x -> y = log(exp(x)+1)
            # If y = -w_real, then x = log(exp(-w_real)-1)
            # Requires exp(-w_real) > 1, so -w_real > 0, w_real < 0 (guaranteed by clamp)
            return torch.log(torch.exp(-w_real)-1 + 1e-10)
        else: raise NotImplementedError

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.real_type == 'none':
            w_real = -self.inv_w_real
        elif self.real_type == 'exp':
            w_real = -torch.exp(self.inv_w_real)
        elif self.real_type == 'relu':
            # Apply relu to the inverse, then negate
            w_real = -F.relu(self.inv_w_real + self.real_tolerance) - self.real_tolerance # Ensure negative
        elif self.real_type == 'sigmoid':
            w_real = -F.sigmoid(self.inv_w_real)
        elif self.real_type == 'softplus':
            w_real = -F.softplus(self.inv_w_real)
        else: raise NotImplementedError
        w = w_real + 1j * self.w_imag # (n_ssm, N) where N is N/2 complex
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state (N should be N/2 complex)
        rate: sampling rate factor
        L: target length

        returns:
        k_B: (C, H, L) convolution kernel (generally C=1)
        k_state: (B, C, H, L) output from initial state (None if state is None)
        """

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            try:
                self._setup_C(self.l_max)
            except NotImplementedError as e:
                 # Catch if power function is missing
                 log.error(f"Error during _setup_C initialization: {e}. Kernel calculation might be incorrect if L > l_max.")
                 # Initialize L anyway to prevent repeated errors
                 self.L.fill_(self.l_max)


        # Handle sampling rate logic
        if L is None:
            if self.L.item() == 0: raise ValueError("Kernel length L must be specified if l_max is None or 0")
            L = round(self.L.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate*L)
        if hasattr(self, 'L') and self.L.item() > 0: # Check if L is initialized
            while continuous_L > self.L.item():
                 try:
                     self._setup_C(continuous_L)
                 except NotImplementedError as e:
                     log.error(f"Error during _setup_C doubling: {e}. Using kernel up to length {self.L.item()}.")
                     # Prevent infinite loop if power fails
                     continuous_L = self.L.item()
                     break
            # Length for discrete kernel calculation
            discrete_L = round(self.L.item()/rate)
        else: # If L was never initialized (e.g., l_max=None or 0)
            # Cannot setup C without a length, assume L is the max length needed
            discrete_L = L
            # We might need to initialize self.L here if it's used elsewhere
            # self.L.fill_(continuous_L) # Potential side effect


        dt = torch.exp(self.log_dt) * rate # (H)
        B = _r2c(self.B) # (1, n_ssm, N)
        C = _r2c(self.C) # (C, H, N) - C is already C_tilde if _setup_C ran
        P = _r2c(self.P) # (R, n_ssm, N)
        Q = P.conj()
        w = self._w() # (n_ssm, N)

        # Broadcast parameters to same hidden features H
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast) # (1, H, N)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast) # (R, H, N)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast) # (R, H, N)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast) # (H, N)

        # Apply bandlimiting if specified
        if self.bandlimit is not None:
            freqs = w.imag.abs() / (2*math.pi)
            freqs = dt[:, None] / rate * freqs # Normalize by dt and rate
            mask = torch.where(freqs < self.bandlimit * 0.5, 1.0, 0.0)
            C = C * mask # Apply mask to C

        # Get FFT nodes
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=(rate==1.0)) # (L/2+1)

        # Handle state input B_aug = [s B]
        if state is not None:
            # "Unbilinear" the state: s = 1/dt * (I + dt/2 A) @ state
            # A = w - P Q^*
            s = state # Assume state is already complex (B, H, N)
            sA = (
                s * w # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, Q, P) # (B H N)
            )
            s = s / dt.unsqueeze(0).unsqueeze(-1) + sA / 2 # (B H N)
            B_aug = torch.cat([s, B], dim=0) # (B+1, H, N)
        else:
            B_aug = B # (1, H, N)

        # Stack B_aug and P, C and Q for convenient batching
        B_stacked = torch.cat([B_aug, P], dim=0) # (B+1+R, H, N)
        C_stacked = torch.cat([C, Q], dim=0) # (C+R, H, N)

        # Calculate resolvent at omega nodes: (zI - A)^-1 = (z-w)^-1 + (z-w)^-1 P [I - Q(z-w)^-1 P]^-1 Q (z-w)^-1
        # Output is (..., L/2+1)
        v = B_stacked.unsqueeze(1) * C_stacked.unsqueeze(0) # (B+1+R, C+R, H, N)

        # Use Keops/CUDA if available
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v.contiguous(), z.contiguous(), w.contiguous(), symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            # Fallback for cauchy naive needs shapes (..., N), (..., L), (..., N) -> (..., L)
            # v is (B+1+R, C+R, H, N)
            # z is (L/2+1)
            # w is (H, N)
            # Need to broadcast and contract carefully
            # r = cauchy_naive(v, z, w) # Original call, likely incorrect shape handling
            # Reshape v: (batch_dims, H, N) -> ( (B+1+R)*(C+R), H, N )
            v_reshaped = v.view(-1, self.H, self.N)
            # Broadcast z: (L/2+1) -> (1, L/2+1)
            z_b = z.unsqueeze(0)
            # Broadcast w: (H, N) -> (1, H, N)
            w_b = w.unsqueeze(0)
            # Apply cauchy_naive per H dimension? No, per N dimension.
            # Need z - w term: z (L) - w (N) -> (N L)
            # v / (z-w) -> sum over N -> (L)
            # Let's assume cauchy_naive handles broadcasting correctly if dimensions match after unsqueezing
            # z: (L/2+1) -> (1, 1, 1, L/2+1)
            # w: (H, N) -> (1, 1, H, N, 1)
            # v: (B+1+R, C+R, H, N) -> (B+1+R, C+R, H, N, 1)
            z_ = z.view(1, 1, 1, -1)
            w_ = w.view(1, 1, self.H, self.N, 1)
            v_ = v.view(v.shape[0], v.shape[1], self.H, self.N, 1)
            cauchy_matrix = v_ / (z_ - w_) # Broadcasting: (B+1+R, C+R, H, N, L/2+1)
            r = stats.sum(cauchy_matrix, dim=-2) # Sum over N -> (B+1+R, C+R, H, L/2+1)


        # Low-rank Woodbury correction
        if self.rank > 0:
            r00 = r[:-self.rank, :-self.rank, :, :] # (B+1, C, H, L/2+1)
            r01 = r[:-self.rank, -self.rank:, :, :] # (B+1, R, H, L/2+1)
            r10 = r[-self.rank:, :-self.rank, :, :] # (R,   C, H, L/2+1)
            r11 = r[-self.rank:, -self.rank:, :, :] # (R,   R, H, L/2+1)

            r11 = rearrange(r11, "a b h l -> h l a b") # (H, L/2+1, R, R)
            # Invert I + r11
            Id = torch.eye(self.rank, dtype=r.dtype, device=r.device)
            inv_term = torch.linalg.solve(Id + r11, r10) # Solve (I+r11)X = r10 for X
            # inv_term shape (H, L/2+1, R, C)
            correction = contract('b r h l, h l r c -> b c h l', r01, inv_term) # (B+1, C, H, L/2+1)
            k_f = r00 - correction
        else: # rank=0
            k_f = r # (B+1, C, H, L/2+1)

        # Final correction for the bilinear transform (depends on B vs C)
        # S4 paper suggests multiplying by dt in the B term's discretization
        # Let's assume r already incorporates necessary factors from C and (z-A)^-1
        # We need the effect of B_bar = dt * (I - dt/2 A)^-1 B
        # The resolvent r includes (I - dt/2 A)^-1 implicitly via z
        # So we just need to multiply by dt * B? B is already in v. Multiply by dt.
        k_f = k_f * dt[None, None, :, None] # (B+1, C, H, L/2+1)

        # Optional factor from bilinear transform of output C? Usually absorbed.
        # k_f = k_f * 2 / (1 + omega) # Might be needed if C is also transformed

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # Truncate to target length L
        k = k[..., :L]

        # Extract state contribution and kernel
        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)

        return k_B, k_state

    @torch.no_grad()
    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w() # (n_ssm, N)
        B = _r2c(self.B) # (1, n_ssm, N)
        P = _r2c(self.P) # (R, n_ssm, N)
        Q = P.conj()

        # Repeat w shape properly
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast) # (1, H, N)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast) # (R, H, N)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast) # (R, H, N)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast) # (H, N)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt) # (H)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R_term = 2*contract('r h n, h n, s h n -> h r s', Q, D, P).real # (H R R)
        R = torch.eye(self.rank, dtype=w.dtype, device=w.device) + R_term # (H R R)
        Q_D = rearrange(Q*D, 'r h n -> h r n') # (H R N)
        try:
            # R = torch.linalg.solve(R, Q_D) # Solve R X = Q_D -> X = R^-1 Q_D
            # Need R^-1:
            R_inv = torch.linalg.inv(R) # (H R R)
            R = contract('h r s, h s n -> h r n', R_inv, Q_D) # R = R^-1 Q_D (H R N)
        except torch.linalg.LinAlgError: # Catch potential singularity
            log.warning("Warning: Linear step setup encountered singular matrix R. Using pseudo-inverse.")
            R_inv = torch.linalg.pinv(R)
            R = contract('h r s, h s n -> h r n', R_inv, Q_D) # R = R_pinv Q_D (H R N)

        R = rearrange(R, 'h r n -> r h n') # (R H N)

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (R H N) R = R^-1 Q D
            "P": P, # (R H N)
            "Q": Q, # (R H N)
            "B": B, # (1 H N)
            "E": (2.0 / dt.unsqueeze(-1) + w), # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        u: (B, H) input
        state: (B, H, N) state (complex N/2)
        Returns: (B, H, N) next state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
            batch_dim = False
        else:
            batch_dim = True # Input has batch dim B

        if state is None: # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)
            if batch_dim: state = state.unsqueeze(0).repeat(u.shape[0], 1, 1) # Add batch dim if u has it
        elif not batch_dim: # If u is None but state is provided (for dA)
             state = state.squeeze(0) # Remove batch dim if state had it

        step_params = self.step_params.copy()
        # Assume state is complex N/2, use conjugate symmetry
        contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N]

        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (R H N) R = R^-1 Q D
        P = step_params["P"]  # (R H N)
        Q = step_params["Q"]  # (R H N)
        B = step_params["B"]  # (1 H N)

        # Add batch dim to u if needed for broadcasting with state
        if batch_dim and u.ndim == 2: u = u.unsqueeze(0) # (1, B, H) -> (B, H) ? No, u is (B, H)

        # Calculation: new_state = D * ( (E * state - P @ Q @ state) + 2 * B * u - P @ R @ ( (E*state - P@Q@state) + 2*B*u ) )
        # Let state_ = (E * state - P @ Q @ state) + 2 * B * u
        state_ = E * state - contract_fn(P, Q, state) # (B H N)
        state_ = state_ + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        # new_state = D * (state_ - P @ R @ state_)
        new_state = D * (state_ - contract_fn(P, R, state_)) # (B H N)

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """
        self._setup_linear()
        C_ = _r2c(self.C) # For dtype/device

        # Calculate dA by stepping identity matrix state with zero input
        # state should be complex N/2 for _step_state_linear
        state_eye = torch.eye(self.N, dtype=C_.dtype, device=C_.device).unsqueeze(-2) # (N, 1, N)
        dA = self._step_state_linear(state=state_eye) # (N, H, N)
        dA = rearrange(dA, "m h n -> h m n") # (H, N, N)

        # Calculate dB by stepping zero state with ones input
        u_ones = torch.ones(self.H, dtype=C_.dtype, device=C_.device)
        dB = self._step_state_linear(u=u_ones) # (1, H, N)
        dB = rearrange(dB, '1 h n -> h n') # (H, N)

        # Conjugate symmetry means dA/dB should be related to conj(dA)/conj(dB)
        # For stepping, we might need the full 2N state version if not using linear step
        # Let's return the N/2 complex versions for now
        return dA, dB


    def _setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self.dA, self.dB = self._setup_state() # Get complex N/2 versions

        # Calculate original C (dC) from C_tilde (self.C)
        C_tilde = _r2c(self.C) # (C, H, N) complex
        if not hasattr(self, 'L') or self.L.item() == 0:
            # If L=0, C_tilde = C
            self.dC = C_tilde
        else:
            # C = C_tilde (I - dA^L) where dA is complex N/2 diagonal matrix
            # dA_L = power(self.L.item(), self.dA)[0] # Inefficient
            if not hasattr(self, 'dA_power'): # dA_power might not be initialized if _setup_C wasn't called
                 # Calculate dA^L directly if needed (less efficient)
                 log.warning("Calculating dA^L on the fly in _setup_step.")
                 dA_L, _ = power(self.L.item(), self.dA)
                 # self.dA_power = dA_L # Optionally cache
            else:
                 dA_L = self.dA_power # Use cached power

            I = torch.eye(self.dA.size(-1), device=dA_L.device, dtype=dA_L.dtype).unsqueeze(0) # Add H dim
            # dC = C_tilde @ (I - dA_L) # Matrix multiply for diagonal dA_L
            self.dC = C_tilde * (I - dA_L) # Element-wise for diagonal dA_L

        self._step_mode = mode
        # No special contractions needed for linear/diagonal step with complex N/2 state


    def default_state(self, *batch_shape, device=None):
        C_ = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C_.dtype, device=C_.device if device is None else device)
        return state

    def step(self, u, state):
        """ Step state using discretized parameters dA, dB, dC (complex N/2 version) """
        # Assumes _setup_step has been called
        # dA: (H, N, N) diagonal -> (H, N)
        # dB: (H, N)
        # dC: (C, H, N)
        # u: (B H)
        # state: (B H N) complex
        next_state = self.dA * state + self.dB * u.unsqueeze(-1) # Elementwise dA * state
        y = contract('c h n, b h n -> b c h', self.dC, next_state) # contract over N
        return y.real, next_state # Return real part of output

class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        A, B, C, log_dt,
        L=None,
        disc='bilinear',
        real_type='exp',
        lr=None,
        bandlimit=None,
        **kernel_args # Accept extra args
    ):

        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        # Rank of low-rank correction (rank=0 for diagonal)
        assert A.size(-1) == B.size(-1) # N/2 complex
        self.H = log_dt.size(-1)
        self.N = A.size(-1) # N here is N/2 complex
        assert A.size(-2) == B.size(-2) # Number of independent SSMs trained (n_ssm)
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        # C shape is (channels, H, N/2) complex
        self.C = nn.Parameter(_c2r(C)) # Store as (channels, H, N/2, 2) real

        # Register parameters
        if lr is None or isinstance(lr, float): lr_dict = {}
        else: lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr=lr_dict.get('dt', lr))
        # A is diagonal, stored as (n_ssm, N) complex. Store real/imag separately.
        self.register("inv_A_real", self._A_init(A.real), lr=lr_dict.get('A', lr))
        self.register("A_imag", A.imag, lr=lr_dict.get('A', lr))
        # B is stored as (n_ssm, N) complex
        self.register("B", _c2r(B), lr=lr_dict.get('B', lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4) # Ensure strictly negative
        if self.real_type == 'none':
            return -A_real
        elif self.real_type == 'exp':
            return torch.log(-A_real + 1e-10) # Add epsilon
        # Add other real_types if needed, matching SSKernelNPLR
        else: raise NotImplementedError(f"real_type {self.real_type} not supported")

    def _A(self):
        # Get the internal A (diagonal) parameter
        if self.real_type == 'none':
            A_real = -self.inv_A_real
        elif self.real_type == 'exp':
            A_real = -torch.exp(self.inv_A_real)
        # Add other real_types if needed
        else: raise NotImplementedError(f"real_type {self.real_type} not supported")
        A = A_real + 1j * self.A_imag # (n_ssm, N) where N is N/2 complex
        return A

    def forward(self, L, state=None, rate=1.0, u=None): # u is unused, kept for compatibility
        """
        state: (B, H, N) initial state (N should be N/2 complex)
        rate: sampling rate factor
        L: target length

        returns:
        K: (C, H, L) convolution kernel (generally C=1)
        K_state: (B, C, H, L) output from initial state (None if state is None)
        """

        dt = torch.exp(self.log_dt) * rate # (H)
        C = _r2c(self.C) # (C, H, N) complex
        A = self._A() # (n_ssm, N) complex
        B = _r2c(self.B) # (n_ssm, N) complex

        # Repeat A and B to H dimensions
        A = repeat(A, 't n -> (v t) n', v=self.repeat) # (H, N) complex
        B = repeat(B, 't n -> (v t) n', v=self.repeat) # (H, N) complex

        # Apply bandlimiting
        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2*math.pi)
            mask = torch.where(freqs < self.bandlimit * 0.5, 1.0, 0.0)
            C = C * mask # Apply mask to C

        # Discretize A and B
        dtA = A * dt.unsqueeze(-1) # (H, N)
        if self.disc == 'zoh':
            dA = torch.exp(dtA)
            dB = B * (torch.exp(dtA) - 1.) / A # Add eps to A?
            # Handle A=0 case for ZOH dB
            dB = torch.where(A.abs() < 1e-7, B * dt.unsqueeze(-1), dB)
        elif self.disc == 'bilinear':
            dA = (1. + dtA/2) / (1. - dtA/2)
            dB = B * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1)
        elif self.disc == 'dss':
            # DSS discretization logic (simplified, assumes A.real < 0)
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device) # [H N L]
            S = P.exp() # [H N L]
            num = torch.exp(dtA) - 1.0 # [H N]
            den = A # [H N]
            # Inline reciprocal
            x = den
            x_conj = _resolve_conj(x)
            r = x_conj / (x*x_conj + 1e-7)
            dB = B * num * r # [H N]
            dC = C # [C H N] - Note: C is not modified in DSS discretization for kernel calculation
            # Calculate kernel K = contract('chn,hnl->chl', dC, S)
            K = contract('chn,hnl->chl', dC, S).to(torch.float) # Ensure float output
            K_state = None # State calculation not directly supported by this kernel path
            return K, K_state
        else: raise NotImplementedError(f"Discretization {self.disc} not supported")

        # Augment B with state contribution: B_aug = [s B]
        if state is not None:
            # If bilinear: s = state * (1 + dtA/2) / dt
            # If zoh: s = state * A * exp(dtA) / (exp(dtA)-1) = state * A / (1 - exp(-dtA))
            if self.disc == 'bilinear':
                s = state * (1. + dtA/2) / dt.unsqueeze(-1)
            elif self.disc == 'zoh':
                # Handle exp(dtA)=1 case (i.e., dtA=0) using limit A*state
                s = torch.where(dtA.abs() < 1e-7, A * state, state * A * dA / (dA - 1.))
            B_aug = torch.cat([s, B.unsqueeze(0)], dim=0) # (B+1, H, N)
        else:
            B_aug = B.unsqueeze(0) # (1, H, N)

        # Calculate C * B_aug
        C_term = (C.unsqueeze(0) * B_aug.unsqueeze(1)) # (B+1, C, H, N)

        # Calculate kernel using Vandermonde multiplication
        # K = C * log_vandermonde(dB, log(dA), L)
        # Need C_term * Vandermonde(dA^l)
        K = log_vandermonde(C_term.view(-1, self.H, self.N), dA.log(), L) # ( (B+1)*C, H, L )
        K = K.view(-1, self.channels, self.H, L) # (B+1, C, H, L)

        # Extract state contribution and kernel
        if state is not None:
            K_state = K[:-1, :, :, :] # (B, C, H, L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C, H, L)

        return K, K_state

    def _setup_step(self):
        """ Setup dA and dB for recurrent stepping """
        dt = torch.exp(self.log_dt) # (H)
        A = self._A() # (n_ssm, N)
        B = _r2c(self.B) # (n_ssm, N)
        C = _r2c(self.C) # (C, H, N)

        # Repeat A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat) # (H, N)
        B = repeat(B, 't n -> (v t) n', v=self.repeat) # (H, N)

        # Discretize A and B
        dtA = A * dt.unsqueeze(-1)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            # Handle A=0 case
            self.dB = torch.where(A.abs() < 1e-7, B * dt.unsqueeze(-1), B * (torch.exp(dtA)-1.) / A) # (H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2) # (H N)
            self.dB = B * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1) # (H N)
        else: raise NotImplementedError(f"Discretization {self.disc} not supported for step")

        self.dC = C # (C H N)

    def default_state(self, *batch_shape, device=None):
        C_ = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C_.dtype, device=C_.device if device is None else device)
        return state

    def step(self, u, state):
        """ Recurrent step """
        # dA: (H N), dB: (H N), dC: (C H N)
        # u: (B H), state: (B H N) complex
        next_state = self.dA * state + self.dB * u.unsqueeze(-1) # Elementwise multiply dA * state
        y = contract("c h n, b h n -> b c h", self.dC, next_state) # contract over N
        return 2*y.real, next_state # Return real part of output

    def forward_state(self, u, state):
        """ Forward the state through a sequence """
        self._setup_step() # Ensure dA, dB are calculated
        AL = self.dA ** u.size(-1) # dA^L
        # Compute scan sum_i dA^(L-1-i) dB u_i efficiently
        # Use log_vandermonde_transpose: contract_L(V * u * v) where V = Vandermonde(log(dA), L)
        # Need v = dB, x = log(dA)
        # u needs to be flipped: u.flip(-1)
        v = log_vandermonde_transpose(u.flip(-1).to(self.dA), self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


# --- SSKernel Wrapper ---
class SSKernel(nn.Module):
    """ Wrapper around SSKernel parameterizations. """
    def __init__(
        self, H, N=64, L=None, measure="legs", rank=1, channels=1,
        dt_min=0.001, dt_max=0.1, deterministic=False, lr=None,
        mode="nplr", n_ssm=None, verbose=False, measure_args={}, **kernel_args
    ):
        super().__init__()
        self.N = N // 2 # Kernel implementations expect N/2 complex state size
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt (Fixed dt for standard S4)
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # Generate A, B, C parameters based on measure and mode
        if mode == "nplr":
            w, P, B, V = combination(measure, self.N*2, rank, self.n_ssm, **measure_args) # Pass N*2 to combination for NPLR
            C = torch.randn(channels, self.H, self.N, dtype=cdtype) # N/2 complex
            # Transform C to V basis: C = C V^*
            C = contract('chn, hmn -> chm', C, V.conj().transpose(-1, -2))
            self.kernel = SSKernelNPLR(
                w, P, B, C, log_dt, L=L, lr=lr, verbose=verbose, **kernel_args
            )
        elif mode == "diag":
            # A represents eigenvalues w for diagonal mode
            w, P, B, V = combination(measure, self.N*2, rank, self.n_ssm, diagonal=True, **measure_args) # Pass N*2, P is ignored
            C = torch.randn(channels, self.H, self.N, dtype=cdtype) # N/2 complex
            self.kernel = SSKernelDiag(
                w, B, C, log_dt, L=L, lr=lr, **kernel_args
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    def forward(self, state=None, L=None, rate=1.0):
        # Pass arguments to the specific kernel implementation
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """ Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N) complex N/2 state
        u: (B, H, L)

        Returns: (B, H, N) complex N/2 state
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)
        else: # Fallback for kernels without optimized state forwarding
            self._setup_step() # Ensure dA, dB, dC are calculated
            new_state = state
            for i in range(u.size(-1)):
                _, new_state = self.step(u[..., i], new_state)
            return new_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        return self.kernel.step(u, state, **kwargs)

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)