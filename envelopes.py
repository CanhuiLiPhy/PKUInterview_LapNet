"""
Multiplicative envelope functions for PyTorch NNVMC.

This module implements envelope functions similar to lapnet's envelopes.
"""

import torch
import torch.nn as nn
import enum
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import math

class EnvelopeType(enum.Enum):
    """The point at which the envelope is applied."""
    PRE_ORBITAL = "pre_orbital"
    PRE_DETERMINANT = "pre_determinant" 
    POST_DETERMINANT = "post_determinant"

class EnvelopeLabel(enum.Enum):
    """Available multiplicative envelope functions."""
    ISOTROPIC = "isotropic"
    ABS_ISOTROPIC = "abs_isotropic"
    DIAGONAL = "diagonal"
    FULL = "full"
    NULL = "null"
    STO = "sto"
    STO_POLY = "sto_poly"
    OUTPUT = "output"
    EXACT_CUSP = "exact_cusp"

class EnvelopeBase(nn.Module):
    """Base class for envelope functions."""
    
    def __init__(self, envelope_type: EnvelopeType):
        super().__init__()
        self.envelope_type = envelope_type
    
    def forward(self, 
                ae: torch.Tensor, 
                r_ae: torch.Tensor, 
                r_ee: torch.Tensor) -> torch.Tensor:
        """Apply envelope function.
        
        Args:
            ae: atom-electron vectors [batch, n_electrons, n_atoms, 3]
            r_ae: atom-electron distances [batch, n_electrons, n_atoms, 1]
            r_ee: electron-electron distances [batch, n_electrons, n_electrons, 1]
            
        Returns:
            Envelope values
        """
        raise NotImplementedError

class IsotropicEnvelope(EnvelopeBase):
    """Isotropic exponentially decaying envelope."""
    
    def __init__(self, 
                 n_atoms: int, 
                 output_dim: int,
                 is_abs: bool = False):
        super().__init__(EnvelopeType.PRE_DETERMINANT)
        self.is_abs = is_abs
        
        # Learnable parameters - initialize like standard implementation
        self.pi = nn.Parameter(torch.ones(n_atoms, output_dim))
        self.sigma = nn.Parameter(torch.ones(n_atoms, output_dim))
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Compute isotropic envelope, matching standard implementation."""
        # r_ae: [batch, n_electrons, n_atoms, 1]
        # sigma: [n_atoms, output_dim]
        
        r_ae_squeezed = r_ae.squeeze(-1)  # [batch, n_electrons, n_atoms]
        
        # Broadcast to match standard implementation behavior
        # Standard: r_ae * sigma where r_ae is (N, natom, 1) and sigma is (natom, output_dim)
        # Result should be (N, natom, output_dim), then sum over natom (axis=1)
        
        # Expand dimensions for broadcasting
        r_ae_expanded = r_ae_squeezed.unsqueeze(-1)  # [batch, n_electrons, n_atoms, 1]
        sigma_expanded = self.sigma.unsqueeze(0).unsqueeze(0)  # [1, 1, n_atoms, output_dim]
        
        # Compute r_ae * sigma: [batch, n_electrons, n_atoms, output_dim]
        r_sigma_product = r_ae_expanded * sigma_expanded
        
        if self.is_abs:
            exp_term = torch.exp(-torch.abs(r_sigma_product))
        else:
            exp_term = torch.exp(-r_sigma_product)
        
        # Apply pi and sum over atoms (matching standard implementation axis=1 -> dim=2)
        pi_expanded = self.pi.unsqueeze(0).unsqueeze(0)  # [1, 1, n_atoms, output_dim]
        result = torch.sum(exp_term * pi_expanded, dim=2)  # [batch, n_electrons, output_dim]
        
        return result

class DiagonalEnvelope(EnvelopeBase):
    """Diagonal exponentially decaying envelope."""
    
    def __init__(self, 
                 n_atoms: int, 
                 output_dim: int,
                 n_dim: int = 3):
        super().__init__(EnvelopeType.PRE_DETERMINANT)
        
        self.pi = nn.Parameter(torch.ones(n_atoms, output_dim))
        self.sigma = nn.Parameter(torch.ones(n_atoms, n_dim, output_dim))
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Compute diagonal envelope."""
        # ae: [batch, n_electrons, n_atoms, 3]
        # sigma: [n_atoms, 3, output_dim]
        
        # Compute anisotropic distance
        ae_sigma = ae.unsqueeze(-1) * self.sigma.unsqueeze(0).unsqueeze(0)  # [batch, n_electrons, n_atoms, 3, output_dim]
        r_ae_sigma = torch.norm(ae_sigma, dim=3)  # [batch, n_electrons, n_atoms, output_dim]
        
        exp_term = torch.exp(-r_ae_sigma)
        return torch.sum(exp_term * self.pi, dim=2)

class FullEnvelope(EnvelopeBase):
    """Fully anisotropic exponentially decaying envelope."""
    
    def __init__(self, 
                 n_atoms: int, 
                 output_dim: int,
                 n_dim: int = 3):
        super().__init__(EnvelopeType.PRE_DETERMINANT)
        
        self.pi = nn.Parameter(torch.ones(n_atoms, output_dim))
        # Initialize sigma as identity matrices - use clone() to avoid memory sharing
        sigma_init = torch.eye(n_dim).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_atoms, output_dim).clone()
        self.sigma = nn.Parameter(sigma_init)
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Compute fully anisotropic envelope using safe tensor operations."""
        # ae: [batch, n_electrons, n_atoms, 3]
        # sigma: [3, 3, n_atoms, output_dim]
        
        batch_size, n_electrons, n_atoms, n_dim = ae.shape
        output_dim = self.sigma.shape[-1]
        
        # Use einsum to avoid loops and memory sharing issues
        # This is equivalent to applying sigma transformation but safer
        # ae: [batch, n_electrons, n_atoms, 3] -> [batch, n_electrons, 3, n_atoms]
        ae_transposed = ae.permute(0, 1, 3, 2)  # [batch, n_electrons, 3, n_atoms]
        
        # Apply sigma: [3, 3, n_atoms, output_dim] @ [batch, n_electrons, 3, n_atoms] 
        # Result: [batch, n_electrons, 3, n_atoms, output_dim]
        ae_sigma = torch.einsum('ijkl,mnij->mnikl', self.sigma, ae_transposed)
        
        # Compute norms: [batch, n_electrons, n_atoms, output_dim]
        r_ae_sigma = torch.norm(ae_sigma, dim=2)
        
        exp_term = torch.exp(-r_ae_sigma)
        return torch.sum(exp_term * self.pi, dim=2)

class NullEnvelope(EnvelopeBase):
    """No-op (identity) envelope."""
    
    def __init__(self):
        super().__init__(EnvelopeType.PRE_DETERMINANT)
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Return ones (identity operation)."""
        batch_size, n_electrons = ae.shape[:2]
        return torch.ones(batch_size, n_electrons, 1, device=ae.device)

class STOEnvelope(EnvelopeBase):
    """Slater-type orbital envelope."""
    
    def __init__(self, 
                 n_atoms: int, 
                 output_dim: int,
                 n_dim: int = 3):
        super().__init__(EnvelopeType.PRE_ORBITAL)
        
        # Initialize pi to ones instead of zeros to avoid zero output
        self.pi = nn.Parameter(torch.ones(n_atoms, output_dim))
        sigma_init = torch.eye(n_dim).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_atoms, output_dim).clone()
        self.sigma = nn.Parameter(sigma_init)
        # Log order of polynomial (initialize to reasonable value)
        self.n = nn.Parameter(-2.0 * torch.ones(n_atoms, output_dim))  # Less extreme initial value
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Compute STO envelope using safe tensor operations."""
        batch_size, n_electrons, n_atoms, n_dim = ae.shape
        output_dim = self.sigma.shape[-1]
        
        # Use einsum to avoid problematic loops and memory sharing
        # ae: [batch, n_electrons, n_atoms, 3] -> [batch, n_electrons, 3, n_atoms]
        ae_transposed = ae.permute(0, 1, 3, 2)  # [batch, n_electrons, 3, n_atoms]
        
        # Apply sigma: [3, 3, n_atoms, output_dim] @ [batch, n_electrons, 3, n_atoms] 
        # Result: [batch, n_electrons, 3, n_atoms, output_dim]
        ae_sigma = torch.einsum('ijkl,mnij->mnikl', self.sigma, ae_transposed)
        
        # Compute norms: [batch, n_electrons, n_atoms, output_dim]
        r_ae_sigma = torch.norm(ae_sigma, dim=2)
        
        # Use the exact formula from standard implementation with numerical safeguards:
        # exp(-r_ae_sigma + exp(n) * log(r_ae_sigma))
        
        # Clamp distances to avoid numerical issues
        r_ae_sigma_safe = torch.clamp(r_ae_sigma, min=1e-10, max=50.0)
        
        # exp(n) gives the actual polynomial order (n is in log space)
        # Clamp exp(n) to reasonable range to avoid numerical issues
        exp_n = torch.exp(torch.clamp(self.n, min=-10.0, max=5.0))
        
        # Use the exact formula matching standard implementation
        log_r_ae_sigma = torch.log(r_ae_sigma_safe)
        exp_r_ae = torch.exp(-r_ae_sigma_safe + exp_n * log_r_ae_sigma)
        
        # Sum over atoms (axis=1 in standard implementation corresponds to dim=2 here)
        out = torch.sum(exp_r_ae * self.pi, dim=2)
        return out

class OutputEnvelope(EnvelopeBase):
    """Anisotropic envelope applied to determinants in log space."""
    
    def __init__(self, 
                 n_atoms: int,
                 n_dim: int = 3):
        super().__init__(EnvelopeType.POST_DETERMINANT)
        
        self.pi = nn.Parameter(torch.ones(n_atoms))  # Initialize to ones instead of zeros
        sigma_init = torch.eye(n_dim).unsqueeze(-1).expand(-1, -1, n_atoms).clone()
        self.sigma = nn.Parameter(sigma_init)
    
    def forward(self, ae: torch.Tensor, r_ae: torch.Tensor, r_ee: torch.Tensor) -> torch.Tensor:
        """Apply envelope in log space using safe tensor operations."""
        batch_size, n_electrons, n_atoms, n_dim = ae.shape
        
        # Use einsum to avoid loops and memory sharing
        # ae: [batch, n_electrons, n_atoms, 3] -> [batch, n_electrons, 3, n_atoms]
        ae_transposed = ae.permute(0, 1, 3, 2)  # [batch, n_electrons, 3, n_atoms]
        
        # Apply sigma: [3, 3, n_atoms] @ [batch, n_electrons, 3, n_atoms] 
        # Result: [batch, n_electrons, 3, n_atoms]
        ae_sigma = torch.einsum('ijk,mnij->mnik', self.sigma, ae_transposed)
        
        # Compute norms: [batch, n_electrons, n_atoms]
        r_ae_sigma = torch.norm(ae_sigma, dim=2)
        
        # Sum over electrons and atoms in log space
        log_env = torch.sum(torch.log(torch.sum(torch.exp(-r_ae_sigma + self.pi), dim=2) + 1e-12))
        return log_env

def get_envelope(envelope_label: EnvelopeLabel, 
                 n_atoms: int,
                 output_dim: Optional[int] = None,
                 n_dim: int = 3,
                 **kwargs) -> EnvelopeBase:
    """Factory function to create envelope functions.
    
    Args:
        envelope_label: Type of envelope to create
        n_atoms: Number of atoms
        output_dim: Output dimension (required for most envelopes)
        n_dim: Spatial dimensions (default 3)
        **kwargs: Additional parameters
        
    Returns:
        Envelope function instance
    """
    if envelope_label == EnvelopeLabel.ISOTROPIC:
        return IsotropicEnvelope(n_atoms, output_dim, is_abs=False)
    elif envelope_label == EnvelopeLabel.ABS_ISOTROPIC:
        return IsotropicEnvelope(n_atoms, output_dim, is_abs=True)
    elif envelope_label == EnvelopeLabel.DIAGONAL:
        return DiagonalEnvelope(n_atoms, output_dim, n_dim)
    elif envelope_label == EnvelopeLabel.FULL:
        return FullEnvelope(n_atoms, output_dim, n_dim)
    elif envelope_label == EnvelopeLabel.NULL:
        return NullEnvelope()
    elif envelope_label == EnvelopeLabel.STO:
        return STOEnvelope(n_atoms, output_dim, n_dim)
    elif envelope_label == EnvelopeLabel.OUTPUT:
        return OutputEnvelope(n_atoms, n_dim)
    else:
        raise ValueError(f"Envelope type {envelope_label} not implemented")

# Helper functions for constructing input features
def construct_input_features(electron_coords: torch.Tensor, 
                           atom_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct input features following lapnet convention.
    
    Args:
        electron_coords: [batch_size, n_electrons, 3]
        atom_coords: [n_atoms, 3]
        
    Returns:
        (ae, ee, r_ae, r_ee): atom-electron vectors/distances and electron-electron distances
    """
    batch_size, n_electrons, _ = electron_coords.shape
    n_atoms = atom_coords.shape[0]
    
    # Atom-electron vectors and distances
    # ae: [batch, n_electrons, n_atoms, 3]
    ae = electron_coords.unsqueeze(2) - atom_coords.unsqueeze(0).unsqueeze(0)
    r_ae = torch.norm(ae, dim=3, keepdim=True)  # [batch, n_electrons, n_atoms, 1]
    
    # Electron-electron distances
    # ee: [batch, n_electrons, n_electrons, 3]
    ee = electron_coords.unsqueeze(2) - electron_coords.unsqueeze(1)
    r_ee = torch.norm(ee, dim=3, keepdim=True)  # [batch, n_electrons, n_electrons, 1]
    
    return ae, ee, r_ae, r_ee 