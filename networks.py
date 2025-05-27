"""
Neural network architectures for PyTorch NNVMC.

This module implements LapNet-style neural network wave functions with:
- Transformer-based architectures
- Multi-determinant wave functions
- Jastrow factors for electron correlation
- Envelope functions for proper boundary conditions
- Support for multi-electron systems with spin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math

from transformer_blocks import TransformerLayer, MultiHeadAttention, Dense, CrossAttentionLayer
from envelopes import IsotropicEnvelope, DiagonalEnvelope, FullEnvelope, construct_input_features, get_envelope, EnvelopeLabel, STOEnvelope, NullEnvelope, OutputEnvelope

class LapNetWaveFunction(nn.Module):
    """
    LapNet-style neural network wave function for quantum Monte Carlo.
    
    Architecture:
    1. Feature extraction from electron coordinates
    2. Transformer layers for electron interactions
    3. Multi-determinant output with Jastrow correlation
    4. Envelope functions for proper boundary conditions
    """
    
    def __init__(self,
                 n_electrons_up: int,
                 n_electrons_down: int,
                 n_atoms: int = 1,
                 d_model: int = 64,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 n_determinants: int = 4,
                 use_layer_norm: bool = True,
                 use_jastrow: bool = True,
                 envelope_type: str = 'isotropic',
                 jastrow_init: float = 0.5,
                 atom_coords: Optional[torch.Tensor] = None,
                 nuclear_charges: Optional[torch.Tensor] = None):
        """
        Args:
            n_electrons_up: Number of spin-up electrons
            n_electrons_down: Number of spin-down electrons  
            n_atoms: Number of atoms
            d_model: Hidden dimension size
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            n_determinants: Number of determinants in the wave function
            use_layer_norm: Whether to use layer normalization
            use_jastrow: Whether to include Jastrow factor
            envelope_type: Type of envelope function ('isotropic', 'diagonal', 'full')
            jastrow_init: Initial scale for Jastrow factor
            atom_coords: Atomic coordinates [n_atoms, 3]
            nuclear_charges: Nuclear charges [n_atoms]
        """
        super().__init__()
        
        self.n_electrons_up = n_electrons_up
        self.n_electrons_down = n_electrons_down
        self.n_electrons = n_electrons_up + n_electrons_down
        self.n_atoms = n_atoms
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_determinants = n_determinants
        self.use_jastrow = use_jastrow
        self.envelope_type = envelope_type
        
        # Set default atom coordinates and charges
        if atom_coords is None:
            atom_coords = torch.zeros(n_atoms, 3)
        if nuclear_charges is None:
            nuclear_charges = torch.ones(n_atoms)
            
        self.register_buffer('atom_coords', atom_coords)
        self.register_buffer('nuclear_charges', nuclear_charges)
        
        # Update n_atoms based on actual atom_coords
        self.n_atoms = atom_coords.shape[0]
        
        # Input feature extraction
        # Calculate input dimension: for each atom we have (3+1) features per atom + 1 spin feature
        # scale_r_ae: [n_atoms, 1], ae * scale_r_ae / r_ae: [n_atoms, 3], spin: [1]
        input_dim = self.n_atoms * (3 + 1) + 1  # Distance + directional features + spin
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # # Positional encoding (optional)
        # self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_electrons)
        
        # Transformer layers for dual-stream architecture
        self.transformer_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                use_layernorm=False  # Disable internal LayerNorm, use external ones
            ) for _ in range(n_layers)
        ])
        
        # MLP layers for dense stream
        self.dense_mlps = nn.ModuleList([
            Dense(d_model, d_model, activation='tanh')
            for _ in range(n_layers)
        ])
        
        # Sparse stream layers (two per layer like LapNet)
        self.sparse_mlps = nn.ModuleList([
            nn.ModuleList([
                Dense(d_model, d_model, activation='tanh'),
                Dense(d_model, d_model, activation='tanh')
            ]) for _ in range(n_layers)
        ])
        
        # Envelope function for boundary conditions (create before determinant heads)
        self.envelope = self._create_envelope(envelope_type, d_model)
        
        # Output layers for determinants
        self.determinant_heads = nn.ModuleList([
            DeterminantHead(
                d_model=d_model,
                n_electrons_up=n_electrons_up,
                n_electrons_down=n_electrons_down,
                n_atoms=n_atoms,
                envelope=self.envelope,
                atom_coords=self.atom_coords
            ) for _ in range(n_determinants)
        ])
        
        # Jastrow factor for electron correlation
        if use_jastrow:
            self.jastrow = JastrowFactor(
                n_electrons=self.n_electrons,
                n_atoms=n_atoms,
                init_scale=jastrow_init
            )
            # Store electron configuration for Jastrow factor
            self.n_electrons_up = n_electrons_up
            self.n_electrons_down = n_electrons_down
        
        # Layer normalization - separate for each layer like LapNet
        if use_layer_norm:
            # Create separate layer norms for each transformer layer (3 per layer like LapNet)
            self.sparse_layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)  # ln1 for sparse stream
            ])
            self.dense_layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)   # ln2 for dense stream
            ])
            self.output_layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)   # ln3 for output
            ])
        else:
            self.sparse_layer_norms = None
            self.dense_layer_norms = None
            self.output_layer_norms = None
            
        # Initialize parameters
        self._init_parameters()
    
    def _create_envelope(self, envelope_type: str, d_model: int) -> nn.Module:
        """Create envelope function based on type."""
        try:
            # Map string names to EnvelopeLabel enum
            envelope_map = {
                'isotropic': EnvelopeLabel.ISOTROPIC,
                'abs_isotropic': EnvelopeLabel.ABS_ISOTROPIC,
                'diagonal': EnvelopeLabel.DIAGONAL,
                'full': EnvelopeLabel.FULL,
                'null': EnvelopeLabel.NULL,
                'sto': EnvelopeLabel.STO,
                'sto_poly': EnvelopeLabel.STO_POLY,
                'output': EnvelopeLabel.OUTPUT,
                'exact_cusp': EnvelopeLabel.EXACT_CUSP
            }
            
            if envelope_type.lower() in envelope_map:
                envelope_label = envelope_map[envelope_type.lower()]
                return get_envelope(
                    envelope_label=envelope_label,
                    n_atoms=self.n_atoms,
                    output_dim=d_model,
                    n_dim=3
                )
            else:
                print(f"Warning: Unknown envelope type '{envelope_type}', defaulting to isotropic")
                return get_envelope(
                    envelope_label=EnvelopeLabel.ISOTROPIC,
                    n_atoms=self.n_atoms,
                    output_dim=d_model,
                    n_dim=3
                )
        except Exception as e:
            print(f"Error creating envelope: {e}, falling back to isotropic")
            return IsotropicEnvelope(self.n_atoms, d_model)
    
    def _init_parameters(self):
        """Initialize network parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, electron_coords: torch.Tensor, spins: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute log|ψ|.
        
        Args:
            electron_coords: Electron coordinates [batch_size, n_electrons, 3]
            spins: Electron spins [batch_size, n_electrons] (+1 for up, -1 for down)
            
        Returns:
            log|ψ|: Log magnitude of wave function [batch_size]
        """
        batch_size, n_electrons, _ = electron_coords.shape
        
        # Construct input features like in LapNet
        ae, ee, r_ae, r_ee = construct_input_features(electron_coords, self.atom_coords)
        
        # Apply LapNet distance feature engineering
        scale_r_ae = torch.log(1.0 + r_ae)  # Log distance transform
        # Combine scaled distances with directional features
        # ae: [batch, n_electrons, n_atoms, 3], r_ae: [batch, n_electrons, n_atoms, 1]
        # scale_r_ae: [batch, n_electrons, n_atoms, 1]
        directional_features = ae * scale_r_ae / (r_ae + 1e-12)  # [batch, n_electrons, n_atoms, 3]
        input_features = torch.cat((scale_r_ae, directional_features), dim=-1)  # [batch, n_electrons, n_atoms, 4]
        input_features = input_features.reshape(batch_size, n_electrons, -1)  # [batch, n_electrons, n_atoms*4]
        
        # Add spin features (explicit spin encoding like in LapNet)
        spin_features = spins.unsqueeze(-1)  # [batch_size, n_electrons, 1]
        input_features = torch.cat((input_features, spin_features), dim=-1)
        
        # Initial linear projection to d_model
        hs = self.input_projection(input_features)  # sparse stream
        hd = hs.clone()  # dense stream (start identical)
        
        # Apply dual-stream transformer layers (LapNet architecture)
        for i, layer in enumerate(self.transformer_layers):
            # Apply external LayerNorm before attention (like standard LapNet)
            if self.sparse_layer_norms is not None:
                hs_norm = self.sparse_layer_norms[i](hs)  # ln1 for sparse stream
                hd_norm = self.dense_layer_norms[i](hd)   # ln2 for dense stream
            else:
                hs_norm = hs
                hd_norm = hd
            
            # Cross-attention: sparse stream provides Q,K, dense stream provides V (correct LapNet direction)
            cross_attn_output = layer(hs_norm, hd_norm)  # Fixed: hs_norm first (Q,K), hd_norm second (V)
            hd_new = hd + cross_attn_output
            
            # Apply LayerNorm 3 after cross-attention like LapNet
            if self.output_layer_norms is not None:
                hd_new_norm = self.output_layer_norms[i](hd_new)
            else:
                hd_new_norm = hd_new
                
            # Apply MLP to dense stream  
            hd = hd_new + self.dense_mlps[i](hd_new_norm)
            
            # Keep sparsity in hs with residual connections
            hs = hs + self.sparse_mlps[i][0](hs)
            hs = hs + self.sparse_mlps[i][1](hs)  # Two sparse updates like LapNet
        
        # Use dense stream for final output
        features = hd
        
        # Compute determinants
        log_determinants = []
        for det_head in self.determinant_heads:
            log_det = det_head(features, electron_coords, spins)
            log_determinants.append(log_det)
        
        # Combine determinants (log-sum-exp for numerical stability)
        log_determinants = torch.stack(log_determinants, dim=1)  # [batch_size, n_determinants]
        log_det_sum = torch.logsumexp(log_determinants, dim=1)  # [batch_size]
        
        # Add Jastrow factor if enabled
        if self.use_jastrow:
            log_jastrow = self.jastrow(electron_coords, self.atom_coords, self.n_electrons_up)
            log_det_sum = log_det_sum + log_jastrow
        
        # Envelope is now applied at orbital level (PRE_DETERMINANT), no need for post-processing
        log_psi = log_det_sum
        
        return log_psi


class DeterminantHead(nn.Module):
    """Head for computing a single Slater determinant."""
    
    def __init__(self, d_model: int, n_electrons_up: int, n_electrons_down: int, n_atoms: int, envelope=None, atom_coords=None):
        super().__init__()
        
        self.n_electrons_up = n_electrons_up
        self.n_electrons_down = n_electrons_down
        self.n_electrons = n_electrons_up + n_electrons_down
        self.n_atoms = n_atoms
        
        # Separate networks for spin-up and spin-down electrons
        if n_electrons_up > 0:
            self.up_orbitals = OrbitalNetwork(d_model, n_electrons_up, n_atoms, envelope, atom_coords)
        if n_electrons_down > 0:
            self.down_orbitals = OrbitalNetwork(d_model, n_electrons_down, n_atoms, envelope, atom_coords)
    
    def forward(self, features: torch.Tensor, electron_coords: torch.Tensor, spins: torch.Tensor) -> torch.Tensor:
        """
        Compute log determinant for this head.
        
        Args:
            features: Processed features [batch_size, n_electrons, d_model]
            electron_coords: Electron coordinates [batch_size, n_electrons, 3]
            spins: Electron spins [batch_size, n_electrons]
            
        Returns:
            log_det: Log determinant [batch_size]
        """
        batch_size = features.shape[0]
        device = features.device
        
        log_det_total = torch.zeros(batch_size, device=device)
        
        # Compute spin-up determinant
        if self.n_electrons_up > 0:
            up_indices = (spins > 0).nonzero(as_tuple=True)[1]  # Get indices of spin-up electrons
            if len(up_indices) > 0:
                up_features = features[:, up_indices[:self.n_electrons_up], :]  # [batch_size, n_up, d_model]
                up_coords = electron_coords[:, up_indices[:self.n_electrons_up], :]  # [batch_size, n_up, 3]
                
                up_orbitals = self.up_orbitals(up_features, up_coords)  # [batch_size, n_up, n_up]
                log_det_up = self._log_determinant(up_orbitals)
                log_det_total = log_det_total + log_det_up
        
        # Compute spin-down determinant  
        if self.n_electrons_down > 0:
            down_indices = (spins < 0).nonzero(as_tuple=True)[1]  # Get indices of spin-down electrons
            if len(down_indices) > 0:
                down_features = features[:, down_indices[:self.n_electrons_down], :]  # [batch_size, n_down, d_model]
                down_coords = electron_coords[:, down_indices[:self.n_electrons_down], :]  # [batch_size, n_down, 3]
                
                down_orbitals = self.down_orbitals(down_features, down_coords)  # [batch_size, n_down, n_down]
                log_det_down = self._log_determinant(down_orbitals)
                log_det_total = log_det_total + log_det_down
        
        return log_det_total
    
    def _log_determinant(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute log determinant in a numerically stable way."""
        try:
            # Use LU decomposition for stability
            sign, log_det = torch.linalg.slogdet(matrix)
            # Handle negative determinants by taking abs
            log_det = torch.where(sign > 0, log_det, log_det)  # Just use log_det as is
            return log_det
        except:
            # Fallback to simple determinant
            det = torch.det(matrix)
            return torch.log(torch.abs(det) + 1e-12)


class OrbitalNetwork(nn.Module):
    """Network for computing molecular orbitals with envelope application."""
    
    def __init__(self, d_model: int, n_orbitals: int, n_atoms: int, envelope=None, atom_coords=None):
        super().__init__()
        
        self.n_orbitals = n_orbitals
        self.n_atoms = n_atoms
        self.envelope = envelope
        
        # Store atom coordinates
        if atom_coords is not None:
            self.register_buffer('atom_coords', atom_coords)
        else:
            self.register_buffer('atom_coords', torch.zeros(n_atoms, 3))
        
        # Simple orbital network
        self.orbital_net = nn.Sequential(
            Dense(d_model, d_model, activation='gelu'),
            Dense(d_model, n_orbitals)
        )
        
    def forward(self, features: torch.Tensor, electron_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute orbital matrix with PRE_DETERMINANT envelope application.
        
        Args:
            features: Features [batch_size, n_electrons, d_model]
            electron_coords: Coordinates [batch_size, n_electrons, 3]
            
        Returns:
            orbitals: Orbital matrix [batch_size, n_electrons, n_orbitals]
        """
        batch_size, n_electrons, d_model = features.shape
        
        # Compute raw orbitals
        orbitals = self.orbital_net(features)  # [batch_size, n_electrons, n_orbitals]
        
        # Apply envelope at orbital level (PRE_DETERMINANT like LapNet)
        if self.envelope is not None:
            ae, ee, r_ae, r_ee = construct_input_features(electron_coords, self.atom_coords)
            envelope_factor = self.envelope(ae, r_ae, r_ee)  # [batch_size, n_electrons, ...]
            
            # Apply envelope multiplication (broadcast appropriately)
            # Ensure envelope_factor can broadcast with orbitals
            if envelope_factor.dim() > 2:
                # Sum over extra dimensions to get [batch_size, n_electrons] or similar
                envelope_factor = envelope_factor.sum(dim=tuple(range(2, envelope_factor.dim())))
            
            if envelope_factor.dim() == 2:  # [batch_size, n_electrons]
                envelope_factor = envelope_factor.unsqueeze(-1)  # [batch_size, n_electrons, 1]
            
            orbitals = orbitals * envelope_factor
        
        return orbitals


class JastrowFactor(nn.Module):
    """Jastrow factor for electron correlation using standard α²/(α+r) form."""
    
    def __init__(self, n_electrons: int, n_atoms: int, init_scale: float = 0.5):
        super().__init__()
        
        self.n_electrons = n_electrons
        self.n_atoms = n_atoms
        
        # Parameters for parallel (same spin) electron pairs
        self.alpha_par = nn.Parameter(torch.ones(1) * init_scale)
        
        # Parameters for antiparallel (different spin) electron pairs
        self.alpha_anti = nn.Parameter(torch.ones(1) * init_scale)
    
    def forward(self, electron_coords: torch.Tensor, atom_coords: torch.Tensor, n_electrons_up: int) -> torch.Tensor:
        """
        Compute Jastrow factor using the correct α²/(α+r) form.
        
        Args:
            electron_coords: [batch_size, n_electrons, 3]
            atom_coords: [n_atoms, 3]
            n_electrons_up: Number of spin-up electrons
            
        Returns:
            log_jastrow: [batch_size]
        """
        batch_size, n_electrons, _ = electron_coords.shape
        device = electron_coords.device
        
        # Compute electron-electron distances
        log_jastrow = torch.zeros(batch_size, device=device)
        
        if n_electrons > 1:
            # Create spin assignment (assuming first n_up are spin-up, rest are spin-down)
            # This should be passed as input ideally, but we'll infer from the structure
            n_up = n_electrons_up
            n_down = n_electrons - n_up
            
            # Same spin pairs (parallel)
            # Spin-up pairs
            if n_up > 1:
                for i in range(n_up):
                    for j in range(i + 1, n_up):
                        r_ij = torch.norm(electron_coords[:, i] - electron_coords[:, j], dim=1)
                        r_ij = torch.clamp(r_ij, min=1e-10)
                        # Use the correct Jastrow form: -0.25 * α²/(α + r)
                        log_jastrow += -0.25 * (self.alpha_par**2) / (self.alpha_par + r_ij)
            
            # Spin-down pairs
            if n_down > 1:
                for i in range(n_up, n_electrons):
                    for j in range(i + 1, n_electrons):
                        r_ij = torch.norm(electron_coords[:, i] - electron_coords[:, j], dim=1)
                        r_ij = torch.clamp(r_ij, min=1e-10)
                        # Use the correct Jastrow form: -0.25 * α²/(α + r)
                        log_jastrow += -0.25 * (self.alpha_par**2) / (self.alpha_par + r_ij)
            
            # Different spin pairs (antiparallel)
            for i in range(n_up):
                for j in range(n_up, n_electrons):
                    r_ij = torch.norm(electron_coords[:, i] - electron_coords[:, j], dim=1)
                    r_ij = torch.clamp(r_ij, min=1e-10)
                    # Use the correct Jastrow form: -0.5 * α²/(α + r)
                    log_jastrow += -0.5 * (self.alpha_anti**2) / (self.alpha_anti + r_ij)
        
        return log_jastrow


class SimpleWaveFunction(nn.Module):
    """Simplified wave function for testing and comparison."""
    
    def __init__(self, n_electrons: int, hidden_dim: int = 64):
        super().__init__()
        
        self.n_electrons = n_electrons
        self.hidden_dim = hidden_dim
        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(n_electrons * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, electron_coords: torch.Tensor, spins: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        batch_size = electron_coords.shape[0]
        
        # Flatten coordinates
        coords_flat = electron_coords.view(batch_size, -1)
        
        # Pass through network
        log_psi = self.network(coords_flat).squeeze(-1)
        
        return log_psi


def make_wave_function(config: Dict[str, Any]) -> nn.Module:
    """Factory function for creating wave functions."""
    
    # Extract configuration
    n_electrons_up = config['n_electrons_up']
    n_electrons_down = config['n_electrons_down']
    
    # Use simple wave function for basic cases
    if config.get('simple', False):
        return SimpleWaveFunction(
            n_electrons=n_electrons_up + n_electrons_down,
            hidden_dim=config.get('d_model', 64)
        )
    
    # Default to LapNet wave function
    return make_lapnet_wave_function(config)


def make_lapnet_wave_function(config: Dict[str, Any]) -> LapNetWaveFunction:
    """Factory function for creating LapNet wave functions."""
    
    # Get atom coordinates and charges if provided
    atom_coords = None
    nuclear_charges = None
    
    if 'atom_coords' in config:
        atom_coords = torch.tensor(config['atom_coords'], dtype=torch.float32)
    if 'nuclear_charges' in config:
        nuclear_charges = torch.tensor(config['nuclear_charges'], dtype=torch.float32)
    
    return LapNetWaveFunction(
        n_electrons_up=config['n_electrons_up'],
        n_electrons_down=config['n_electrons_down'],
        n_atoms=config.get('n_atoms', 1),
        d_model=config.get('d_model', 64),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 8),
        n_determinants=config.get('n_determinants', 4),
        use_layer_norm=config.get('use_layer_norm', True),
        use_jastrow=config.get('use_jastrow', True),
        envelope_type=config.get('envelope_type', 'isotropic'),
        jastrow_init=config.get('jastrow_init', 0.5),
        atom_coords=atom_coords,
        nuclear_charges=nuclear_charges
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_wave_function():
    """Test the wave function implementation."""
    print("🧪 Testing Wave Function Implementation")
    print("=" * 50)
    
    # Test configuration
    config = {
        'n_electrons_up': 1,
        'n_electrons_down': 1,
        'n_atoms': 1,
        'd_model': 32,
        'n_layers': 2,
        'n_heads': 4,
        'n_determinants': 2,
        'use_layer_norm': True,
        'use_jastrow': True,
        'envelope_type': 'isotropic',
        'atom_coords': [[0.0, 0.0, 0.0]],
        'nuclear_charges': [2.0]
    }
    
    # Create wave function
    wave_fn = make_lapnet_wave_function(config)
    
    print(f"Wave function parameters: {count_parameters(wave_fn):,}")
    
    # Test forward pass
    batch_size = 4
    n_electrons = config['n_electrons_up'] + config['n_electrons_down']
    
    electron_coords = torch.randn(batch_size, n_electrons, 3) * 0.5
    spins = torch.tensor([1, -1]).unsqueeze(0).expand(batch_size, -1).float()
    
    try:
        log_psi = wave_fn(electron_coords, spins)
        
        print(f"Input shape: {electron_coords.shape}")
        print(f"Output shape: {log_psi.shape}")
        print(f"Output range: [{log_psi.min():.6f}, {log_psi.max():.6f}]")
        print(f"All finite: {torch.isfinite(log_psi).all()}")
        
        print("✅ Wave function test passed!")
        
    except Exception as e:
        print(f"❌ Wave function test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_wave_function() 