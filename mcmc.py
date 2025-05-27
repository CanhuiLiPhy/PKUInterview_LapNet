"""
Markov Chain Monte Carlo (MCMC) sampling methods for PyTorch NNVMC.

This module implements MCMC samplers for quantum Monte Carlo using PyTorch,
following the lapnet implementation patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable, Optional, Any

# 默认MCMC步数和预热步数
DEFAULT_MCMC_STEPS = 100
MCMC_BURN_IN = 50

def _harmonic_mean(x: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Calculates the harmonic mean of each electron distance to the nuclei.
    
    Args:
        x: electron positions. Shape (batch, n_electrons, 1, 3)
        atoms: atom positions. Shape (n_atoms, 3)
        
    Returns:
        Array of shape (batch, n_electrons, 1, 1), where the (i, j, 0, 0) element is
        the harmonic mean of the distance of the j-th electron of the i-th MCMC
        configuration to all atoms.
    """
    # ae: [batch, n_electrons, n_atoms, 3]
    ae = x - atoms[None, None, :, :]  # broadcast atoms to match x dimensions
    r_ae = torch.norm(ae, dim=-1, keepdim=True)  # [batch, n_electrons, n_atoms, 1]
    # Harmonic mean: 1/mean(1/r)
    return 1.0 / torch.mean(1.0 / (r_ae + 1e-12), dim=-2, keepdim=True)  # [batch, n_electrons, 1, 1]

def _log_prob_gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates the log probability of Gaussian with diagonal covariance.
    
    Args:
        x: Positions. Shape (batch, n_electrons, 1, 3)
        mu: means of Gaussian distribution. Same shape as or broadcastable to x.
        sigma: standard deviation of the distribution. Same shape as or broadcastable to x.
        mask: Optional mask for selective updates
        
    Returns:
        Log probability with shape (batch,)
    """
    diff_sq = (x - mu) ** 2 / (sigma ** 2 + 1e-12)
    
    if mask is not None:
        numer = torch.sum(-0.5 * diff_sq, dim=[1, 2, 3], where=mask)
        denom = x.shape[-1] * torch.sum(torch.log(sigma + 1e-12), dim=[1, 2, 3], where=mask)
    else:
        numer = torch.sum(-0.5 * diff_sq, dim=[1, 2, 3])
        denom = x.shape[-1] * torch.sum(torch.log(sigma + 1e-12), dim=[1, 2, 3])
        
    return numer - denom

class MetropolisHastings(nn.Module):
    """Metropolis-Hastings MCMC sampler for quantum Monte Carlo."""
    
    def __init__(self, 
                 step_size: float = 0.02,
                 n_steps: int = DEFAULT_MCMC_STEPS,
                 blocks: int = 1,
                 atoms: Optional[torch.Tensor] = None,
                 debug: bool = False):
        """Initialize the Metropolis-Hastings sampler.
        
        Args:
            step_size: Standard deviation for the proposal distribution
            n_steps: Number of MCMC steps per iteration
            blocks: Number of blocks for electron updates
            atoms: Atom positions for adaptive step sizes
            debug: Whether to print debug info
        """
        super(MetropolisHastings, self).__init__()
        self.step_size = step_size
        self.n_steps = n_steps
        self.blocks = blocks
        self.atoms = atoms
        self.debug = debug
    
    def mh_update(self, 
                 params: Any,
                 f: Callable,
                 x1: torch.Tensor,
                 lp_1: torch.Tensor,
                 num_accepts: int,
                 stddev: float,
                 step_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Performs one Metropolis-Hastings step using an all-electron move.
        
        Args:
            params: Wave function parameters (not used in our case)
            f: Wave function that returns log|psi|
            x1: Initial MCMC configurations. Shape (batch, n_electrons*3)
            lp_1: log probability of f evaluated at x1
            num_accepts: Number of MH move proposals accepted
            stddev: width of Gaussian move proposal
            step_idx: Current step index for block updates
            
        Returns:
            (x_new, lp_new, num_accepts): Updated configurations, log probs, and accept count
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Handle block updates
        if self.blocks > 1:
            block_size = x1.shape[-1] // self.blocks
            block_pos = step_idx % self.blocks
            mask = torch.zeros_like(x1)
            start_idx = block_pos * block_size
            end_idx = min((block_pos + 1) * block_size, x1.shape[-1])
            mask[:, start_idx:end_idx] = 1.0
        else:
            mask = torch.ones_like(x1)
        
        if self.atoms is None:  # symmetric proposal, same stddev everywhere
            proposed_move = stddev * torch.randn_like(x1)
            x2 = x1 + proposed_move * mask
            lp_2 = 2.0 * f(x2)  # log prob of proposal
            ratio = lp_2 - lp_1
        else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
            # Reshape for 3D operations
            n = x1.shape[0]
            x1_3d = x1.reshape(n, -1, 1, 3)
            mask_3d = mask.reshape(n, -1, 1, 3)
            
            hmean1 = _harmonic_mean(x1_3d, self.atoms)  # [batch, n_electrons, 1, 1]
            
            proposed_move = stddev * hmean1 * torch.randn_like(x1_3d)
            x2_3d = x1_3d + proposed_move * mask_3d
            x2 = x2_3d.reshape(n, -1)
            
            lp_2 = 2.0 * f(x2)  # log prob of proposal
            hmean2 = _harmonic_mean(x2_3d, self.atoms)  # needed for probability of reverse jump
            
            # Calculate forward and reverse probabilities
            lq_1 = _log_prob_gaussian(x1_3d, x2_3d, stddev * hmean1, mask=mask_3d)  # forward
            lq_2 = _log_prob_gaussian(x2_3d, x1_3d, stddev * hmean2, mask=mask_3d)  # reverse
            ratio = lp_2 + lq_2 - lp_1 - lq_1
        
        # Accept/reject step
        rnd = torch.log(torch.rand(batch_size, device=device) + 1e-12)
        cond = ratio > rnd
        
        x_new = torch.where(cond.unsqueeze(-1), x2, x1)
        lp_new = torch.where(cond, lp_2, lp_1)
        num_accepts += cond.sum().item()
        
        return x_new, lp_new, num_accepts
    
    def forward(self, 
                samples: torch.Tensor, 
                log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
                n_steps: Optional[int] = None) -> Tuple[torch.Tensor, float]:
        """Perform MCMC sampling.
        
        Args:
            samples: Initial samples [batch_size, n_electrons, 3] or [batch_size, dim]
            log_prob_fn: Target distribution log probability function
            n_steps: Number of MCMC steps (overrides self.n_steps if provided)
            
        Returns:
            Tuple containing:
                - New samples after MCMC
                - Acceptance ratio
        """
        steps = n_steps if n_steps is not None else self.n_steps
        device = samples.device
        
        # Ensure proper shape for processing
        if len(samples.shape) == 3:
            # [batch, n_electrons, 3] -> [batch, n_electrons*3]
            samples_flat = samples.view(samples.shape[0], -1)
            reshape_output = True
            original_shape = samples.shape
        else:
            samples_flat = samples
            reshape_output = False
        
        batch_size = samples_flat.shape[0]
        
        # Initial log probability
        with torch.no_grad():
            current_log_prob = 2.0 * log_prob_fn(samples if not reshape_output else samples)
        
        current_samples = samples_flat.clone()
        num_accepts = 0
        
        # Total steps considering blocks
        total_steps = self.blocks * steps
        
        # Execute MCMC steps
        for step in range(total_steps):
            current_samples, current_log_prob, num_accepts = self.mh_update(
                params=None,
                f=lambda x: log_prob_fn(x.view(original_shape) if reshape_output else x),
                x1=current_samples,
                lp_1=current_log_prob,
                num_accepts=num_accepts,
                stddev=self.step_size,
                step_idx=step
            )
        
        # Calculate acceptance rate
        if total_steps > 0:
            accept_ratio = num_accepts / (batch_size * total_steps)
        else:
            accept_ratio = 0.0  # No steps taken, no acceptance ratio
        
        if self.debug:
            print(f"MCMC finished with accept_ratio: {accept_ratio:.4f}")
        
        # Reshape output if needed
        if reshape_output:
            current_samples = current_samples.view(original_shape)
        
        return current_samples, accept_ratio
    
    def run_burn_in(self, 
                   samples: torch.Tensor, 
                   log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
                   burn_in_steps: int = MCMC_BURN_IN) -> torch.Tensor:
        """Run burn-in period to equilibrate MCMC chain.
        
        Args:
            samples: Initial samples
            log_prob_fn: Target distribution log probability function
            burn_in_steps: Number of burn-in steps
            
        Returns:
            Equilibrated samples
        """
        if self.debug:
            print(f"Starting burn-in with {burn_in_steps} steps")
        
        # If no burn-in steps, return original samples
        if burn_in_steps <= 0:
            if self.debug:
                print("No burn-in steps required, returning original samples")
            return samples
        
        samples_burned, _ = self.forward(samples, log_prob_fn, n_steps=burn_in_steps)
        return samples_burned

def mcmc_step(wave_function: Callable[[torch.Tensor], torch.Tensor],
              electron_coords: torch.Tensor,
              step_size: float = 0.02,
              n_steps: int = 10,
              atoms: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """
    Single MCMC step function for compatibility.
    
    Args:
        wave_function: Wave function that returns log|psi|
        electron_coords: Electron coordinates [batch_size, n_electrons, 3]
        step_size: MCMC step size
        n_steps: Number of MCMC steps
        atoms: Atom positions for adaptive step sizes
        
    Returns:
        (new_coords, accept_ratio): Updated coordinates and acceptance ratio
    """
    sampler = MetropolisHastings(step_size=step_size, n_steps=n_steps, atoms=atoms)
    
    def log_prob_fn(coords):
        # Wave function should return log|psi| (we multiply by 2 in the sampler)
        return wave_function(coords)
    
    return sampler(electron_coords, log_prob_fn)

# 保留向后兼容的函数接口
def mcmc_sampling(initial_samples, target_fn, n_steps, step_size):
    """
    Metropolis-Hastings MCMC采样的函数接口（向后兼容）
    
    Args:
        initial_samples: [batch, dim] 初始样本
        target_fn: 目标分布的对数概率函数
        n_steps: 采样步数
        step_size: 步长
        
    Returns:
        [batch, dim] 采样样本
    """
    sampler = MetropolisHastings(step_size=step_size, n_steps=n_steps)
    samples, _ = sampler(initial_samples, target_fn)
    return samples 