"""
Training script for PyTorch NNVMC.

This module implements the training loop for neural network variational Monte Carlo
with Adam optimizer, progress tracking, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import argparse

from mcmc import MetropolisHastings
from hamiltonian import LocalEnergyHamiltonian, potential_energy
from loss import VMCLoss, make_loss_fn
from networks import make_wave_function, make_lapnet_wave_function

class VMCTrainer:
    """Variational Monte Carlo trainer."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            config: Training configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        
        # Setup system
        self.setup_system()
        
        # Initialize wave function
        try:
            self.wave_function = make_lapnet_wave_function(config['system']).to(device)
        except:
            # Fallback to legacy interface
            self.wave_function = make_wave_function(config['system']).to(device)
        
        # Initialize Hamiltonian
        hamiltonian_config = config['hamiltonian']
        self.hamiltonian = LocalEnergyHamiltonian(
            atom_coords=self.atom_coords,
            nuclear_charges=self.nuclear_charges,
            method=hamiltonian_config.get('method', 'log'),
            grad_clip=hamiltonian_config.get('grad_clip', 50.0),
            include_electron_electron=False

        ).to(device)
        
        # Initialize loss function
        self.loss_fn = make_loss_fn(
            loss_type=config['loss'].get('type', 'vmc'),
            **config['loss'].get('params', {})
        )
        
        # Initialize optimizer
        self.optimizer = self.setup_optimizer()
        
        # Initialize MCMC sampler
        self.mcmc_sampler = MetropolisHastings(
            step_size=config['mcmc']['step_size'],
            n_steps=config['mcmc']['n_steps'],
            debug=config.get('debug', False)
        )
        
        # MCMC step size adaptation parameters
        self.mcmc_config = config['mcmc']
        self.adapt_step_size = config['mcmc'].get('adapt_step_size', True)
        self.step_size_min = config['mcmc'].get('step_size_min', 0.001)
        self.step_size_max = config['mcmc'].get('step_size_max', 0.5)
        self.target_accept_rate = config['mcmc'].get('target_accept_rate', 0.5)
        self.accept_rate_window = config['mcmc'].get('accept_rate_window', 10)
        
        # Training tracking
        self.history = {
            'energy': [],
            'variance': [],
            'acceptance_rate': [],
            'step_size': [],  # Track step size changes
            'loss': [],
            'step': []
        }
        
        # Current electron coordinates
        self.electron_coords = None
        
    def setup_system(self):
        """Setup the quantum system."""
        system_config = self.config['system']
        
        # Atom coordinates
        if 'atom_coords' in system_config:
            self.atom_coords = torch.tensor(
                system_config['atom_coords'], 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            # Default: single atom at origin (hydrogen)
            self.atom_coords = torch.zeros(1, 3, device=self.device)
        
        # Nuclear charges
        if 'nuclear_charges' in system_config:
            self.nuclear_charges = torch.tensor(
                system_config['nuclear_charges'], 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            # Default: all charges are 1
            self.nuclear_charges = torch.ones(
                self.atom_coords.shape[0], 
                device=self.device
            )
        
        # Electron spins
        n_electrons_up = system_config['n_electrons_up']
        n_electrons_down = system_config['n_electrons_down']
        self.n_electrons = n_electrons_up + n_electrons_down
        
        # Create spin array
        self.spins = torch.ones(self.n_electrons, device=self.device)
        self.spins[n_electrons_up:] = -1  # spin-down electrons
        
    def setup_optimizer(self):
        """Setup Adam optimizer."""
        optim_config = self.config['optimizer']
        
        return optim.Adam(
            self.wave_function.parameters(),
            lr=optim_config['lr'],
            betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.999)),
            eps=optim_config.get('eps', 1e-8),
            weight_decay=optim_config.get('weight_decay', 0.0)
        )
    
    def initialize_electrons(self, batch_size: int) -> torch.Tensor:
        """根据原子电荷比例初始化电子位置。"""
        init_width = self.config['mcmc'].get('init_width', 1.0)
        
        # 初始化电子坐标列表
        electron_coords = []
        n_electrons_up = self.config['system']['n_electrons_up']
        n_electrons_down = self.config['system']['n_electrons_down']
        n_atoms = self.atom_coords.shape[0]
        
        # 计算每个原子的相对权重（基于电荷）
        total_charge = self.nuclear_charges.sum().item()
        charge_weights = self.nuclear_charges / total_charge
        
        # 按照电荷比例分配自旋向上电子
        electrons_per_atom_up = torch.floor(charge_weights * n_electrons_up).int()
        # 按照电荷比例分配自旋向下电子
        electrons_per_atom_down = torch.floor(charge_weights * n_electrons_down).int()
        
        # 初始化各原子周围的电子
        for i in range(n_atoms):
            atom_pos = self.atom_coords[i]
            
            # 自旋向上电子
            for _ in range(electrons_per_atom_up[i].item()):
                pos = atom_pos + init_width * torch.randn(batch_size, 3, device=self.device)
                electron_coords.append(pos)
            
            # 自旋向下电子
            for _ in range(electrons_per_atom_down[i].item()):
                pos = atom_pos + init_width * torch.randn(batch_size, 3, device=self.device)
                electron_coords.append(pos)
        
        # 处理剩余电子（由于向下取整可能有未分配的电子）
        remaining_up = n_electrons_up - electrons_per_atom_up.sum().item()
        remaining_down = n_electrons_down - electrons_per_atom_down.sum().item()
        
        # 对于剩余的电子，按照电荷大小优先分配给电荷较大的原子
        if remaining_up > 0 or remaining_down > 0:
            # 获取按电荷降序排列的原子索引
            _, sorted_indices = torch.sort(self.nuclear_charges, descending=True)
            
            # 分配剩余的自旋向上电子
            atom_idx = 0
            for _ in range(remaining_up):
                atom_pos = self.atom_coords[sorted_indices[atom_idx]]
                pos = atom_pos + init_width * torch.randn(batch_size, 3, device=self.device)
                electron_coords.append(pos)
                # 循环使用原子
                atom_idx = (atom_idx + 1) % n_atoms
            
            # 分配剩余的自旋向下电子
            atom_idx = 0
            for _ in range(remaining_down):
                atom_pos = self.atom_coords[sorted_indices[atom_idx]]
                pos = atom_pos + init_width * torch.randn(batch_size, 3, device=self.device)
                electron_coords.append(pos)
                # 循环使用原子
                atom_idx = (atom_idx + 1) % n_atoms
        
        return torch.stack(electron_coords, dim=1)  # [batch_size, n_electrons, 3]    
        
    def training_step(self, electron_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform one VMC training step with custom gradient computation.
        
        Implements the VMC gradient formula: ∇E = ⟨(E_local - E) * ∇log|ψ|⟩
        This is fundamentally different from standard backpropagation.
        """
        batch_size = electron_coords.shape[0]
        device = electron_coords.device
        
        # Expand spins to batch
        spins_batch = self.spins.unsqueeze(0).expand(batch_size, -1)
        
        # Step 1: Compute local energies (forward pass without gradients for energy statistics only)
        with torch.no_grad():
            # For energy statistics, we only need potential energy (kinetic energy calculation requires gradients)
            # We'll compute a rough energy estimate for outlier detection
            from hamiltonian import potential_energy
            
            # Compute only potential energy for statistics (no gradients needed)
            potential_energies = potential_energy(
                electron_coords, 
                self.hamiltonian.atom_coords, 
                self.hamiltonian.nuclear_charges,
                min_distance=self.hamiltonian.min_distance,
                include_electron_electron= False
            )
            
            # Use potential energy as a proxy for outlier detection
            # This is not the full local energy but sufficient for outlier filtering
            local_energies = potential_energies  # Only potential part
            
            # Handle outliers and compute statistics
            finite_mask = torch.isfinite(local_energies)
            if not finite_mask.all():
                # Replace non-finite values with median of finite values
                finite_energies = local_energies[finite_mask]
                if len(finite_energies) == 0:
                    print("Warning: No finite local energies found!")
                    return torch.tensor(0.0, device=device, requires_grad=True), {
                        'energy_mean': torch.tensor(float('nan')),
                        'energy_var': torch.tensor(float('nan')),
                        'n_finite': 0,
                        'outlier_mask': finite_mask
                    }
                median_energy = torch.median(finite_energies)
                local_energies = torch.where(finite_mask, local_energies, median_energy)
            
            # Remove outliers if configured
            loss_config = self.config['loss']['params']
            if loss_config.get('remove_outliers', True):
                outlier_width = loss_config.get('outlier_width', 10.0)
                energy_mean = torch.mean(local_energies[finite_mask])
                energy_dev = torch.mean(torch.abs(local_energies[finite_mask] - energy_mean))
                
                outlier_mask = (
                    (local_energies >= energy_mean - outlier_width * energy_dev) &
                    (local_energies <= energy_mean + outlier_width * energy_dev) &
                    finite_mask
                )
            else:
                outlier_mask = finite_mask
            
            # Note: energy_mean and energy_var here are based on potential energy only
            # The actual local energy (kinetic + potential) will be computed with gradients below
            valid_energies = local_energies[outlier_mask]
            if len(valid_energies) == 0:
                energy_mean = torch.tensor(0.0, device=device)
                energy_var = torch.tensor(0.0, device=device)
            else:
                energy_mean = torch.mean(valid_energies)
                energy_var = torch.var(valid_energies)
        
        # Step 2: Compute log|ψ| with gradients for custom VMC gradient
        # Enable gradients for parameters
        self.optimizer.zero_grad()
        
        # Ensure electron_coords requires gradient for Hamiltonian computation
        electron_coords_grad = electron_coords.clone().detach().requires_grad_(True)
        
        # Compute log|ψ| with gradient tracking
        log_psi = self.wave_function(electron_coords_grad, spins_batch)
        
        # Step 3: Implement VMC custom gradient computation
        # We need to compute ∇E = ⟨(E_local - E) * ∇log|ψ|⟩
        
        # Recompute local energies with gradients for gradient computation
        def wave_fn_with_grad(coords):
            return self.wave_function(coords, spins_batch)
        
        def hamiltonian_fn_with_grad(wave_fn_inner, coords):
            return self.hamiltonian(wave_fn_inner, coords)
        
        local_energies_grad = hamiltonian_fn_with_grad(wave_fn_with_grad, electron_coords_grad)
        
        # Update energy statistics with the real local energies (kinetic + potential)
        with torch.no_grad():
            # Use the complete local energies for final statistics
            real_local_energies = local_energies_grad.detach()
            
            # Apply the same outlier mask but update statistics
            valid_real_energies = real_local_energies[outlier_mask]
            if len(valid_real_energies) > 0:
                energy_mean = torch.mean(valid_real_energies)
                energy_var = torch.var(valid_real_energies)
            # Keep the original potential-based values if no valid energies
        
        # Apply energy clipping for gradient computation (if configured)
        clip_energy = loss_config.get('clip_local_energy', 5.0)
        if clip_energy > 0:
            # Use median-based clipping but preserve gradients
            if outlier_mask.sum() > 0:
                # Don't use no_grad context when processing tensors that need gradients
                valid_energies_for_clip = local_energies_grad[outlier_mask]
                
                # Compute clipping parameters
                with torch.no_grad():
                    # Only compute statistics without gradients
                    valid_energies_detached = valid_energies_for_clip.detach()
                    median_energy = torch.median(valid_energies_detached)
                    energy_dev = torch.mean(torch.abs(valid_energies_detached - median_energy))
                
                # Clip local energies for gradient computation (with gradients preserved)
                clipped_energies = torch.clamp(
                    local_energies_grad,
                    min=median_energy - clip_energy * energy_dev,
                    max=median_energy + clip_energy * energy_dev
                )
                
                # Renormalize difference to have zero mean
                if outlier_mask.sum() > 0:
                    clipped_mean = torch.sum(clipped_energies * outlier_mask.float()) / outlier_mask.sum()
                    energy_diff = clipped_energies - clipped_mean
                else:
                    energy_diff = clipped_energies - energy_mean
            else:
                energy_diff = local_energies_grad - energy_mean
        else:
            energy_diff = local_energies_grad - energy_mean
        
        # Apply outlier mask to energy differences
        energy_diff = energy_diff * outlier_mask.float()
        
        # Step 4: Compute VMC gradient: ∇E = ⟨(E_local - E) * ∇log|ψ|⟩
        # This is the key difference from standard backpropagation!
        
        # IMPORTANT: Detach energy_diff to prevent it from participating in gradient computation
        # In VMC, E_local should be treated as constants, only ∇log|ψ| should have gradients
        energy_diff_detached = energy_diff.detach()
        
        # Compute the VMC loss using the custom gradient formula
        # We use the fact that autograd will compute ∇log|ψ| when we backprop through log_psi
        
        effective_batch_size = outlier_mask.sum().float()
        if effective_batch_size > 0:
            # Compute weighted sum for VMC gradient
            # Note: energy_diff is detached, so gradients only flow through log_psi
            vmc_loss = torch.sum(energy_diff_detached * log_psi) / effective_batch_size
        else:
            vmc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Step 5: Backward pass with custom VMC gradient
        vmc_loss.backward()
        
        # Step 6: Gradient clipping
        grad_clip = self.config['optimizer'].get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.wave_function.parameters(), grad_clip)
        
        # Step 7: Optimizer step
        self.optimizer.step()
        
        # Step 8: Prepare auxiliary data
        aux_data = {
            'energy_mean': energy_mean.detach(),
            'energy_var': energy_var.detach(),
            'energy_std': torch.sqrt(energy_var).detach(),
            'local_energies': real_local_energies,  # Use real complete local energies
            'n_finite': finite_mask.sum().item(),
            'n_outliers': (finite_mask.sum() - outlier_mask.sum()).item(),
            'effective_batch_size': effective_batch_size.item(),
            'outlier_mask': outlier_mask,
            'vmc_loss': vmc_loss.detach()
        }
        
        # Return the mean energy as the loss (for monitoring), not the VMC loss
        return energy_mean, aux_data
    
    def mcmc_step(self, electron_coords: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Perform MCMC sampling step."""
        batch_size = electron_coords.shape[0]
        spins_batch = self.spins.unsqueeze(0).expand(batch_size, -1)

        def log_prob_fn(coords):
            with torch.no_grad():
                log_psi = self.wave_function(coords, spins_batch)
                return 2.0 * log_psi  # |ψ|²
        
        return self.mcmc_sampler(electron_coords, log_prob_fn)
    
    def adapt_mcmc_step_size(self, current_accept_rate: float) -> bool:
        """
        动态调整MCMC步长基于接受率
        
        Args:
            current_accept_rate: 当前接受率
            
        Returns:
            bool: 是否调整了步长
        """
        if not self.adapt_step_size:
            return False
            
        old_step_size = self.mcmc_sampler.step_size
        adjusted = False
        
        # 接受率太低（< 0.3），显著减少步长  
        if current_accept_rate < 0.3:
            new_step_size = max(old_step_size / 3.0, self.step_size_min)
            if new_step_size != old_step_size:
                self.mcmc_sampler.step_size = new_step_size
                adjusted = True
                print(f"📉📉 MCMC步长大幅减少: {old_step_size:.4f} → {new_step_size:.4f} (接受率={current_accept_rate:.3f})")
                
        # 接受率偏低（0.3-0.6），小幅减少步长
        elif current_accept_rate < 0.4:
            new_step_size = max(old_step_size / 1.2, self.step_size_min)
            if new_step_size != old_step_size:
                self.mcmc_sampler.step_size = new_step_size
                adjusted = True
                print(f"📉 MCMC步长减少: {old_step_size:.4f} → {new_step_size:.4f} (接受率={current_accept_rate:.3f})")
                
        # 接受率很高（> 0.8），增加步长但更保守
        elif current_accept_rate > 0.8:
            new_step_size = min(old_step_size * 1.2, self.step_size_max)  # 从1.5改为1.2，更保守
            if new_step_size != old_step_size:
                self.mcmc_sampler.step_size = new_step_size
                adjusted = True
                print(f"📈 MCMC步长增加: {old_step_size:.4f} → {new_step_size:.4f} (接受率={current_accept_rate:.3f})")
        
        return adjusted
    
    def train(self, 
              n_iterations: int,
              batch_size: int,
              burn_in_steps: int = 100,
              save_frequency: int = 100,
              plot_frequency: int = 50) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            n_iterations: Number of training iterations
            batch_size: Batch size
            burn_in_steps: Number of burn-in MCMC steps
            save_frequency: Frequency to save checkpoints
            plot_frequency: Frequency to update plots
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Wave function parameters: {sum(p.numel() for p in self.wave_function.parameters()):,}")
        
        # Initialize electron coordinates
        self.electron_coords = self.initialize_electrons(batch_size)
        
        # Burn-in MCMC
        print(f"Performing burn-in with {burn_in_steps} steps...")
        self.electron_coords = self.mcmc_sampler.run_burn_in(
            self.electron_coords, 
            lambda coords: 2.0 * self.wave_function(
                coords, 
                self.spins.unsqueeze(0).expand(batch_size, -1)
            ),
            burn_in_steps
        )
        
        # Setup progress bar
        pbar = tqdm(range(n_iterations), desc="Training")
        
        # Setup plotting to save to file instead of showing
        plot_data = {'steps': [], 'energy': [], 'variance': [], 'acceptance_rate': []}
        
        start_time = time.time()
        
        for iteration in pbar:
            # MCMC step
            self.electron_coords, accept_rate = self.mcmc_step(self.electron_coords)
            
            # Training step
            loss, aux_data = self.training_step(self.electron_coords)
            
            # Record history
            energy = aux_data['energy_mean'].item()
            variance = aux_data['energy_var'].item()
            
            self.history['energy'].append(energy)
            self.history['variance'].append(variance)
            self.history['acceptance_rate'].append(accept_rate)
            self.history['step_size'].append(self.mcmc_sampler.step_size)  # Record current step size
            self.history['loss'].append(loss.item())
            self.history['step'].append(iteration)
            
            # Adapt MCMC step size based on acceptance rate
            if self.adapt_step_size and iteration > 0 and (iteration + 1) % self.accept_rate_window == 0:
                # Calculate recent average acceptance rate
                recent_accept_rates = self.history['acceptance_rate'][-self.accept_rate_window:]
                avg_accept_rate = sum(recent_accept_rates) / len(recent_accept_rates)
                self.adapt_mcmc_step_size(avg_accept_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'Energy': f'{energy:.4f}',
                'Var': f'{variance:.4f}',
                'Accept': f'{accept_rate:.3f}',
                'StepSize': f'{self.mcmc_sampler.step_size:.4f}',  # Show current step size
                'Loss': f'{loss.item():.4f}',
                'Finite': f"{aux_data['n_finite']}/{batch_size}",
                'Outliers': aux_data['n_outliers']
            })
            
            # Save plot data
            if plot_frequency > 0 and (iteration + 1) % plot_frequency == 0:
                plot_data['steps'].append(iteration)
                plot_data['energy'].append(energy)
                plot_data['variance'].append(variance)
                plot_data['acceptance_rate'].append(accept_rate)
            
            # Save checkpoint
            if save_frequency > 0 and (iteration + 1) % save_frequency == 0:
                self.save_checkpoint(iteration + 1)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save final plots to file instead of showing
        if plot_frequency > 0 and len(self.history['step']) > 0:
            self.save_training_plots()
        
        return self.history
    
    def save_training_plots(self):
        """Save training plots to file instead of showing."""
        if not self.history['step']:
            return
        
        steps = self.history['step']
        
        # Create figure with 2x3 layout to include step size plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle('NNVMC Training Progress')
        
        # Energy plot
        axes[0, 0].plot(steps, self.history['energy'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Energy')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Energy (a.u.)')
        axes[0, 0].grid(True)
        
        # Variance plot
        axes[0, 1].plot(steps, self.history['variance'], 'r-', alpha=0.7)
        axes[0, 1].set_title('Energy Variance')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].grid(True)
        
        # Loss plot
        axes[0, 2].plot(steps, self.history['loss'], 'm-', alpha=0.7)
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        # Acceptance rate plot
        axes[1, 0].plot(steps, self.history['acceptance_rate'], 'g-', alpha=0.7)
        axes[1, 0].set_title('MCMC Acceptance Rate')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Acceptance Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True)
        
        # MCMC Step Size plot
        if 'step_size' in self.history and len(self.history['step_size']) > 0:
            axes[1, 1].plot(steps, self.history['step_size'], 'orange', alpha=0.7)
            axes[1, 1].set_title('MCMC Step Size')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Step Size')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Step Size Data', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('MCMC Step Size')
        
        # Combined Accept Rate & Step Size plot
        if 'step_size' in self.history and len(self.history['step_size']) > 0:
            ax_accept = axes[1, 2]
            ax_step = ax_accept.twinx()
            
            line1 = ax_accept.plot(steps, self.history['acceptance_rate'], 'g-', alpha=0.7, label='Accept Rate')
            line2 = ax_step.plot(steps, self.history['step_size'], 'orange', alpha=0.7, label='Step Size')
            
            ax_accept.set_xlabel('Iteration')
            ax_accept.set_ylabel('Acceptance Rate', color='g')
            ax_step.set_ylabel('Step Size', color='orange')
            ax_accept.set_ylim(0, 1)
            ax_accept.grid(True)
            ax_accept.set_title('Accept Rate & Step Size')
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_accept.legend(lines, labels, loc='upper right')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Combined Data', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Accept Rate & Step Size')
        
        plt.tight_layout()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = self.config.get('plots_dir', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, f'training_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Training plots saved to {plot_path}")
    
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint with error handling."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'wave_function_state_dict': self.wave_function.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'electron_coords': self.electron_coords.cpu() if self.electron_coords is not None else None
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pt')
        
        # Try saving with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                return
            except (TimeoutError, OSError, IOError) as e:
                print(f"Checkpoint save attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"Failed to save checkpoint after {max_retries} attempts. Continuing training...")
                    return
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.wave_function.load_state_dict(checkpoint['wave_function_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if checkpoint['electron_coords'] is not None:
            self.electron_coords = checkpoint['electron_coords'].to(self.device)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['iteration']

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for hydrogen atom."""
    return {
        'system': {
            'n_electrons_up': 1,
            'n_electrons_down': 0,
            'n_atoms': 1,
            'atom_coords': [[0.0, 0.0, 0.0]],
            'nuclear_charges': [1.0],
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'n_determinants': 4,
            'dropout': 0.1,
            'use_layer_norm': True,
            'jastrow_init': 0.5
        },
        'mcmc': {
            'step_size': 0.02,
            'n_steps': 10,
            'init_width': 1.0
        },
        'hamiltonian': {
            'method': 'log'
        },
        'loss': {
            'type': 'vmc',
            'params': {
                'clip_local_energy': 5.0,
                'remove_outliers': True,
                'outlier_width': 10.0
            }
        },
        'optimizer': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0.0,
            'grad_clip': 1.0
        },
        'training': {
            'n_iterations': 1000,
            'batch_size': 64,
            'burn_in_steps': 100,
            'save_frequency': 200,
            'plot_frequency': 50
        },
        'checkpoint_dir': 'checkpoints'
    }

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='PyTorch NNVMC Training')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        print("Using default configuration for hydrogen atom")
    
    # Create trainer
    trainer = VMCTrainer(config, device)
    
    # Load checkpoint if provided
    start_iteration = 0
    if args.checkpoint:
        start_iteration = trainer.load_checkpoint(args.checkpoint)
    
    # Training parameters
    training_config = config['training']
    n_iterations = training_config['n_iterations'] - start_iteration
    
    if n_iterations > 0:
        # Train
        history = trainer.train(
            n_iterations=n_iterations,
            batch_size=training_config['batch_size'],
            burn_in_steps=training_config['burn_in_steps'] if start_iteration == 0 else 0,
            save_frequency=training_config['save_frequency'],
            plot_frequency=training_config['plot_frequency']
        )
        
        # Save final results
        results = {
            'config': config,
            'history': history,
            'final_energy': history['energy'][-1] if history['energy'] else None,
            'final_variance': history['variance'][-1] if history['variance'] else None
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = config.get('results_dir', 'results_json')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        if history['energy']:
            print(f"Final energy: {history['energy'][-1]:.6f} ± {np.sqrt(history['variance'][-1]):.6f}")
    else:
        print("Training already completed according to checkpoint")

if __name__ == '__main__':
    main() 