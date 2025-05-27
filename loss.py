"""
Loss functions for PyTorch NNVMC.

This module implements loss functions for variational quantum Monte Carlo.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any

class VMCLoss(nn.Module):
    """Variational Monte Carlo loss function."""
    
    def __init__(self, 
                 clip_local_energy: float = 5.0,
                 remove_outliers: bool = True,
                 outlier_width: float = 10.0):
        """
        Args:
            clip_local_energy: 局部能量剪切阈值
            remove_outliers: 是否移除异常值
            outlier_width: 异常值检测宽度
        """
        super().__init__()
        self.clip_local_energy = clip_local_energy
        self.remove_outliers = remove_outliers
        self.outlier_width = outlier_width
    
    def forward(self, 
                wave_fn: Callable[[torch.Tensor], torch.Tensor],
                hamiltonian_fn: Callable[[Callable, torch.Tensor], torch.Tensor],
                electron_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算VMC损失函数
        
        Args:
            wave_fn: 波函数，返回log|ψ|
            hamiltonian_fn: 哈密顿量函数
            electron_coords: 电子坐标 [batch_size, n_electrons, 3]
            
        Returns:
            (loss, aux_data): 损失值和辅助数据
        """
        batch_size = electron_coords.shape[0]
        device = electron_coords.device
        
        # 计算局部能量
        local_energies = hamiltonian_fn(wave_fn, electron_coords)
        
        # 异常值处理
        if self.remove_outliers:
            local_energies, outlier_mask = self._remove_outliers(local_energies)
        else:
            outlier_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # 能量剪切
        if self.clip_local_energy > 0:
            local_energies = torch.clamp(
                local_energies,
                min=-self.clip_local_energy,
                max=self.clip_local_energy
            )
        
        # 计算损失（能量期望值）
        loss = torch.mean(local_energies)
        
        # 计算辅助统计量
        aux_data = {
            'local_energies': local_energies.detach(),
            'energy_mean': loss.detach(),
            'energy_var': torch.var(local_energies).detach(),
            'energy_std': torch.std(local_energies).detach(),
            'outlier_mask': outlier_mask,
            'n_outliers': torch.sum(~outlier_mask).item(),
            'effective_batch_size': torch.sum(outlier_mask).item()
        }
        
        return loss, aux_data
    
    def _remove_outliers(self, energies: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """移除局部能量异常值"""
        median_energy = torch.median(energies)
        mad = torch.median(torch.abs(energies - median_energy))  # Median Absolute Deviation
        
        # 定义异常值边界
        lower_bound = median_energy - self.outlier_width * mad
        upper_bound = median_energy + self.outlier_width * mad
        
        # 创建掩码
        outlier_mask = (energies >= lower_bound) & (energies <= upper_bound)
        
        # 只保留非异常值
        if torch.sum(outlier_mask) > 0:
            filtered_energies = energies[outlier_mask]
        else:
            # 如果所有值都被认为是异常值，保留原始值
            filtered_energies = energies
            outlier_mask = torch.ones_like(energies, dtype=torch.bool)
        
        return filtered_energies, outlier_mask

class ReweightedVMCLoss(nn.Module):
    """Reweighted VMC loss with importance sampling."""
    
    def __init__(self, 
                 clip_local_energy: float = 5.0,
                 remove_outliers: bool = True,
                 outlier_width: float = 10.0,
                 center_energies: bool = True):
        """
        Args:
            clip_local_energy: 局部能量剪切阈值
            remove_outliers: 是否移除异常值
            outlier_width: 异常值检测宽度
            center_energies: 是否中心化能量
        """
        super().__init__()
        self.clip_local_energy = clip_local_energy
        self.remove_outliers = remove_outliers
        self.outlier_width = outlier_width
        self.center_energies = center_energies
    
    def forward(self, 
                wave_fn: Callable[[torch.Tensor], torch.Tensor],
                hamiltonian_fn: Callable[[Callable, torch.Tensor], torch.Tensor],
                electron_coords: torch.Tensor,
                target_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算重权重VMC损失函数
        
        Args:
            wave_fn: 波函数
            hamiltonian_fn: 哈密顿量函数
            electron_coords: 当前电子坐标
            target_coords: 目标电子坐标（用于重权重）
            
        Returns:
            (loss, aux_data): 损失值和辅助数据
        """
        # 计算当前配置的局部能量
        local_energies = hamiltonian_fn(wave_fn, electron_coords)
        
        # 计算权重（概率比）
        log_psi_current = wave_fn(electron_coords)
        log_psi_target = wave_fn(target_coords)
        log_weights = 2.0 * (log_psi_target - log_psi_current)
        weights = torch.exp(log_weights.clamp(max=10))  # 避免权重过大
        
        # 权重归一化
        weights = weights / torch.sum(weights)
        
        # 异常值处理
        if self.remove_outliers:
            local_energies, outlier_mask = self._remove_outliers(local_energies)
            weights = weights[outlier_mask]
        else:
            outlier_mask = torch.ones_like(local_energies, dtype=torch.bool)
        
        # 能量剪切
        if self.clip_local_energy > 0:
            local_energies = torch.clamp(
                local_energies,
                min=-self.clip_local_energy,
                max=self.clip_local_energy
            )
        
        # 中心化能量
        if self.center_energies:
            energy_mean = torch.sum(weights * local_energies)
            local_energies = local_energies - energy_mean
        
        # 计算加权损失
        loss = torch.sum(weights * local_energies)
        
        # 辅助数据
        aux_data = {
            'local_energies': local_energies.detach(),
            'weights': weights.detach(),
            'energy_mean': torch.sum(weights * local_energies).detach(),
            'energy_var': torch.sum(weights * local_energies**2).detach(),
            'outlier_mask': outlier_mask,
            'n_outliers': torch.sum(~outlier_mask).item(),
            'effective_batch_size': torch.sum(outlier_mask).item()
        }
        
        return loss, aux_data
    
    def _remove_outliers(self, energies: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """移除局部能量异常值"""
        median_energy = torch.median(energies)
        mad = torch.median(torch.abs(energies - median_energy))
        
        lower_bound = median_energy - self.outlier_width * mad
        upper_bound = median_energy + self.outlier_width * mad
        
        outlier_mask = (energies >= lower_bound) & (energies <= upper_bound)
        
        if torch.sum(outlier_mask) > 0:
            filtered_energies = energies[outlier_mask]
        else:
            filtered_energies = energies
            outlier_mask = torch.ones_like(energies, dtype=torch.bool)
        
        return filtered_energies, outlier_mask

class VarianceRegularizedLoss(nn.Module):
    """VMC loss with variance regularization."""
    
    def __init__(self, 
                 variance_weight: float = 0.1,
                 clip_local_energy: float = 5.0,
                 remove_outliers: bool = True,
                 outlier_width: float = 10.0):
        """
        Args:
            variance_weight: 方差正则化权重
            clip_local_energy: 局部能量剪切阈值
            remove_outliers: 是否移除异常值
            outlier_width: 异常值检测宽度
        """
        super().__init__()
        self.variance_weight = variance_weight
        self.vmc_loss = VMCLoss(clip_local_energy, remove_outliers, outlier_width)
    
    def forward(self, 
                wave_fn: Callable[[torch.Tensor], torch.Tensor],
                hamiltonian_fn: Callable[[Callable, torch.Tensor], torch.Tensor],
                electron_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算方差正则化的VMC损失
        
        Args:
            wave_fn: 波函数
            hamiltonian_fn: 哈密顿量函数
            electron_coords: 电子坐标
            
        Returns:
            (loss, aux_data): 损失值和辅助数据
        """
        # 计算基础VMC损失
        energy_loss, aux_data = self.vmc_loss(wave_fn, hamiltonian_fn, electron_coords)
        
        # 计算方差项
        local_energies = aux_data['local_energies']
        energy_var = torch.var(local_energies)
        
        # 总损失 = 能量期望 + λ * 方差
        total_loss = energy_loss + self.variance_weight * energy_var
        
        # 更新辅助数据
        aux_data['variance_loss'] = energy_var
        aux_data['total_loss'] = total_loss
        aux_data['energy_loss'] = energy_loss
        
        return total_loss, aux_data

def make_loss_fn(loss_type: str = 'vmc', **kwargs) -> nn.Module:
    """
    Loss function factory.
    
    Args:
        loss_type: 损失函数类型 ('vmc', 'reweighted', 'variance_regularized')
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    if loss_type == 'vmc':
        return VMCLoss(**kwargs)
    elif loss_type == 'reweighted':
        return ReweightedVMCLoss(**kwargs)
    elif loss_type == 'variance_regularized':
        return VarianceRegularizedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# 辅助函数
def compute_energy_stats(local_energies: torch.Tensor) -> Dict[str, float]:
    """计算能量统计量"""
    return {
        'mean': local_energies.mean().item(),
        'std': local_energies.std().item(),
        'var': local_energies.var().item(),
        'min': local_energies.min().item(),
        'max': local_energies.max().item(),
        'median': local_energies.median().item()
    } 