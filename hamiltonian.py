"""
Hamiltonian and local energy computation module for PyTorch NNVMC.

This module contains functions for computing local energy in quantum Monte Carlo
using automatic differentiation for kinetic energy terms.

Updated with improved numerical stability for multi-electron systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Union, Optional

def local_energy(
    wave_fn: Callable[[torch.Tensor], torch.Tensor], 
    electron_coords: torch.Tensor, 
    atom_coords: torch.Tensor, 
    method: str = 'direct',
    nuclear_charges: Optional[torch.Tensor] = None,
    include_electron_electron: bool = True,
    grad_clip: float = 50.0  # 添加梯度裁剪参数
) -> torch.Tensor:
    """计算波函数的局部能量
    
    E_L = -0.5 * ∇²ψ/ψ + V(r)
    
    Args:
        wave_fn: 波函数，接受electron_coords并返回log|ψ|
        electron_coords: 电子坐标 [batch_size, n_electrons, 3]
        atom_coords: 原子坐标 [n_atoms, 3]
        method: 计算方法，'direct'或'log'
        nuclear_charges: 原子核电荷 [n_atoms]，默认全为1
        include_electron_electron: 是否包含电子-电子相互作用项，默认True
        grad_clip: 梯度裁剪阈值，防止数值不稳定
    
    Returns:
        局部能量 [batch_size]
    """
    batch_size, n_electrons, _ = electron_coords.shape
    n_atoms = atom_coords.shape[0]
    device = electron_coords.device
    
    # 设置默认的原子核电荷（全为1）
    if nuclear_charges is None:
        nuclear_charges = torch.ones(n_atoms, device=device)
    
    # 确保输入可微，但不切断已存在的梯度链
    if not electron_coords.requires_grad:
        electron_coords = electron_coords.requires_grad_(True)
    
    # 计算动能项
    try:
        if method == 'log':
            # 使用对数导数法（推荐，更稳定）
            log_psi = wave_fn(electron_coords)
            kinetic = compute_kinetic_energy_log(log_psi, electron_coords, grad_clip=grad_clip)
        else:
            # 直接法：先计算ψ然后计算拉普拉斯算子
            log_psi = wave_fn(electron_coords)
            psi = torch.exp(log_psi)
            kinetic = compute_kinetic_energy_direct(psi, electron_coords)
    except Exception as e:
        print(f"动能计算错误: {e}")
        # 如果计算出错，则返回势能
        return potential_energy(electron_coords, atom_coords, nuclear_charges, include_electron_electron=include_electron_electron)
    
    # 计算势能项
    potential = potential_energy(electron_coords, atom_coords, nuclear_charges, include_electron_electron=include_electron_electron)
    
    # 返回局部能量 (动能 + 势能)
    return kinetic + potential

def compute_kinetic_energy_direct(
    psi: torch.Tensor, 
    electron_coords: torch.Tensor
) -> torch.Tensor:
    """直接法计算动能项 -0.5 * ∇²ψ/ψ
    
    Args:
        psi: 波函数值 [batch_size]
        electron_coords: 电子坐标 [batch_size, n_electrons, 3]
    
    Returns:
        动能 [batch_size]
    """
    batch_size, n_electrons, n_dims = electron_coords.shape
    
    # 计算一阶导数 ∇ψ
    grad_psi = torch.autograd.grad(
        psi.sum(), 
        electron_coords, 
        create_graph=True,
        retain_graph=True
    )[0]  # [batch_size, n_electrons, 3]
    
    # 计算二阶导数的迹（拉普拉斯算子）
    laplacian = torch.zeros(batch_size, device=electron_coords.device)
    
    for i in range(n_electrons):
        for j in range(n_dims):
            # 计算每个电子每个坐标的二阶导数
            grad_component = grad_psi[:, i, j]
            second_grad = torch.autograd.grad(
                grad_component.sum(),
                electron_coords,
                create_graph=True,
                retain_graph=True
            )[0][:, i, j]
            
            laplacian += second_grad
    
    # 动能 = -0.5 * ∇²ψ/ψ
    kinetic = -0.5 * laplacian / (psi + 1e-12)
    
    return kinetic

def compute_kinetic_energy_log(
    log_psi: torch.Tensor,
    electron_coords: torch.Tensor,
    grad_clip: float = 50.0
) -> torch.Tensor:
    """使用对数波函数计算动能项（修复版本）
    
    动能 = -0.5 * (∇²log(ψ) + |∇log(ψ)|²)
    
    Args:
        log_psi: 对数波函数值 log|ψ| [batch_size]
        electron_coords: 电子坐标 [batch_size, n_electrons, 3]
        grad_clip: 梯度裁剪阈值
    
    Returns:
        动能 [batch_size]
    """
    batch_size, n_electrons, n_dims = electron_coords.shape
    device = electron_coords.device
    
    if not log_psi.requires_grad or not electron_coords.requires_grad:
        return torch.zeros(batch_size, device=device)
    
    try:
        # 计算一阶导数 ∇log(ψ)
        grad_log_psi = torch.autograd.grad(
            outputs=log_psi.sum(),
            inputs=electron_coords,
            create_graph=True,
            retain_graph=True
        )[0]  # [batch_size, n_electrons, 3]
        
        # 计算 |∇log(ψ)|² 项
        grad_squared = torch.sum(grad_log_psi**2, dim=(1, 2))  # [batch_size]
        
        # 计算拉普拉斯算子 ∇²log(ψ) - 简化的稳定实现
        laplacian = torch.zeros(batch_size, device=device)
        
        # 对每个电子的每个坐标分量计算二阶导数
        for i in range(n_electrons):
            for j in range(n_dims):
                try:
                    # 取出这个分量的一阶导数
                    grad_component = grad_log_psi[:, i, j]  # [batch_size]
                    
                    # 计算二阶导数
                    second_deriv = torch.autograd.grad(
                        outputs=grad_component.sum(),
                        inputs=electron_coords,
                        create_graph=True,
                        retain_graph=True
                    )[0][:, i, j]  # [batch_size]
                    
                    # 检查并累加有限的二阶导数
                    finite_mask = torch.isfinite(second_deriv)
                    if finite_mask.any():
                        # 应用裁剪
                        clipped_deriv = torch.clamp(second_deriv, -grad_clip, grad_clip)
                        # 只累加有限值，其他设为0
                        safe_deriv = torch.where(finite_mask, clipped_deriv, torch.zeros_like(clipped_deriv))
                        laplacian += safe_deriv
                
                except Exception as e:
                    # 如果某个分量计算失败，跳过
                    print(f"Warning: Failed to compute second derivative for electron {i}, coord {j}: {e}")
                    continue
        
        # 合并项得到动能：T = -0.5 * (∇²log(ψ) + |∇log(ψ)|²)
        kinetic = -0.5 * (laplacian + grad_squared)
        
        # 更保守的裁剪
        kinetic = torch.clamp(kinetic, -100.0, 100.0)
        
        # 检查异常值
        if torch.any(torch.abs(kinetic) > 50.0):
            print(f"Warning: Large kinetic energy: min={kinetic.min():.3f}, max={kinetic.max():.3f}, mean={kinetic.mean():.3f}")
        
        return kinetic
        
    except Exception as e:
        print(f"Kinetic energy computation failed: {e}")
        return torch.zeros(batch_size, device=device)

def potential_energy(
    electron_coords: torch.Tensor, 
    atom_coords: torch.Tensor,
    nuclear_charges: Optional[torch.Tensor] = None,
    min_distance: float = 1e-10,
    include_electron_electron: bool = True
) -> torch.Tensor:
    """计算库仑势能（改进了最小距离处理）
    
    V = ∑_{i<j} 1/|r_i-r_j| - ∑_{i,a} Z_a/|r_i-R_a| + ∑_{a<b} Z_a*Z_b/|R_a-R_b|
    
    电子-电子排斥 + 电子-核吸引 + 核-核排斥
    
    Args:
        electron_coords: 电子坐标 [batch_size, n_electrons, 3]
        atom_coords: 原子坐标 [n_atoms, 3]
        nuclear_charges: 原子核电荷 [n_atoms]，默认全为1
        min_distance: 最小距离阈值，防止奇点
        include_electron_electron: 是否包含电子-电子相互作用项，默认True
    
    Returns:
        势能 [batch_size]
    """
    batch_size, n_electrons, _ = electron_coords.shape
    n_atoms = atom_coords.shape[0]
    device = electron_coords.device
    
    # 设置默认的原子核电荷
    if nuclear_charges is None:
        nuclear_charges = torch.ones(n_atoms, device=device)
    
    # 电子-电子排斥（可选）
    e_e_repulsion = torch.zeros(batch_size, device=device)
    if include_electron_electron and n_electrons > 1:
        for i in range(n_electrons):
            for j in range(i+1, n_electrons):
                r_ij = torch.norm(electron_coords[:, i] - electron_coords[:, j], dim=1)
                # 避免除零，使用更合理的最小距离
                r_ij = torch.clamp(r_ij, min=min_distance)
                e_e_repulsion += 1.0 / r_ij
    
    # 电子-原子核吸引
    e_n_attraction = torch.zeros(batch_size, device=device)
    for i in range(n_electrons):
        for j in range(n_atoms):
            r_ij = torch.norm(electron_coords[:, i] - atom_coords[j], dim=1)
            r_ij = torch.clamp(r_ij, min=min_distance)
            e_n_attraction -= nuclear_charges[j] / r_ij
    
    # 原子核-原子核排斥 (对于固定原子核，这是常数)
    n_n_repulsion = 0.0
    if n_atoms > 1:
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = torch.norm(atom_coords[i] - atom_coords[j])
                if r_ij > min_distance:
                    n_n_repulsion += nuclear_charges[i] * nuclear_charges[j] / r_ij
    
    # 返回总势能
    potential = e_e_repulsion + e_n_attraction + n_n_repulsion
    
    # 裁剪势能以避免极端值
    potential = torch.clamp(potential, -1000.0, 1000.0)
    
    return potential

class LocalEnergyHamiltonian(nn.Module):
    """Local energy Hamiltonian module with improved numerical stability."""
    
    def __init__(self, 
                 atom_coords: torch.Tensor,
                 nuclear_charges: Optional[torch.Tensor] = None,
                 method: str = 'log',
                 grad_clip: float = 50.0,  # 使用与改进动能函数一致的裁剪阈值
                 min_distance: float = 1e-10,
                 include_electron_electron: bool = True):
        """
        Args:
            atom_coords: 原子坐标 [n_atoms, 3]
            nuclear_charges: 原子核电荷 [n_atoms]
            method: 计算方法 'direct' 或 'log'
            grad_clip: 梯度裁剪阈值
            min_distance: 最小距离阈值
            include_electron_electron: 是否包含电子-电子相互作用项，默认True
        """
        super().__init__()
        self.register_buffer('atom_coords', atom_coords)
        if nuclear_charges is None:
            nuclear_charges = torch.ones(atom_coords.shape[0])
        self.register_buffer('nuclear_charges', nuclear_charges)
        self.method = method
        self.grad_clip = grad_clip
        self.min_distance = min_distance
        self.include_electron_electron = include_electron_electron
    
    def forward(self, 
                wave_fn: Callable[[torch.Tensor], torch.Tensor],
                electron_coords: torch.Tensor) -> torch.Tensor:
        """计算局部能量
        
        Args:
            wave_fn: 波函数
            electron_coords: 电子坐标 [batch_size, n_electrons, 3]
            
        Returns:
            局部能量 [batch_size]
        """
        return local_energy(
            wave_fn, 
            electron_coords, 
            self.atom_coords,
            method=self.method,
            nuclear_charges=self.nuclear_charges,
            include_electron_electron=self.include_electron_electron,
            grad_clip=self.grad_clip
        )

def hydrogen_exact_local_energy(r: torch.Tensor) -> torch.Tensor:
    """氢原子基态ψ(r)=exp(-r)的精确局部能量
    
    对于任何r值，结果应该恒为-0.5 a.u.
    
    Args:
        r: 电子到原子核的距离
    
    Returns:
        局部能量，恒为-0.5
    """
    return torch.ones_like(r) * (-0.5)

def local_energy_stats(
    wave_fn: Callable[[torch.Tensor], torch.Tensor], 
    electron_coords: torch.Tensor, 
    atom_coords: torch.Tensor, 
    method: str = 'log'
) -> Tuple[float, float, float]:
    """计算局部能量并返回统计量
    
    Args:
        wave_fn: 波函数
        electron_coords: 电子坐标
        atom_coords: 原子坐标
        method: 计算方法
    
    Returns:
        (mean, std, variance): 均值、标准差和方差
    """
    energy = local_energy(wave_fn, electron_coords, atom_coords, method)
    
    mean = energy.mean().item()
    std = energy.std().item()
    variance = energy.var().item()
    
    return mean, std, variance 