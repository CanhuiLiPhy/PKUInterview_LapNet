"""
PyTorch Neural Network Variational Monte Carlo (NNVMC) package.

This package provides implementations of neural network wave functions
for quantum Monte Carlo calculations using PyTorch.
"""

from .networks import NeuralWaveFunction, make_wave_function
from .mcmc import MetropolisHastings, mcmc_step
from .hamiltonian import LocalEnergyHamiltonian, local_energy
from .loss import VMCLoss, make_loss_fn
from .train import VMCTrainer, create_default_config

__version__ = "0.1.0"
__author__ = "PyTorch NNVMC Team"

__all__ = [
    'NeuralWaveFunction',
    'make_wave_function', 
    'MetropolisHastings',
    'mcmc_step',
    'LocalEnergyHamiltonian',
    'local_energy',
    'VMCLoss',
    'make_loss_fn',
    'VMCTrainer',
    'create_default_config'
] 