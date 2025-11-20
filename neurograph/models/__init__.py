"""
NeuroGraph Models Module
Provides GNN architectures for brain connectomics
"""

from .residual_gnn import ResidualGNNs
from .cmu_residual_gnn import CMUResidualGNNs, CMUResidualGNNsSimple

__all__ = ['ResidualGNNs', 'CMUResidualGNNs', 'CMUResidualGNNsSimple']
