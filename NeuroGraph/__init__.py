"""
NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics

A comprehensive framework for graph neural network experiments on brain connectivity data.
Includes datasets, models, and utilities for both HCP and CMU Brain datasets.
"""

__version__ = "2.0.0"

# Expose main classes
from neurograph.datasets import NeuroGraphDataset, CMUBrainDataset, NeuroGraphDynamic
from neurograph.models import ResidualGNNs, CMUResidualGNNs, CMUResidualGNNsSimple
from neurograph.utils import fix_seed
from neurograph.data import BrainDataLoader, BrainGraphDataset, create_data_splits

__all__ = [
    # Version
    '__version__',
    # Datasets
    'NeuroGraphDataset',
    'CMUBrainDataset',
    'NeuroGraphDynamic',
    # Models
    'ResidualGNNs',
    'CMUResidualGNNs',
    'CMUResidualGNNsSimple',
    # Utilities
    'fix_seed',
    # Data loaders
    'BrainDataLoader',
    'BrainGraphDataset',
    'create_data_splits',
]
