# CMU Brain Dataset with NeuroGraph GNN Architecture

## Overview

This repository adapts the **NeuroGraph GNN architecture** (originally designed for HCP datasets) to work with **CMU Brain connectivity data**. The adaptation is minimal—we keep the same ResidualGNNs architecture but create a wrapper to handle the different input format (70×70 adjacency matrices with rich node features).

## Dataset

**CMU Brain Dataset**
- **114 subjects** with structural brain connectivity graphs
- **70 nodes** per graph (brain regions)
- **76 node features** per node (derived from graph topology + position encoding)
- **Edge weights**: fiber tract counts (log-transformed)

### Tasks (Binary Classification)

1. **Sex Classification**: Male vs Female (50 F, 64 M)
2. **Math Ability**: High vs Low FSIQ (median split on Full-Scale IQ)
3. **Creativity**: High vs Low CAQ (median split on Creativity Achievement Questionnaire)

## Architecture

We use the **ResidualGNNs** architecture from NeuroGraph with minimal modifications:

```
Input: Graph with node features (70 nodes × 76 features)
  ↓
GNN Layers (3 layers, hidden_dim=32)
  - GCNConv / GATConv / SAGEConv / GINConv
  - Tanh activation
  ↓
Global Pooling (mean aggregation)
  ↓
MLP Classifier (64 → 32 → 32 → 2 classes)
  - BatchNorm + ReLU + Dropout
  ↓
Output: Binary classification
```

### Key Files

- **`cmu_brain_adapter.py`**: Dataset adapter that loads CMU .mat files and creates PyG Data objects
- **`cmu_model_wrapper.py`**: Model wrapper (CMUResidualGNNs) that adapts the architecture for CMU data
- **`main_cmu.py`**: Training script with 10-fold cross-validation
- **`data/CMUBrain/data_loader.py`**: Loads .mat files and computes node features
- **`data/CMUBrain/dataset.py`**: PyTorch Geometric dataset wrapper

## Usage

### Quick Start

```bash
# Test the dataset adapter
python3 cmu_brain_adapter.py

# Run training on sex classification task (quick test)
python3 main_cmu.py --task sex --model GCNConv --runs 2 --folds 3 --epochs 50

# Run full experiment (10 runs × 10 folds)
./run_cmu.sh
```

### Command-Line Arguments

```bash
python3 main_cmu.py \
    --task sex                # Task: sex, math, creativity
    --model GCNConv          # GNN: GCNConv, GATConv, SAGEConv, GINConv, TransformerConv
    --hidden 32              # GNN hidden dimension
    --hidden_mlp 64          # MLP hidden dimension
    --num_layers 3           # Number of GNN layers
    --epochs 200             # Training epochs
    --batch_size 16          # Batch size
    --lr 1e-4                # Learning rate
    --weight_decay 0.0005    # L2 regularization
    --dropout 0.5            # Dropout rate
    --runs 10                # Number of runs
    --folds 10               # K-fold cross-validation
    --device cuda            # Device: cuda or cpu
```

### Run All Experiments

```bash
# Run all tasks with all models
chmod +x run_cmu_full.sh
./run_cmu_full.sh
```

This will test:
- **4 GNN models**: GCN, GAT, GraphSAGE, GIN
- **3 tasks**: Sex, Math, Creativity
- **10 runs × 10 folds** for each combination

## Node Features

Each node (brain region) has **76 features**:

1. **Network statistics** (6 features):
   - In-strength, out-strength, total strength (z-scored)
   - Clustering coefficient
   - Betweenness centrality
   - Eigenvector centrality

2. **Position encoding** (70 features):
   - One-hot encoding of node ID (ROI identity)

This rich feature set allows the GNN to learn both topological patterns and region-specific information.

## Results Format

Results are saved to `results/cmu_<task>_results.csv`:

```csv
Run 1,0.650000,0.080000,0.720000,0.060000
Run 2,0.630000,0.090000,0.700000,0.070000
...
Final,0.640000,0.085000,0.710000,0.065000
```

Columns: `Run, Accuracy_Mean, Accuracy_Std, AUROC_Mean, AUROC_Std`

## Comparison with NeuroGraph

### Similarities
- ✅ Same ResidualGNNs architecture
- ✅ Same GNN layers (GCN, GAT, SAGE, GIN, etc.)
- ✅ Same training procedure (10-fold CV)
- ✅ Same evaluation metrics (Accuracy, AUROC)

### Differences
- **Data format**: CMU uses 70×70 adjacency matrices vs HCP's 100×100 correlation matrices
- **Node features**: CMU uses topology + position encoding vs HCP's correlation values
- **Dataset size**: 114 subjects vs HCP's 1,078 subjects
- **Model wrapper**: `CMUResidualGNNs` handles different input dimensions

## Dependencies

```bash
pip install torch torch_geometric scipy scikit-learn pandas networkx
```

## File Structure

```
NeuroGraph/
├── cmu_brain_adapter.py          # Dataset adapter
├── cmu_model_wrapper.py          # Model wrapper
├── main_cmu.py                   # Training script
├── run_cmu.sh                    # Run single experiment
├── run_cmu_full.sh               # Run all experiments
├── utils.py                      # Original NeuroGraph utils
├── data/
│   └── CMUBrain/
│       ├── data_loader.py        # Load .mat files
│       ├── dataset.py            # PyG dataset wrapper
│       ├── raw/
│       │   └── brainnetworks/
│       │       ├── metainfo.csv  # Subject metadata
│       │       └── smallgraphs/  # *.mat files (70×70 adjacency)
│       └── processed/            # Processed PyG graphs
└── results/                      # Experimental results
```

## Citation

If you use this code, please cite both the original NeuroGraph paper and acknowledge the CMU Brain dataset:

```bibtex
@article{said2023neurograph,
  title={NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics},
  author={Said, Anwar and Bayrak, Roza G and Derr, Tyler and Shabbir, Mudassir and Moyer, Daniel and Chang, Catie and Koutsoukos, Xenofon},
  journal={arXiv preprint arXiv:2306.06202},
  year={2023}
}
```

## Notes

- The CMU dataset is relatively small (114 subjects), so results may have higher variance than HCP
- We use binary classification (median splits) for all tasks to maintain consistency
- Edge weights are log-transformed to reduce the impact of outliers
- Early stopping is applied when validation accuracy plateaus for 50 epochs
