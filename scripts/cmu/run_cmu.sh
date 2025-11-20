#!/bin/bash

# Run CMU Brain experiments with NeuroGraph GNN architecture
# Minimal adaptation - uses existing ResidualGNNs from utils.py

echo "========================================================================"
echo "CMU Brain Classification with NeuroGraph GNN Architecture"
echo "========================================================================"
echo ""

# Configuration
TASK="sex"           # Task: sex, math, or creativity
MODEL="GCNConv"      # GNN model: GCNConv, GATConv, SAGEConv, GINConv, etc.
HIDDEN=32            # Hidden dimension
HIDDEN_MLP=64        # MLP hidden dimension
LAYERS=3             # Number of GNN layers
EPOCHS=100           # Training epochs
BATCH_SIZE=16        # Batch size
LR=1e-4              # Learning rate
DEVICE="cuda"        # Device: cuda or cpu
RUNS=10              # Number of runs
FOLDS=10             # K-fold cross-validation

echo "Configuration:"
echo "  Task:        $TASK"
echo "  Model:       $MODEL"
echo "  Hidden dims: $HIDDEN (GNN), $HIDDEN_MLP (MLP)"
echo "  Layers:      $LAYERS"
echo "  Epochs:      $EPOCHS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Learn rate:  $LR"
echo "  Runs:        $RUNS"
echo "  Folds:       $FOLDS"
echo ""
echo "------------------------------------------------------------------------"
echo ""

# Change to repository root
cd "$(dirname "$0")/../.."

python3 experiments/cmu/train_cmu.py \
    --task $TASK \
    --model $MODEL \
    --hidden $HIDDEN \
    --hidden_mlp $HIDDEN_MLP \
    --num_layers $LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --runs $RUNS \
    --folds $FOLDS

echo ""
echo "========================================================================"
echo "Experiment completed!"
echo "Results saved to: outputs/results/cmu_${TASK}_results.csv"
echo "========================================================================"
