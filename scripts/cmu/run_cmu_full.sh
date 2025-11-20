#!/bin/bash

# Full CMU Brain experiments with NeuroGraph GNN architecture
# Tests multiple models and tasks

echo "========================================================================"
echo "CMU Brain Full Experiments - NeuroGraph GNN Architecture"
echo "========================================================================"
echo ""

# Change to repository root
cd "$(dirname "$0")/../.."

# Create results directory
mkdir -p outputs/results

# Run experiments for each task
TASKS=("sex" "math" "creativity")
MODELS=("GCNConv" "GATConv" "SAGEConv" "GINConv")

EPOCHS=200
RUNS=10
FOLDS=10
BATCH_SIZE=16
LR=1e-4

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Task: ${TASK^^}"
    echo "========================================================================"
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "Running ${MODEL} on ${TASK}..."
        echo "------------------------------------------------------------------------"
        
        python3 experiments/cmu/train_cmu.py \
            --task $TASK \
            --model $MODEL \
            --hidden 32 \
            --hidden_mlp 64 \
            --num_layers 3 \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --device cuda \
            --runs $RUNS \
            --folds $FOLDS
        
        echo ""
    done
done

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "Results saved in: results/"
echo "========================================================================"
