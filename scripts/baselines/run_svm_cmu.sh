#!/bin/bash

# Run SVM baseline on CMU Brain Dataset
# Parallel to HCP SVM baseline for comparison

echo "========================================================================"
echo "SVM Baseline on CMU Brain Dataset"
echo "========================================================================"
echo ""

# Configuration
TASK="sex"           # Task: sex, math, or creativity
KERNEL="rbf"         # Kernel: rbf or linear
FEATURES="adjacency" # Features: adjacency (recommended), mean, or flatten
RUNS=10              # Number of runs
FOLDS=10             # K-fold CV

echo "Configuration:"
echo "  Task:        $TASK"
echo "  Kernel:      $KERNEL"
echo "  Features:    $FEATURES"
echo "  Runs:        $RUNS"
echo "  Folds:       $FOLDS"
echo ""
echo "------------------------------------------------------------------------"
echo ""

# Change to repository root
cd "$(dirname "$0")/../.."

python3 experiments/baselines/svm_cmu_baseline.py \
    --task $TASK \
    --kernel $KERNEL \
    --features $FEATURES \
    --runs $RUNS \
    --folds $FOLDS

echo ""
echo "========================================================================"
echo "Experiment completed!"
echo "Results saved to: outputs/results/svm_cmu_${TASK}_${KERNEL}_${FEATURES}_*.csv"
echo "========================================================================"
