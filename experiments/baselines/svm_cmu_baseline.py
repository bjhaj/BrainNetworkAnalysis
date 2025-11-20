"""
SVM Baseline for CMU Brain Dataset
Simple and effective baseline using graph-level features
Parallel to HCP SVM baseline for comparison across datasets
"""

import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from neurograph.datasets import CMUBrainDataset
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sex', choices=['sex', 'math', 'creativity'],
                    help='Classification task')
parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf'])
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--features', type=str, default='adjacency', choices=['mean', 'flatten', 'adjacency'],
                    help='Feature extraction method: mean (average pooling), flatten (all node features), or adjacency (upper triangular of connectivity matrix)')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

print("="*80)
print(f"SVM Baseline on CMU Brain Dataset - Task: {args.task.upper()}")
print(f"Kernel: {args.kernel}")
print(f"Features: {args.features}")
print(f"Runs: {args.runs}, Folds: {args.folds}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Create results directory
results_dir = REPO_ROOT / 'outputs' / 'results'
results_dir.mkdir(parents=True, exist_ok=True)
results_dir = str(results_dir)

# Load dataset
print("\n[1] Loading CMU Brain dataset...")
dataset = CMUBrainDataset(root=str(REPO_ROOT / 'data' / 'CMUBrain'), task=args.task)
print(f"    Dataset size: {len(dataset)}")
print(f"    Number of node features: {dataset.num_features}")
print(f"    Number of classes: {dataset.num_classes}")

# Extract labels
labels = np.array([data.y.item() for data in dataset])
print(f"    Class distribution: {np.bincount(labels)}")
print(f"    Class 0: {np.sum(labels == 0)} samples")
print(f"    Class 1: {np.sum(labels == 1)} samples")

# Extract graph-level features
print("\n[2] Extracting graph-level features...")
if args.features == 'mean':
    print("    Using mean-pooled node features as graph representation")
    print("    WARNING: This loses graph structure - use 'adjacency' for better performance!")
    graph_features = []
    for data in tqdm(dataset, desc="Extracting features"):
        features = data.x.mean(dim=0).cpu().numpy()
        graph_features.append(features)
    graph_features = np.array(graph_features)
elif args.features == 'flatten':
    print("    Using flattened node features (all nodes) as graph representation")
    graph_features = []
    for data in tqdm(dataset, desc="Extracting features"):
        features = data.x.flatten().cpu().numpy()
        graph_features.append(features)
    graph_features = np.array(graph_features)
elif args.features == 'adjacency':
    print("    Using upper triangular of adjacency matrix as graph representation")
    print("    This preserves the connectivity pattern (recommended for brain graphs)")
    graph_features = []
    for data in tqdm(dataset, desc="Extracting features"):
        # Reconstruct adjacency matrix from edge_index and edge_weight
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        # Fill adjacency matrix with edge weights
        adj_matrix[edge_index[0], edge_index[1]] = edge_weight

        # Extract upper triangular (excluding diagonal) - contains all unique edges
        upper_tri_indices = np.triu_indices(num_nodes, k=1)
        features = adj_matrix[upper_tri_indices].cpu().numpy()
        graph_features.append(features)
    graph_features = np.array(graph_features)
    print(f"    Number of edges (unique connections): {graph_features.shape[1]}")

print(f"    Feature vector dimension: {graph_features.shape[1]}")

# K-fold cross-validation with multiple runs
print(f"\n[3] Running {args.folds}-fold cross-validation with {args.runs} runs...")
print("="*80)

all_results = []

for run in range(args.runs):
    print(f"\nRun {run + 1}/{args.runs}")
    print("-" * 80)

    # Set seed for this run
    run_seed = args.seed + run
    np.random.seed(run_seed)

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=run_seed)

    fold_accs = []
    fold_aurocs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(graph_features, labels)):
        # Split data
        X_train, X_test = graph_features[train_idx], graph_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM
        svm = SVC(kernel=args.kernel, probability=True, random_state=run_seed)
        svm.fit(X_train, y_train)

        # Predict
        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)

        fold_accs.append(acc)
        fold_aurocs.append(auroc)

        # Store results
        all_results.append({
            'Run': run + 1,
            'Fold': fold + 1,
            'Accuracy': acc,
            'AUROC': auroc,
            'Task': args.task,
            'Kernel': args.kernel,
            'Features': args.features,
            'Train_Size': len(train_idx),
            'Test_Size': len(test_idx)
        })

    # Print run summary
    print(f"  Run {run + 1} - Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}, "
          f"AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

# Save detailed results
df_detailed = pd.DataFrame(all_results)
df_detailed.to_csv(f'{results_dir}/svm_cmu_{args.task}_{args.kernel}_{args.features}_detailed.csv', index=False)

# Compute overall statistics
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nSVM ({args.kernel} kernel) on CMU Brain - {args.task.upper()} task:")
print(f"  Mean Accuracy:  {df_detailed['Accuracy'].mean():.4f} ± {df_detailed['Accuracy'].std():.4f}")
print(f"  Mean AUROC:     {df_detailed['AUROC'].mean():.4f} ± {df_detailed['AUROC'].std():.4f}")
print(f"  Best Accuracy:  {df_detailed['Accuracy'].max():.4f}")
print(f"  Worst Accuracy: {df_detailed['Accuracy'].min():.4f}")
print(f"  Best AUROC:     {df_detailed['AUROC'].max():.4f}")
print(f"  Worst AUROC:    {df_detailed['AUROC'].min():.4f}")

# Create summary
summary_data = {
    'Task': args.task,
    'Kernel': args.kernel,
    'Features': args.features,
    'Runs': args.runs,
    'Folds': args.folds,
    'Mean_Accuracy': df_detailed['Accuracy'].mean(),
    'Std_Accuracy': df_detailed['Accuracy'].std(),
    'Mean_AUROC': df_detailed['AUROC'].mean(),
    'Std_AUROC': df_detailed['AUROC'].std(),
    'Best_Accuracy': df_detailed['Accuracy'].max(),
    'Worst_Accuracy': df_detailed['Accuracy'].min(),
    'Feature_Dim': graph_features.shape[1],
    'Dataset_Size': len(dataset)
}
df_summary = pd.DataFrame([summary_data])
df_summary.to_csv(f'{results_dir}/svm_cmu_{args.task}_{args.kernel}_{args.features}_summary.csv', index=False)

print(f"\n[INFO] Results saved to {results_dir}/svm_cmu_{args.task}_{args.kernel}_{args.features}_*.csv")
print("\n" + "="*80)
print(f"Experiment complete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
