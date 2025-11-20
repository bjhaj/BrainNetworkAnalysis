"""
SVM Baseline for NeuroGraph HCPGender Dataset
Simple and effective baseline using graph-level features
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
from neurograph.datasets import NeuroGraphDataset
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf'])
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--features', type=str, default='mean', choices=['mean', 'flatten', 'adjacency', 'graph2vec'],
                    help='Feature extraction method: mean (average pooling), flatten (all node features), adjacency (upper triangular of correlation matrix), or graph2vec (pre-computed embeddings)')
parser.add_argument('--embedding-file', type=str, default=None,
                    help='Path to .npz file containing graph2vec embeddings (required if --features graph2vec)')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

print("="*80)
print(f"SVM Baseline on NeuroGraph {args.dataset} Dataset")
print(f"Kernel: {args.kernel}")
print(f"Features: {args.features}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Create results directory
results_dir = REPO_ROOT / 'outputs' / 'results'
results_dir.mkdir(parents=True, exist_ok=True)
results_dir = str(results_dir)

# Load dataset
print("\n[1] Loading dataset...")
dataset = NeuroGraphDataset(root=str(REPO_ROOT / 'data'), name=args.dataset)
print(f"    Dataset size: {len(dataset)}")
print(f"    Number of node features: {dataset.num_features}")
print(f"    Number of classes: {dataset.num_classes}")

# Extract labels
labels = np.array([data.y.item() for data in dataset])
print(f"    Class distribution: {np.bincount(labels)}")

# Extract graph-level features
print("\n[2] Extracting graph-level features...")
if args.features == 'graph2vec':
    if args.embedding_file is None:
        raise ValueError("--embedding-file must be specified when using --features graph2vec")
    print(f"    Loading pre-computed Graph2Vec embeddings from: {args.embedding_file}")
    data_npz = np.load(args.embedding_file)
    graph_features = data_npz['embeddings']
    loaded_labels = data_npz['labels']
    
    # Verify labels match
    if not np.array_equal(labels, loaded_labels):
        print("    WARNING: Labels in embedding file don't match dataset labels!")
    print(f"    Loaded embeddings shape: {graph_features.shape}")
elif args.features == 'mean':
    print("    Using mean-pooled node features as graph representation")
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
    print("    Using upper triangular of correlation matrix as graph representation")
    print("    This preserves functional connectivity patterns")
    graph_features = []
    for data in tqdm(dataset, desc="Extracting features"):
        # For HCP: data.x IS the correlation matrix
        corr_matrix = data.x  # Shape: [num_nodes, num_nodes]
        num_nodes = corr_matrix.shape[0]

        # Extract upper triangle (excluding diagonal) - contains all unique edges
        upper_tri_indices = np.triu_indices(num_nodes, k=1)
        features = corr_matrix[upper_tri_indices].cpu().numpy()
        graph_features.append(features)

    graph_features = np.array(graph_features)
    print(f"    Number of unique connections: {graph_features.shape[1]}")

print(f"    Feature vector dimension: {graph_features.shape[1]}")

# 10-fold cross-validation
print("\n[3] Running 10-fold cross-validation...")
print("="*80)

all_results = []

for run in range(args.runs):
    print(f"\n--- Run {run + 1}/{args.runs} ---")
    
    # Set seed for this run
    run_seed = args.seed + run
    np.random.seed(run_seed)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run_seed)
    
    fold_accs = []
    fold_aurocs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        # Split data
        X_train, X_test = graph_features[train_idx], graph_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        if args.kernel == 'linear':
            svm = SVC(kernel='linear', probability=True, random_state=run_seed)
        else:  # rbf
            svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=run_seed)
        
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = svm.predict(X_test_scaled)
        y_prob = svm.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auroc = roc_auc_score(y_test, y_prob)
        except:
            auroc = 0.5
        
        fold_accs.append(acc)
        fold_aurocs.append(auroc)
        
        all_results.append({
            'Run': run + 1,
            'Fold': fold + 1,
            'Accuracy': acc,
            'AUROC': auroc
        })
    
    # Print run summary
    print(f"  Run {run + 1} - Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}, "
          f"AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

# Save detailed results
df_detailed = pd.DataFrame(all_results)
df_detailed.to_csv(f'{results_dir}/svm_{args.kernel}_{args.features}_detailed.csv', index=False)

# Compute overall statistics
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nSVM ({args.kernel.upper()} kernel):")
print(f"  Mean Accuracy:  {df_detailed['Accuracy'].mean():.4f} ± {df_detailed['Accuracy'].std():.4f}")
print(f"  Mean AUROC:     {df_detailed['AUROC'].mean():.4f} ± {df_detailed['AUROC'].std():.4f}")

# Save summary
summary_data = {
    'Model': f'SVM_{args.kernel}_{args.features}',
    'Mean_Accuracy': df_detailed['Accuracy'].mean(),
    'Std_Accuracy': df_detailed['Accuracy'].std(),
    'Mean_AUROC': df_detailed['AUROC'].mean(),
    'Std_AUROC': df_detailed['AUROC'].std(),
    'Runs': args.runs,
    'Features': args.features,
    'Feature_Dim': graph_features.shape[1]
}
df_summary = pd.DataFrame([summary_data])
df_summary.to_csv(f'{results_dir}/svm_{args.kernel}_{args.features}_summary.csv', index=False)

print(f"\n[INFO] Results saved to {results_dir}/svm_{args.kernel}_{args.features}_*.csv")
print("\n" + "="*80)
print(f"Experiment complete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
