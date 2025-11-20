"""
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

Fast Weisfeiler-Lehman Hash SVM using PyTorch Geometric directly
No NetworkX conversion - works on edge_index tensors directly
"""

import argparse
import numpy as np
import torch
from collections import defaultdict, Counter
from neurograph.datasets import NeuroGraphDataset
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--wl-iterations', type=int, default=5)
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf'])
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

print("="*80)
print(f"Fast WL Hash + SVM on {args.dataset} (Direct PyG)")
print(f"WL iterations: {args.wl_iterations}")
print(f"SVM kernel: {args.kernel}, C={args.C}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

os.makedirs('results', exist_ok=True)

# Load dataset
print("\n[1] Loading dataset...")
dataset = NeuroGraphDataset(root='data/', name=args.dataset)
print(f"    Dataset size: {len(dataset)}")
labels = np.array([data.y.item() for data in dataset])
print(f"    Class distribution: {np.bincount(labels)}")


def compute_wl_features(data, num_iterations=5):
    """
    Compute WL hash features directly from PyG Data object
    """
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index.cpu().numpy()
    
    # Initialize node labels (discretize node features)
    node_labels = (data.x.mean(dim=1).cpu().numpy() * 10).astype(int)
    
    # Build adjacency list
    adj_list = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].append(dst)
    
    # WL iterations
    feature_vectors = []
    label_lookup = {}
    next_label = max(node_labels) + 1
    
    for iteration in range(num_iterations + 1):
        # Count label frequencies (graph fingerprint)
        label_counts = Counter(node_labels)
        feature_vectors.append(label_counts)
        
        if iteration == num_iterations:
            break
        
        # WL relabeling
        new_labels = np.zeros(num_nodes, dtype=int)
        for node in range(num_nodes):
            # Get sorted neighbor labels
            neighbors = adj_list[node]
            neighbor_labels = sorted([node_labels[n] for n in neighbors])
            
            # Create signature: (current_label, neighbor_labels_tuple)
            signature = (node_labels[node], tuple(neighbor_labels))
            
            # Assign new label
            if signature not in label_lookup:
                label_lookup[signature] = next_label
                next_label += 1
            new_labels[node] = label_lookup[signature]
        
        node_labels = new_labels
    
    return feature_vectors


def build_feature_matrix(dataset, num_iterations=5):
    """
    Build feature matrix for all graphs using WL hashing
    """
    print(f"\n[2] Computing WL hash features (h={num_iterations})...")
    
    all_features = []
    all_label_sets = set()
    
    # First pass: collect all features and unique labels
    for data in tqdm(dataset, desc="Computing WL hashes"):
        features = compute_wl_features(data, num_iterations)
        all_features.append(features)
        
        for counter in features:
            all_label_sets.update(counter.keys())
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_label_sets))}
    num_features = len(label_to_idx)
    
    print(f"    Total unique WL labels: {num_features}")
    
    # Second pass: build feature matrix
    feature_matrix = np.zeros((len(dataset), num_features))
    
    for graph_idx, features in enumerate(all_features):
        for counter in features:
            for label, count in counter.items():
                feature_matrix[graph_idx, label_to_idx[label]] += count
    
    # Normalize
    feature_matrix = normalize(feature_matrix, norm='l2')
    
    return feature_matrix


# Build features
feature_matrix = build_feature_matrix(dataset, args.wl_iterations)
print(f"    Feature matrix shape: {feature_matrix.shape}")

# Cross-validation
print("\n[3] Running 10-fold cross-validation...")
print("="*80)

all_results = []

for run in range(args.runs):
    print(f"\n--- Run {run + 1}/{args.runs} ---")
    
    run_seed = args.seed + run
    np.random.seed(run_seed)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run_seed)
    
    fold_accs = []
    fold_aurocs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        X_train = feature_matrix[train_idx]
        X_test = feature_matrix[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Train SVM
        svm = SVC(kernel=args.kernel, C=args.C, probability=True, random_state=run_seed)
        svm.fit(X_train, y_train)
        
        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)[:, 1]
        
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
    
    print(f"  Run {run + 1} - Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}, "
          f"AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

# Save results
df_detailed = pd.DataFrame(all_results)
df_detailed.to_csv(ff'{str(REPO_ROOT / "outputs" / "results")}/fast_wl_svm_h{args.wl_iterations}_detailed.csv', index=False)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nFast WL Hash (h={args.wl_iterations}) + SVM ({args.kernel}):")
print(f"  Mean Accuracy:  {df_detailed['Accuracy'].mean():.4f} ± {df_detailed['Accuracy'].std():.4f}")
print(f"  Mean AUROC:     {df_detailed['AUROC'].mean():.4f} ± {df_detailed['AUROC'].std():.4f}")

summary_data = {
    'Model': f'Fast_WL_h{args.wl_iterations}_SVM_{args.kernel}',
    'Mean_Accuracy': df_detailed['Accuracy'].mean(),
    'Std_Accuracy': df_detailed['Accuracy'].std(),
    'Mean_AUROC': df_detailed['AUROC'].mean(),
    'Std_AUROC': df_detailed['AUROC'].std(),
    'WL_Iterations': args.wl_iterations,
    'Num_Features': feature_matrix.shape[1],
    'Kernel': args.kernel,
    'C': args.C,
    'Runs': args.runs
}
df_summary = pd.DataFrame([summary_data])
df_summary.to_csv(ff'{str(REPO_ROOT / "outputs" / "results")}/fast_wl_svm_h{args.wl_iterations}_summary.csv', index=False)

print(f"\n[INFO] Results saved to results/fast_wl_svm_h{args.wl_iterations}_*.csv")
print("\n" + "="*80)
print(f"Complete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
