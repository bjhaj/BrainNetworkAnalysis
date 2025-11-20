"""
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

Weisfeiler-Lehman Kernel + SVM with FULL Node Features
Uses actual node feature vectors instead of discretized labels
Should perform much better than standard WL kernel
"""

import argparse
import numpy as np
import torch
import networkx as nx
from neurograph.datasets import NeuroGraphDataset
from grakel import GraphKernel
from grakel.utils import graph_from_networkx
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os
import hashlib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--wl-iterations', type=int, default=5)
parser.add_argument('--normalize-kernel', action='store_true')
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--feature-bins', type=int, default=100, help='Number of bins for feature discretization')
args = parser.parse_args()

print("="*80)
print(f"WL Kernel + SVM with FULL Node Features on {args.dataset}")
print(f"WL iterations: {args.wl_iterations}, Feature bins: {args.feature_bins}")
print(f"SVM C={args.C}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

os.makedirs('results', exist_ok=True)

# Load dataset
print("\n[1] Loading dataset...")
dataset = NeuroGraphDataset(root='data/', name=args.dataset)
print(f"    Dataset size: {len(dataset)}")
labels = np.array([data.y.item() for data in dataset])
print(f"    Class distribution: {np.bincount(labels)}")
print(f"    Node feature dim: {dataset[0].x.shape[1]}")

# Convert to NetworkX format with FULL node features
print("\n[2] Converting graphs with full node features...")
nx_graphs = []

# Collect all node features to compute global statistics
all_node_features = []
for data in dataset:
    all_node_features.append(data.x.cpu().numpy())
all_node_features = np.vstack(all_node_features)

# Compute quantiles for each feature dimension
feature_quantiles = []
for dim in range(all_node_features.shape[1]):
    quantiles = np.percentile(all_node_features[:, dim], 
                              np.linspace(0, 100, args.feature_bins + 1))
    feature_quantiles.append(quantiles)

print(f"    Computed {args.feature_bins} quantile bins per feature dimension")

def hash_feature_vector(feature_vec, quantiles_list):
    """Hash high-dimensional feature vector to discrete label using quantiles"""
    discretized = []
    for val, quantiles in zip(feature_vec, quantiles_list):
        bin_idx = np.searchsorted(quantiles, val, side='right') - 1
        bin_idx = max(0, min(bin_idx, len(quantiles) - 2))
        discretized.append(bin_idx)
    
    # Create hash from discretized vector
    hash_str = ','.join(map(str, discretized))
    hash_val = int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)
    return hash_val

for data in tqdm(dataset, desc="Converting"):
    edge_index = data.edge_index.cpu().numpy()
    
    # Create undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    
    # Add node labels using FULL feature vectors (hashed)
    for node in range(data.x.shape[0]):
        node_feature_vec = data.x[node].cpu().numpy()
        # Hash the full feature vector to a discrete label
        node_label = hash_feature_vector(node_feature_vec, feature_quantiles)
        G.nodes[node]['label'] = node_label
    
    # Add edges
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    nx_graphs.append(G)

# Convert to GraKeL format
print("\n[3] Converting to GraKeL format...")
grakel_graphs = []
for G in tqdm(nx_graphs, desc="Converting to GraKeL"):
    grakel_graph = list(graph_from_networkx([G], node_labels_tag='label'))[0]
    grakel_graphs.append(grakel_graph)

# Compute WL kernel matrix
print(f"\n[4] Computing WL kernel matrix (h={args.wl_iterations})...")
print("    This may take several minutes...")

wl_kernel = GraphKernel(
    kernel=[{"name": "weisfeiler_lehman", "n_iter": args.wl_iterations},
            {"name": "vertex_histogram"}],
    normalize=args.normalize_kernel
)

kernel_matrix = wl_kernel.fit_transform(grakel_graphs)
print(f"    Kernel matrix shape: {kernel_matrix.shape}")
print(f"    Kernel matrix stats: mean={kernel_matrix.mean():.4f}, std={kernel_matrix.std():.4f}")

# Cross-validation
print("\n[5] Running 10-fold cross-validation...")
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
        # Split kernel matrix
        K_train = kernel_matrix[np.ix_(train_idx, train_idx)]
        K_test = kernel_matrix[np.ix_(test_idx, train_idx)]
        
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Train SVM with precomputed kernel
        svm = SVC(kernel='precomputed', C=args.C, probability=True, random_state=run_seed)
        svm.fit(K_train, y_train)
        y_pred = svm.predict(K_test)
        y_prob = svm.predict_proba(K_test)[:, 1]
        
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
df_detailed.to_csv(ff'{str(REPO_ROOT / "outputs" / "results")}/wl_kernel_features_svm_h{args.wl_iterations}_detailed.csv', index=False)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nWL Kernel with FULL Features (h={args.wl_iterations}) + SVM:")
print(f"  Mean Accuracy:  {df_detailed['Accuracy'].mean():.4f} ± {df_detailed['Accuracy'].std():.4f}")
print(f"  Mean AUROC:     {df_detailed['AUROC'].mean():.4f} ± {df_detailed['AUROC'].std():.4f}")

summary_data = {
    'Model': f'WL_Features_h{args.wl_iterations}_SVM',
    'Mean_Accuracy': df_detailed['Accuracy'].mean(),
    'Std_Accuracy': df_detailed['Accuracy'].std(),
    'Mean_AUROC': df_detailed['AUROC'].mean(),
    'Std_AUROC': df_detailed['AUROC'].std(),
    'WL_Iterations': args.wl_iterations,
    'Feature_Bins': args.feature_bins,
    'C': args.C,
    'Runs': args.runs
}
df_summary = pd.DataFrame([summary_data])
df_summary.to_csv(ff'{str(REPO_ROOT / "outputs" / "results")}/wl_kernel_features_svm_h{args.wl_iterations}_summary.csv', index=False)

print(f"\n[INFO] Results saved to results/wl_kernel_features_svm_h{args.wl_iterations}_*.csv")
print("\n" + "="*80)
print(f"Complete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
