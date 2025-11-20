"""
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

Graph2Vec Embedding Generator for NeuroGraph datasets
Pre-computes graph embeddings that can be used with SVM or other classifiers
Uses karateclub's Graph2Vec implementation
"""

import argparse
import numpy as np
import torch
import networkx as nx
from neurograph.datasets import NeuroGraphDataset
from karateclub import Graph2Vec
from tqdm import tqdm
import pickle
import os
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--dimensions', type=int, default=128, help='Embedding dimension')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--output', type=str, default='embeddings', help='Output directory')
args = parser.parse_args()

print("="*80)
print(f"Graph2Vec Embedding Generation for {args.dataset}")
print(f"Embedding dimensions: {args.dimensions}")
print(f"Training epochs: {args.epochs}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Create output directory
os.makedirs(args.output, exist_ok=True)

# Load dataset
print("\n[1] Loading dataset...")
dataset = NeuroGraphDataset(root='data/', name=args.dataset)
print(f"    Dataset size: {len(dataset)}")
print(f"    Number of node features: {dataset.num_features}")
print(f"    Number of classes: {dataset.num_classes}")

# Extract labels
labels = np.array([data.y.item() for data in dataset])
print(f"    Class distribution: {np.bincount(labels)}")

# Convert PyG graphs to NetworkX format
print("\n[2] Converting graphs to NetworkX format...")
nx_graphs = []
for data in tqdm(dataset, desc="Converting graphs"):
    edge_index = data.edge_index.cpu().numpy()
    
    # Create undirected NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0]))
    
    # Add edges
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    nx_graphs.append(G)

print(f"    Converted {len(nx_graphs)} graphs")

# Train Graph2Vec
print("\n[3] Training Graph2Vec model...")
print("    This may take several minutes...")
model = Graph2Vec(
    dimensions=args.dimensions,
    epochs=args.epochs,
    learning_rate=0.025,
    min_count=5,
    wl_iterations=2,
    workers=1  # karateclub doesn't support multiprocessing well
)

try:
    model.fit(nx_graphs)
    print("    Training complete!")
except Exception as e:
    print(f"    Error during training: {e}")
    print("    This might be due to graph structure issues. Trying with simpler parameters...")
    model = Graph2Vec(
        dimensions=args.dimensions,
        epochs=args.epochs // 2,
        learning_rate=0.05,
        min_count=1,
        wl_iterations=2,
        workers=1
    )
    model.fit(nx_graphs)
    print("    Training complete with adjusted parameters!")

# Extract embeddings
print("\n[4] Extracting embeddings...")
embeddings = model.get_embedding()
print(f"    Embedding shape: {embeddings.shape}")

# Save embeddings and labels
output_path = os.path.join(args.output, f'{args.dataset}_graph2vec_{args.dimensions}d.npz')
np.savez(output_path, 
         embeddings=embeddings, 
         labels=labels,
         dimensions=args.dimensions,
         epochs=args.epochs)

print(f"\n[5] Saved embeddings to: {output_path}")

# Save the model
model_path = os.path.join(args.output, f'{args.dataset}_graph2vec_{args.dimensions}d_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"    Saved model to: {model_path}")

# Print statistics
print("\n" + "="*80)
print("Embedding Statistics:")
print(f"  Shape: {embeddings.shape}")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std:  {embeddings.std():.4f}")
print(f"  Min:  {embeddings.min():.4f}")
print(f"  Max:  {embeddings.max():.4f}")
print("="*80)

print(f"\nComplete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nTo use these embeddings with SVM, run:")
print(f"  python svm_baseline.py --features graph2vec --embedding-file {output_path}")
print("="*80)
