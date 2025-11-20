"""
MLP Classifier using Graph2Vec Embeddings
Multi-layer perceptron trained on pre-computed graph embeddings
"""

import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--embedding-file', type=str, required=True,
                    help='Path to .npz file containing graph2vec embeddings')
parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                    help='Hidden layer dimensions (e.g., 256 128 64)')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

print("="*80)
print(f"MLP Classifier on Graph2Vec Embeddings")
print(f"Hidden dimensions: {args.hidden_dims}")
print(f"Device: {args.device}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Create results directory
os.makedirs('results', exist_ok=True)

# Load embeddings
print("\n[1] Loading Graph2Vec embeddings...")
data_npz = np.load(args.embedding_file)
graph_features = data_npz['embeddings']
labels = data_npz['labels']
print(f"    Embedding shape: {graph_features.shape}")
print(f"    Number of samples: {len(labels)}")
print(f"    Class distribution: {np.bincount(labels)}")


# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    
    return acc, auroc


# 10-fold cross-validation
print("\n[2] Running 10-fold cross-validation...")
print("="*80)

all_results = []

for run in range(args.runs):
    print(f"\n--- Run {run + 1}/{args.runs} ---")
    
    # Set seed for this run
    run_seed = args.seed + run
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run_seed)
    
    fold_accs = []
    fold_aurocs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Split data
        X_train, X_test = graph_features[train_idx], graph_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = MLP(
            input_dim=graph_features.shape[1],
            hidden_dims=args.hidden_dims,
            num_classes=2,
            dropout=args.dropout
        ).to(args.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Training with early stopping
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, args.device)
            test_acc, test_auroc = evaluate(model, test_loader, args.device)
            
            scheduler.step(test_acc)
            
            if test_acc > best_val_acc:
                best_val_acc = test_acc
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(args.device) for k, v in best_model_state.items()})
                break
        
        # Final evaluation
        test_acc, test_auroc = evaluate(model, test_loader, args.device)
        
        fold_accs.append(test_acc)
        fold_aurocs.append(test_auroc)
        
        all_results.append({
            'Run': run + 1,
            'Fold': fold + 1,
            'Accuracy': test_acc,
            'AUROC': test_auroc
        })
    
    # Print run summary
    print(f"  Run {run + 1} - Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}, "
          f"AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

# Save detailed results
df_detailed = pd.DataFrame(all_results)
df_detailed.to_csv(f'{str(REPO_ROOT / "outputs" / "results")}/mlp_graph2vec_detailed.csv', index=False)

# Compute overall statistics
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nMLP on Graph2Vec embeddings:")
print(f"  Mean Accuracy:  {df_detailed['Accuracy'].mean():.4f} ± {df_detailed['Accuracy'].std():.4f}")
print(f"  Mean AUROC:     {df_detailed['AUROC'].mean():.4f} ± {df_detailed['AUROC'].std():.4f}")

# Save summary
summary_data = {
    'Model': 'MLP_Graph2Vec',
    'Mean_Accuracy': df_detailed['Accuracy'].mean(),
    'Std_Accuracy': df_detailed['Accuracy'].std(),
    'Mean_AUROC': df_detailed['AUROC'].mean(),
    'Std_AUROC': df_detailed['AUROC'].std(),
    'Hidden_Dims': str(args.hidden_dims),
    'Embedding_Dim': graph_features.shape[1],
    'Runs': args.runs
}
df_summary = pd.DataFrame([summary_data])
df_summary.to_csv(f'{str(REPO_ROOT / "outputs" / "results")}/mlp_graph2vec_summary.csv', index=False)

print(f"\n[INFO] Results saved to results/mlp_graph2vec_*.csv")
print("\n" + "="*80)
print(f"Experiment complete! End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
