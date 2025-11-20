"""
Training script for CMU Brain Dataset using NeuroGraph GNN Architecture
Minimal adaptation of main.py to work with CMU data
"""
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neurograph.datasets import CMUBrainDataset
from neurograph.models import CMUResidualGNNs
from neurograph.utils import fix_seed
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import os
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv, ChebConv, SGConv, GraphConv

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sex', 
                    choices=['sex', 'math', 'creativity'],
                    help='Classification task')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--hidden_mlp', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

# Use paths relative to repository root
path = REPO_ROOT / "outputs" / "checkpoints"
res_path = REPO_ROOT / "outputs" / "results"
root = REPO_ROOT / "data" / "CMUBrain"

# Create directories if they don't exist
path.mkdir(parents=True, exist_ok=True)
res_path.mkdir(parents=True, exist_ok=True)

# Convert to strings for compatibility
path = str(path) + "/"
res_path = str(res_path) + "/"
root = str(root)

def logger(info):
    """Log results to file"""
    f = open(os.path.join(res_path, f'cmu_{args.task}_results.csv'), 'a')
    print(info, file=f)
    f.close()

# Fix random seed
fix_seed(args.seed)

# Load dataset
print(f"\n{'='*80}")
print(f"CMU Brain Classification - Task: {args.task.upper()}")
print(f"{'='*80}\n")

dataset = CMUBrainDataset(root=root, task=args.task)
print(f"Dataset loaded successfully!")
print(f"  Number of graphs: {len(dataset)}")
print(f"  Number of features: {dataset.num_features}")
print(f"  Number of classes: {dataset.num_classes}")
print(f"  Average nodes per graph: {dataset[0].num_nodes}")

# Set args based on dataset
args.num_features = dataset.num_features
args.num_classes = dataset.num_classes

# Extract labels for stratified splitting
labels = [d.y.item() for d in dataset]
label_counts = np.bincount(labels)
print(f"  Label distribution: {label_counts}")
print()

# Training functions
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(args.device)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * len(data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader, model):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)
        prob = F.softmax(out, dim=1)[:, 1] if args.num_classes == 2 else None
        
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        if prob is not None:
            y_prob.extend(prob.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    
    # Compute AUROC for binary classification
    auroc = 0.5
    if args.num_classes == 2 and len(y_prob) > 0:
        try:
            auroc = roc_auc_score(y_true, np.array(y_prob))
        except:
            pass
    
    return acc, auroc

# Cross-validation
print(f"Starting {args.folds}-fold cross-validation with {args.runs} runs...")
print(f"Model: {args.model}, Hidden: {args.hidden}, Layers: {args.num_layers}")
print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
print(f"{'='*80}\n")

all_test_accs = []
all_test_aurocs = []
seeds = [123, 124, 125, 126, 127, 128, 129, 130, 131, 132]

for run_idx in range(args.runs):
    print(f"\n{'='*80}")
    print(f"RUN {run_idx + 1}/{args.runs}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    fix_seed(seeds[run_idx])
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seeds[run_idx])
    fold_accs = []
    fold_aurocs = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        # Further split train_val into train and val
        train_val_labels = [labels[i] for i in train_val_idx]
        train_size = int(0.875 * len(train_val_idx))  # 87.5% of train_val for training
        
        np.random.seed(seeds[run_idx])
        perm = np.random.permutation(len(train_val_idx))
        train_idx = train_val_idx[perm[:train_size]]
        val_idx = train_val_idx[perm[train_size:]]
        
        # Create data loaders
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]
        
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        
        # Initialize model
        gnn = eval(args.model)
        model = CMUResidualGNNs(args, train_dataset, args.hidden, args.hidden_mlp, 
                                args.num_layers, gnn).to(args.device)
        
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Training
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            loss = train(train_loader, model, optimizer)
            val_acc, val_auroc = test(val_loader, model)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                if epoch > int(args.epochs / 2):
                    torch.save(
                        model.state_dict(),
                        os.path.join(path, f'cmu_{args.task}_fold{fold_idx}_best.pkl')
                    )
            else:
                patience_counter += 1
            
            if patience_counter >= args.early_stopping:
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Fold {fold_idx+1}/{args.folds} | Epoch {epoch+1:3d} | "
                      f"Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val AUROC: {val_auroc:.4f}")
        
        # Load best model and test
        try:
            model.load_state_dict(torch.load(
                os.path.join(path, f'cmu_{args.task}_fold{fold_idx}_best.pkl')
            ))
        except:
            pass
        
        test_acc, test_auroc = test(test_loader, model)
        fold_accs.append(test_acc)
        fold_aurocs.append(test_auroc)
        
        print(f"  Fold {fold_idx+1}/{args.folds} | Test Acc: {test_acc:.4f} | Test AUROC: {test_auroc:.4f}")
    
    # Run statistics
    run_acc_mean = np.mean(fold_accs)
    run_acc_std = np.std(fold_accs)
    run_auroc_mean = np.mean(fold_aurocs)
    run_auroc_std = np.std(fold_aurocs)
    
    all_test_accs.extend(fold_accs)
    all_test_aurocs.extend(fold_aurocs)
    
    elapsed = time.time() - start_time
    
    print(f"\n  Run {run_idx+1} Summary:")
    print(f"    Accuracy:  {run_acc_mean:.4f} ± {run_acc_std:.4f}")
    print(f"    AUROC:     {run_auroc_mean:.4f} ± {run_auroc_std:.4f}")
    print(f"    Time:      {elapsed:.1f}s")
    
    # Log results
    logger(f"Run {run_idx+1},{run_acc_mean:.6f},{run_acc_std:.6f},{run_auroc_mean:.6f},{run_auroc_std:.6f}")

# Final statistics
print(f"\n{'='*80}")
print(f"FINAL RESULTS - {args.task.upper()}")
print(f"{'='*80}")
print(f"Model: {args.model}")
print(f"Total folds across {args.runs} runs: {len(all_test_accs)}")
print(f"\nAccuracy:  {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")
print(f"AUROC:     {np.mean(all_test_aurocs):.4f} ± {np.std(all_test_aurocs):.4f}")
print(f"{'='*80}\n")

# Save final summary
logger(f"\nFinal,{np.mean(all_test_accs):.6f},{np.std(all_test_accs):.6f},"
       f"{np.mean(all_test_aurocs):.6f},{np.std(all_test_aurocs):.6f}")
