"""
Simplified CMU training - matches NeuroGraph's approach more closely
Single train/val/test split per run, like HCP experiments
"""
import sys
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neurograph.datasets import CMUBrainDataset
from neurograph.models import CMUResidualGNNsSimple
from neurograph.utils import fix_seed
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv, ChebConv, SGConv, GraphConv

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sex')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

# Paths
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
    f = open(os.path.join(res_path, f'cmu_{args.task}_simple.csv'), 'a')
    print(info, file=f)
    f.close()

# Load dataset
print(f"\n{'='*80}")
print(f"CMU Brain - {args.task.upper()} (Simplified Training)")
print(f"{'='*80}\n")

fix_seed(args.seed)
dataset = CMUBrainDataset(root=root, task=args.task)

print(f"Dataset: {len(dataset)} graphs")
print(f"Features: {dataset.num_features}")
print(f"Classes: {dataset.num_classes}")

labels = [d.y.item() for d in dataset]
print(f"Label distribution: {np.bincount(labels)}\n")

args.num_features = dataset.num_features
args.num_classes = dataset.num_classes

# Training functions
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data.y)
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
    
    acc = accuracy_score(y_true, y_pred)
    auroc = 0.5
    if args.num_classes == 2 and len(y_prob) > 0:
        try:
            auroc = roc_auc_score(y_true, np.array(y_prob))
        except:
            pass
    
    return acc, auroc

# Run experiments
print(f"Training: {args.runs} runs, 70/10/20 train/val/test split")
print(f"Model: {args.model}, Hidden: {args.hidden}, Layers: {args.num_layers}")
print(f"LR: {args.lr}, Weight decay: {args.weight_decay}, Batch: {args.batch_size}")
print(f"{'='*80}\n")

test_accs = []
test_aurocs = []
seeds = [123, 124, 125, 126, 127, 128, 129, 130, 131, 132]

for run_idx in range(args.runs):
    print(f"Run {run_idx+1}/{args.runs}", end=" ")
    start = time.time()
    fix_seed(seeds[run_idx])
    
    # 70/10/20 split like NeuroGraph
    train_tmp, test_indices = train_test_split(
        list(range(len(labels))),
        test_size=0.2,
        stratify=labels,
        random_state=seeds[run_idx],
        shuffle=True
    )
    
    tmp_labels = [labels[i] for i in train_tmp]
    train_indices, val_indices = train_test_split(
        train_tmp,
        test_size=0.125,  # 0.125 of 80% = 10% of total
        stratify=tmp_labels,
        random_state=seeds[run_idx],
        shuffle=True
    )
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    
    # Initialize model - use simpler version
    gnn = eval(args.model)
    model = CMUResidualGNNsSimple(
        args, train_dataset, args.hidden, args.hidden, args.num_layers, gnn
    ).to(args.device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        loss = train(train_loader, model, optimizer)
        val_acc, val_auroc = test(val_loader, model)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch
            # Save best model
            if epoch > args.epochs // 3:
                torch.save(model.state_dict(), 
                          os.path.join(path, f'cmu_{args.task}_run{run_idx}_best.pkl'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping:
            break
    
    # Load best and test
    try:
        model.load_state_dict(torch.load(
            os.path.join(path, f'cmu_{args.task}_run{run_idx}_best.pkl')
        ))
    except:
        pass
    
    test_acc, test_auroc = test(test_loader, model)
    test_accs.append(test_acc)
    test_aurocs.append(test_auroc)
    
    elapsed = time.time() - start
    print(f"| Best epoch: {best_epoch:3d} | Test Acc: {test_acc:.4f} | AUROC: {test_auroc:.4f} | Time: {elapsed:.1f}s")
    
    logger(f"{run_idx+1},{test_acc:.6f},{test_auroc:.6f}")

# Final results
print(f"\n{'='*80}")
print(f"FINAL RESULTS - {args.task.upper()}")
print(f"{'='*80}")
print(f"Accuracy:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print(f"AUROC:     {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}")
print(f"{'='*80}\n")

logger(f"Final,{np.mean(test_accs):.6f},{np.std(test_accs):.6f},{np.mean(test_aurocs):.6f},{np.std(test_aurocs):.6f}")
