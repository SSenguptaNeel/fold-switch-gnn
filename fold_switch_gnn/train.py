"""
train.py
========
Training script for FoldSwitchGCN.

Given the small dataset (~20 proteins), uses leave-one-out or k-fold CV
rather than a fixed train/test split, which is standard practice in
structural bioinformatics with limited labelled data.

Run:
    python -m fold_switch_gnn.train --root ./cache --epochs 100 --folds 5
"""

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np

from fold_switch_gnn.graphs.dataset import FoldSwitchDataset
from fold_switch_gnn.models.model import FoldSwitchGCN


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    return np.array(all_probs), np.array(all_labels)


def run_cv(dataset, args, device):
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_aucs, fold_mccs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n── Fold {fold+1}/{args.folds} ──────────────────────────────")
        train_data = [dataset[i] for i in train_idx]
        val_data   = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=args.batch_size)

        model = FoldSwitchGCN(
            in_channels=dataset.num_node_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        # class-weighted BCE for imbalanced labels
        pos_weight = torch.tensor(
            [sum(l==0 for l in labels) / (sum(l==1 for l in labels) + 1e-6)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc = 0.0
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % 20 == 0 or epoch == args.epochs:
                probs, true = evaluate(model, val_loader, device)
                if len(np.unique(true)) > 1:
                    auc = roc_auc_score(true, probs)
                    best_auc = max(best_auc, auc)
                print(f"  Epoch {epoch:3d}  loss={loss:.4f}  val_AUC={auc:.3f}")

        probs, true = evaluate(model, val_loader, device)
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
        mcc = matthews_corrcoef(true, preds)
        fold_aucs.append(auc)
        fold_mccs.append(mcc)
        print(f"  Fold {fold+1} final  AUC={auc:.3f}  MCC={mcc:.3f}")

    print(f"\n{'='*50}")
    print(f"CV AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"CV MCC: {np.mean(fold_mccs):.3f} ± {np.std(fold_mccs):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       default="./cache")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--folds",      type=int,   default=5)
    parser.add_argument("--hidden_dim", type=int,   default=64)
    parser.add_argument("--num_layers", type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = FoldSwitchDataset(root=args.root)
    dataset.summary()

    run_cv(dataset, args, device)


if __name__ == "__main__":
    main()
