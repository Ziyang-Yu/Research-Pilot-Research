#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-encoded training only: load saved embeddings/labels and train an MLP.
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------
# Configuration
# ---------------

# Encoder removed (pre-encoded flow only)

TRAIN_SPLIT_PCT = 0.8           # 80/20 split via stable hash on title
EPOCHS = 30                     # total epochs for training
BATCH_SIZE_TRAIN = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42

torch.is_deterministic = True
torch.backends.cudnn.deterministic = True

# ----------------------------
# Utilities and set seed
# ----------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

 # (DB-related utilities removed for train-only script)

"""Pre-encoded training only; encoder code removed."""

# ---------------------------------
# Multi-task MLP classifier (Independent Networks)
# ---------------------------------
class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, num_tasks: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Independent networks for each task (no shared layers)
        self.task_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # Get independent outputs for each task
        task_outputs = []
        for network in self.task_networks:
            task_outputs.append(network(x))
        
        # Stack outputs: (batch_size, num_tasks)
        return torch.stack(task_outputs, dim=1).squeeze(-1)

# ---------------------------------
# Multi-task XGBoost classifier
# ---------------------------------
# Remove XGBoost multi-task wrapper

# ---------------------------------
# Datasets
# ---------------------------------
class MultiTaskTensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0]
        self.X = X.float()
        self.y = y.float()  # Shape: (batch_size, num_tasks)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------
# Training and Evaluation
# ---------------------------------
def evaluate_model(model, test_X, test_y, device: str, num_tasks: int, model_type: str = "mlp") -> Dict[str, float]:
    """Evaluate model on test set and return metrics for each task."""
    if test_X is None or test_X.shape[0] == 0:
        print("[Eval] No test set available.")
        return {}
    
    if model_type == "mlp":
        # PyTorch MLP evaluation
        model.eval()
        with torch.no_grad():
            logits = model(test_X)  # Shape: (batch_size, num_tasks)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            y = test_y  # Shape: (batch_size, num_tasks)
    else:
        raise ValueError("Unsupported model_type for evaluation")
    
    # Calculate metrics for each task
    task_metrics = {}
    overall_acc = 0.0
    
    for task_idx in range(num_tasks):
        task_preds = preds[:, task_idx].cpu().numpy()
        task_y = y[:, task_idx].cpu().numpy()
        
        # Calculate metrics using sklearn for consistency
        task_acc = accuracy_score(task_y, task_preds)
        task_precision = precision_score(task_y, task_preds, zero_division=0)
        task_recall = recall_score(task_y, task_preds, zero_division=0)
        task_f1 = f1_score(task_y, task_preds, zero_division=0)
        
        overall_acc += task_acc
        
        task_metrics[f'task_{task_idx}'] = {
            'accuracy': task_acc,
            'precision': task_precision,
            'recall': task_recall,
            'f1': task_f1
        }

    # and along index 1
    
    if type(preds) == torch.Tensor and type(y) == torch.Tensor:
        preds = torch.prod(preds, axis=1)
        y = torch.prod(y, axis=1)
    elif type(preds) == np.ndarray and type(y) == np.ndarray:
        preds = np.prod(preds, axis=1)
        y = np.prod(y, axis=1)
    else:
        raise ValueError(f"Unsupported type: {type(preds)}")

    # overall metrics
    overall_acc = accuracy_score(y, preds)
    overall_precision = precision_score(y, preds, zero_division=0)
    overall_recall = recall_score(y, preds, zero_division=0)
    overall_f1 = f1_score(y, preds, zero_division=0)

    # Overall metrics 
    # overall_acc /= num_tasks
    # overall_precision = sum(m['precision'] for m in task_metrics.values()) / num_tasks
    # overall_recall = sum(m['recall'] for m in task_metrics.values()) / num_tasks
    # overall_f1 = sum(m['f1'] for m in task_metrics.values()) / num_tasks
    
    metrics = {
        'overall_accuracy': overall_acc,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'task_metrics': task_metrics
    }
    
    print(f"[Eval-{model_type.upper()}] size={int(y.shape[0])}  overall_acc={overall_acc:.4f}  overall_precision={overall_precision:.4f}  overall_recall={overall_recall:.4f}  overall_f1={overall_f1:.4f}")
    
    # Print per-task metrics
    for task_idx in range(num_tasks):
        task_m = task_metrics[f'task_{task_idx}']
        print(f"  Task {task_idx}: acc={task_m['accuracy']:.4f}  precision={task_m['precision']:.4f}  recall={task_m['recall']:.4f}  f1={task_m['f1']:.4f}")
    
    return metrics

def train_mlp_model(model: nn.Module, train_X: torch.Tensor, train_y: torch.Tensor, 
                    test_X: torch.Tensor, test_y: torch.Tensor, device: str, 
                    num_tasks: int, epochs: int = EPOCHS) -> List[Dict[str, float]]:
    """Train MLP model and return evaluation history."""
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create data loaders
    train_dataset = MultiTaskTensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=False)
    
    # Move data to device
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    test_X = test_X.to(device) if test_X is not None else None
    test_y = test_y.to(device) if test_y is not None else None
    
    evaluation_history = []
    
    print(f"[Train-MLP] Starting training for {epochs} epochs...")
    print(f"[Train-MLP] Training samples: {train_X.shape[0]}")
    print(f"[Train-MLP] Test samples: {test_X.shape[0] if test_X is not None else 0}")
    print(f"[Train-MLP] Number of tasks: {num_tasks}")
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        n_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)  # Shape: (batch_size, num_tasks)
            
            optimizer.zero_grad()
            logits = model(batch_X)  # Shape: (batch_size, num_tasks)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            n_samples += batch_X.size(0)
        
        avg_loss = running_loss / max(1, n_samples)
        print(f"[Train-MLP] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  (n={n_samples})")
        
        # Evaluation
        if test_X is not None and test_y is not None:
            metrics = evaluate_model(model, test_X, test_y, device, num_tasks, model_type="mlp")
            evaluation_history.append(metrics)
    
    return evaluation_history

# Remove XGBoost trainer

# ---------------------------------
# Model Comparison
# ---------------------------------
# Remove model comparison

# ---------------------------------
# Main flow
# ---------------------------------
def main():
    global EPOCHS, BATCH_SIZE_TRAIN, LEARNING_RATE
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train MLP from pre-encoded BGE-M3 embeddings")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing tensors (X_all.pt, y_all.pt, train_idx.pt, test_idx.pt, meta.json)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_TRAIN, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--save_model", type=str, default=None, help="Optional path to save trained MLP (state_dict)")
    
    args = parser.parse_args()

    # Update configuration based on arguments
    save_dir = args.save_dir
    print(f"[Config] Using encoder: bge_m3 (pre-encoded)")
    print(f"[Config] Data dir: {save_dir}")

    # ----------------------
    # TRAIN MODE (only)
    # ----------------------

    # Override training hyperparameters from CLI
    EPOCHS = args.epochs
    BATCH_SIZE_TRAIN = args.batch_size
    LEARNING_RATE = args.learning_rate
    print(f"[Config] Epochs: {EPOCHS}  Batch size: {BATCH_SIZE_TRAIN}  LR: {LEARNING_RATE}")

    # Load tensors
    X_all_path = os.path.join(save_dir, "X_all.pt")
    y_all_path = os.path.join(save_dir, "y_all.pt")
    train_idx_path = os.path.join(save_dir, "train_idx.pt")
    test_idx_path = os.path.join(save_dir, "test_idx.pt")
    if not (os.path.exists(X_all_path) and os.path.exists(y_all_path) and os.path.exists(train_idx_path) and os.path.exists(test_idx_path)):
        raise FileNotFoundError("Missing one or more required files: X_all.pt, y_all.pt, train_idx.pt, test_idx.pt")

    X_all = torch.load(X_all_path)
    y_all = torch.load(y_all_path)
    train_idx = torch.load(train_idx_path)
    test_idx = torch.load(test_idx_path)
    input_dim = int(X_all.shape[1])
    num_tasks = int(y_all.shape[1])
    print(f"[Load] X_all={tuple(X_all.shape)}  y_all={tuple(y_all.shape)}  train_n={int(train_idx.shape[0])}  test_n={int(test_idx.shape[0])}")

    # Build splits
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    print(f"[Train] X_train shape: {X_train.shape}")
    print(f"[Train] y_train shape: {y_train.shape}")
    print(f"[Train] X_test shape: {X_test.shape}")
    print(f"[Train] y_test shape: {y_test.shape}")

    # Initialize and train MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp_model = MultiTaskMLP(in_dim=input_dim, num_tasks=num_tasks).to(device)
    print(f"[Model] Initialized MultiTaskMLP with input_dim={input_dim}, num_tasks={num_tasks} on device={device}")

    _ = train_mlp_model(
        model=mlp_model,
        train_X=X_train,
        train_y=y_train,
        test_X=X_test,
        test_y=y_test,
        device=device,
        num_tasks=num_tasks,
        epochs=EPOCHS
    )

    # Optionally save trained model
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(mlp_model.state_dict(), args.save_model)
        print(f"[Save] Saved trained MLP state_dict to {args.save_model}")

    print("\n[Main] Training completed successfully.")
    return

    # (Training and split removed for simple encode-and-save workflow)

# Remove global metrics summary for XGBoost/compare

if __name__ == "__main__":
    main()
