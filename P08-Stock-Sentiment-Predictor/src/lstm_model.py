"""
lstm_model.py — LSTM classifier: definition, training (with warmup & early stopping),
                and evaluation utilities.
P08: Stock Market Sentiment and Price Movement Predictor

Classes
-------
LSTMClassifier          – 2-layer stacked LSTM → FC(128→64) → FC(64→1) (raw logits).

Functions
---------
make_dataloader(X, y)   – Wrap numpy arrays in a PyTorch DataLoader.
train_lstm(...)         – Full training loop with LR warmup + early stopping.
evaluate_lstm(...)      – Accuracy, AUC-ROC, classification report on the test set.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, SEED, RESULTS_DIR,
)


# ──────────────────────────────────────────────────────────────
# 1.  Model architecture
# ──────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Stacked LSTM for binary price-movement classification.

    Architecture
    ~~~~~~~~~~~~
    Input  → ``(batch, lookback, num_features)``
    LSTM   → 2 layers, hidden_size=128, dropout=0.2
    Dense  → 128 → 64 (ReLU) → 1 (raw logit)
    Output → raw logit score (apply sigmoid externally for probability).

    Parameters
    ----------
    input_size : int
        Number of input features per time-step.
    hidden_size : int
        LSTM hidden-state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate applied between LSTM layers and in the FC head.
    """

    def __init__(self, input_size: int,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, lookback, num_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` — raw logit scores.
        """
        lstm_out, _ = self.lstm(x)          # (batch, lookback, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
        return self.fc(last_hidden).squeeze(-1)


# ──────────────────────────────────────────────────────────────
# 2.  DataLoader helper
# ──────────────────────────────────────────────────────────────

def make_dataloader(X: np.ndarray, y: np.ndarray,
                    batch_size: int = BATCH_SIZE,
                    shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays into a PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray
        Features, shape ``(N, lookback, features)``.
    y : np.ndarray
        Labels, shape ``(N,)``.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle (disable for val/test).

    Returns
    -------
    DataLoader
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ──────────────────────────────────────────────────────────────
# 3.  Training with LR warmup + early stopping
# ──────────────────────────────────────────────────────────────

def _warmup_lr(optimizer, epoch: int, warmup_epochs: int, target_lr: float):
    """Linearly ramp the learning rate during the first *warmup_epochs*."""
    lr = target_lr * (epoch / warmup_epochs)
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               input_size: int | None = None,
               epochs: int = EPOCHS,
               lr: float = 1e-3,
               warmup_epochs: int = 5,
               patience: int = 7,
               batch_size: int = BATCH_SIZE,
               device: str | None = None,
               save_path: str | None = None,
               pos_weight=None) -> tuple:
    """Train an :class:`LSTMClassifier` with LR warmup and early stopping.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features ``(N, lookback, F)`` and labels ``(N,)``.
    X_val, y_val : np.ndarray
        Validation features and labels.
    input_size : int, optional
        Number of features per time-step.  Inferred from *X_train* if
        not provided.
    epochs : int
        Maximum training epochs (default from ``config.EPOCHS``).
    lr : float
        Peak learning rate (reached after warmup).
    warmup_epochs : int
        Number of epochs for linear LR warmup.
    patience : int
        Early-stopping patience (based on validation loss).
    batch_size : int
        Mini-batch size.
    device : str, optional
        ``'cuda'`` or ``'cpu'``.  Auto-detected if ``None``.

    Returns
    -------
    model : LSTMClassifier
        Trained model (with best weights restored).
    history : dict
        ``{'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}``.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if input_size is None:
        input_size = X_train.shape[2]

    # ── DataLoaders ──
    train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_dataloader(X_val,   y_val,   batch_size, shuffle=False)

    # ── Model / loss / optimiser ──
    model     = LSTMClassifier(input_size=input_size).to(device)
    
    if pos_weight is None:
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        computed_pos_weight = class_weights[1] / class_weights[0]
        pos_weight = torch.tensor([computed_pos_weight], dtype=torch.float32, device=device)
    elif not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    else:
        pos_weight = pos_weight.to(device)
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history     = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val    = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    wait        = 0

    print(f"\n{'═' * 65}")
    print(f"  Training LSTMClassifier  |  device={device}  |  "
          f"epochs={epochs}  lr={lr}")
    print(f"  warmup={warmup_epochs}  patience={patience}  "
          f"batch={batch_size}")
    print(f"{'═' * 65}\n")

    for epoch in range(1, epochs + 1):
        # ── LR warmup ──
        if epoch <= warmup_epochs:
            _warmup_lr(optimizer, epoch, warmup_epochs, lr)

        # ── Train ──
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_loss += criterion(preds, y_b).item() * X_b.size(0)
                correct  += ((preds > 0.5).float() == y_b).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ── Logging ──
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:>3}/{epochs}  │  "
              f"Train Loss {train_loss:.4f}  │  "
              f"Val Loss {val_loss:.4f}  │  "
              f"Val Acc {val_acc:.4f}  │  "
              f"LR {cur_lr:.6f}")

        # ── Early stopping ──
        if val_loss < best_val:
            best_val     = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(patience {patience} exhausted).\n")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"  ✓ Best val loss: {best_val:.4f}")

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ Model saved → {save_path}\n")

    return model, history


# ──────────────────────────────────────────────────────────────
# 4.  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_lstm(model: nn.Module,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  device: str | None = None) -> dict:
    """Evaluate a trained LSTM on the test set.

    Computes accuracy, AUC-ROC, classification report, and ROC
    curve data.

    Parameters
    ----------
    model : nn.Module
        Trained :class:`LSTMClassifier`.
    X_test : np.ndarray
        Test features, shape ``(N, lookback, features)``.
    y_test : np.ndarray
        True labels, shape ``(N,)``.
    device : str, optional
        Auto-detected if ``None``.

    Returns
    -------
    dict
        ``{'accuracy', 'auc_roc', 'report', 'report_dict',
           'fpr', 'tpr', 'y_pred', 'y_prob'}``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device).eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_t)).cpu().numpy()

    y_pred = (y_prob > 0.5).astype(int)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report_str  = classification_report(y_test, y_pred,
                                        target_names=['Down ↓', 'Up ↑'],
                                        zero_division=0)
    report_dict = classification_report(y_test, y_pred,
                                        target_names=['Down ↓', 'Up ↑'],
                                        output_dict=True,
                                        zero_division=0)

    print(f"\n{'═' * 50}")
    print(f"  Test Evaluation")
    print(f"{'═' * 50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"{'─' * 50}")
    print(report_str)

    return {
        'accuracy':    acc,
        'auc_roc':     auc,
        'report':      report_str,
        'report_dict': report_dict,
        'fpr':         fpr,
        'tpr':         tpr,
        'y_pred':      y_pred,
        'y_prob':      y_prob,
    }
