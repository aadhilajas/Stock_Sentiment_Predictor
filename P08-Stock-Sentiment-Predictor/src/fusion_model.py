"""
fusion_model.py — Merge price features with sentiment, run 3 comparative experiments,
                  and persist results.
P08: Stock Market Sentiment and Price Movement Predictor

Functions
---------
merge_sentiment(price_df, sentiment_df)
    Inner-join price features and daily sentiment on Date.
build_fusion_features(price_df, sentiment_df, ...)
    Full pipeline: merge → scale → create sequences.
train_sentiment_mlp(X_train, y_train, X_val, y_val)
    Lightweight MLP baseline using only sentiment features.
run_experiments(stock_data, sentiment_daily, ...)
    Execute the 3-experiment comparison and save results.
plot_experiment_comparison(results_df)
    Bar chart of accuracy + AUC across experiments.
compute_sentiment_correlation(stock_data, sentiment_daily)
    Per-stock correlation heatmap between sentiment & 1-day return.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    STOCKS, LOOKBACK, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, SEED, RESULTS_DIR,
)
from src.feature_engineering import (
    add_features, add_target, create_sequences,
    split_data, scale_features,
)
from src.lstm_model import (
    LSTMClassifier, train_lstm, evaluate_lstm,
)


# ──────────────────────────────────────────────────────────────
# 1.  Merge price + sentiment on Date
# ──────────────────────────────────────────────────────────────

def merge_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    sentiment_col: str = "daily_sentiment",
) -> pd.DataFrame:
    """Inner-join a price DataFrame with daily sentiment scores on Date.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV + engineered features, Date-indexed.
    sentiment_df : pd.DataFrame
        Must contain ``daily_sentiment`` column, Date-indexed.
    sentiment_col : str
        Column name in *sentiment_df* to merge.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``sentiment_score`` column appended.
    """
    merged = price_df.copy()

    # Align sentiment to price index
    sent = sentiment_df[[sentiment_col]].rename(
        columns={sentiment_col: "sentiment_score"}
    )
    merged = merged.join(sent, how="left")

    # Forward-fill missing sentiment (weekends / holidays)
    merged["sentiment_score"] = merged["sentiment_score"].ffill().fillna(0.0)

    print(f"[MERGE]  {len(merged)} rows after merging sentiment "
          f"(nulls filled: {merged['sentiment_score'].isna().sum()})")
    return merged


# ──────────────────────────────────────────────────────────────
# 2.  Build fusion feature matrix
# ──────────────────────────────────────────────────────────────

def build_fusion_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge, scale, and create sequences with sentiment appended.

    Parameters
    ----------
    price_df : pd.DataFrame
        Raw OHLCV stock data (Date-indexed).
    sentiment_df : pd.DataFrame
        Daily sentiment scores (Date-indexed, ``daily_sentiment`` col).
    lookback : int
        Sliding window length.

    Returns
    -------
    X : np.ndarray  — ``(samples, lookback, num_features)``
    y : np.ndarray  — ``(samples,)``
    feature_cols : list[str]
    """
    df = add_features(price_df)
    df = add_target(df)
    df = merge_sentiment(df, sentiment_df)

    # Feature columns = all numeric except target
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != "target"]

    df_scaled, _ = scale_features(df, feature_cols)
    X, y = create_sequences(df_scaled, feature_cols=feature_cols, lookback=lookback)
    return X, y, feature_cols


# ──────────────────────────────────────────────────────────────
# 3.  Sentiment-only MLP baseline
# ──────────────────────────────────────────────────────────────

class SentimentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # raw logit, no sigmoid
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_sentiment_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    patience: int = 7,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
    save_path: str | None = None,
    pos_weight=None,
) -> tuple:
    """Train an MLP on sentiment-only features (flattened sequences).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.  X shape ``(N, lookback, 1)`` — will be flattened.
    X_val, y_val : np.ndarray
        Validation data.
    epochs : int
    lr : float
    patience : int
    batch_size : int
    device : str, optional

    Returns
    -------
    model : SentimentMLP
    history : dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Flatten: (N, lookback, F) → (N, lookback*F)
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_va_flat = X_val.reshape(X_val.shape[0], -1)
    input_dim = X_tr_flat.shape[1]

    train_ds = TensorDataset(torch.tensor(X_tr_flat, dtype=torch.float32),
                             torch.tensor(y_train,   dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_va_flat, dtype=torch.float32),
                             torch.tensor(y_val,     dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = SentimentMLP(input_dim).to(device)
    
    if pos_weight is None:
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        # Use the balanced class weight for the positive class (Up=1).
        pos_weight = torch.tensor([float(class_weights[1])], dtype=torch.float32, device=device)
    elif not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    else:
        pos_weight = pos_weight.to(device)
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_wts, wait = float("inf"), None, 0

    print(f"\n{'═' * 60}")
    print(f"  Training SentimentMLP  |  input_dim={input_dim}  |  "
          f"device={device}")
    print(f"{'═' * 60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        t_loss = running / len(train_ds)

        model.eval()
        v_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                v_loss += criterion(preds, yb).item() * xb.size(0)
                probs = torch.sigmoid(preds)
                correct += ((probs > 0.5).float() == yb).sum().item()
        v_loss /= len(val_ds)
        v_acc = correct / len(val_ds)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  │  "
                  f"Train {t_loss:.4f}  Val {v_loss:.4f}  Acc {v_acc:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            best_wts = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch}.\n")
                break

    model.load_state_dict(best_wts)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ Model saved → {save_path}\n")
    return model, history


def evaluate_mlp(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str | None = None,
) -> dict:
    """Evaluate an MLP on the test set (flattens input automatically).

    Parameters
    ----------
    model : nn.Module
    X_test : np.ndarray — ``(N, lookback, F)``
    y_test : np.ndarray
    device : str, optional

    Returns
    -------
    dict with ``accuracy`` and ``auc_roc``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    X_flat = X_test.reshape(X_test.shape[0], -1)
    xt = torch.tensor(X_flat, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_prob = torch.sigmoid(model(xt)).cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"  MLP  →  Acc {acc:.4f}  |  AUC {auc:.4f}")
    return {"accuracy": acc, "auc_roc": auc, "y_pred": y_pred, "y_prob": y_prob}


# ──────────────────────────────────────────────────────────────
# 4.  Run 3 experiments
# ──────────────────────────────────────────────────────────────

def run_experiments(
    stock_data: dict[str, pd.DataFrame],
    sentiment_daily: pd.DataFrame,
    lookback: int = LOOKBACK,
    device: str | None = None,
) -> pd.DataFrame:
    """Execute three comparative experiments and save results.

    Experiment 1 — LSTM on **price features only**.
    Experiment 2 — MLP on **sentiment features only**.
    Experiment 3 — LSTM on **price + sentiment** (fusion).

    Parameters
    ----------
    stock_data : dict
        ``{ticker: OHLCV DataFrame}`` for each stock.
    sentiment_daily : pd.DataFrame
        Daily sentiment scores (Date-indexed, ``daily_sentiment`` col).
    lookback : int
    device : str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``Experiment``, ``Accuracy``, ``AUC-ROC``.
        Also saved to ``results/experiment_results.csv``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Prepare data per stock and concatenate ──
    price_Xs, price_ys = [], []
    sent_Xs,  sent_ys  = [], []
    fusion_Xs, fusion_ys = [], []

    for ticker, raw_df in stock_data.items():
        print(f"\n── Preparing {ticker} ──")

        # Price-only
        df_p = add_features(raw_df.copy())
        df_p = add_target(df_p)
        feat_p = [c for c in df_p.select_dtypes(include=[np.number]).columns
                  if c != "target"]
        df_p_s, _ = scale_features(df_p, feat_p)
        Xp, yp = create_sequences(df_p_s, feature_cols=feat_p, lookback=lookback)
        price_Xs.append(Xp)
        price_ys.append(yp)

        # Sentiment-only
        df_s = merge_sentiment(df_p, sentiment_daily)
        sent_cols = ["sentiment_score"]
        df_s_s, _ = scale_features(df_s, sent_cols)
        Xs, ys = create_sequences(df_s_s, feature_cols=sent_cols, lookback=lookback)
        sent_Xs.append(Xs)
        sent_ys.append(ys)

        # Fusion (price + sentiment)
        fuse_cols = feat_p + ["sentiment_score"]
        df_f = merge_sentiment(df_p, sentiment_daily)
        df_f_s, _ = scale_features(df_f, fuse_cols)
        Xf, yf = create_sequences(df_f_s, feature_cols=fuse_cols, lookback=lookback)
        fusion_Xs.append(Xf)
        fusion_ys.append(yf)

    X_price  = np.concatenate(price_Xs)
    y_price  = np.concatenate(price_ys)
    X_sent   = np.concatenate(sent_Xs)
    y_sent   = np.concatenate(sent_ys)
    X_fusion = np.concatenate(fusion_Xs)
    y_fusion = np.concatenate(fusion_ys)

    results = []

    # ── Experiment 1: Price-only LSTM ──
    print(f"\n{'█' * 60}")
    print("  EXPERIMENT 1 — LSTM (Price Features Only)")
    print(f"{'█' * 60}")
    sp1 = split_data(X_price, y_price)
    # Compute pos_weight for price-only experiment
    cw1 = compute_class_weight('balanced', classes=np.array([0, 1]), y=sp1['y_train'])
    pos_w1 = torch.tensor([float(cw1[1])], dtype=torch.float32)
    m1, _ = train_lstm(sp1["X_train"], sp1["y_train"],
                       sp1["X_val"], sp1["y_val"], device=device,
                       pos_weight=pos_w1,
                       save_path=os.path.join(RESULTS_DIR, 'exp1_price_only.pth'))
    e1 = evaluate_lstm(m1, sp1["X_test"], sp1["y_test"], device=device)
    results.append({"Experiment": "Price-Only LSTM",
                    "Accuracy": round(e1["accuracy"], 4),
                    "AUC-ROC": round(e1["auc_roc"], 4)})

    # ── Experiment 2: Sentiment-only MLP ──
    print(f"\n{'█' * 60}")
    print("  EXPERIMENT 2 — MLP (Sentiment Features Only)")
    print(f"{'█' * 60}")
    sp2 = split_data(X_sent, y_sent)
    # Compute pos_weight for sentiment-only experiment
    cw2 = compute_class_weight('balanced', classes=np.array([0, 1]), y=sp2['y_train'])
    pos_w2 = torch.tensor([float(cw2[1])], dtype=torch.float32)
    m2, _ = train_sentiment_mlp(sp2["X_train"], sp2["y_train"],
                                sp2["X_val"], sp2["y_val"], device=device,
                                pos_weight=pos_w2,
                                save_path=os.path.join(RESULTS_DIR, 'exp2_sentiment_only.pth'))
    e2 = evaluate_mlp(m2, sp2["X_test"], sp2["y_test"], device=device)
    results.append({"Experiment": "Sentiment-Only MLP",
                    "Accuracy": round(e2["accuracy"], 4),
                    "AUC-ROC": round(e2["auc_roc"], 4)})

    # ── Experiment 3: Fusion LSTM ──
    print(f"\n{'█' * 60}")
    print("  EXPERIMENT 3 — LSTM (Price + Sentiment Fusion)")
    print(f"{'█' * 60}")
    sp3 = split_data(X_fusion, y_fusion)
    # Compute pos_weight for fusion experiment
    cw3 = compute_class_weight('balanced', classes=np.array([0, 1]), y=sp3['y_train'])
    pos_w3 = torch.tensor([float(cw3[1])], dtype=torch.float32)
    m3, _ = train_lstm(sp3["X_train"], sp3["y_train"],
                       sp3["X_val"], sp3["y_val"], device=device,
                       pos_weight=pos_w3,
                       save_path=os.path.join(RESULTS_DIR, 'exp3_fusion.pth'))
    e3 = evaluate_lstm(m3, sp3["X_test"], sp3["y_test"], device=device)
    results.append({"Experiment": "Fusion LSTM (Price+Sent)",
                    "Accuracy": round(e3["accuracy"], 4),
                    "AUC-ROC": round(e3["auc_roc"], 4)})

    # ── Save results ──
    results_df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED]  Experiment results → {csv_path}\n")
    print(results_df.to_string(index=False))

    return results_df


# ──────────────────────────────────────────────────────────────
# 5.  Comparison bar chart
# ──────────────────────────────────────────────────────────────

def plot_experiment_comparison(results_df: pd.DataFrame):
    """Generate a grouped bar chart of Accuracy & AUC-ROC per experiment.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: ``Experiment``, ``Accuracy``, ``AUC-ROC``.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    w = 0.3

    bars1 = ax.bar(x - w / 2, results_df["Accuracy"], w,
                   label="Accuracy", color="#2563eb", edgecolor="white")
    bars2 = ax.bar(x + w / 2, results_df["AUC-ROC"], w,
                   label="AUC-ROC", color="#16a34a", edgecolor="white")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Experiment"], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison — Accuracy & AUC-ROC",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 6.  Sentiment ↔ price-movement correlation heatmap
# ──────────────────────────────────────────────────────────────

def compute_sentiment_correlation(
    stock_data: dict[str, pd.DataFrame],
    sentiment_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compute per-stock correlation between daily sentiment and 1-day return.

    Parameters
    ----------
    stock_data : dict
        ``{ticker: OHLCV DataFrame}``.
    sentiment_daily : pd.DataFrame
        Date-indexed, ``daily_sentiment`` column.

    Returns
    -------
    corr_df : pd.DataFrame
        Correlation matrix (tickers × [daily_sentiment, 1d_return]).
    fig : matplotlib Figure
        Heatmap visualisation.
    """
    records = []
    for ticker, df in stock_data.items():
        df = df.copy()
        df["1d_return"] = df["Close"].pct_change()
        merged = df[["1d_return"]].join(
            sentiment_daily[["daily_sentiment"]], how="left"
        )
        merged = merged.ffill().dropna()
        corr = merged["daily_sentiment"].corr(merged["1d_return"])
        records.append({"Ticker": ticker, "Correlation": round(corr, 4)})

    corr_df = pd.DataFrame(records).set_index("Ticker")

    # Heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr_df[["Correlation"]],
        annot=True, fmt=".4f", cmap="RdYlGn", center=0,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Sentiment ↔ 1-Day Return Correlation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return corr_df, fig
