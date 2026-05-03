"""
feature_engineering.py — Transform raw OHLCV data into model-ready features.
P08: Stock Market Sentiment and Price Movement Predictor

Functions
---------
add_features(df)               – Compute technical indicators & derived columns.
add_target(df)                 – Create binary next-day direction label.
create_sequences(df, lookback) – Sliding-window arrays for LSTM input.
split_data(X, y, ...)          – Chronological train / val / test split.
scale_features(df, cols)       – Min-Max scaling helper.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import LOOKBACK


# ──────────────────────────────────────────────────────────────
# Helper: manual RSI & MACD (avoids pandas_ta / numba dependency)
# ──────────────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with alpha=1/length)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ──────────────────────────────────────────────────────────────
# 1.  Feature computation
# ──────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from a raw OHLCV DataFrame.

    Features added
    ~~~~~~~~~~~~~~
    * **daily_return** — ``(Close - Close_prev) / Close_prev``
    * **return_std_20** — rolling 20-day standard deviation of daily returns
    * **RSI_14** — Relative Strength Index (period 14)
    * **MACD_12_26_9** — MACD line
    * **MACDs_12_26_9** — Signal line
    * **MACDh_12_26_9** — MACD histogram

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``Close`` and ``Volume`` columns,
        indexed by Date.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new feature columns appended and all
        NaN rows dropped.
    """
    df = df.copy()

    # ── Daily returns ──
    df['daily_return'] = df['Close'].pct_change()

    # ── Rolling 20-day std of returns ──
    df['return_std_20'] = df['daily_return'].rolling(window=20).std()

    # ── RSI (14) ──
    df['RSI_14'] = _compute_rsi(df['Close'], length=14)

    # ── MACD (12, 26, 9) ──
    macd, signal, hist = _compute_macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD_12_26_9']  = macd
    df['MACDs_12_26_9'] = signal
    df['MACDh_12_26_9'] = hist

    # ── Drop rows with NaN introduced by indicators ──
    df = df.dropna()

    print(f"[FEATURES]  Added 6 features → {df.shape[1]} columns, "
          f"{len(df)} rows remain after NaN drop.")
    return df


# ──────────────────────────────────────────────────────────────
# 2.  Target label
# ──────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary classification target for next-day direction.

    ``target = 1`` if tomorrow's Close > today's Close, else ``0``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Copy with a new ``target`` column; the last row is dropped
        because it has no "next day".
    """
    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.iloc[:-1]                          # drop last row (no label)
    df = df.dropna()

    up   = (df['target'] == 1).sum()
    down = (df['target'] == 0).sum()
    print(f"[TARGET]  ↑ {up} ({up / len(df):.1%})  |  "
          f"↓ {down} ({down / len(df):.1%})  |  Total {len(df)}")
    return df


# ──────────────────────────────────────────────────────────────
# 3.  Sequence creation
# ──────────────────────────────────────────────────────────────

def create_sequences(df: pd.DataFrame,
                     feature_cols: list | None = None,
                     lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """Convert a feature DataFrame into sliding-window numpy arrays.

    For each time-step *t* the model sees the preceding *lookback*
    rows of features and must predict ``target[t]``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``target`` column and at least one feature column.
    feature_cols : list, optional
        Column names to use as features.  If ``None``, every numeric
        column **except** ``target`` is used.
    lookback : int
        Number of past time-steps per sample.

    Returns
    -------
    X : np.ndarray, shape ``(samples, lookback, num_features)``
    y : np.ndarray, shape ``(samples,)``
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != 'target']

    features = df[feature_cols].values
    targets  = df['target'].values

    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback : i])
        y.append(targets[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[SEQUENCES]  X {X.shape}  |  y {y.shape}  "
          f"(lookback={lookback}, features={len(feature_cols)})")
    return X, y


# ──────────────────────────────────────────────────────────────
# 4.  Chronological train / val / test split
# ──────────────────────────────────────────────────────────────

def split_data(X: np.ndarray,
               y: np.ndarray,
               train_ratio: float = 0.70,
               val_ratio: float = 0.15) -> dict:
    """Split arrays chronologically into train / validation / test sets.

    **No shuffling** — preserves temporal ordering.

    Parameters
    ----------
    X : np.ndarray
        Feature array, shape ``(N, lookback, features)``.
    y : np.ndarray
        Label array, shape ``(N,)``.
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).
        Test gets the remainder ``1 - train_ratio - val_ratio``.

    Returns
    -------
    dict
        ``{'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'}``
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    splits = {
        'X_train': X[:train_end],       'y_train': y[:train_end],
        'X_val':   X[train_end:val_end], 'y_val':   y[train_end:val_end],
        'X_test':  X[val_end:],          'y_test':  y[val_end:],
    }

    print(f"[SPLIT]  Train {len(splits['X_train'])} | "
          f"Val {len(splits['X_val'])} | "
          f"Test {len(splits['X_test'])} "
          f"(ratio {train_ratio}/{val_ratio}/{1 - train_ratio - val_ratio:.2f})")
    return splits


# ──────────────────────────────────────────────────────────────
# 5.  Scaling helper
# ──────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame,
                   feature_cols: list) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max scale selected columns **in-place** (on a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : list
        Column names to scale.

    Returns
    -------
    df_scaled : pd.DataFrame
        Copy of *df* with scaled feature columns.
    scaler : MinMaxScaler
        Fitted scaler (use for inverse transforms later).
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    print(f"[SCALE]  Scaled {len(feature_cols)} columns with MinMaxScaler.")
    return df_scaled, scaler
