"""
sentiment_pipeline.py — FinBERT fine-tuning, inference, and aggregation utilities.
P08: Stock Market Sentiment and Price Movement Predictor

Classes / Functions
-------------------
load_finbert()                   – Load pre-trained ProsusAI/finbert tokenizer + model.
load_phrasebank()                – Download Financial PhraseBank (sentences_allagree).
fine_tune_finbert(...)           – Fine-tune FinBERT for 3-class sentiment (3 epochs).
get_sentiment_score(text)        – Predict sentiment for a single headline string.
batch_sentiment(headlines_list)  – Process a list → DataFrame with label & score.
aggregate_daily_sentiment(df)    – Daily average sentiment score per date.
"""

import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm

from src.config import SEED, RESULTS_DIR


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_MAP     = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID   = {"negative": 0, "neutral": 1, "positive": 2}
NUM_LABELS    = 3

# Sentiment numeric mapping for downstream fusion
SENTIMENT_NUMERIC = {"positive": 1, "neutral": 0, "negative": -1}

# Fine-tuned model save directory
FINETUNED_DIR = os.path.join(RESULTS_DIR, "finbert_finetuned")


# ──────────────────────────────────────────────────────────────
# 1.  Dataset
# ──────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenised sentences + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_phrasebank() -> tuple[list[str], list[int]]:
    """Download Financial PhraseBank (sentences_allagree split).

    The dataset uses label encoding:
      - 0 = negative
      - 1 = neutral
      - 2 = positive

    Returns
    -------
    sentences : list[str]
    labels : list[int]
    """
    print("[DATA]  Loading Financial PhraseBank (sentences_allagree) …")
    ds = load_dataset(
        "takala/financial_phrasebank",
        "sentences_allagree",
    )
    df = pd.DataFrame(ds["train"])
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    print(f"[DATA]  {len(sentences)} sentences loaded.  "
          f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    return sentences, labels


# ──────────────────────────────────────────────────────────────
# 2.  Load pre-trained FinBERT
# ──────────────────────────────────────────────────────────────

def load_finbert(model_path: str | None = None,
                 device: str | None = None):
    """Load FinBERT tokenizer and model.

    Parameters
    ----------
    model_path : str, optional
        Path to a fine-tuned checkpoint directory.
        If ``None``, loads the base ``ProsusAI/finbert``.
    device : str, optional
        ``'cuda'`` or ``'cpu'``.  Auto-detected if ``None``.

    Returns
    -------
    tokenizer, model, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    src = model_path or FINBERT_MODEL
    print(f"[LOAD]  Loading FinBERT from {src}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=NUM_LABELS,
    )
    model = model.to(device)
    return tokenizer, model, device


# ──────────────────────────────────────────────────────────────
# 3.  Fine-tuning
# ──────────────────────────────────────────────────────────────

def fine_tune_finbert(
    sentences: list[str],
    labels: list[int],
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    test_size: float = 0.15,
    device: str | None = None,
) -> tuple:
    """Fine-tune ProsusAI/finbert on 3-class sentiment classification.

    Parameters
    ----------
    sentences : list[str]
        Input texts.
    labels : list[int]
        Integer labels (0=negative, 1=neutral, 2=positive).
    epochs : int
        Number of training epochs (default 3).
    lr : float
        Learning rate (default 2e-5).
    batch_size : int
        Mini-batch size (default 16).
    test_size : float
        Fraction held out for evaluation.
    device : str, optional
        Auto-detected if ``None``.

    Returns
    -------
    model : AutoModelForSequenceClassification
        Fine-tuned model (best checkpoint).
    tokenizer : AutoTokenizer
    eval_metrics : dict
        ``{'accuracy', 'precision', 'recall', 'f1', 'report'}``
    history : dict
        ``{'train_loss': [...], 'eval_loss': [...]}``
    """
    tokenizer, model, device = load_finbert(device=device)

    # ── Train / eval split (stratified) ──
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        sentences, labels, test_size=test_size,
        random_state=SEED, stratify=labels,
    )
    print(f"[SPLIT]  Train {len(train_texts)} | Eval {len(eval_texts)}")

    # ── Tokenise ──
    train_enc = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=128, return_tensors="pt")
    eval_enc  = tokenizer(eval_texts,  truncation=True, padding=True,
                          max_length=128, return_tensors="pt")

    train_ds = SentimentDataset(train_enc, train_labels)
    eval_ds  = SentimentDataset(eval_enc,  eval_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False)

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "eval_loss": []}
    best_eval_loss = float('inf')
    best_state = None

    print(f"\n{'═' * 60}")
    print(f"  Fine-tuning FinBERT  |  epochs={epochs}  lr={lr}  "
          f"batch={batch_size}")
    print(f"{'═' * 60}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]",
                          leave=False):
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            labels_b  = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention)
            loss = criterion(outputs.logits, labels_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * input_ids.size(0)

        train_loss = running / len(train_ds)

        # ── Evaluate ──
        model.eval()
        eval_loss_sum = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                labels_b  = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention)
                eval_loss_sum += criterion(outputs.logits, labels_b).item() * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels_b.cpu().tolist())

        eval_loss = eval_loss_sum / len(eval_ds)
        acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)

        # ── Track best checkpoint ──
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"  Epoch {epoch}/{epochs}  │  "
                  f"Train Loss {train_loss:.4f}  │  "
                  f"Eval Loss {eval_loss:.4f}  │  "
                  f"Eval Acc {acc:.4f}  ★ best")
        else:
            print(f"  Epoch {epoch}/{epochs}  │  "
                  f"Train Loss {train_loss:.4f}  │  "
                  f"Eval Loss {eval_loss:.4f}  │  "
                  f"Eval Acc {acc:.4f}")

    # ── Restore best weights ──
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  ✓ Restored best checkpoint (eval_loss={best_eval_loss:.4f})")

    # ── Final evaluation metrics (re-evaluate with best weights) ──
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            labels_b  = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_b.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=[0, 1, 2],
    )
    report_str = classification_report(
        all_labels, all_preds,
        target_names=["negative", "neutral", "positive"],
    )
    print(f"\n{report_str}")

    eval_metrics = {
        "accuracy":  acc,
        "precision": precision.tolist(),
        "recall":    recall.tolist(),
        "f1":        f1.tolist(),
        "report":    report_str,
    }

    # ── Save ──
    os.makedirs(FINETUNED_DIR, exist_ok=True)
    model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"[SAVED]  Fine-tuned model → {FINETUNED_DIR}")

    return model, tokenizer, eval_metrics, history


# ──────────────────────────────────────────────────────────────
# 4.  Single-text inference
# ──────────────────────────────────────────────────────────────

def get_sentiment_score(
    text: str,
    tokenizer=None,
    model=None,
    device: str | None = None,
) -> dict:
    """Predict sentiment for a **single** headline string.

    Parameters
    ----------
    text : str
        Financial headline / sentence.
    tokenizer, model : optional
        Pre-loaded tokenizer & model.  If ``None``, they are loaded
        from the fine-tuned checkpoint (or base FinBERT if not found).
    device : str, optional

    Returns
    -------
    dict
        ``{'sentiment_label': str, 'sentiment_score': float, 'class_id': int}``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None or model is None:
        src = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else None
        tokenizer, model, device = load_finbert(model_path=src, device=device)

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    cls_id = torch.argmax(probs).item()

    return {
        "sentiment_label": LABEL_MAP[cls_id],
        "sentiment_score": probs[cls_id].item(),
        "class_id":        cls_id,
    }


# ──────────────────────────────────────────────────────────────
# 5.  Batch inference
# ──────────────────────────────────────────────────────────────

def batch_sentiment(
    headlines: list[str],
    tokenizer=None,
    model=None,
    device: str | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Predict sentiment for a list of headlines.

    Parameters
    ----------
    headlines : list[str]
        Texts to classify.
    tokenizer, model : optional
        Pre-loaded objects.
    device : str, optional
    batch_size : int

    Returns
    -------
    pd.DataFrame
        Columns: ``text``, ``sentiment_label``, ``sentiment_score``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None or model is None:
        src = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else None
        tokenizer, model, device = load_finbert(model_path=src, device=device)

    model.eval()
    records = []

    for i in tqdm(range(0, len(headlines), batch_size), desc="Batch sentiment"):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        for j, pred in enumerate(preds):
            records.append({
                "text":             batch[j],
                "sentiment_label":  LABEL_MAP[pred.item()],
                "sentiment_score":  probs[j][pred].item(),
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# 6.  Daily aggregation
# ──────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """Compute daily average sentiment score.

    Maps labels to numeric values (positive=+1, neutral=0,
    negative=−1), multiplies by confidence score, and averages
    per date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *date_col*, *score_col*, and *label_col*.
    date_col : str
    score_col : str
    label_col : str

    Returns
    -------
    pd.DataFrame
        Indexed by date with column ``daily_sentiment``.
    """
    df = df.copy()
    df["_numeric"] = df[label_col].map(SENTIMENT_NUMERIC)
    df["_weighted"] = df["_numeric"] * df[score_col]

    daily = (
        df.groupby(date_col)["_weighted"]
        .mean()
        .rename("daily_sentiment")
        .to_frame()
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    print(f"[AGGREGATE]  {len(daily)} unique dates, "
          f"sentiment range [{daily['daily_sentiment'].min():.3f}, "
          f"{daily['daily_sentiment'].max():.3f}]")
    return daily
