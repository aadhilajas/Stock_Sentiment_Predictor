# 📈 Stock Sentiment Predictor (P08)
### AI-Powered Stock Market Analysis with LSTM + FinBERT

**B.Tech CSE AI/ML Project** · Lovely Professional University

---

## 🎯 Overview

A production-ready Python ML pipeline that predicts stock price movements by fusing LSTM-based price momentum analysis with FinBERT sentiment classification. The project includes:

- **Data Pipeline**: OHLCV data ingestion, feature engineering, and sequence creation
- **Models**: Stacked LSTM classifiers, FinBERT sentiment analyzer, ensemble fusion model
- **Analytics**: Backtesting framework with strategy evaluation and risk metrics
- **Dashboard**: Modern Streamlit frontend with glassmorphism design and real-time updates

**Project Stack**: PyTorch · TensorFlow · FinBERT · Streamlit · Plotly · Python 3.11+

---

## 📁 Project Structure

```
P08-Stock-Sentiment-Predictor/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_github_push.py               # Git setup helper (legacy)
│
├── src/                               # Core ML pipeline
│   ├── __init__.py
│   ├── config.py                      # Global constants & seeds
│   ├── data_pipeline.py               # OHLCV download & caching
│   ├── feature_engineering.py         # Technical indicators & sequences
│   ├── lstm_model.py                  # LSTM classifier with warmup
│   ├── sentiment_pipeline.py          # FinBERT sentiment analysis
│   ├── fusion_model.py                # 3-way experiment runner
│   └── backtesting.py                 # Strategy simulation
│
├── notebooks/                         # Jupyter execution scripts
│   ├── 01_data_pipeline.ipynb         # Load, engineer, scale data
│   ├── 02_lstm_model.ipynb            # Train LSTM on price features
│   ├── 03_finbert_sentiment.ipynb     # Generate FinBERT scores
│   ├── 04_fusion_evaluation.ipynb     # Compare 3 models + backtest
│   ├── data/                          # Notebook-local data
│   └── results/                       # Notebook outputs
│
├── data/                              # Raw stock data
│   ├── AAPL_ohlcv.csv
│   ├── MSFT_ohlcv.csv
│   ├── GOOGL_ohlcv.csv
│   ├── AMZN_ohlcv.csv
│   ├── TSLA_ohlcv.csv
│   └── download_instructions.md
│
├── results/                           # Trained models & outputs
│   ├── lstm_model.pth                 # Combined LSTM checkpoint
│   ├── exp1_price_only.pth            # Price-only LSTM
│   ├── exp2_sentiment_only.pth        # Sentiment-only MLP
│   ├── exp3_fusion.pth                # Fusion LSTM
│   ├── backtest_AAPL.pth              # AAPL backtest model
│   ├── backtest_TSLA.pth              # TSLA backtest model
│   ├── *.npy                          # Preprocessed sequences
│   ├── *.pkl                          # StandardScaler instances
│   ├── *.png                          # Generated charts
│   ├── experiment_results.csv         # Model metrics
│   ├── backtest_plot.png              # Strategy comparison
│   └── finbert_finetuned/             # Fine-tuned FinBERT weights
│
├── app/                               # Streamlit dashboard
│   └── app.py                         # Modern responsive UI
│
├── run_all.py                         # Execute entire pipeline
└── run_all_notebooks.py               # Batch notebook runner
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline (All 4 Notebooks)

```bash
# Option A: Sequential execution
python run_all_notebooks.py

# Option B: Launch Streamlit to run individual notebooks
streamlit run app/app.py
```

### 3. View the Dashboard

```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the data pipeline
python -m src.data_pipeline

# 5. Launch the dashboard
streamlit run app/app.py
```

## 📊 Stocks Tracked

| Ticker | Company       |
|--------|---------------|
| AAPL   | Apple Inc.    |
| MSFT   | Microsoft     |
| GOOGL  | Alphabet      |
| AMZN   | Amazon        |
| TSLA   | Tesla Inc.    |

**Date range:** 2019-01-01 → 2024-12-31

## 🧠 Model Architecture

### LSTM Classifier
- **Input:** 30-day sliding windows of scaled OHLCV + technical indicators
- **Architecture:** 2-layer stacked LSTM (hidden=128) → FC(64) → Sigmoid
- **Output:** P(price goes up tomorrow)

### FinBERT Sentiment
- Pre-trained `ProsusAI/finbert` for financial text
- Classifies headlines as **positive / negative / neutral**
- Aggregated daily sentiment scores

### Late Fusion
- Logistic regression on `[LSTM_prob, sentiment_score]`
- Also supports weighted-average ensemble

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix & ROC-AUC
- Backtesting: Total Return, Sharpe Ratio, Max Drawdown

## 🛠️ Tech Stack
- **Python 3.10+**
- **PyTorch** — LSTM model
- **HuggingFace Transformers** — FinBERT
- **yfinance** — stock data
- **pandas-ta** — technical indicators
- **Streamlit** — interactive dashboard
- **Plotly / Matplotlib / Seaborn** — visualisation

## 👨‍💻 Author
BTech CSE AI/ML Student — Lovely Professional University

## 📄 License
Academic use only.
