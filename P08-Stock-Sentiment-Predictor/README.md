# P08 Stock Market Sentiment and Price Movement Predictor

End-to-end machine learning project for next-day stock direction prediction using price-based deep learning and financial-news sentiment analysis.

## Overview

This repository implements a four-stage pipeline:

1. Data collection and feature engineering for five equities (AAPL, MSFT, GOOGL, AMZN, TSLA).
2. Sequence modeling with a stacked LSTM on technical and price features.
3. Financial sentiment modeling with FinBERT.
4. Fusion experiments, evaluation, backtesting, and dashboard visualization.

The project is organized for reproducible experimentation and includes a Streamlit dashboard for presentation and analysis.

## Key Features

- Multi-asset OHLCV preprocessing with cached local CSV files.
- Technical indicators including rolling statistics, RSI, and MACD.
- Stacked LSTM training pipeline with validation and class-balance handling.
- FinBERT-based sentiment inference and daily sentiment aggregation.
- Comparative experiments: price-only, sentiment-only, and fusion models.
- Backtesting utilities with strategy return, buy-and-hold return, Sharpe ratio, and drawdown views.
- Streamlit application with per-ticker prediction, sentiment, and backtest sections.

## Repository Structure

```text
P08-Stock-Sentiment-Predictor/
├── app/
│   └── app.py
├── data/
│   ├── AAPL_ohlcv.csv
│   ├── AMZN_ohlcv.csv
│   ├── GOOGL_ohlcv.csv
│   ├── MSFT_ohlcv.csv
│   ├── TSLA_ohlcv.csv
│   └── download_instructions.md
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_lstm_model.ipynb
│   ├── 03_finbert_sentiment.ipynb
│   └── 04_fusion_evaluation.ipynb
├── results/
│   ├── experiment_results.csv
│   ├── daily_sentiment.csv
│   └── finbert_finetuned/
├── src/
│   ├── backtesting.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── fusion_model.py
│   ├── lstm_model.py
│   └── sentiment_pipeline.py
├── GITHUB_PUSH_GUIDE.md
├── PROJECT_SUMMARY.md
├── README.md
├── requirements.txt
└── run_all_notebooks.py
```

## Tech Stack

- Python
- PyTorch
- Transformers (Hugging Face)
- Datasets
- scikit-learn
- pandas and NumPy
- yfinance
- Streamlit and Plotly
- Matplotlib and Seaborn

## Results Snapshot

The latest recorded metrics in results/experiment_results.csv are:

| Experiment | Accuracy | AUC-ROC |
| --- | ---: | ---: |
| Price-Only LSTM | 0.5265 | 0.5336 |
| Sentiment-Only MLP | 0.6252 | 0.6612 |
| Fusion LSTM (Price+Sent) | 0.5265 | 0.5144 |

Interpretation: sentiment-only modeling currently provides the strongest classification performance in this artifact set.

## Quick Start

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the notebook pipeline

```bash
python run_all_notebooks.py
```

### 4) Launch the dashboard

```bash
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

## Reproducibility Notes

- The application reads model and data artifacts from results/.
- If required artifacts are missing, portions of the dashboard may show fallback or empty states.
- For best consistency, run the notebook pipeline before launching the dashboard.

## Academic Context

This project was developed as an AI/ML academic project in the FinTech and algorithmic-trading domain.

## Disclaimer

This repository is for academic and educational use. It is not financial advice.

## License

Academic use only.
