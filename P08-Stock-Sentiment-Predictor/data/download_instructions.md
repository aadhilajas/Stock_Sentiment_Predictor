# Data Download Instructions

## Overview
This project uses two primary data sources:

### 1. Stock Price Data (via yfinance)
Stock price data is downloaded automatically via the `yfinance` library in the data pipeline.

**Stocks tracked:** AAPL, MSFT, GOOGL, AMZN, TSLA  
**Date range:** 2019-01-01 to 2024-12-31  
**Features:** Open, High, Low, Close, Adj Close, Volume

No manual download is required — run `src/data_pipeline.py` or `notebooks/01_data_pipeline.ipynb`.

### 2. Financial News / Sentiment Data
Sentiment analysis is performed using **FinBERT** (ProsusAI/finbert) on financial news headlines.

Possible news sources:
- [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) (HuggingFace Datasets)
- Twitter/X financial tweets (via API or Kaggle datasets)
- Reuters / Bloomberg headlines (if accessible)

### Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python src/data_pipeline.py
```

### Storage
- Raw data will be stored in `data/raw/`
- Processed data will be stored in `data/processed/`

These directories are created automatically by the pipeline.
