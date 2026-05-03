"""
data_pipeline.py — Download, cache, validate and load stock market OHLCV data.
P08: Stock Market Sentiment and Price Movement Predictor

Functions
---------
download_stock(ticker)     – Download a single ticker via yfinance, with CSV caching.
validate_data(df, ticker)  – Check for missing values, zero-volume rows, and min length.
load_all_stocks()          – Return a dict {ticker: DataFrame} for every configured ticker.
run_pipeline()             – End-to-end: download → validate → report for all tickers.
"""

import os
import pandas as pd
import yfinance as yf

from src.config import STOCKS, START_DATE, END_DATE, DATA_RAW_DIR

# Minimum number of trading-day rows we expect for ≈6 years of data
MIN_ROWS = 500


# ──────────────────────────────────────────────────────────────
# 1.  Download (with caching)
# ──────────────────────────────────────────────────────────────

def download_stock(ticker: str,
                   start: str = START_DATE,
                   end: str = END_DATE,
                   data_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """Download OHLCV data for *ticker* via yfinance, or load from cache.

    If ``data/{ticker}_ohlcv.csv`` already exists on disk the function
    loads and returns it directly, skipping the network call.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``'AAPL'``.
    start : str
        Start date in ``'YYYY-MM-DD'`` format.
    end : str
        End date in ``'YYYY-MM-DD'`` format.
    data_dir : str
        Directory in which to store / look for CSV files.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns:
        Open, High, Low, Close, Adj Close, Volume.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{ticker}_ohlcv.csv")

    # ── Cache hit ──
    if os.path.exists(csv_path):
        print(f"[CACHE]  {ticker} — loading from {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        return df

    # ── Cache miss → download ──
    print(f"[DOWNLOAD]  {ticker} — fetching from Yahoo Finance "
          f"({start} → {end}) ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"[WARNING]  {ticker} — yfinance returned an empty DataFrame!")
        return df

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Persist to CSV
    df.to_csv(csv_path)
    print(f"[SAVED]  {ticker} — {len(df)} rows → {csv_path}")
    return df


# ──────────────────────────────────────────────────────────────
# 2.  Validation
# ──────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, ticker: str = "UNKNOWN") -> bool:
    """Run sanity checks on a stock OHLCV DataFrame.

    Checks performed:
    1. **Minimum length** — at least ``MIN_ROWS`` (500) rows.
    2. **Missing values** — warns if any NaN/null cells exist.
    3. **Zero-volume rows** — warns if any rows have Volume == 0.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (expects a ``'Volume'`` column).
    ticker : str
        Ticker symbol, used only for logging.

    Returns
    -------
    bool
        ``True`` if all checks pass, ``False`` otherwise.
    """
    passed = True
    print(f"\n{'─' * 50}")
    print(f"  Validating {ticker}  ({len(df)} rows)")
    print(f"{'─' * 50}")

    # Check 1 — sufficient length
    if len(df) < MIN_ROWS:
        print(f"  ✗ FAIL  Row count {len(df)} < minimum {MIN_ROWS}")
        passed = False
    else:
        print(f"  ✓ PASS  Row count {len(df)} ≥ {MIN_ROWS}")

    # Check 2 — missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"  ✗ WARN  {total_missing} missing value(s) found:")
        for col, cnt in missing[missing > 0].items():
            print(f"          {col}: {cnt}")
        passed = False
    else:
        print(f"  ✓ PASS  No missing values")

    # Check 3 — zero-volume rows
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > 0:
            print(f"  ✗ WARN  {zero_vol} row(s) with Volume == 0")
            passed = False
        else:
            print(f"  ✓ PASS  No zero-volume rows")
    else:
        print(f"  ✗ WARN  'Volume' column not found — skipping volume check")
        passed = False

    return passed


# ──────────────────────────────────────────────────────────────
# 3.  Load all stocks
# ──────────────────────────────────────────────────────────────

def load_all_stocks(tickers: list = None,
                    data_dir: str = DATA_RAW_DIR) -> dict:
    """Download (or load from cache) OHLCV data for every configured ticker.

    Parameters
    ----------
    tickers : list, optional
        List of ticker symbols. Defaults to ``config.STOCKS``.
    data_dir : str
        Directory containing (or that will contain) the CSV files.

    Returns
    -------
    dict
        ``{ticker: pd.DataFrame}`` mapping for each ticker.
    """
    if tickers is None:
        tickers = STOCKS

    stock_data: dict[str, pd.DataFrame] = {}
    print(f"\n{'═' * 55}")
    print(f"  Loading {len(tickers)} stocks: {', '.join(tickers)}")
    print(f"{'═' * 55}\n")

    for ticker in tickers:
        df = download_stock(ticker, data_dir=data_dir)
        stock_data[ticker] = df

    print(f"\n[INFO]  Loaded {len(stock_data)} tickers successfully.\n")
    return stock_data


# ──────────────────────────────────────────────────────────────
# 4.  Full pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(tickers: list = None) -> dict:
    """End-to-end pipeline: download → validate → report.

    Downloads data for every ticker (with caching), validates each
    DataFrame, and prints a summary table.

    Parameters
    ----------
    tickers : list, optional
        Ticker symbols. Defaults to ``config.STOCKS``.

    Returns
    -------
    dict
        ``{ticker: pd.DataFrame}`` mapping for each ticker.
    """
    stock_data = load_all_stocks(tickers)

    # ── Validate each ticker ──
    results = {}
    for ticker, df in stock_data.items():
        ok = validate_data(df, ticker)
        results[ticker] = ok

    # ── Summary ──
    print(f"\n{'═' * 55}")
    print("  Validation Summary")
    print(f"{'═' * 55}")
    for ticker, ok in results.items():
        status = "✓ PASS" if ok else "✗ ISSUES"
        rows = len(stock_data[ticker])
        date_range = (f"{stock_data[ticker].index.min().date()} → "
                      f"{stock_data[ticker].index.max().date()}")
        print(f"  {ticker:>5}  {status:>10}  {rows:>5} rows  {date_range}")
    print(f"{'═' * 55}\n")

    return stock_data


# ──────────────────────────────────────────────────────────────
# 5.  CLI entry-point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
