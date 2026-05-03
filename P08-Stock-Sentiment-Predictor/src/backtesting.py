"""
backtesting.py — Simulate a prediction-driven trading strategy and compare
                 against a Buy-and-Hold baseline.
P08: Stock Market Sentiment and Price Movement Predictor

Functions
---------
backtest(prices, predictions, ...)
    Run a long-only strategy driven by binary model signals.
plot_backtest(result, ticker, ...)
    Plot cumulative returns: strategy vs Buy-and-Hold.
print_summary(result, ticker)
    Print a formatted metrics table.
compare_stocks(stock_results)
    Side-by-side comparison table + combined plot for ≥ 2 stocks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import RESULTS_DIR


# ──────────────────────────────────────────────────────────────
# 1.  Core backtest engine
# ──────────────────────────────────────────────────────────────

def backtest(
    prices: np.ndarray | pd.Series,
    predictions: np.ndarray,
    true_labels: np.ndarray | None = None,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.01,
    dates: pd.DatetimeIndex | None = None,
) -> dict:
    """Simulate a long-only strategy from binary model predictions.

    Strategy logic
    ~~~~~~~~~~~~~~
    * **predict 1 (price up)** → buy / hold the stock for that day.
    * **predict 0 (price down)** → sell / stay in cash for that day.

    Daily return on a "buy" day equals the stock's actual daily return;
    on a "cash" day the return is zero.

    Parameters
    ----------
    prices : array-like
        Actual closing prices aligned with the test period.
    predictions : np.ndarray
        Binary model predictions (1 = up, 0 = down).
    true_labels : np.ndarray, optional
        True direction labels — used only for accuracy logging.
    initial_capital : float
        Starting portfolio value.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation (default 0.01).
    dates : pd.DatetimeIndex, optional
        Date index for time-axis plotting.

    Returns
    -------
    dict
        Keys: ``strategy_cum``, ``buyhold_cum``, ``dates``,
        ``strategy_return``, ``buyhold_return``, ``sharpe_ratio``,
        ``max_drawdown``, ``daily_strat_returns``, ``predictions``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    # Daily simple returns
    daily_returns = np.diff(prices) / prices[:-1]          # length N-1
    signals       = predictions[1:]                         # align with returns

    # Truncate to matching length
    n = min(len(daily_returns), len(signals))
    daily_returns = daily_returns[:n]
    signals       = signals[:n]

    # Strategy returns: earn market return only when signal == 1
    strat_returns = daily_returns * signals

    # Cumulative portfolio values
    strat_cum = initial_capital * np.cumprod(1 + strat_returns)
    bh_cum    = initial_capital * np.cumprod(1 + daily_returns)

    # ── Metrics ──
    strategy_return = (strat_cum[-1] / initial_capital) - 1
    buyhold_return  = (bh_cum[-1]    / initial_capital) - 1

    # Sharpe ratio (annualised, excess over risk-free)
    excess = strat_returns - risk_free_rate / 252
    sharpe = (np.mean(excess) / (np.std(excess) + 1e-9)) * np.sqrt(252)

    # Max drawdown
    peak     = np.maximum.accumulate(strat_cum)
    drawdown = (peak - strat_cum) / peak
    max_dd   = np.max(drawdown)

    # Dates alignment
    if dates is not None:
        dates = dates[1: 1 + n]

    return {
        "strategy_cum":         strat_cum,
        "buyhold_cum":          bh_cum,
        "dates":                dates,
        "strategy_return":      strategy_return,
        "buyhold_return":       buyhold_return,
        "sharpe_ratio":         sharpe,
        "max_drawdown":         max_dd,
        "daily_strat_returns":  strat_returns,
        "predictions":          signals,
    }


# ──────────────────────────────────────────────────────────────
# 2.  Plot cumulative returns
# ──────────────────────────────────────────────────────────────

def plot_backtest(
    result: dict,
    ticker: str = "STOCK",
    save: bool = False,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot cumulative portfolio value: strategy vs Buy-and-Hold.

    Parameters
    ----------
    result : dict
        Output of :func:`backtest`.
    ticker : str
        Stock symbol (used in title).
    save : bool
        If ``True``, save the figure to *save_path*.
    save_path : str, optional
        Defaults to ``results/backtest_{ticker}.png``.
    ax : matplotlib Axes, optional
        If provided, draw on this axes instead of creating a new figure.

    Returns
    -------
    fig or None
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()

    x = result["dates"] if result["dates"] is not None else np.arange(len(result["strategy_cum"]))

    ax.plot(x, result["strategy_cum"],
            label="LSTM Strategy", color="#2563eb", linewidth=1.6)
    ax.plot(x, result["buyhold_cum"],
            label="Buy & Hold", color="#dc2626", linewidth=1.4, linestyle="--")

    ax.set_title(f"{ticker} — Cumulative Returns (Strategy vs Buy & Hold)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date" if result["dates"] is not None else "Trading Day")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    if result["dates"] is not None:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate(rotation=30)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = save_path or os.path.join(RESULTS_DIR, f"backtest_{ticker}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SAVED]  Plot → {path}")

    if own_fig:
        plt.tight_layout()
        return fig
    return None


# ──────────────────────────────────────────────────────────────
# 3.  Summary table
# ──────────────────────────────────────────────────────────────

def print_summary(result: dict, ticker: str = "STOCK"):
    """Print a formatted metrics table for one backtest run.

    Parameters
    ----------
    result : dict
        Output of :func:`backtest`.
    ticker : str
        Stock symbol.
    """
    sr  = result["strategy_return"]
    bhr = result["buyhold_return"]
    sh  = result["sharpe_ratio"]
    mdd = result["max_drawdown"]

    print(f"\n┌{'─' * 48}┐")
    print(f"│  {ticker:^44s}  │")
    print(f"├{'─' * 48}┤")
    print(f"│  Strategy Cumulative Return  :  {sr:>+10.2%}       │")
    print(f"│  Buy & Hold Return           :  {bhr:>+10.2%}       │")
    print(f"│  Sharpe Ratio (rf=1%)        :  {sh:>+10.4f}       │")
    print(f"│  Max Drawdown                :  {mdd:>10.2%}       │")
    print(f"└{'─' * 48}┘")


# ──────────────────────────────────────────────────────────────
# 4.  Multi-stock comparison
# ──────────────────────────────────────────────────────────────

def compare_stocks(
    stock_results: dict[str, dict],
    save: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Side-by-side comparison of backtest results for multiple stocks.

    Parameters
    ----------
    stock_results : dict
        ``{ticker: backtest_result_dict}``.
    save : bool
        Save the combined comparison plot.
    save_path : str, optional
        Defaults to ``results/backtest_plot.png``.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per stock.
    """
    # ── Summary table ──
    rows = []
    for ticker, res in stock_results.items():
        rows.append({
            "Ticker":             ticker,
            "Strategy Return":    f"{res['strategy_return']:+.2%}",
            "Buy&Hold Return":    f"{res['buyhold_return']:+.2%}",
            "Sharpe Ratio":       f"{res['sharpe_ratio']:+.4f}",
            "Max Drawdown":       f"{res['max_drawdown']:.2%}",
        })
    summary_df = pd.DataFrame(rows)

    # ── Print individual summaries ──
    for ticker, res in stock_results.items():
        print_summary(res, ticker)

    # ── Combined plot ──
    n_stocks = len(stock_results)
    fig, axes = plt.subplots(1, n_stocks, figsize=(8 * n_stocks, 5),
                             squeeze=False)

    for i, (ticker, res) in enumerate(stock_results.items()):
        plot_backtest(res, ticker=ticker, ax=axes[0, i])

    fig.suptitle("Backtest Comparison — Strategy vs Buy & Hold",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = save_path or os.path.join(RESULTS_DIR, "backtest_plot.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n[SAVED]  Combined plot → {path}")

    plt.show()

    # ── Print summary table ──
    print(f"\n{'═' * 70}")
    print("  Backtest Comparison Summary")
    print(f"{'═' * 70}")
    print(summary_df.to_string(index=False))
    print(f"{'═' * 70}\n")

    return summary_df
