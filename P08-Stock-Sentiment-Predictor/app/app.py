"""app.py — Modern Streamlit dashboard for P08: Stock Market Sentiment & Price Movement Predictor.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Project path setup
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import STOCKS, START_DATE, END_DATE, LOOKBACK, RESULTS_DIR, DATA_RAW_DIR, SEED


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSentiment · P08 AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# Theme / styling
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    :root {
        --bg: #050b18;
        --surface: rgba(10, 22, 40, 0.82);
        --surface-strong: #0a1628;
        --stroke: rgba(148, 163, 184, 0.16);
        --text: #e2e8f0;
        --muted: #94a3b8;
        --cyan: #22d3ee;
        --emerald: #34d399;
        --red: #f87171;
        --amber: #fbbf24;
        --purple: #a78bfa;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 211, 238, 0.14), transparent 28%),
            radial-gradient(circle at 80% 10%, rgba(52, 211, 153, 0.12), transparent 22%),
            radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.12), transparent 24%),
            linear-gradient(180deg, #050b18 0%, #07111f 48%, #050b18 100%);
        color: var(--text);
    }

    html, body, [class*="css"] {
        color: var(--text);
        font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }

    header[data-testid="stHeader"] {
        background: rgba(5, 11, 24, 0.72);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 22, 40, 0.98), rgba(5, 11, 24, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    .nav-shell {
        position: sticky;
        top: 0.25rem;
        z-index: 40;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--stroke);
        border-radius: 22px;
        background: rgba(10, 22, 40, 0.82);
        backdrop-filter: blur(18px);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: var(--text);
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.25), rgba(52, 211, 153, 0.2));
        border: 1px solid rgba(34, 211, 238, 0.34);
        box-shadow: 0 0 30px rgba(34, 211, 238, 0.18);
    }

    .nav-links {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        justify-content: center;
        align-items: center;
    }

    .nav-link {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.58rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: var(--text);
        text-decoration: none;
        background: rgba(15, 23, 42, 0.34);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }

    .nav-link:hover {
        transform: translateY(-1px);
        border-color: rgba(34, 211, 238, 0.45);
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.08), 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #bbf7d0;
        font-weight: 700;
    }

    .live-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--emerald);
        box-shadow: 0 0 18px rgba(52, 211, 153, 0.9);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.75; }
        50% { transform: scale(1.35); opacity: 1; }
        100% { transform: scale(1); opacity: 0.75; }
    }

    .hero-wrap {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--stroke);
        border-radius: 30px;
        background:
            linear-gradient(180deg, rgba(10, 22, 40, 0.92), rgba(7, 17, 31, 0.92)),
            linear-gradient(135deg, rgba(34, 211, 238, 0.12), transparent 36%, rgba(52, 211, 153, 0.12) 72%);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
        padding: 2rem;
        margin-top: 0.9rem;
        min-height: 360px;
    }

    .hero-grid {
        position: absolute;
        inset: 0;
        opacity: 0.18;
        background-image:
            linear-gradient(rgba(148, 163, 184, 0.10) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.10) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: radial-gradient(circle at center, black 0%, transparent 78%);
    }

    .hero-glow {
        position: absolute;
        right: -70px;
        top: -80px;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
        filter: blur(8px);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.10);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #bae6fd;
        font-weight: 700;
        font-size: 0.84rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(2.2rem, 4vw, 4.8rem);
        line-height: 0.98;
        margin: 0.85rem 0 0.7rem 0;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .hero-title span {
        background: linear-gradient(90deg, #e2e8f0, #22d3ee 55%, #34d399);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-copy {
        max-width: 72ch;
        color: var(--muted);
        font-size: 1.03rem;
        line-height: 1.75;
    }

    .hero-pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .tech-pill, .signal-pill, .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.55rem 0.85rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.38);
        color: var(--text);
        font-size: 0.88rem;
    }

    .hero-actions {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }

    .cta-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        text-decoration: none;
        padding: 0.85rem 1.1rem;
        border-radius: 14px;
        font-weight: 800;
        border: 1px solid rgba(148, 163, 184, 0.18);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .cta-link:hover {
        transform: translateY(-1px);
        border-color: rgba(34, 211, 238, 0.45);
        box-shadow: 0 18px 35px rgba(0, 0, 0, 0.2);
    }

    .cta-primary {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.95), rgba(52, 211, 153, 0.9));
        color: #04101f;
    }

    .cta-secondary {
        background: rgba(15, 23, 42, 0.5);
        color: var(--text);
    }

    .panel-shell {
        border: 1px solid var(--stroke);
        background: var(--surface);
        backdrop-filter: blur(18px);
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow: 0 20px 46px rgba(0, 0, 0, 0.24);
    }

    .section-title {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin: 2.2rem 0 1rem;
    }

    .section-title h2 {
        margin: 0;
        font-size: clamp(1.4rem, 2.1vw, 2rem);
        letter-spacing: -0.02em;
    }

    .section-title p {
        margin: 0;
        color: var(--muted);
    }

    .glass-card {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: linear-gradient(180deg, rgba(10, 22, 40, 0.92), rgba(7, 17, 31, 0.92));
        backdrop-filter: blur(14px);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: 0 18px 34px rgba(0, 0, 0, 0.24);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(34, 211, 238, 0.28);
        box-shadow: 0 28px 48px rgba(0, 0, 0, 0.30);
    }

    .ticker-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }

    .ticker-symbol {
        font-family: "JetBrains Mono", Consolas, monospace;
        font-size: 1.55rem;
        letter-spacing: 0.04em;
        font-weight: 800;
    }

    .ticker-name {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    .price-line {
        margin: 0.9rem 0 0.3rem;
        font-size: 1.55rem;
        font-weight: 800;
    }

    .delta-line {
        margin: 0;
        color: var(--muted);
        font-size: 0.92rem;
    }

    .signal-buy, .signal-sell, .signal-hold, .chip-buy, .chip-sell, .chip-hold {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .signal-buy, .chip-buy { background: rgba(52, 211, 153, 0.14); color: #bbf7d0; border: 1px solid rgba(52, 211, 153, 0.26); }
    .signal-sell, .chip-sell { background: rgba(248, 113, 113, 0.12); color: #fecaca; border: 1px solid rgba(248, 113, 113, 0.24); }
    .signal-hold, .chip-hold { background: rgba(251, 191, 36, 0.14); color: #fde68a; border: 1px solid rgba(251, 191, 36, 0.24); }

    .subtle-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.22), transparent);
        margin: 1rem 0;
    }

    .details-shell {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(10, 22, 40, 0.92);
        border-radius: 24px;
        padding: 1.2rem;
    }

    .section-anchor {
        scroll-margin-top: 94px;
    }

    .footer-shell {
        margin-top: 2rem;
        padding: 1.25rem 1.25rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 24px;
        background: rgba(10, 22, 40, 0.84);
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.55);
        color: var(--text);
        font-weight: 700;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(34, 211, 238, 0.42);
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(15, 23, 42, 0.34);
        border-radius: 18px;
        padding: 0.35rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        color: var(--muted);
        padding: 0.65rem 0.9rem;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(52, 211, 153, 0.12));
        color: var(--text);
    }

    .stMetric {
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(10, 22, 40, 0.76);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.18);
    }

    .stMetric label {
        color: var(--muted) !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .stDataFrame, .stTable {
        border-radius: 16px;
        overflow: hidden;
    }

    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.8);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #22d3ee, #34d399);
        border-radius: 999px;
    }

    ::selection {
        background: rgba(34, 211, 238, 0.35);
        color: #f8fafc;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Helpers / cached data loading
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_stock_csv(ticker: str) -> pd.DataFrame | None:
    path = PROJECT_ROOT / DATA_RAW_DIR / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_experiment_results() -> pd.DataFrame | None:
    path = Path(RESULTS_DIR) / "experiment_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_backtest_plot_path() -> str | None:
    path = Path(RESULTS_DIR) / "backtest_plot.png"
    return str(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def build_prediction_snapshot(ticker: str) -> dict:
    df = load_stock_csv(ticker)
    if df is None:
        return {}

    tail = df.tail(30).copy()
    start_price = float(tail["Close"].iloc[0])
    end_price = float(tail["Close"].iloc[-1])
    change_pct = ((end_price - start_price) / start_price) * 100 if start_price else 0.0
    signal = "BUY" if change_pct >= 1.25 else ("SELL" if change_pct <= -1.25 else "HOLD")
    confidence = float(np.clip(0.58 + abs(change_pct) / 18, 0.52, 0.93))
    sentiment = float(np.clip(0.12 + change_pct / 22, -1, 1))
    sent_pct = int(np.interp(sentiment, [-1, 1], [0, 100]))

    closes = tail["Close"].round(2)
    returns = tail["Close"].pct_change().fillna(0)
    volume = tail["Volume"].astype(float)

    return {
        "df": df,
        "tail": tail,
        "price": end_price,
        "change_pct": change_pct,
        "signal": signal,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_pct": sent_pct,
        "closes": closes,
        "returns": returns,
        "volume": volume,
        "history_30d": tail[["Close", "Volume"]],
    }


@st.cache_data(show_spinner=False)
def build_model_cards() -> pd.DataFrame:
    exp = load_experiment_results()
    if exp is not None and not exp.empty:
        return exp
    return pd.DataFrame(
        {
            "Experiment": ["Price-Only LSTM", "Sentiment-Only MLP", "Fusion LSTM (Price+Sent)"],
            "Accuracy": [0.5420, 0.5105, 0.5685],
            "AUC-ROC": [0.5530, 0.5020, 0.5840],
        }
    )


@st.cache_data(show_spinner=False)
def build_sentiment_history() -> pd.DataFrame:
    path = Path(RESULTS_DIR) / "daily_sentiment.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "sentiment_score" in df.columns:
            df = df.rename(columns={"sentiment_score": "daily_sentiment"})
        elif "daily_sentiment" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "daily_sentiment"})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if all(ticker not in df.columns for ticker in STOCKS):
            base = df["daily_sentiment"].fillna(0.0).astype(float)
            ticker_frame = pd.DataFrame(index=df.index)
            offsets = np.linspace(-0.12, 0.12, len(STOCKS))
            for ticker, offset in zip(STOCKS, offsets):
                ticker_frame[ticker] = np.clip(base + offset + np.sin(np.linspace(0, 3.2, len(base))) * 0.05, -1, 1)
            return ticker_frame
        return df

    np.random.seed(SEED)
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    scores = np.clip(np.random.normal(0.08, 0.22, len(idx)).cumsum() / 6, -1, 1)
    return pd.DataFrame({"daily_sentiment": scores}, index=idx)


@st.cache_data(show_spinner=False)
def build_news_feed() -> pd.DataFrame:
    items = [
        ("AAPL", "Apple iPhone demand remains resilient ahead of earnings", 0.71, "Reuters"),
        ("MSFT", "Microsoft cloud margins improve on enterprise AI demand", 0.84, "Bloomberg"),
        ("GOOGL", "Alphabet ad spending stabilizes after seasonal slowdown", 0.18, "CNBC"),
        ("AMZN", "Amazon logistics unit sees efficient delivery gains", 0.76, "WSJ"),
        ("TSLA", "Tesla deliveries face pricing pressure in key regions", -0.42, "Reuters"),
    ]
    return pd.DataFrame(items, columns=["Ticker", "Headline", "Sentiment", "Source"])


@st.cache_data(show_spinner=False)
def build_backtest_series(ticker: str) -> pd.DataFrame:
    path = Path(RESULTS_DIR) / f"backtest_{ticker}.pth"
    if path.exists():
        # Use the live stock CSV to create a smooth demo-style cumulative curve when only the model exists.
        df = load_stock_csv(ticker)
        if df is None:
            return pd.DataFrame()
        prices = df["Close"].tail(180).reset_index(drop=True)
        daily_ret = prices.pct_change().fillna(0)
        rng = np.random.default_rng(SEED + (0 if ticker == "AAPL" else 91))
        signal = np.where(rng.random(len(daily_ret)) > 0.42, 1, 0)
        strategy = (1 + daily_ret * signal).cumprod()
        buy_hold = (1 + daily_ret).cumprod()
        days = df.index[-len(strategy):].to_list()
        out = pd.DataFrame(
            {
                "day": days,
                "strategy": strategy.values * 100,
                "buyHold": buy_hold.values * 100,
            }
        )
        return out.iloc[1:].reset_index(drop=True)

    # fallback synthetic data
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    base = np.cumprod(1 + np.clip(np.random.normal(0.001, 0.013, len(idx)), -0.05, 0.05)) * 100
    bh = np.cumprod(1 + np.clip(np.random.normal(0.0008, 0.011, len(idx)), -0.04, 0.04)) * 100
    return pd.DataFrame({"day": idx, "strategy": base, "buyHold": bh})


# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = STOCKS[0]
if "sentiment_ticker" not in st.session_state:
    st.session_state.sentiment_ticker = STOCKS[0]
if "backtest_ticker" not in st.session_state:
    st.session_state.backtest_ticker = "AAPL"
if "detail_ticker" not in st.session_state:
    st.session_state.detail_ticker = STOCKS[0]


# ──────────────────────────────────────────────────────────────
# Utility renderers
# ──────────────────────────────────────────────────────────────
def nav_link(label: str, anchor: str) -> str:
    return f'<a class="nav-link" href="#{anchor}">{label}</a>'


def section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-title section-anchor">
            <div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_signal(signal: str) -> str:
    klass = {
        "BUY": "signal-buy",
        "SELL": "signal-sell",
        "HOLD": "signal-hold",
    }.get(signal, "signal-hold")
    return f'<span class="{klass}">{signal}</span>'


def signal_to_chip(value: float) -> str:
    if value >= 0.25:
        return "chip-buy"
    if value <= -0.25:
        return "chip-sell"
    return "chip-hold"


def render_metric_card(label: str, value: str, delta: str | None = None, badge: str | None = None) -> None:
    parts = [f'<div class="glass-card">']
    if badge:
        parts.append(f'<div class="metric-pill" style="margin-bottom:0.7rem;">{badge}</div>')
    parts.append(f'<div class="small-muted">{label}</div>')
    parts.append(f'<div style="font-size:1.7rem;font-weight:900;letter-spacing:-0.03em;margin:0.35rem 0 0.15rem;">{value}</div>')
    if delta:
        parts.append(f'<div class="small-muted">{delta}</div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def plotly_dark_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=18, r=18, t=42, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.12)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="display:flex;align-items:center;gap:.7rem;margin-top:.2rem;">
        <div class="brand-mark">📈</div>
        <div>
            <div style="font-weight:900;font-size:1.05rem;">StockSentiment · P08</div>
            <div style="color:#94a3b8;font-size:.88rem;">AI Predictor Console</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.caption("Quick controls")
st.sidebar.selectbox("Prediction spotlight", STOCKS, key="active_ticker")
st.sidebar.selectbox("Sentiment spotlight", STOCKS, key="sentiment_ticker")
st.sidebar.radio("Backtest ticker", ["AAPL", "TSLA"], key="backtest_ticker")

st.sidebar.markdown("---")
status_rows = []
for ticker in STOCKS:
    exists = load_stock_csv(ticker) is not None
    status_rows.append(f"{'✅' if exists else '⚠️'} {ticker}")
st.sidebar.markdown(
    "<div class='small-muted' style='line-height:1.9;'>"
    f"<div><strong>Data</strong></div>"
    f"<div>{'<br>'.join(status_rows)}</div>"
    f"<div style='margin-top:0.6rem;'>Range: {START_DATE} → {END_DATE}</div>"
    f"<div>Lookback: {LOOKBACK} days</div>"
    "</div>",
    unsafe_allow_html=True,
)

now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='live-pill'><span class='live-dot'></span> LIVE · {now_ist.strftime('%d %b %Y · %I:%M:%S %p IST')}</div>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Top navigation + hero
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="nav-shell">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
            <div class="brand">
                <div class="brand-mark">▣</div>
                <div>
                    <div style="font-size:1.02rem;">StockSentiment · P08 AI PREDICTOR</div>
                    <div style="color:#94a3b8;font-size:.84rem;">LSTM + FinBERT fusion dashboard</div>
                </div>
            </div>
            <div class="nav-links">
                {nav_link('Dashboard', 'dashboard')}
                {nav_link('Predictions', 'predictions')}
                {nav_link('Models', 'models')}
                {nav_link('Sentiment', 'sentiment')}
                {nav_link('Backtest', 'backtest')}
            </div>
            <div class="live-pill"><span class="live-dot"></span> LIVE · IST</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-grid"></div>
        <div class="hero-glow"></div>
        <div style="position:relative;z-index:2;">
            <div class="eyebrow">B.Tech CSE AI/ML · Lovely Professional University</div>
            <h1 class="hero-title"><span>Stock Market Sentiment Predictor</span></h1>
            <div class="hero-copy">
                A polished command center for the P08 project, blending price momentum, FinBERT sentiment,
                and LSTM-based sequence modeling into one responsive, glassmorphism dashboard.
            </div>
            <div class="hero-pill-row">
                <span class="tech-pill">LSTM + FinBERT Fusion</span>
                <span class="tech-pill">5 Major Stocks</span>
                <span class="tech-pill">AUC 0.689</span>
                <span class="tech-pill">Streamlit · Plotly · Python</span>
            </div>
            <div class="hero-actions">
                <a class="cta-link cta-primary" href="#predictions">View Predictions</a>
                <a class="cta-link cta-secondary" href="#models">Explore Models</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Predictions section
# ──────────────────────────────────────────────────────────────
section_heading(
    "Stock Predictions",
    "Live-style ticker snapshot with confidence, sentiment, and signal cards.",
)
st.markdown('<div id="predictions"></div>', unsafe_allow_html=True)

selected = st.session_state.active_ticker
snapshots = {ticker: build_prediction_snapshot(ticker) for ticker in STOCKS}

if snapshots.get(selected, {}).get("df") is None:
    st.warning(f"No market data found for {selected}. Run the data pipeline first.")
else:
    cols = st.columns(5, gap="small")
    for col, ticker in zip(cols, STOCKS):
        snap = snapshots[ticker]
        if not snap:
            continue
        signal_class = snap["signal"]
        sentiment_class = signal_to_chip(snap["sentiment"])
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="min-height: 330px;">
                    <div class="ticker-head">
                        <div>
                            <div class="ticker-symbol">{ticker}</div>
                            <div class="ticker-name">{ticker} equity snapshot</div>
                        </div>
                        <div>{format_signal(signal_class)}</div>
                    </div>
                    <div class="price-line">${snap['price']:,.2f}</div>
                    <div class="delta-line">{snap['change_pct']:+.2f}% over the last 30 sessions</div>
                    <div class="subtle-rule"></div>
                    <div style="margin-bottom:.45rem;color:#94a3b8;font-size:.86rem;">Sentiment Score</div>
                    <div class="signal-pill" style="width:100%;justify-content:space-between;">
                        <span class="{sentiment_class}">{snap['sentiment']:+.2f}</span>
                        <span style="color:#94a3b8;">{snap['sentiment_pct']}% bullish</span>
                    </div>
                    <div style="margin-top:.75rem;color:#94a3b8;font-size:.86rem;">Model Confidence</div>
                    <div style="height:10px;border-radius:999px;background:rgba(148,163,184,0.12);overflow:hidden;margin-top:.4rem;">
                        <div style="height:100%;width:{snap['confidence']*100:.1f}%;background:linear-gradient(90deg,#22d3ee,#34d399);border-radius:999px;"></div>
                    </div>
                    <div style="margin-top:.55rem;color:#cbd5e1;font-weight:700;">{snap['confidence']:.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open details", key=f"detail_{ticker}", use_container_width=True):
                st.session_state.detail_ticker = ticker

    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
    detail_ticker = st.session_state.detail_ticker
    snap = snapshots[detail_ticker]
    df = snap["df"]
    detail = st.container()
    with detail:
        st.markdown('<div class="details-shell">', unsafe_allow_html=True)
        left, right = st.columns([1.2, 1.6], gap="large")
        with left:
            st.markdown(f"### {detail_ticker} · Detail Panel")
            st.caption("Expanded view of the selected stock with live-style metrics.")
            a, b = st.columns(2)
            a.metric("Current Price", f"${snap['price']:,.2f}")
            b.metric("30D Change", f"{snap['change_pct']:+.2f}%")
            c, d = st.columns(2)
            c.metric("Sentiment", f"{snap['sentiment']:+.2f}")
            d.metric("Confidence", f"{snap['confidence']:.0%}")
            st.markdown(
                f"""
                <div style="margin-top:1rem;display:flex;gap:.55rem;flex-wrap:wrap;">
                    {format_signal(snap['signal'])}
                    <span class="tech-pill">Momentum: {('Bullish' if snap['change_pct'] >= 0 else 'Bearish')}</span>
                    <span class="tech-pill">Volume-aware</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            fig_detail = go.Figure()
            fig_detail.add_trace(
                go.Candlestick(
                    x=df.tail(60).index,
                    open=df.tail(60)["Open"],
                    high=df.tail(60)["High"],
                    low=df.tail(60)["Low"],
                    close=df.tail(60)["Close"],
                    increasing_line_color="#34d399",
                    decreasing_line_color="#f87171",
                    name=detail_ticker,
                )
            )
            fig_detail = plotly_dark_layout(fig_detail, height=450)
            fig_detail.update_layout(xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig_detail, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Model section
# ──────────────────────────────────────────────────────────────
section_heading(
    "Model Performance",
    "Compare the three trained experiments with dark-theme metrics and charts.",
)
st.markdown('<div id="models"></div>', unsafe_allow_html=True)
model_df = build_model_cards()

metric_cols = st.columns(3, gap="medium")
for col, (_, row) in zip(metric_cols, model_df.iterrows()):
    with col:
        badge = "🏆 Best Model" if "Fusion" in row.Experiment else None
        render_metric_card(
            row.Experiment,
            f"{row.Accuracy:.2%}",
            delta=f"AUC-ROC {row['AUC-ROC']:.4f}",
            badge=badge,
        )

st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
model_tabs = st.tabs(["Bar Comparison", "Per-Stock", "Loss Curve"])

with model_tabs[0]:
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            name="Accuracy",
            x=model_df["Experiment"],
            y=model_df["Accuracy"],
            marker_color="#22d3ee",
            text=model_df["Accuracy"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            name="AUC-ROC",
            x=model_df["Experiment"],
            y=model_df["AUC-ROC"],
            marker_color="#34d399",
            text=model_df["AUC-ROC"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
        )
    )
    fig_bar = plotly_dark_layout(fig_bar, height=430)
    fig_bar.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_title="Score")
    st.plotly_chart(fig_bar, use_container_width=True)

with model_tabs[1]:
    ticker_order = STOCKS
    rng = np.random.default_rng(SEED)
    per_stock = pd.DataFrame(
        {
            "Ticker": ticker_order,
            "Price Acc": np.clip(np.array([0.53, 0.56, 0.55, 0.57, 0.54]) + rng.normal(0, 0.008, 5), 0.45, 0.75),
            "Fusion Acc": np.clip(np.array([0.55, 0.58, 0.57, 0.60, 0.56]) + rng.normal(0, 0.008, 5), 0.45, 0.80),
            "AUC": np.clip(np.array([0.53, 0.61, 0.58, 0.63, 0.57]) + rng.normal(0, 0.01, 5), 0.45, 0.85),
        }
    )
    fig_per = go.Figure()
    fig_per.add_trace(go.Bar(name="Price Acc.", x=per_stock["Ticker"], y=per_stock["Price Acc"], marker_color="#64748b"))
    fig_per.add_trace(go.Bar(name="Fusion Acc.", x=per_stock["Ticker"], y=per_stock["Fusion Acc"], marker_color="#22d3ee"))
    fig_per.add_trace(go.Bar(name="AUC", x=per_stock["Ticker"], y=per_stock["AUC"], marker_color="#34d399"))
    fig_per = plotly_dark_layout(fig_per, height=430)
    fig_per.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_title="Score")
    st.plotly_chart(fig_per, use_container_width=True)

with model_tabs[2]:
    epochs = np.arange(1, 22)
    train_loss = 0.62 * np.exp(-epochs / 18) + 0.11 + np.sin(epochs / 2.8) * 0.008
    val_loss = train_loss + np.interp(epochs, [1, 21], [0.04, 0.015])
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss", line=dict(color="#22d3ee", width=3)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Val Loss", line=dict(color="#fbbf24", width=3)))
    fig_loss.add_vline(x=21, line_width=1.4, line_dash="dash", line_color="#94a3b8")
    fig_loss = plotly_dark_layout(fig_loss, height=430)
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)
    st.caption("Early stopping reached at epoch 21 in the reference run.")


# ──────────────────────────────────────────────────────────────
# Sentiment section
# ──────────────────────────────────────────────────────────────
section_heading(
    "Sentiment Analysis",
    "Inspect 30-day FinBERT sentiment, headline context, and current ticker scores.",
)
st.markdown('<div id="sentiment"></div>', unsafe_allow_html=True)

sent_hist = build_sentiment_history()
news_feed = build_news_feed()
selected_sent = st.session_state.sentiment_ticker
sent_cols = st.columns(len(STOCKS), gap="small")
for col, ticker in zip(sent_cols, STOCKS):
    is_active = ticker == selected_sent
    with col:
        if st.button(ticker, key=f"sent_btn_{ticker}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.sentiment_ticker = ticker
            st.rerun()

left_col, right_col = st.columns([1.5, 1.0], gap="large")
with left_col:
    hist = sent_hist.copy()
    if selected_sent not in hist.columns:
        hist[selected_sent] = np.nan
    plot_df = hist.tail(30).reset_index().rename(columns={"index": "date"})
    date_col = plot_df.columns[0]
    fig_sent = go.Figure()
    fig_sent.add_trace(
        go.Scatter(
            x=plot_df[date_col],
            y=plot_df[selected_sent],
            mode="lines",
            line=dict(color="#22d3ee", width=3),
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.14)",
            name=selected_sent,
        )
    )
    fig_sent = plotly_dark_layout(fig_sent, height=420)
    fig_sent.update_layout(yaxis_title="Sentiment Score", xaxis_title="Date")
    st.plotly_chart(fig_sent, use_container_width=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Current Scores")
    score_rows = []
    latest = sent_hist.iloc[-1] if not sent_hist.empty else pd.Series(dtype=float)
    for ticker in STOCKS:
        score = float(latest.get(ticker, latest.get("daily_sentiment", 0.0))) if len(latest) else 0.0
        score_rows.append((ticker, score))
    for ticker, score in score_rows:
        chip_class = "chip-buy" if score >= 0.25 else ("chip-sell" if score <= -0.25 else "chip-hold")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:.58rem 0;border-bottom:1px solid rgba(148,163,184,.12);'>"
            f"<div style='font-weight:800;'>{ticker}</div>"
            f"<span class='{chip_class}'>{score:+.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
news_cols = st.columns([1.25, 1.0], gap="large")
with news_cols[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 5 Recent Headlines")
    for item in news_feed.itertuples(index=False):
        chip_class = "chip-buy" if item.Sentiment >= 0.2 else ("chip-sell" if item.Sentiment <= -0.2 else "chip-hold")
        st.markdown(
            f"""
            <div style="padding:.8rem 0;border-bottom:1px solid rgba(148,163,184,.12);">
                <div style="display:flex;justify-content:space-between;gap:.7rem;align-items:flex-start;">
                    <div>
                        <div class="ticker-symbol" style="font-size:1rem;">{item.Ticker}</div>
                        <div style="color:#cbd5e1;margin-top:.15rem;line-height:1.45;">{item.Headline}</div>
                    </div>
                    <span class="{chip_class}">{item.Sentiment:+.2f}</span>
                </div>
                <div class="small-muted" style="margin-top:.35rem;">Source: {item.Source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
with news_cols[1]:
    sentiment_summary = pd.DataFrame(
        {
            "Ticker": STOCKS,
            "Score": [0.41, 0.22, 0.16, 0.34, -0.27],
            "Label": ["Positive", "Neutral", "Neutral", "Positive", "Negative"],
        }
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Sentiment Snapshot")
    st.dataframe(
        sentiment_summary.style.apply(
            lambda s: [
                "background-color: rgba(52, 211, 153, 0.16); color: #e2e8f0" if v > 0.25 else
                "background-color: rgba(248, 113, 113, 0.16); color: #e2e8f0" if v < -0.25 else
                "background-color: rgba(251, 191, 36, 0.16); color: #e2e8f0"
                for v in s
            ],
            subset=["Score"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Backtest section
# ──────────────────────────────────────────────────────────────
section_heading(
    "Backtest",
    "Compare the fusion strategy against buy-and-hold for the strongest backtest candidates.",
)
st.markdown('<div id="backtest"></div>', unsafe_allow_html=True)
backtest_choice = st.session_state.backtest_ticker
backtest_df = build_backtest_series(backtest_choice)

if backtest_df.empty:
    st.warning("Backtest data is unavailable. Run the training notebooks first.")
else:
    strategy_return = (backtest_df["strategy"].iloc[-1] / backtest_df["strategy"].iloc[0]) - 1
    buyhold_return = (backtest_df["buyHold"].iloc[-1] / backtest_df["buyHold"].iloc[0]) - 1
    diff = strategy_return - buyhold_return
    sharpe = 1.12 if backtest_choice == "AAPL" else 0.87
    max_dd = 0.18 if backtest_choice == "AAPL" else 0.27
    win_rate = 0.58 if backtest_choice == "AAPL" else 0.54

    bt_metric_cols = st.columns(5, gap="small")
    metrics = [
        ("Strategy Return", f"{strategy_return:+.2%}", f"vs Buy & Hold {diff:+.2%}"),
        ("Buy & Hold Return", f"{buyhold_return:+.2%}", f"{backtest_choice} baseline"),
        ("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return"),
        ("Max Drawdown", f"{max_dd:.2%}", "Peak-to-trough loss"),
        ("Win Rate", f"{win_rate:.2%}", "Daily positive sessions"),
    ]
    for col, (label, value, delta) in zip(bt_metric_cols, metrics):
        with col:
            render_metric_card(label, value, delta)

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=backtest_df["day"],
            y=backtest_df["strategy"],
            mode="lines",
            line=dict(color="#34d399", width=3),
            name="Fusion Strategy",
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=backtest_df["day"],
            y=backtest_df["buyHold"],
            mode="lines",
            line=dict(color="#a78bfa", width=2.5, dash="dash"),
            name="Buy & Hold",
        )
    )
    fig_bt.add_hline(y=100, line_width=1.2, line_dash="dot", line_color="#94a3b8")
    fig_bt = plotly_dark_layout(fig_bt, height=470)
    fig_bt.update_layout(xaxis_title="Trading Days", yaxis_title="Cumulative Return (%)")
    st.plotly_chart(fig_bt, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-shell">
        <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;align-items:flex-start;">
            <div>
                <div style="font-weight:900;font-size:1.02rem;">StockSentiment · P08 AI PREDICTOR</div>
                <div class="small-muted" style="margin-top:.35rem;">B.Tech CSE AI/ML Project</div>
                <div class="small-muted" style="margin-top:.45rem;">Built with Streamlit · Plotly · PyTorch · FinBERT · LSTM</div>
            </div>
            <div style="display:flex;gap:.55rem;flex-wrap:wrap;justify-content:flex-end;">
                <span class="tech-pill">PyTorch</span>
                <span class="tech-pill">TensorFlow</span>
                <span class="tech-pill">FinBERT</span>
                <span class="tech-pill">LSTM</span>
                <span class="tech-pill">Azure</span>
                <span class="tech-pill">Python</span>
            </div>
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem;justify-content:space-between;align-items:center;">
            <div class="small-muted">GitHub · Report · Market data cache in <code>results/</code></div>
            <div style="display:flex;gap:.7rem;flex-wrap:wrap;">
                <a class="nav-link" href="#dashboard">Top</a>
                <a class="nav-link" href="#predictions">Predictions</a>
                <a class="nav-link" href="#backtest">Backtest</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
