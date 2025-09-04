# train.py — nightly trainer for CNN-LSTM (multi-horizon)
# Primary data: Stooq (free, stable). Fallback: yfinance with retries.
# Reads tickers from Google Sheets.

import os, re, sys, time, io, requests, csv, hashlib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import gspread
import argparse
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# --- Constants ---
WINDOW = 60
HORIZONS = [1, 3, 5, 10, 20]
CACHE_DIR = "data_cache"  # persisted CSV cache for robustness

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Train models using symbols from a Google Sheet.")
    p.add_argument("--model", choices=["cnn-lstm", "linear"], default="cnn-lstm")
    p.add_argument("--sheet-id", required=True)
    p.add_argument("--worksheet", required=True)
    p.add_argument("--symbol-column", default="Ticker")
    p.add_argument("--period", default=os.getenv("PERIOD", "1y"),
                   help="How far back: 6mo, 1y, 3y, 5y")
    p.add_argument("--interval", default=os.getenv("INTERVAL", "1d"),
                   help="Bar size. Stooq supports 1d only; others fall back to yfinance.")
    return p.parse_args()

# ---------- Ticker utils ----------

def clean_tickers(raw):
    """Keep only plausible ticker symbols (no spaces), dedupe in order."""
    out = []
    for t in raw:
        if not t:
            continue
        t = str(t).strip().upper()
        if re.fullmatch(r"[\^A-Z0-9.\-=]{1,15}", t):
            out.append(t)
    # dedupe preserving order
    return list(dict.fromkeys(out))

def to_stooq_symbol(t):
    """
    Map common US tickers to Stooq format:
      - lowercase
      - '.' → '-'  (BRK.B → brk-b)
      - add '.us' suffix for US listings
    """
    t = t.lower().replace(".", "-")
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t

# ---------- Period helpers (daily only) ----------

def period_to_days(period_str: str) -> int:
    ps = period_str.strip().lower()
    if ps.endswith("y"):
        return int(ps[:-1]) * 365
    if ps.endswith("mo"):
        return int(ps[:-2]) * 30
    if ps in ("max", "all"):
        return 10 * 365  # cap to 10y for training practicality
    # default
    return 365

def trim_period_daily(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df
    days = period_to_days(period)
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff.tz_localize(None)]

# ---------- HTTP / sessions ----------

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s

# ---------- Stooq primary source (daily EOD) ----------

def fetch_stooq_one(ticker: str, session: requests.Session) -> pd.DataFrame:
    """
    Download daily CSV from Stooq for one ticker.
    URL form: https://stooq.com/q/d/l/?s=aapl.us&i=d
    """
    sym = to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    for i in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and r.text and "Date,Open,High,Low,Close,Volume" in r.text.splitlines()[0]:
                df = pd.read_csv(io.StringIO(r.text))
                # Normalize cols
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
                df = df.rename(columns={"Close": "close", "Volume": "vol"})
                df["symbol"] = ticker
                return df[["Date", "close", "vol", "symbol"]]
        except Exception:
            pass
        time.sleep(1 + i)
    return pd.DataFrame()

# ---------- yfinance fallback (supports intraday too) ----------

def yf_download(ticker, period, interval, session, tries=4, backoff=3) -> pd.DataFrame:
    last = None
    for i in range(1, tries + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30,
                session=session,
            )
            if df is not None and not df.empty:
                return df
            last = "empty dataframe"
        except Exception as e:
            last = e
        time.sleep(backoff * i)
    # alt path
    try:
        df = yf.Ticker(ticker, session=session).history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty:
            return df
    except Exception as e2:
        last = e2
    print(f"Warning: yfinance failed for '{ticker}': {last}")
    return pd.DataFrame()

# ---------- Cache helpers ----------

def cache_path(ticker: str, source: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ticker.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{source}_{safe}.csv")

def load_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["Date"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass

# ---------- Unified history fetch ----------

def get_history(symbols=("AAPL",), period="1y", interval="1d") -> pd.DataFrame:
    """
    Primary: Stooq (daily only). Fallback: yfinance for anything else or if Stooq fails.
    Uses a local CSV cache so transient network failures don't kill the run.
    """
    session = make_session()
    frames = []

    for s in symbols:
        # Try Stooq only for daily interval
        df = pd.DataFrame()
        if interval == "1d":
            cpath = cache_path(s, "stooq")
            df = load_cache(cpath)
            if df.empty:
                df = fetch_stooq_one(s, session)
                if not df.empty:
                    save_cache(df, cpath)
            if not df.empty:
                df = trim_period_daily(df, period)
                frames.append(df.rename(columns={"Date": "time"}))
                time.sleep(0.5)
                continue  # next ticker

        # Fallback to Yahoo Finance for intraday or if Stooq failed
        cpath = cache_path(s, "yfinance")
        yfd = load_cache(cpath)
        if yfd.empty:
            yfd = yf_download(s, period=period, interval=interval, session=session)
            if not yfd.empty:
                yfd = yfd.reset_index()  # ensure 'Date' column present
                if "Date" not in yfd.columns:
                    # sometimes index is named 'Datetime'
                    if "Datetime" in yfd.columns:
                        yfd = yfd.rename(columns={"Date
