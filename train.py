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
    URL form: https://stooq.com/q/d/l/?
