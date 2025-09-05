# train.py — nightly trainer for CNN-LSTM (multi-horizon)
# Primary data: Stooq (free, stable). Fallback: yfinance with retries.
# Reads tickers from Google Sheets. Includes EarlyStopping & CLI flags.
# Supports reading historical data from a Google Sheet 'prices' tab
# or downloading (Stooq primary, yfinance fallback) with a local cache.

import os, re, sys, io, time, argparse, requests
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import gspread
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ------------ constants ------------
WINDOW = 60
HORIZONS = [1, 3, 5, 10, 20]
CACHE_DIR = "data_cache"

# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser(description="Train stock models from Google Sheets watchlists.")
    p.add_argument("--model", choices=["cnn-lstm", "linear"], default="cnn-lstm")
    p.add_argument("--sheet-id", required=True)
    p.add_argument("--worksheet", required=True)
    p.add_argument("--symbol-column", default="Ticker")
    p.add_argument("--period", default=os.getenv("PERIOD", "1y"))
    p.add_argument("--interval", default=os.getenv("INTERVAL", "1d"))
    # training controls
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 40)))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 256)))
    p.add_argument("--patience", type=int, default=int(os.getenv("PATIENCE", 5)))
    # prices tab mode
    p.add_argument("--use-prices-tab", action="store_true",
                   help="Read from 'prices' worksheet instead of downloading")
    p.add_argument("--prices-worksheet", default=os.getenv("PRICES_TAB", "prices"),
                   help="Name of the prices worksheet")
    return p.parse_args()

# ------------ ticker helpers ------------
def clean_tickers(raw):
    out, seen = [], set()
    for t in raw:
        if not t: 
            continue
        t = str(t).strip().upper()
        if not re.full
