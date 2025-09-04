#!/usr/bin/env python3
"""
Train a Linear (Ridge) model to predict next-day close using simple technical features.

Artifacts saved:
- models/linear_ALL.pkl
- models/scaler_ALL.pkl
- outputs/predictions_linear.csv

Optional Google Sheets usage (if environment variables present and gspread installed):
- Read tickers from a sheet
- Write latest predictions back to a sheet

Environment variables (all optional):
- TICKERS: e.g. "AAPL,MSFT,SPY" (default if no Google Sheet is configured)
- LOOKBACK_DAYS: e.g. "1825" (default 1825 ~ 5y)
- HORIZON: prediction horizon in days (default 1)
- TRAIN_SPLIT: fraction for train (default 0.8)
- YF_INTERVAL: e.g. "1d" (default)
- SHEET_ID: Google Sheet ID to read/write
- SHEET_TICKERS_TAB: tab name that has tickers in column A (default "Tickers")
- SHEET_RESULTS_TAB: tab name for writing predictions (default "Predictions_Linear")
- SHEET_TICKERS_RANGE: e.g. "A2:A" (default)
"""

import os
import sys
import math
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# Optional: Google Sheets hook
# ----------------------------
def try_import_gs():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        return gspread, Credentials
    except Exception:
        return None, None

GSPREAD, GCREDS = try_import_gs()

def read_tickers_from_sheet():
    sheet_id = os.getenv("SHEET_ID")
    tab = os.getenv("SHEET_TICKERS_TAB", "Tickers")
    rng = os.getenv("SHEET_TICKERS_RANGE", "A2:A")
    if not (GSPREAD and GCREDS and sheet_id):
        return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")).exists():
            creds = GCREDS.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=scopes)
        elif Path("creds.json").exists():
            creds = GCREDS.from_service_account_file("creds.json", scopes=scopes)
        else:
            return None
        gc = GSPREAD.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        values = ws.get(rng)
        tickers = [v[0].strip().upper() for v in values if v and v[0].strip()]
        return tickers or None
    except Exception:
        return None

def write_predictions_to_sheet(df):
    sheet_id = os.getenv("SHEET_ID")
    tab = os.getenv("SHEET_RESULTS_TAB", "Predictions_Linear")
    if not (GSPREAD and GCREDS and sheet_id):
        return
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")).exists():
            creds = GCREDS.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=scopes)
        elif Path("creds.json").exists():
            creds = GCREDS.from_service_account_file("creds.json", scopes=scopes)
        else:
            return
        gc = GSPREAD.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        try:
            ws = sh.worksheet(tab)
        except Exception:
            ws = sh.add_worksheet(title=tab, rows="1000", cols="20")
        # Prepare data
        out = [df.columns.tolist()] + df.astype(str).values.tolist()
        ws.clear()
        ws.update("A1", out)
    except Exception:
        pass

# ----------------------------
# Data utilities
# ----------------------------
def fetch_data(ticker, lookback_days=1825, interval="1d"):
    import yfinance as yf
    end = datetime.utcnow()
    start = end - timedelta(days=int(lookback_days))
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, auto_adjust=True, progress=False, group_by="column")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    df = df.rename_axis("Date").reset_index()
    df["Ticker"] = ticker
    return df

def add_features(df):
    # Basic technicals on Close
    close = df["Close"]
    df["Return1"] = close.pct_change(1)
    for l in [1,2,3,5]:
        df[f"Lag{l}"] = close.shift(l)
    for w in [5,10,20]:
        df[f"SMA{w}"] = close.rolling(w).mean()
        df[f"STD{w}"] = close.rolling(w).std()
    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll = 14
    rs = up.rolling(roll).mean() / dn.rolling(roll).mean()
    df["RSI14"] = 100 - (100 / (1 + rs))
    # Target: next-day Close
    df["Target_Close_h1"] = close.shift(-1)
    return df

def build_dataset(tickers, lookback_days, interval, horizon=1):
    frames = []
    for t in tickers:
        raw = fetch_data(t, lookback_days, interval)
        feat = add_features(raw)
        frames.append(feat)
    data = pd.concat(frames, ignore_index=True)

    # --- NEW: ensure flat, string columns (handles any MultiIndex tuples) ---
    data.columns = [
        c if isinstance(c, str)
        else "_".join([str(x) for x in c if x is not None])
        for c in data.columns
    ]

    
    # Keep rows with complete features and target
    feature_cols = [c for c in data.columns if c.startswith(("Lag","SMA","STD","RSI","Return1"))]
    X = data[feature_cols]
    y = data["Target_Close_h1"]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    meta = data.loc[mask, ["Date","Ticker","Close"]].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True), feature_cols

# ----------------------------
# Train / Evaluate
# ----------------------------
def train_and_eval(X, y):
    split = float(os.getenv("TRAIN_SPLIT", "0.8"))
    n = len(X)
    k = int(n * split)
    X_train, X_test = X.iloc[:k], X.iloc[k:]
    y_train, y_test = y.iloc[:k], y.iloc[k:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Ridge with CV over a few alphas
    model = RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 50.0, 100.0]))
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    mae  = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"[Linear] n_train={len(X_train)} n_test={len(X_test)}")
    print(f"[Linear] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} Alpha*={getattr(model, 'alpha_', 'N/A')}")

    return model, scaler, preds, y_test

# ----------------------------
# Main
# ----------------------------
def main():
    # Resolve tickers (Google Sheet first, fallback to env)
    tickers = read_tickers_from_sheet()
    if not tickers:
        tickers = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL,MSFT,SPY").split(",") if t.strip()]
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "1825"))
    interval = os.getenv("YF_INTERVAL", "1d")
    horizon = int(os.getenv("HORIZON", "1"))
    if horizon != 1:
        print("[Info] This script is configured for HORIZON=1 (next-day). Other horizons not implemented.")

    print(f"Tickers: {tickers}")
    print(f"Lookback days: {lookback_days} | Interval: {interval}")

    X, y, meta, feature_cols = build_dataset(tickers, lookback_days, interval, horizon=horizon)

    if len(X) < 200:
        raise RuntimeError("Not enough data after feature engineering to train. Try increasing LOOKBACK_DAYS.")

    model, scaler, preds, y_test = train_and_eval(X, y)

    # Save artifacts
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/linear_ALL.pkl")
    joblib.dump(scaler, "models/scaler_ALL.pkl")
    print("Saved: models/linear_ALL.pkl, models/scaler_ALL.pkl")

    # Build predictions table for the holdout
    out = meta.iloc[int(len(meta)*float(os.getenv('TRAIN_SPLIT','0.8'))):].copy()
    out["y_true_next_close"] = y_test.values
    out["y_pred_next_close"] = preds
    out = out[["Date","Ticker","Close","y_true_next_close","y_pred_next_close"]]

    Path("outputs").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/predictions_linear.csv", index=False)
    print("Saved: outputs/predictions_linear.csv")

    # Optional: write to Google Sheet
    try:
        write_predictions_to_sheet(out.tail(200))  # limit size
        print("Wrote predictions to Google Sheet (if configured).")
    except Exception as e:
        print(f"Skipped writing to Google Sheet: {e}")

if __name__ == "__main__":
    sys.exit(main())
