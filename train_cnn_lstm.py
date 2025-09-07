#!/usr/bin/env python3
# train_cnn_lstm.py — Google Finance only
#
# Reads OHLCV from a Google Sheet "prices" worksheet that is populated
# by GOOGLEFINANCE formulas (e.g., via append_prices.py). Trains a CNN+LSTM
# on stacked sequences across tickers and saves the scaler + model.

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import gspread
import gspread_dataframe as gd

from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# ---------- Defaults ----------
DEFAULT_LOOKBACK = 60
DEFAULT_PRICES_WS = "prices"
REQUIRED_COLS = ["Date","Ticker","Open","High","Low","Close","Volume"]

Path("models").mkdir(exist_ok=True)

def auth():
    # Uses ~/.config/gspread/service_account.json by default, or GOOGLE_APPLICATION_CREDENTIALS if set.
    return gspread.service_account()

def read_prices(sheet_id: str, prices_ws: str) -> pd.DataFrame:
    gc = auth()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_ws)
    # Evaluate formulas so we get numeric values for training
    df = gd.get_as_dataframe(ws, evaluate_formulas=True, header=0)
    if df is None or df.empty:
        raise SystemExit(f"'{prices_ws}' is empty.")

    # Normalize headers / ensure required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in '{prices_ws}': {missing}")

    df = df[REQUIRED_COLS].dropna(subset=["Date","Ticker"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Cast numeric columns defensively
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Some days (holidays) may be blank from GOOGLEFINANCE; drop rows with missing price data
    df = df.dropna(subset=["Open","High","Low","Close"]).sort_values(["Ticker","Date"])
    return df

def build_dataset(series_2d: np.ndarray, lookback: int):
    """series_2d: shape (N, 5) columns = [o,h,l,close,v] scaled"""
    X, y = [], []
    for i in range(len(series_2d) - lookback):
        X.append(series_2d[i:i+lookback])
        y.append(series_2d[i+lookback][3])  # index 3 is 'close'
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")

def collect_features(df: pd.DataFrame, symbols: list[str], lookback: int) -> np.ndarray:
    frames = []
    skipped = []
    for s in symbols:
        sub = df[df["Ticker"] == s].sort_values("Date")
        # Require at least lookback+1 rows to produce one target
        if len(sub) <= lookback:
            skipped.append((s, len(sub)))
            continue
        f = (sub.rename(columns={"Open":"o","High":"h","Low":"l","Close":"close","Volume":"v"})
               [["o","h","l","close","v"]].dropna())
        if len(f) <= lookback:
            skipped.append((s, len(f)))
            continue
        frames.append(f.values)

    if not frames:
        detail = "; ".join([f"{s}:{n}" for s,n in skipped]) or "no data"
        raise SystemExit(f"No usable data after filtering (lookback={lookback}). Details: {detail}")

    all_rows = np.vstack(frames)
    print(f"Collected rows across {len(frames)} tickers: {all_rows.shape[0]}")
    return all_rows

def build_model(lookback: int) -> Sequential:
    model = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(lookback, 5)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)  # predicts scaled 'close'
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True, help="Google Sheet ID containing the prices worksheet")
    ap.add_argument("--prices-worksheet", default=DEFAULT_PRICES_WS, help="Worksheet name (default: prices)")
    ap.add_argument("--symbols", default="", help="Comma-separated tickers. If omitted, use all in prices sheet.")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--output-prefix", default="ALL", help="Used in saved scaler/model filenames")
    args = ap.parse_args()

    # Load prices (evaluated values from GOOGLEFINANCE formulas)
    prices = read_prices(args.sheet_id, args.prices_worksheet)

    # Choose symbols
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = prices["Ticker"].dropna().astype(str).str.upper().unique().tolist()

    if not symbols:
        raise SystemExit("No symbols provided or found in prices sheet.")

    print(f"Training on {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols)>10 else ''}")
    lookback = args.lookback

    # Assemble feature matrix from sheet (Google Finance-derived)
    all_rows = collect_features(prices, symbols, lookback)

    # Scale 5 features
    scaler = MinMaxScaler()
    all_scaled = scaler.fit_transform(all_rows)

    # Save scaler
    scaler_path = Path("models") / f"scaler_{args.output_prefix}.joblib"
    joblib.dump(scaler, scaler_path.as_posix())
    print(f"Saved {scaler_path}")

    # Build sequences
    X, y = build_dataset(all_scaled, lookback)
    print("X:", X.shape, "y:", y.shape)

    if len(X) == 0:
        raise SystemExit("Not enough data to form sequences. Try reducing --lookback or adding more history.")

    # Model
    model = build_model(lookback)
    model.summary()

    # Train
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=args.val_split, verbose=1)

    # Save model
    model_path = Path("models") / f"cnn_lstm_{args.output_prefix}.keras"
    model.save(model_path.as_posix())
    print(f"Saved {model_path}")

if __name__ == "__main__":
    main()
