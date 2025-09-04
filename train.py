# train.py — nightly trainer for CNN-LSTM (multi-horizon)
# Integrates with Google Sheets. Hardened yfinance fetch with retries/fallback.

import os
import re
import sys
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import gspread
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# --- Constants ---
WINDOW = 60
HORIZONS = [1, 3, 5, 10, 20]

# ---------- Helpers ----------

def clean_tickers(raw):
    """Keep only plausible ticker symbols (no spaces), dedupe in order."""
    out = []
    for t in raw:
        if not t:
            continue
        t = str(t).strip().upper()
        # allow A-Z, digits, ^, ., -, =
        if re.fullmatch(r"[\^A-Z0-9.\-=]{1,15}", t):
            out.append(t)
    return list(dict.fromkeys(out))

def make_session():
    """Session with friendly headers to avoid some 403/HTML responses."""
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

def _download_yf(ticker, period, interval, session, tries=4, backoff=3):
    """Download one ticker with retries + fallback to Ticker.history."""
    last_err = None
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
            last_err = "empty dataframe"
        except Exception as e:
            last_err = e
        time.sleep(backoff * i)

    # Fallback path
    try:
        df = yf.Ticker(ticker, session=session).history(
            period=period, interval=interval, auto_adjust=True
        )
        if df is not None and not df.empty:
            return df
    except Exception as e2:
        last_err = e2

    print(f"Warning: yfinance failed for '{ticker}': {last_err}")
    return pd.DataFrame()

def get_history(symbols=("AAPL",), period="1y", interval="1d"):
    """Fetch historical OHLCV via yfinance with retries/fallback."""
    print(f"Fetching {period} of {interval} data for symbols: {', '.join(symbols)}")
    frames, session = [], make_session()
    for s in symbols:
        df = _download_yf(s, period=period, interval=interval, session=session)
        if df.empty:
            print(f"Warning: No data found for symbol '{s}'. Skipping.")
            continue
        need = {"Close", "Volume"}
        if not need.issubset(df.columns):
            print(f"Warning: Missing expected columns for '{s}' (have {list(df.columns)}). Skipping.")
            continue
        h = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "vol"})
        h["symbol"] = s
        frames.append(h)
        time.sleep(1)  # be nice to Yahoo

    if not frames:
        print("Error: No data could be fetched for any symbols. Exiting.")
        sys.exit(1)

    out = pd.concat(frames)
    out.index.name = "time"
    return out.reset_index()

def make_features(df):
    """Engineer features from the raw stock data."""
    df = df.copy()
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def windowize(df, window=WINDOW, horizons=HORIZONS):
    """Create windowed sequences for training the time-series model."""
    feats = ["close", "ret1", "ma10", "ma50", "vlog"]
    scaler = MinMaxScaler()
    df = df.copy()
    df[feats] = scaler.fit_transform(df[feats].values)

    Xs, Ys = [], []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) <= window + max(horizons):
            print(f"Warning: Not enough data for symbol '{s}' to create windows. Skipping.")
            continue
        for i in range(len(g) - window - max(horizons)):
            x = g.loc[i:i+window-1, feats].values
            future = [g.loc[i+window-1+h, "close"] for h in horizons]
            Xs.append(x)
            Ys.append(future)

    if not Xs:
        print("Error: Could not create any training windows from the provided data. Exiting.")
        sys.exit(1)

    X = np.array(Xs)
    y = np.array(Ys, dtype=np.float32)
    return X, y, scaler, feats

def build_model(n_features, window=WINDOW, n_out=len(HORIZONS)):
    """Build and compile the CNN-LSTM Keras model."""
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Train models using symbols from a Google Sheet.")
    parser.add_argument("--model", choices=["cnn-lstm", "linear"], default="cnn-lstm")
    parser.add_argument("--sheet-id", required=True)
    parser.add_argument("--worksheet", required=True)
    parser.add_argument("--symbol-column", default="Ticker")
    parser.add_argument("--period", default=os.getenv("PERIOD", "1y"),
                        help="How far back to fetch (e.g., 6mo, 1y, 3y, 5y)")
    parser.add_argument("--interval", default=os.getenv("INTERVAL", "1d"),
                        help="Bar size (e.g., 1d, 1h, 30m, 5m)")
    args = parser.parse_args()

    print(
        f"Received arguments: model='{args.model}', sheet-id='{args.sheet_id}', "
        f"worksheet='{args.worksheet}', symbol-column='{args.symbol_column}', "
        f"period='{args.period}', interval='{args.interval}'"
    )

    # 1) Sheets auth
    print("Authenticating with Google Sheets...")
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
        or os.path.expanduser("~/.config/gspread/service_account.json")
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(args.sheet_id)
    ws = sh.worksheet(args.worksheet)

    # 2) Read symbols (case-insensitive header)
    header_row = ws.row_values(1)
    lower = [h.strip().lower() for h in header_row]
    try:
        col_index = lower.index(args.symbol_column.strip().lower()) + 1
    except ValueError:
        print(f"Error: Column '{args.symbol_column}' not found in worksheet '{args.worksheet}'.")
        sys.exit(1)

    symbols = ws.col_values(col_index)[1:]  # skip header
    symbols = [s for s in symbols if s]
    symbols = clean_tickers(symbols)
    if not symbols:
        print("Error: No valid tickers found in the specified column. Exiting.")
        sys.exit(1)
    print(f"Using tickers: {symbols}")

    # 3) Data → features
    raw = get_history(symbols, period=args.period, interval=args.interval)
    feat = make_features(raw)

    # Linear branch
    if args.model == "linear":
        df = feat.dropna(subset=["ma10", "ma50"])
        if df.empty:
            print("Error: Not enough data to train linear model.")
            sys.exit(1)
        X = df[["ma10", "ma50"]].values
        y = df["close"].values
        model = LinearRegression()
        model.fit(X, y)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/linear_model.pkl")
        print("✅ Saved models/linear_model.pkl")
        return

    # CNN-LSTM branch
    X, y, scaler, feats = windowize(feat)
    print(f"Created training dataset with shape X: {X.shape}, y: {y.shape}")
    model = build_model(n_features=X.shape[-1])
    model.fit(X, y, epochs=8, batch_size=256, validation_split=0.1, verbose=2)

    # 4) Save artifacts
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_ALL.keras")
    joblib.dump({"scaler": scaler, "feats": feats}, "models/scaler_ALL.pkl")
    print("✅ Saved models/cnn_lstm_ALL.keras and models/scaler_ALL.pkl")

if __name__ == "__main__":
    main()
