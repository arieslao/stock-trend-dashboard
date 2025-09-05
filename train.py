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
        if not re.fullmatch(r"[\^A-Z0-9.\-=]{1,15}", t):
            continue
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def to_stooq_symbol(t):
    t = t.lower().replace(".", "-")
    if not t.endswith(".us"):
        t += ".us"
    return t

# ------------ period helpers ------------
def period_to_days(period_str: str) -> int:
    ps = str(period_str).strip().lower()
    if ps.endswith("y"):   return int(ps[:-1]) * 365
    if ps.endswith("mo"):  return int(ps[:-2]) * 30
    if ps in ("max", "all"): return 3650
    return 365

def trim_period_daily(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    cutoff = now.normalize() - pd.Timedelta(days=period_to_days(period))
    return d[d["date"] >= cutoff]

# ------------ http session ------------
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s

# ------------ stooq/yf download + cache ------------
def fetch_stooq_one(ticker: str, session: requests.Session) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={to_stooq_symbol(ticker)}&i=d"
    for i in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and r.text.startswith("Date,Open,High,Low,Close,Volume"):
                df = pd.read_csv(io.StringIO(r.text))
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
                # normalize to lower-case
                df = df.rename(columns={"Date":"date","Close": "close", "Volume": "vol"})
                df["symbol"] = ticker
                return df[["date","close","vol","symbol"]]
        except Exception:
            pass
        time.sleep(1+i)
    return pd.DataFrame()

def yf_download(ticker, period, interval, session, tries=4, backoff=3) -> pd.DataFrame:
    last = None
    for i in range(1, tries+1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=False,
                             timeout=30, session=session)
            if df is not None and not df.empty:
                df = df.reset_index()
                if "Date" not in df.columns and "Datetime" in df.columns:
                    df = df.rename(columns={"Datetime":"Date"})
                if {"Date","Close","Volume"}.issubset(df.columns):
                    df = df.rename(columns={"Date":"date","Close":"close","Volume":"vol"})
                    df["symbol"] = ticker
                    return df[["date","close","vol","symbol"]]
            last = "empty"
        except Exception as e:
            last = e
        time.sleep(backoff*i)
    print(f"Warning: yfinance failed for '{ticker}': {last}")
    return pd.DataFrame()

def cache_path(ticker: str, source: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{source}_{ticker.replace('/','_')}.csv")

def load_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # normalize cached to lower-case
            df.columns = [c.strip().lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass

def get_history(symbols=("AAPL",), period="1y", interval="1d") -> pd.DataFrame:
    """Primary: Stooq (daily). Fallback: yfinance. Uses local CSV cache too."""
    session = make_session()
    frames = []
    for s in symbols:
        df = pd.DataFrame()
        if interval == "1d":
            cpath = cache_path(s, "stooq")
            df = load_cache(cpath)
            if df.empty:
                df = fetch_stooq_one(s, session)
                if not df.empty: save_cache(df, cpath)
            if not df.empty:
                df = trim_period_daily(df, period)
                frames.append(df.rename(columns={"date":"time"}))
                time.sleep(0.4)
                continue
        # fallback to yahoo
        cpath = cache_path(s, "yfinance")
        yfd = load_cache(cpath)
        if yfd.empty:
            yfd = yf_download(s, period=period, interval=interval, session=session)
            if not yfd.empty: save_cache(yfd, cpath)
        if not yfd.empty:
            frames.append(yfd.rename(columns={"date":"time"}))
        time.sleep(0.4)

    if not frames:
        print("Error: No data could be fetched for any symbols.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)

# ------------ prices tab loader (case-insensitive) ------------
def load_prices_from_sheet(gc, sheet_id, prices_tab, symbols, period="1y"):
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_tab)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        raise SystemExit(f"'{prices_tab}' is empty.")

    # build df then normalize to lower-case
    df = pd.DataFrame(values[1:], columns=[h.strip() for h in values[0]])
    df.columns = [c.strip().lower() for c in df.columns]

    # map common variants
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close":"close"})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close":"close"})
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume":"vol"})

    required = {"date","symbol","close"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"'{prices_tab}' is missing required column(s): {missing}")

    # types
    df["date"]  = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)
    else:
        df["vol"] = 0.0  # keep pipeline stable

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date","close"])

    # filter to requested symbols
    symbols_up = {s.upper() for s in symbols}
    df = df[df["symbol"].isin(symbols_up)].copy()

    # period cutoff
    now = pd.Timestamp.utcnow().tz_localize(None)
    cutoff = now.normalize() - pd.Timedelta(days=period_to_days(period))
    df = df[df["date"] >= cutoff]

    # final columns for the pipeline
    df = df.rename(columns={"date":"time"})
    return df.sort_values(["symbol","time"]).reset_index(drop=True)

# ------------ features & model ------------
def make_features(df):
    df = df.copy()
    if "vol" not in df.columns:
        df["vol"] = 0.0
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def windowize(df, window=WINDOW, horizons=HORIZONS):
    feats = ["close", "ret1", "ma10", "ma50", "vlog"]
    scaler = MinMaxScaler()
    df = df.copy()
    df[feats] = scaler.fit_transform(df[feats].values)
    Xs, Ys = [], []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) <= window + max(horizons):
            print(f"Warning: Not enough data for '{s}' to create windows. Skipping.")
            continue
        for i in range(len(g) - window - max(horizons)):
            x = g.loc[i:i+window-1, feats].values
            future = [g.loc[i+window-1+h, "close"] for h in horizons]
            Xs.append(x); Ys.append(future)
    if not Xs:
        print("Error: Could not create any training windows from the provided data.")
        sys.exit(1)
    X = np.array(Xs); y = np.array(Ys, dtype=np.float32)
    return X, y, scaler, feats

def build_model(n_features, window=WINDOW, n_out=len(HORIZONS)):
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

# ------------ main ------------
def main():
    args = parse_args()
    print(
        f"Args: model={args.model}, sheet-id={args.sheet_id}, worksheet={args.worksheet}, "
        f"symbol-column={args.symbol_column}, period={args.period}, interval={args.interval}, "
        f"epochs={args.epochs}, batch={args.batch_size}, patience={args.patience}, "
        f"use_prices_tab={args.use_prices_tab}, prices_ws={args.prices_worksheet}"
    )

    # Sheets auth
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
        or os.path.expanduser("~/.config/gspread/service_account.json")
    print("Authenticating with Google Sheets...")
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(args.sheet_id)
    ws = sh.worksheet(args.worksheet)

    # Read tickers (case-insensitive header)
    header_row = [h.strip() for h in ws.row_values(1)]
    try:
        col_index = [h.lower() for h in header_row].index(args.symbol_column.lower()) + 1
    except ValueError:
        print(f"Error: Column '{args.symbol_column}' not found in worksheet '{args.worksheet}'.")
        sys.exit(1)
    symbols = clean_tickers(ws.col_values(col_index)[1:])
    if not symbols:
        print("Error: No valid tickers found in watchlist.")
        sys.exit(1)
    print(f"Using tickers: {symbols}")

    # Load prices
    if args.use_prices_tab:
        print(f"Loading data from prices tab '{args.prices_worksheet}'...")
        raw = load_prices_from_sheet(gc, args.sheet_id, args.prices_worksheet, symbols, period=args.period)
    else:
        raw = get_history(symbols, period=args.period, interval=args.interval)
        # get_history returns lower-case date -> 'time'
        raw = raw.rename(columns={"date":"time"}) if "date" in raw.columns else raw

    raw = raw.sort_values(["symbol","time"]).reset_index(drop=True)

    # Features
    feat = make_features(raw)

    # Linear branch
    if args.model == "linear":
        df = feat.dropna(subset=["ma10","ma50"])
        if df.empty:
            print("Error: Not enough data to train linear model.")
            sys.exit(1)
        X = df[["ma10","ma50"]].values
        y = df["close"].values
        model = LinearRegression().fit(X, y)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/linear_model.pkl")
        print("✅ Saved models/linear_model.pkl")
        return

    # CNN-LSTM branch
    X, y, scaler, feats = windowize(feat)
    print(f"Created training dataset with shape X: {X.shape}, y: {y.shape}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience//2), min_lr=1e-5),
    ]
    model = build_model(n_features=X.shape[-1])
    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2,
    )

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_ALL.keras")
    joblib.dump({"scaler": scaler, "feats": feats}, "models/scaler_ALL.pkl")
    print("✅ Saved models/cnn_lstm_ALL.keras and models/scaler_ALL.pkl")

if __name__ == "__main__":
    main()
