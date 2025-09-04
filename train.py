# train.py — nightly trainer for CNN-LSTM (multi-horizon)
# Primary data: Stooq (free, stable). Fallback: yfinance with retries.
# Reads tickers from Google Sheets. Includes EarlyStopping & CLI flags.

import os, re, sys, time, io, requests
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
CACHE_DIR = "data_cache"  # persisted CSV cache for robustness

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Train models using symbols from a Google Sheet.")
    p.add_argument("--model", choices=["cnn-lstm", "linear"], default="cnn-lstm")
    p.add_argument("--sheet-id", required=True)
    p.add_argument("--worksheet", required=True)
    p.add_argument("--symbol-column", default="Ticker")
    p.add_argument("--period", default=os.getenv("PERIOD", "1y"),
                   help="History window: e.g., 6mo, 1y, 3y, 5y")
    p.add_argument("--interval", default=os.getenv("INTERVAL", "1d"),
                   help="Bar size. Stooq supports 1d only; others fall back to yfinance.")
    # NEW: training controls
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 40)),
                   help="Max training epochs (with EarlyStopping)")
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 256)),
                   help="Batch size")
    p.add_argument("--patience", type=int, default=int(os.getenv("PATIENCE", 5)),
                   help="EarlyStopping patience (epochs) on val_loss")
    p.add_argument("--use-prices-tab", action="store_true",
               help="Read historical data from the 'prices' worksheet instead of downloading")
    p.add_argument("--prices-worksheet", default=os.getenv("PRICES_TAB", "prices"),
               help="Name of the worksheet that stores normalized prices")
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


# ---------- Loader ----------

def load_prices_from_sheet(gc, sheet_id, prices_tab, symbols, period="1y"):
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_tab)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        raise SystemExit(f"'{prices_tab}' is empty.")
    header = [h.strip() for h in values[0]]
    idx = {name: header.index(name) for name in ["Date","Symbol","Close","Volume"]}
    rows = values[1:]
    import pandas as pd
    df = pd.DataFrame([[r[idx["Date"]], r[idx["Symbol"]], r[idx["Close"]], r[idx["Volume"]]] 
                      for r in rows if len(r) >= len(header)],
                      columns=["Date","Symbol","Close","Volume"])
    symbols_up = {s.upper() for s in symbols}
    df = df[df["Symbol"].str.upper().isin(symbols_up)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Date","Close"])
    def _days(p): 
        p = str(p).lower()
        return int(p[:-1])*365 if p.endswith("y") else (int(p[:-2])*30 if p.endswith("mo") else 365)
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=_days(period))
    df = df[df["Date"] >= cutoff]
    df = df.rename(columns={"Date":"time","Close":"close","Volume":"vol","Symbol":"symbol"})
    return df.sort_values(["symbol","time"]).reset_index(drop=True)



# ---------- Period helpers (daily only) ----------

def period_to_days(period_str: str) -> int:
    ps = period_str.strip().lower()
    if ps.endswith("y"):
        return int(ps[:-1]) * 365
    if ps.endswith("mo"):
        return int(ps[:-2]) * 30
    if ps in ("max", "all"):
        return 10 * 365  # cap to 10y for practicality
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
            if r.status_code == 200 and r.text:
                lines = r.text.splitlines()
                if lines and "Date,Open,High,Low,Close,Volume" in lines[0]:
                    df = pd.read_csv(io.StringIO(r.text))
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

# ---- Load from Google Sheet 'prices' tab ----
def load_prices_from_sheet(gc, sheet_id, prices_tab, symbols, period="1y"):
    import pandas as pd
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_tab)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        raise SystemExit(f"'{prices_tab}' is empty.")

    header = [h.strip() for h in values[0]]
    required = ["Date", "Symbol", "Close", "Volume"]
    for col in required:
        if col not in header:
            raise SystemExit(f"'{prices_tab}' is missing column '{col}'")
    idx = {name: header.index(name) for name in required}

    rows = values[1:]
    df = pd.DataFrame(
        [[r[idx["Date"]], r[idx["Symbol"]], r[idx["Close"]], r[idx["Volume"]]]
         for r in rows if len(r) >= len(header)],
        columns=["Date","Symbol","Close","Volume"]
    )

    # keep only requested symbols
    symbols_up = {s.upper() for s in symbols}
    df = df[df["Symbol"].str.upper().isin(symbols_up)].copy()

    # types
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Date","Close"])

    # trim to requested period (daily)
    def _days(p):
        p = str(p).lower()
        if p.endswith("y"):  return int(p[:-1]) * 365
        if p.endswith("mo"): return int(p[:-2]) * 30
        return 365
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=_days(period))
    df = df[df["Date"] >= cutoff]

    # final columns
    df = df.rename(columns={"Date":"time","Close":"close","Volume":"vol","Symbol":"symbol"})
    return df.sort_values(["symbol","time"]).reset_index(drop=True)


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
                if "Date" not in yfd.columns and "Datetime" in yfd.columns:
                    yfd = yfd.rename(columns={"Datetime": "Date"})
            if not yfd.empty:
                need = {"Date", "Close", "Volume"}
                if need.issubset(set(yfd.columns)):
                    df2 = yfd[["Date", "Close", "Volume"]].rename(columns={"Close": "close", "Volume": "vol"})
                    df2["symbol"] = s
                    save_cache(df2, cpath)
                    frames.append(df2.rename(columns={"Date": "time"}))
                else:
                    print(f"Warning: yfinance missing expected cols for '{s}' ({list(yfd.columns)[:6]}...)")
            else:
                print(f"Warning: yfinance returned empty for '{s}'")
        else:
            frames.append(yfd.rename(columns={"Date": "time"}))
        time.sleep(0.5)

    if not frames:
        print("Error: No data could be fetched for any symbols. Exiting.")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    return out  # columns: time, close, vol, symbol

# ---------- Feature + model code ----------

def make_features(df):
    df = df.copy()
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
    args = parse_args()
    print(
        f"Received arguments: model='{args.model}', sheet-id='{args.sheet_id}', "
        f"worksheet='{args.worksheet}', symbol-column='{args.symbol_column}', "
        f"period='{args.period}', interval='{args.interval}', "
        f"epochs={args.epochs}, batch_size={args.batch_size}, patience={args.patience}"
    )

    # 1) Sheets auth
    print("Authenticating with Google Sheets...")
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
        or os.path.expanduser("~/.config/gspread/service_account.json")
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(args.sheet_id)
    ws = sh.worksheet(args.worksheet)

    # 2) Read tickers
    header_row = ws.row_values(1)
    lower = [h.strip().lower() for h in header_row]
    try:
        col_index = lower.index(args.symbol_column.strip().lower()) + 1
    except ValueError:
        print(f"Error: Column '{args.symbol_column}' not found in worksheet '{args.worksheet}'.")
        sys.exit(1)

    symbols = ws.col_values(col_index)[1:]
    symbols = clean_tickers(symbols)
    if not symbols:
        print("Error: No valid tickers found. Exiting.")
        sys.exit(1)
    print(f"Using tickers: {symbols}")

    # 3) Load prices → features 
    # Use data from the Google Sheet 'prices' tab if requested; otherwise download
if args.use_prices_tab:
    print(f"Loading data from prices tab '{args.prices_worksheet}'...")
    raw = load_prices_from_sheet(gc, args.sheet_id, args.prices_worksheet, symbols, period=args.period)
else:
    raw = get_history(symbols, period=args.period, interval=args.interval)

    raw = raw.rename(columns={"time": "time"}).sort_values(["symbol", "time"]).reset_index(drop=True)
    feat = make_features(raw)

    # Linear branch
    if args.model == "linear":
        df = feat.dropna(subset=["ma10", "ma50"])
        if df.empty:
            print("Error: Not enough data to train linear model.")
            sys.exit(1)
        X = df[["ma10", "ma50"]].values
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
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, args.patience // 2), min_lr=1e-5
        ),
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

    # 4) Save artifacts
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_ALL.keras")
    joblib.dump({"scaler": scaler, "feats": feats}, "models/scaler_ALL.pkl")
    print("✅ Saved models/cnn_lstm_ALL.keras and models/scaler_ALL.pkl")

if __name__ == "__main__":
    main()
