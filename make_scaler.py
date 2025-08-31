# make_scaler.py
# Build a MinMaxScaler over [open, high, low, close, volume] using Stooq (robust)
# with optional Yahoo fallback (via yfinance). Saves to models/scaler_ALL.pkl.

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import pickle


# ----------------------------- Config ----------------------------------------

DEFAULT_SYMBOLS = "AAPL,MSFT,GOOG,TSLA"
DEFAULT_DAYS = 365
OUTPUT_PATH = os.path.join("models", "scaler_ALL.pkl")

# Stooq symbol quirks (we use .us suffix and some aliases)
STOOQ_ALIAS = {
    "GOOGL": "GOOG",  # Stooq usually lists GOOG
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


# ----------------------------- Utilities -------------------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def to_utc_date(ts: pd.Series | pd.Index) -> pd.Series:
    """Normalize to date (no time)."""
    return pd.to_datetime(ts, errors="coerce").dt.tz_localize(None).dt.normalize()


def trim_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.sort_values("time")
    cutoff = datetime.utcnow() - timedelta(days=days)
    return df[df["time"] >= cutoff].copy()


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: time, open, high, low, close, volume.
    Drops rows with missing values and duplicates.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # lower-case columns when possible
    df = df.rename(columns={c: c.lower() for c in df.columns})
    col_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "close",  # prefer close if adj close only exists (rare in Stooq)
        "volume": "volume",
    }

    # Some sources may include both 'close' and 'adj close'; keep 'close'
    cols = {}
    for src, dst in col_map.items():
        if src in df.columns and dst not in cols:
            cols[src] = dst

    # If we only have adj close, map it to close
    if "close" not in cols and "adj close" in df.columns:
        cols["adj close"] = "close"

    need = ["open", "high", "low", "close", "volume"]
    out = pd.DataFrame()

    # Build outgoing df with needed columns (if present)
    if "time" in df.columns:
        out["time"] = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        out["time"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # sometimes index is datetime
        if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
            out["time"] = pd.to_datetime(df.index, errors="coerce")
        else:
            return pd.DataFrame()

    for want in need:
        # find first source that maps to 'want'
        choices = [k for k, v in cols.items() if v == want]
        src = choices[0] if choices else None
        if src is None or src not in df.columns:
            return pd.DataFrame()
        out[want] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna().drop_duplicates(subset=["time"])
    return out[["time"] + need]


# --------------------------- Stooq fetch (robust) ----------------------------

def stooq_code(symbol: str) -> str:
    s = symbol.upper().strip()
    s = STOOQ_ALIAS.get(s, s).replace(".", "-")
    return f"{s.lower()}.us"


def fetch_stooq(symbol: str) -> pd.DataFrame:
    """
    Robust Stooq fetch with UA, pandas fallback, and http fallback.
    """
    import pandas as pd

    code = stooq_code(symbol)
    url_https = f"https://stooq.com/q/d/l/?s={code}&i=d"
    headers = {"User-Agent": "Mozilla/5.0"}

    # Attempt 1: requests + UA
    try:
        r = requests.get(url_https, headers=headers, timeout=15)
        txt = r.text or ""
        if r.status_code == 200 and txt and not txt.lstrip().startswith("<"):
            df = pd.read_csv(io.StringIO(txt))
        else:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    # Attempt 2: pandas direct
    if df is None or df.empty:
        try:
            df = pd.read_csv(url_https)
        except Exception:
            df = pd.DataFrame()

    # Attempt 3: http (rarely needed, but helps in odd TLS/proxy setups)
    if df is None or df.empty:
        try:
            url_http = f"http://stooq.com/q/d/l/?s={code}&i=d"
            r = requests.get(url_http, headers=headers, timeout=15)
            txt = r.text or ""
            if r.status_code == 200 and txt and not txt.lstrip().startswith("<"):
                df = pd.read_csv(io.StringIO(txt))
            else:
                df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # normalize to standard columns
    df = df.rename(columns=str.lower)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["date"], errors="coerce")
    out = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    return out


# --------------------------- Yahoo fallback (optional) -----------------------

def fetch_yahoo(symbol: str, days: int) -> pd.DataFrame:
    """
    Optional Yahoo fallback via yfinance. Returns normalized OHLCV.
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    try:
        # 'period' argument is simpler with yfinance (e.g., '1y', '6mo').
        # Convert days to the nearest yfinance period.
        if days >= 365:
            period = "1y"
        elif days >= 180:
            period = "6mo"
        elif days >= 90:
            period = "3mo"
        else:
            period = "1mo"

        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        # yfinance DataFrame columns: Date, Open, High, Low, Close, Adj Close, Volume
        df.rename(columns={"Date": "time"}, inplace=True)
        out = normalize_ohlcv(df)
        return out
    except Exception:
        return pd.DataFrame()


# ------------------------------ Fetch orchestrator ---------------------------

def try_fetch(symbol: str, days: int) -> pd.DataFrame:
    """
    Try Stooq first (most reliable for us), then Yahoo (if available).
    """
    df = fetch_stooq(symbol)
    if df is None or df.empty:
        df = fetch_yahoo(symbol, days)

    df = normalize_ohlcv(df)
    if df is None or df.empty:
        return pd.DataFrame()

    return trim_last_days(df, days)


# ------------------------------ Scaler fitting -------------------------------

def fit_scaler(frames: List[pd.DataFrame]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on concatenated OHLCV rows.
    """
    if not frames:
        raise ValueError("No frames passed to fit_scaler.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna()
    if all_df.empty:
        raise ValueError("All frames empty after dropna; cannot fit scaler.")

    feats = ["open", "high", "low", "close", "volume"]
    X = all_df[feats].to_numpy(dtype=float)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X)
    return scaler


# -------------------------------- CLI / Main ---------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build MinMaxScaler over OHLCV using Stooq (with Yahoo fallback)."
    )
    p.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help=f"Comma-separated symbols (default: {DEFAULT_SYMBOLS})",
    )
    p.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"How many most-recent days to use (default: {DEFAULT_DAYS})",
    )
    p.add_argument(
        "--out",
        type=str,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    days = int(args.days)

    print(f"\nBuilding scaler from symbols: {symbols}  (days={days})")

    fetched: List[Tuple[str, pd.DataFrame]] = []
    for s in symbols:
        df = try_fetch(s, days=days)
        if df is not None and not df.empty:
            print(f"  ✅ {s}: {len(df)} rows")
            fetched.append((s, df))
        else:
            print(f"  [skip] {s}: no rows retrieved.")

    if not fetched:
        print("\nNo data fetched for any symbol. Aborting (scaler would be empty).")
        sys.exit(1)

    # Fit scaler
    frames = [df for _, df in fetched]
    try:
        scaler = fit_scaler(frames)
    except Exception as e:
        print(f"\nFailed to fit scaler: {e}")
        sys.exit(1)

    # Save
    ensure_dir(args.out)
    with open(args.out, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Saved scaler to {args.out}")


if __name__ == "__main__":
    # Make pandas quieter about chained assignments, etc.
    pd.options.mode.chained_assignment = None
    main()
