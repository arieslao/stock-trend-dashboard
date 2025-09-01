# log_eod_cnn.py
# End-of-day logger that predicts with your CNN-LSTM and appends one batch/day to Google Sheets.
# Idempotent: it skips if today's EOD rows were already written.

import os, sys, json, requests
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib

# --- TensorFlow for CNN-LSTM ---
try:
    import tensorflow as tf
except Exception:
    tf = None  # will fall back if TF not available

# --- Google Sheets ---
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError as GSpreadAPIError

LA = ZoneInfo("America/Los_Angeles")

# ---------- Settings ----------
WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "AAPL,MSFT,GOOGL,TSLA").split(",") if s.strip()]
DAYS = int(os.getenv("DAYS_HISTORY", "180"))   # bars to look back for features
DEFAULT_LOOKBACK = 60                           # CNN-LSTM sequence length
NOTE_VALUE = "watchlist_eod_cnn"

GOOGLE_SHEETS_SHEET_ID = os.getenv("GOOGLE_SHEETS_SHEET_ID")
GOOGLE_SHEETS_JSON = os.getenv("GOOGLE_SHEETS_JSON")  # full JSON (service account)

FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")

LOG_COLUMNS = [
    "ts_utc", "ts_pt", "symbol", "model", "lookback",
    "days_history", "last_close", "predicted", "pct_change",
    "in_window", "ma10", "ma50", "note"
]

# Small UA for CSV fallback
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (EOD-logger)"}


def now_la() -> datetime:
    return datetime.now(LA)


# ---------- Data fetch (yfinance with lightweight Stooq fallback) ----------
def _yf_download(symbol: str, days: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        # pad extra bars to ensure MA50 + lookback runway
        hist = yf.download(symbol, period=f"{days+80}d", interval="1d", auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            return None
        hist = hist.dropna()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        df = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        cols = ["open","high","low","close","volume"]
        if not set(cols).issubset(df.columns):
            return None
        return df[cols]
    except Exception:
        return None


def _stooq_csv(symbol: str) -> pd.DataFrame | None:
    # Try both "aapl.us" and "aapl" (Stooq uses .us for US tickers)
    for code in (f"{symbol.lower()}.us", symbol.lower()):
        url = f"https://stooq.com/q/d/l/?s={code}&i=d"
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            r.raise_for_status()
            if not r.text or r.text.strip().lower().startswith("<!doctype"):
                continue
            df = pd.read_csv(pd.compat.StringIO(r.text))
            if df is None or df.empty:
                continue
            df.rename(columns={"Date":"time","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df.dropna()
            cols = ["open","high","low","close","volume"]
            if set(cols).issubset(df.columns):
                return df[cols]
        except Exception:
            continue
    return None


def get_prices(symbol: str, days: int) -> pd.DataFrame | None:
    df = _yf_download(symbol, days)
    if df is not None and not df.empty:
        return df
    df = _stooq_csv(symbol)
    if df is not None and not df.empty:
        # keep only the most recent ~days+80 bars
        return df.sort_index().tail(days + 80)
    return None


# ---------- Google Sheets ----------
def _get_gsheet():
    if not (GOOGLE_SHEETS_SHEET_ID and GOOGLE_SHEETS_JSON):
        return None, "Missing GOOGLE_SHEETS env vars"
    try:
        creds_info = json.loads(GOOGLE_SHEETS_JSON)
    except json.JSONDecodeError as e:
        return None, f"Bad GOOGLE_SHEETS_JSON: {e}"
    if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
        creds_info["private_key"] = creds_info["private_key"].strip()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(GOOGLE_SHEETS_SHEET_ID)
        try:
            ws = sh.worksheet("logs")
        except Exception:
            ws = sh.sheet1
        values = ws.get_all_values()
        if not values or values[0] != LOG_COLUMNS:
            ws.update("A1", [LOG_COLUMNS])
        return ws, getattr(creds, "service_account_email", "service-account")
    except GSpreadAPIError as e:
        return None, f"Sheets API error: {e}"
    except Exception as e:
        return None, f"Auth/open failed: {e}"


def _already_logged_today(ws, note_value: str) -> bool:
    """Check if we already wrote EOD rows today (PT) with the given note."""
    try:
        values = ws.get_all_values()
        if not values or len(values) < 2:
            return False
        today = now_la().date()
        note_idx = LOG_COLUMNS.index("note")
        ts_idx = LOG_COLUMNS.index("ts_pt")
        for row in reversed(values[1:]):  # scan from bottom
            if len(row) <= max(note_idx, ts_idx):
                continue
            if row[note_idx] == note_value:
                ts = datetime.strptime(row[ts_idx], "%Y-%m-%d %H:%M:%S")
                if ts.date() == today:
                    return True
                # If we reached yesterday, we can stop
                if ts.date() < today:
                    break
        return False
    except Exception:
        # If check fails, we err on "not logged" (logger is still idempotent by content equality in most cases)
        return False


def append_prediction_rows(ws, rows):
    payload = [[r.get(k, "") for k in LOG_COLUMNS] for r in rows]
    ws.append_rows(payload, value_input_option="RAW")


# ---------- Model / Scaler ----------
def load_cnn_lstm(symbol: str | None = None):
    if tf is None:
        return None
    candidates = []
    if symbol:
        candidates.append(Path("models") / f"{symbol.upper()}_cnn_lstm.keras")
    candidates.append(Path("models") / "cnn_lstm_ALL.keras")
    for p in candidates:
        if p.exists():
            try:
                return tf.keras.models.load_model(p)
            except Exception as e:
                print(f"[warn] load model failed for {p.name}: {e}", file=sys.stderr)
    return None


def load_scaler():
    for p in [
        Path("models") / "scaler_ALL.pkl",
        Path("models") / "scaler.pkl",
        Path("scaler_ALL.pkl"),
        Path("scaler.pkl"),
    ]:
        if p.exists():
            try:
                obj = joblib.load(p)
                if isinstance(obj, dict) and "scaler" in obj:
                    obj = obj["scaler"]
                return obj
            except Exception as e:
                print(f"[warn] load scaler failed for {p.name}: {e}", file=sys.stderr)
    return None


def _scaler_can_api(s) -> bool:
    return hasattr(s, "transform") and hasattr(s, "inverse_transform")


def _scaler_get_scale_min(s):
    """Return (scale_, min_) arrays from sklearn MinMaxScaler, or dict forms."""
    try:
        if hasattr(s, "scale_") and hasattr(s, "min_"):
            return np.asarray(s.scale_), np.asarray(s.min_)
    except Exception:
        pass
    if isinstance(s, dict):
        node = s
        # flat dict
        if "scale_" in node and "min_" in node:
            return np.asarray(node["scale_"]), np.asarray(node["min_"])
        # common nesting
        for key in ("params", "state"):
            sub = node.get(key)
            if isinstance(sub, dict) and "scale_" in sub and "min_" in sub:
                return np.asarray(sub["scale_"]), np.asarray(sub["min_"])
    return None, None


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if {"Open","High","Low","Close","Volume"}.issubset(d.columns):
        d = d.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    elif {"o","h","l","close","v"}.issubset(d.columns):
        d = d.rename(columns={"o":"open","h":"high","l":"low","v":"volume"})
    return d


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    if "close" not in out.columns:
        return pd.DataFrame()
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    return out.dropna()


def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    if tf is None:
        return None
    model = load_cnn_lstm(symbol) or load_cnn_lstm(None)
    if model is None:
        return None

    d = _normalize_ohlcv(df).dropna(subset=FEATURE_COLS)
    if len(d) < lookback:
        return None

    scaler = load_scaler()
    if scaler is None:
        return None

    try:
        n_feats = int(model.input_shape[-1])
    except Exception:
        n_feats = 5

    scale_, min_ = _scaler_get_scale_min(scaler)

    try:
        if n_feats == 1:
            seq = d["close"].tail(lookback).to_numpy().reshape(-1, 1)
            if scale_ is not None and min_ is not None:
                X = (seq * scale_[CLOSE_IDX] + min_[CLOSE_IDX])[np.newaxis, :, :]
            elif _scaler_can_api(scaler):
                X = scaler.transform(seq)[np.newaxis, :, :]
            else:
                return None
        else:
            block = d[FEATURE_COLS].tail(lookback).to_numpy()
            if scale_ is not None and min_ is not None:
                block_scaled = block * scale_ + min_
            elif _scaler_can_api(scaler):
                block_scaled = scaler.transform(block)
            else:
                return None
            X = block_scaled[np.newaxis, :, :]
    except Exception:
        return None

    try:
        y_scaled = float(model.predict(X, verbose=0)[0][0])
    except Exception:
        return None

    try:
        if scale_ is not None and min_ is not None:
            return float((y_scaled - min_[CLOSE_IDX]) / scale_[CLOSE_IDX])
        elif _scaler_can_api(scaler):
            dummy = np.zeros((1, len(FEATURE_COLS)))
            dummy[0, CLOSE_IDX] = y_scaled
            return float(scaler.inverse_transform(dummy)[0, CLOSE_IDX])
        else:
            return None
    except Exception:
        return None


def load_linear_model():
    p = Path("models") / "linear_model.pkl"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def predict_linear(df: pd.DataFrame) -> float | None:
    model = load_linear_model()
    if model is None:
        return None
    feat = featurize(df)
    if feat.empty:
        return None
    try:
        X_last = feat.iloc[[-1]][["MA10","MA50"]]
        return float(model.predict(X_last)[0])
    except Exception:
        return None


# ---------- Main ----------
def main():
    # Run only after market close (≈ 1:00 PM PT). Give a 5 min buffer.
    t = now_la()
    if t.weekday() >= 5:
        print("Weekend — skipping EOD log.")
        return
    close_today = t.replace(hour=13, minute=0, second=0, microsecond=0)
    if t < close_today + timedelta(minutes=5):
        print("Too early (pre-close) — skipping EOD log.")
        return

    ws, info = _get_gsheet()
    if ws is None:
        print(f"Sheets unavailable: {info}", file=sys.stderr)
        sys.exit(1)

    if _already_logged_today(ws, NOTE_VALUE):
        print("EOD already logged today — nothing to do.")
        return

    rows = []
    for sym in WATCHLIST:
        df = get_prices(sym, DAYS)
        if df is None or df.empty or "close" not in df.columns:
            print(f"[skip] {sym}: no data")
            continue

        last = float(df["close"].iloc[-1])

        # 1) Try CNN-LSTM
        pred = predict_next_close_cnnlstm(sym, df, lookback=DEFAULT_LOOKBACK)
        model_used = "CNN-LSTM" if pred is not None else "—"

        # 2) Linear fallback (pretrained)
        if pred is None:
            pred = predict_linear(df)
            if pred is not None:
                model_used = "Linear(model)"

        # 3) Naive fallback
        if pred is None:
            pred = last
            model_used = "Naive"

        pct = 100.0 * (float(pred) - last) / max(1e-12, last)
        ma10 = float(df["close"].rolling(10).mean().iloc[-1]) if len(df) >= 10 else ""
        ma50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else ""

        rows.append({
            "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "ts_pt":  now_la().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym,
            "model":  model_used,
            "lookback": (DEFAULT_LOOKBACK if model_used.startswith("CNN") else ""),
            "days_history": DAYS,
            "last_close": round(last, 4),
            "predicted": round(float(pred), 4),
            "pct_change": round(pct, 4),
            "in_window": False,
            "ma10": round(ma10, 4) if ma10 != "" else "",
            "ma50": round(ma50, 4) if ma50 != "" else "",
            "note": NOTE_VALUE,
        })

    if not rows:
        print("No rows to log.")
        return

    append_prediction_rows(ws, rows)
    print(f"Logged {len(rows)} rows to Google Sheets as {NOTE_VALUE}.")


if __name__ == "__main__":
    main()
