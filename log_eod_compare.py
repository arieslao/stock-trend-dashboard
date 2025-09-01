# log_eod_compare.py
# EOD model-comparison logger: CNN-LSTM vs Linear vs Hybrid.
# - Appends one row per symbol per trading day (PT)
# - Prevents duplicates
# - On each run, updates yesterday's rows with actual close + error metrics

import os, sys, json, requests
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib

# ---------- Optional TF for CNN-LSTM ----------
try:
    import tensorflow as tf
except Exception:
    tf = None  # graceful fallback

# ---------- Google Sheets ----------
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError as GSpreadAPIError

# ---------- Settings ----------
LA = ZoneInfo("America/Los_Angeles")
WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "AAPL,MSFT,GOOGL,TSLA").split(",") if s.strip()]
DAYS = int(os.getenv("DAYS_HISTORY", "180"))
DEFAULT_LOOKBACK = 60

GOOGLE_SHEETS_SHEET_ID = os.getenv("GOOGLE_SHEETS_SHEET_ID")
GOOGLE_SHEETS_JSON = os.getenv("GOOGLE_SHEETS_JSON")

SHEET_NAME = os.getenv("COMPARE_SHEET", "model_compare")  # put results in a dedicated tab

FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (model-compare-eod)"}

COLUMNS = [
    # identity/time
    "ts_utc", "ts_pt", "date_pt", "symbol",
    # inputs
    "last_close",
    # predictions
    "pred_cnn", "pred_linear", "pred_hybrid",
    "pct_cnn", "pct_linear", "pct_hybrid",
    # winner for the day (if you want to pick one)
    "chosen_model", "chosen_pred", "pct_chosen",
    # realized & errors (filled on next run)
    "actual_close",
    "err_cnn", "err_linear", "err_hybrid", "err_chosen",
    # marker
    "note"
]
NOTE_VALUE = "compare_eod"

# ---------- Helpers ----------
def now_la() -> datetime:
    return datetime.now(LA)

def is_weekend(dt_la: datetime) -> bool:
    return dt_la.weekday() >= 5

def previous_business_day(dt_la: datetime) -> datetime:
    d = dt_la.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return datetime(d.year, d.month, d.day, tzinfo=LA)

# ---------- Data fetchers (yfinance + stooq fallback) ----------
def _yf_download(symbol: str, days: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        hist = yf.download(symbol, period=f"{days+80}d", interval="1d", auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            return None
        hist = hist.dropna()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        df = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        cols = ["open","high","low","close","volume"]
        return df[cols] if set(cols).issubset(df.columns) else None
    except Exception:
        return None

def _stooq_csv(symbol: str) -> pd.DataFrame | None:
    for code in (f"{symbol.lower()}.us", symbol.lower()):
        try:
            url = f"https://stooq.com/q/d/l/?s={code}&i=d"
            r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            r.raise_for_status()
            txt = r.text or ""
            if not txt or txt.strip().lower().startswith("<!doctype"):
                continue
            df = pd.read_csv(pd.compat.StringIO(txt))
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
        return df.sort_index().tail(days + 80)
    return None

# ---------- Sheets ----------
def _open_compare_sheet():
    if not (GOOGLE_SHEETS_SHEET_ID and GOOGLE_SHEETS_JSON):
        return None, "Missing GOOGLE_SHEETS_* env vars"
    try:
        creds_info = json.loads(GOOGLE_SHEETS_JSON)
    except json.JSONDecodeError as e:
        return None, f"Bad GOOGLE_SHEETS_JSON: {e}"
    if isinstance(creds_info.get("private_key"), str):
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
            ws = sh.worksheet(SHEET_NAME)
        except Exception:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=2000, cols=len(COLUMNS))

        values = ws.get_all_values()
        if not values or values[0] != COLUMNS:
            ws.clear()
            ws.update("A1", [COLUMNS])
        return ws, getattr(creds, "service_account_email", "service-account")
    except GSpreadAPIError as e:
        return None, f"Sheets API error: {e}"
    except Exception as e:
        return None, f"Auth/open failed: {e}"

def _row_map(values: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(values)}

def _already_logged_today(ws, symbol: str, date_pt_str: str) -> bool:
    try:
        vals = ws.get_all_values()
        if not vals or len(vals) < 2:
            return False
        header = vals[0]
        idx = _row_map(header)
        for row in reversed(vals[1:]):
            if len(row) <= max(idx.get("symbol", 0), idx.get("date_pt", 0)):
                continue
            if row[idx["symbol"]] == symbol and row[idx["date_pt"]] == date_pt_str:
                return True
        return False
    except Exception:
        return False

def append_rows(ws, rows: list[dict]):
    payload = [[r.get(k, "") for k in COLUMNS] for r in rows]
    ws.append_rows(payload, value_input_option="RAW")

def update_yesterday_errors(ws, ydate_str: str, todays_closes: dict[str, float]):
    """Fill actual_close and error columns for rows dated ydate_str."""
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return
    header = vals[0]
    idx = _row_map(header)

    # Build batch updates (few symbols -> per-row updates are fine)
    for r_i, row in enumerate(vals[1:], start=2):
        if len(row) < len(COLUMNS):
            continue
        if row[idx["date_pt"]] != ydate_str:
            continue
        sym = row[idx["symbol"]]
        actual = todays_closes.get(sym)
        if actual is None:
            continue
        # only update if empty
        if row[idx["actual_close"]]:
            continue

        # parse predictions safely
        def _to_f(x):
            try:
                return float(x)
            except Exception:
                return None

        pred_cnn   = _to_f(row[idx["pred_cnn"]])
        pred_lin   = _to_f(row[idx["pred_linear"]])
        pred_hyb   = _to_f(row[idx["pred_hybrid"]])
        chosen     = _to_f(row[idx["chosen_pred"]])

        updates = {}
        updates["actual_close"] = actual
        if pred_cnn  is not None: updates["err_cnn"]    = round(actual - pred_cnn, 6)
        if pred_lin  is not None: updates["err_linear"] = round(actual - pred_lin, 6)
        if pred_hyb  is not None: updates["err_hybrid"] = round(actual - pred_hyb, 6)
        if chosen    is not None: updates["err_chosen"] = round(actual - chosen, 6)

        # Write updates
        for k, v in updates.items():
            ws.update_cell(r_i, idx[k] + 1, v)

# ---------- Models ----------
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

def load_cnn_lstm(symbol: str | None = None):
    if tf is None:
        return None
    paths = []
    if symbol:
        paths.append(Path("models") / f"{symbol.upper()}_cnn_lstm.keras")
    paths.append(Path("models") / "cnn_lstm_ALL.keras")
    for p in paths:
        if p.exists():
            try:
                return tf.keras.models.load_model(p)
            except Exception as e:
                print(f"[warn] failed to load {p.name}: {e}", file=sys.stderr)
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
                print(f"[warn] load scaler failed ({p.name}): {e}", file=sys.stderr)
    return None

def _scaler_can_api(s) -> bool:
    return hasattr(s, "transform") and hasattr(s, "inverse_transform")

def _scaler_get_scale_min(s):
    try:
        if hasattr(s, "scale_") and hasattr(s, "min_"):
            return np.asarray(s.scale_), np.asarray(s.min_)
    except Exception:
        pass
    if isinstance(s, dict):
        node = s
        if "scale_" in node and "min_" in node:
            return np.asarray(node["scale_"]), np.asarray(node["min_"])
        for key in ("params", "state"):
            sub = node.get(key)
            if isinstance(sub, dict) and "scale_" in sub and "min_" in sub:
                return np.asarray(sub["scale_"]), np.asarray(sub["min_"])
    return None, None

def predict_cnn(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> float | None:
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

def pct_change(pred: float | None, last: float) -> float | str:
    if pred is None:
        return ""
    return round(100.0 * (pred - last) / max(1e-12, last), 6)

# ---------- Main ----------
def main():
    t = now_la()
    if is_weekend(t):
        print("Weekend — skip.")
        return

    # after close (≈1:00pm PT), allow a small buffer
    close_today = t.replace(hour=13, minute=0, second=0, microsecond=0)
    if t < close_today + timedelta(minutes=5):
        print("Too early (pre-close) — skip.")
        return

    ws, info = _open_compare_sheet()
    if ws is None:
        print(f"Sheets unavailable: {info}", file=sys.stderr)
        sys.exit(1)

    date_pt_str = t.strftime("%Y-%m-%d")
    rows_to_append = []
    todays_closes: dict[str, float] = {}

    for sym in WATCHLIST:
        df = get_prices(sym, DAYS)
        if df is None or df.empty or "close" not in df.columns:
            print(f"[skip] {sym}: no data")
            continue

        last = float(df["close"].iloc[-1])
        todays_closes[sym] = last

        # prevent duplicate for symbol-day
        if _already_logged_today(ws, sym, date_pt_str):
            print(f"[dup] {sym} already logged for {date_pt_str}")
            continue

        pred_cnn = predict_cnn(sym, df, DEFAULT_LOOKBACK)
        pred_lin = predict_linear(df)

        # Hybrid: mean of available preds
        pred_list = [p for p in (pred_cnn, pred_lin) if p is not None]
        pred_hyb = float(np.mean(pred_list)) if pred_list else None

        # Pick a winner (you can change strategy anytime)
        chosen_model = "Hybrid" if (pred_cnn is not None and pred_lin is not None) else \
                       ("CNN-LSTM" if pred_cnn is not None else \
                        ("Linear" if pred_lin is not None else "Naive"))
        chosen_pred = pred_hyb if chosen_model == "Hybrid" else \
                      (pred_cnn if chosen_model == "CNN-LSTM" else \
                       (pred_lin if chosen_model == "Linear" else last))

        rows_to_append.append({
            "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "ts_pt":  t.strftime("%Y-%m-%d %H:%M:%S"),
            "date_pt": date_pt_str,
            "symbol": sym,
            "last_close": round(last, 6),

            "pred_cnn":   ("" if pred_cnn is None else round(pred_cnn, 6)),
            "pred_linear":("" if pred_lin is None else round(pred_lin, 6)),
            "pred_hybrid":("" if pred_hyb is None else round(pred_hyb, 6)),

            "pct_cnn":    pct_change(pred_cnn, last),
            "pct_linear": pct_change(pred_lin, last),
            "pct_hybrid": pct_change(pred_hyb, last),

            "chosen_model": chosen_model,
            "chosen_pred":  round(float(chosen_pred), 6),
            "pct_chosen":   pct_change(chosen_pred, last),

            "actual_close": "",  # will be filled tomorrow
            "err_cnn":  "",
            "err_linear": "",
            "err_hybrid": "",
            "err_chosen": "",

            "note": NOTE_VALUE,
        })

    if rows_to_append:
        append_rows(ws, rows_to_append)
        print(f"Appended {len(rows_to_append)} rows for {date_pt_str}.")
    else:
        print("Nothing to append today (all dup/no data).")

    # Back-fill yesterday’s errors
    ydate_pt = previous_business_day(t).strftime("%Y-%m-%d")
    update_yesterday_errors(ws, ydate_pt, todays_closes)
    print("Back-fill done for", ydate_pt)


if __name__ == "__main__":
    main()
