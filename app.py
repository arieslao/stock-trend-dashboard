# ---------- Imports ----------
import os, json, requests
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional

# Optional / safe TF import (only used if you pick the CNN-LSTM model)
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # graceful fallback to Linear if TF isn’t available

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

import streamlit as st

# ---------- Constants ----------
FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")
DEFAULT_LOOKBACK = 60  # sequence length for CNN-LSTM

LOG_COLUMNS = [
    "ts_utc", "ts_pt", "symbol", "model", "lookback",
    "days_history", "last_close", "predicted", "pct_change",
    "in_window", "ma10", "ma50", "note"
]

# ---------- Google Sheets (optional) ----------
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


@st.cache_resource(show_spinner=False)
def _get_gsheet():
    """Return (worksheet, account_email) or (None, message). Supports two secrets styles:
       1) GOOGLE_SHEETS_JSON  (recommended: full Google JSON as-is)
       2) [gcp_service_account]  (TOML table with the same fields)
       Sheet id may be in GOOGLE_SHEETS_SHEET_ID or [gsheets].sheet_id.
    """
    if not (gspread and Credentials):
        return None, "gspread / google-auth not installed"

    # ---- sheet id ----
    sheet_id = None
    if hasattr(st, "secrets"):
        sheet_id = (
            st.secrets.get("GOOGLE_SHEETS_SHEET_ID")
            or (st.secrets.get("gsheets", {}) or {}).get("sheet_id")
        )
    sheet_id = sheet_id or os.getenv("GOOGLE_SHEETS_SHEET_ID")
    if not sheet_id:
        return None, "Missing GOOGLE_SHEETS_SHEET_ID"

    # ---- credentials ----
    creds_info = None
    if hasattr(st, "secrets"):
        if "GOOGLE_SHEETS_JSON" in st.secrets:  # full JSON as a string
            try:
                creds_info = json.loads(st.secrets["GOOGLE_SHEETS_JSON"])
            except Exception:
                return None, "GOOGLE_SHEETS_JSON is not valid JSON"
        elif "gcp_service_account" in st.secrets:  # TOML table
            creds_info = dict(st.secrets["gcp_service_account"])
    if creds_info is None and os.getenv("GOOGLE_SHEETS_JSON"):
        try:
            creds_info = json.loads(os.getenv("GOOGLE_SHEETS_JSON"))
        except Exception:
            return None, "Env GOOGLE_SHEETS_JSON is not valid JSON"

    if not creds_info:
        return None, "Missing service account JSON (GOOGLE_SHEETS_JSON or [gcp_service_account])"

    # Normalize the private key (common copy/paste issue)
    if "private_key" in creds_info and isinstance(creds_info["private_key"], str):
        creds_info["private_key"] = creds_info["private_key"].strip()

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        try:
            ws = sh.worksheet("logs")
        except Exception:
            ws = sh.sheet1

        # ensure header row
        values = ws.get_all_values()
        if not values or values[0] != LOG_COLUMNS:
            ws.update("A1", [LOG_COLUMNS])

        return ws, getattr(creds, "service_account_email", "service-account")
    except Exception as e:
        return None, f"Auth/open failed: {e}"


def append_prediction_rows(rows: List[Dict]):
    ws, _ = _get_gsheet()
    if ws is None:
        return
    payload = [[r.get(k, "") for k in LOG_COLUMNS] for r in rows]
    try:
        ws.append_rows(payload, value_input_option="RAW")
    except Exception:
        pass


# ---------- App config ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption(
    "Education only. Data: Yahoo (chart JSON). "
    "Models: CNN-LSTM (default) with Linear MA10/MA50 fallback. "
    "Auto-fetch primes once per session; use the button for ad-hoc refresh."
)

# ---------- Time window helpers (Pacific) ----------
LA = ZoneInfo("America/Los_Angeles")
WINDOWS_PT = [dtime(6, 30), dtime(12, 0)]
WINDOW_WIDTH_MIN = 7


def now_la() -> datetime:
    return datetime.now(LA)


def in_window() -> bool:
    t = now_la()
    for w in WINDOWS_PT:
        start = t.replace(hour=w.hour, minute=w.minute, second=0, microsecond=0)
        end = start + timedelta(minutes=WINDOW_WIDTH_MIN)
        if start <= t <= end:
            return True
    return False


def seconds_until_next_window() -> int:
    t = now_la()
    candidates = []
    for w in WINDOWS_PT:
        start = t.replace(hour=w.hour, minute=w.minute, second=0, microsecond=0)
        if start > t:
            candidates.append(start)
    if not candidates:
        w = WINDOWS_PT[0]
        candidates.append((t + timedelta(days=1)).replace(hour=w.hour, minute=w.minute, second=0, microsecond=0))
    return max(0, int((min(candidates) - t).total_seconds()))


# ---------- Yahoo fetchers ----------
YH_HEADERS = {"User-Agent": "Mozilla/5.0"}


@st.cache_data(ttl=600)
def get_candles(symbol: str, days: int = 180, ignore_windows: bool = False) -> pd.DataFrame:
    if not ignore_windows and not in_window():
        raise RuntimeError("Outside fetch window")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": f"{days}d"}
    r = requests.get(url, params=params, headers=YH_HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json()
    result = (j.get("chart", {}) or {}).get("result") or []
    if not result:
        raise RuntimeError("Yahoo candles: empty result")
    res = result[0]
    ts = res.get("timestamp")
    q = (res.get("indicators", {}) or {}).get("quote", [{}])[0]
    if not ts or not q:
        raise RuntimeError("Yahoo candles: missing keys")
    df = pd.DataFrame({
        "time": pd.to_datetime(ts, unit="s"),
        "o": q.get("open"),
        "h": q.get("high"),
        "l": q.get("low"),
        "close": q.get("close"),
        "v": q.get("volume"),
    }).dropna()
    df.set_index("time", inplace=True)
    return df[["o", "h", "l", "close", "v"]]


# ---------- History cache helpers ----------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with columns named: open, high, low, close, volume."""
    d = df.copy()
    if {"o", "h", "l", "close", "v"}.issubset(d.columns):
        d = d.rename(columns={"o": "open", "h": "high", "l": "low", "v": "volume"})
    elif {"Open", "High", "Low", "Close"}.issubset(d.columns):
        d = d.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
    return d


def _put_cache(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    st.session_state.setdefault("history_cache", {})[symbol] = df
    return df


def _get_cache(symbol: str) -> Optional[pd.DataFrame]:
    return st.session_state.get("history_cache", {}).get(symbol)


def fetch_history(symbol: str, days: int, *, allow_live_fallback: bool = True) -> Optional[pd.DataFrame]:
    """
    1) Return from our in-memory cache if present.
    2) Else try the cached @st.cache_data call (respects windows).
    3) Else (optionally) one-time live fetch with ignore_windows=True, then cache it.
    """
    df = _get_cache(symbol)
    if df is not None and not df.empty:
        return df

    try:
        df2 = get_candles(symbol, days=days)  # may raise if outside window
        if df2 is not None and not df2.empty:
            return _put_cache(symbol, df2)
    except Exception:
        pass

    if allow_live_fallback and not st.session_state.get(f"_once_live_{symbol}", False):
        try:
            df3 = get_candles(symbol, days=days, ignore_windows=True)
            st.session_state[f"_once_live_{symbol}"] = True
            if df3 is not None and not df3.empty:
                return _put_cache(symbol, df3)
        except Exception:
            st.session_state[f"_once_live_{symbol}"] = True

    return None


def prime_watchlist_cache(symbols: list[str], days: int) -> list[str]:
    """Bulk live fetch (ignore windows) and put into cache. Returns list of successes."""
    ok = []
    for s in symbols:
        try:
            df = get_candles(s, days=days, ignore_windows=True)
            if df is not None and not df.empty:
                _put_cache(s, df)
                ok.append(s)
        except Exception:
            pass
    return ok


# ---------- Model helpers ----------
@st.cache_resource(show_spinner=False)
def load_cnn_lstm(symbol: Optional[str] = None):
    if tf is None:
        return None
    candidates = []
    if symbol:
        candidates.append(Path("models") / f"{symbol.upper()}_cnn_lstm.keras")
    candidates.append(Path("models") / "cnn_lstm_ALL.keras")
    for p in candidates:
        if p.exists():
            try:
                model = tf.keras.models.load_model(p)
                st.session_state["cnn_model_path"] = str(p)
                return model
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load model: {e}")
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    for p in [
        Path("models") / "scaler_ALL.pkl",
        Path("models") / "scaler.pkl",
        Path("scaler_ALL.pkl"),
        Path("scaler.pkl"),
    ]:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load scaler: {e}")
    return None

# -------- Robust scaler utilities --------
def _scaler_can_api(s) -> bool:
    return hasattr(s, "transform") and hasattr(s, "inverse_transform")

def _dig(d: dict, *keys):
    """Safely dig into nested dicts."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def _scaler_get_scale_min(s):
    """
    Return (scale_, min_) as numpy arrays if available.
    Supports:
      - real MinMaxScaler (attrs)
      - dict with scale_/min_
      - dict with nested {"params": {...}} or {"state": {...}}
      - dict with only data_min_/data_max_ (+feature_range) -> derive scale_/min_
    """
    # Real sklearn scaler
    try:
        if hasattr(s, "scale_") and hasattr(s, "min_"):
            return np.asarray(s.scale_), np.asarray(s.min_)
    except Exception:
        pass

    # Dict forms (flat or nested)
    if isinstance(s, dict):
        for node in (s, _dig(s, "params"), _dig(s, "state")):
            if isinstance(node, dict):
                # Direct scale_/min_
                if "scale_" in node and "min_" in node:
                    return np.asarray(node["scale_"]), np.asarray(node["min_"])
                if "scale" in node and "min" in node:
                    return np.asarray(node["scale"]), np.asarray(node["min"])
                # Derive from data_min_, data_max_, feature_range
                dmin = node.get("data_min_")
                dmax = node.get("data_max_")
                drng = node.get("data_range_")
                fr = node.get("feature_range", (0.0, 1.0))
                if dmin is not None and (dmax is not None or drng is not None):
                    dmin = np.asarray(dmin, dtype=float)
                    if drng is None:
                        dmax = np.asarray(dmax, dtype=float)
                        drng = dmax - dmin
                    else:
                        drng = np.asarray(drng, dtype=float)
                    fr0, fr1 = float(fr[0]), float(fr[1])
                    scale_ = (fr1 - fr0) / drng
                    min_ = fr0 - dmin * scale_
                    return scale_, min_

    return None, None


# ---------- Features + models ----------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    if 'close' not in out.columns:
        return pd.DataFrame()
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    return out.dropna()

def fit_and_predict(df_features: pd.DataFrame) -> Optional[float]:
    if df_features is None or df_features.empty:
        return None
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    try:
        model = LinearRegression().fit(X, y)
        pred = float(model.predict(df_features.iloc[[-1]][["MA10", "MA50"]])[0])
        return pred
    except Exception:
        return None

def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    """
    Robust CNN-LSTM predictor that supports:
      • 5-feature OHLCV models, or
      • 1-feature (close-only) models
    and works with either a real MinMaxScaler OR a dict with scale_/min_ (or data_min_/data_max_).
    """
    if tf is None:
        return None

    model = load_cnn_lstm(symbol)
    if model is None:
        return None

    # Normalize + clean
    d = _normalize_ohlcv(df).dropna(subset=FEATURE_COLS)
    if len(d) < lookback:
        return None

    scaler = load_scaler()
    if scaler is None:
        return None

    # Expected feature count
    try:
        n_feats = int(model.input_shape[-1])
    except Exception:
        n_feats = 5

    scale_, min_ = _scaler_get_scale_min(scaler)

    try:
        if n_feats == 1:
            seq = d["close"].tail(lookback).to_numpy().reshape(-1, 1)  # (L,1)
            if scale_ is not None and min_ is not None:
                X = (seq * scale_[CLOSE_IDX] + min_[CLOSE_IDX])[np.newaxis, :, :]  # (1,L,1)
            elif _scaler_can_api(scaler):
                X = scaler.transform(seq)[np.newaxis, :, :]
            else:
                return None
        else:
            block = d[FEATURE_COLS].tail(lookback).to_numpy()  # (L,5)
            if scale_ is not None and min_ is not None:
                block_scaled = block * scale_ + min_
            elif _scaler_can_api(scaler):
                block_scaled = scaler.transform(block)
            else:
                return None
            X = block_scaled[np.newaxis, :, :]
    except Exception:
        return None

    # Predict scaled close
    try:
        y_scaled = float(model.predict(X, verbose=0)[0][0])
    except Exception:
        return None

    # Inverse transform the close
    try:
        if scale_ is not None and min_ is not None:
            # y_scaled = y*scale + min  => y = (y_scaled - min)/scale
            return float((y_scaled - min_[CLOSE_IDX]) / scale_[CLOSE_IDX])
        elif _scaler_can_api(scaler):
            dummy = np.zeros((1, len(FEATURE_COLS)), dtype=float)
            dummy[0, CLOSE_IDX] = y_scaled
            return float(scaler.inverse_transform(dummy)[0, CLOSE_IDX])
        else:
            return None
    except Exception:
        return None


# ---------- Sidebar / Controls ----------
default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA"]
with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect("Watchlist", default_watchlist, default=default_watchlist)
    days = st.slider("History (days)", 90, 365, 180, step=5)
    alert_pct = st.slider("Alert threshold (%)", 0.5, 5.0, 2.0, step=0.25)

    st.subheader("Model")
    MODEL_LINEAR = "Linear (MA10/MA50)"
    MODEL_CNN = "CNN-LSTM (pretrained)"
    # Default to CNN-LSTM (index=1)
    model_choice = st.selectbox(
        "Choose model",
        (MODEL_LINEAR, MODEL_CNN),
        index=1,
        help="Linear is simple; CNN-LSTM uses a pretrained deep model (falls back to Linear if unavailable)."
    )
    st.session_state["model_choice"] = model_choice

    mins = seconds_until_next_window() // 60
    if in_window():
        st.success("✅ Inside fetch window (PT). Live pulls allowed.")
    else:
        st.info(f"🕰️ Outside fetch window. Using cache if available. Next window in ~{mins} min.")


def using_cnn() -> bool:
    return st.session_state.get("model_choice") == "CNN-LSTM (pretrained)"


# ---------- Auto-prime once per session ----------
if not st.session_state.get("_session_primed", False):
    st.session_state["_session_primed"] = True
    if watchlist:
        with st.spinner("Priming cache for your watchlist (one-time)…"):
            try:
                prime_watchlist_cache(watchlist, days)
            except Exception as e:
                st.warning(f"Auto-prime skipped: {e}")


# ---------- Diagnostics / Integrations ----------
with st.expander("🔧 Diagnostics & Integrations", expanded=False):
    st.write("Now (PT):", now_la().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("In window:", in_window())
    st.write("Next window (min):", seconds_until_next_window() // 60)

    mdl = load_cnn_lstm()  # generic/all-symbol model
    st.write("CNN-LSTM loaded:", bool(mdl))
    if mdl:
        try:
            st.write("Model input shape:", mdl.input_shape)
            st.write("Model expects features:", mdl.input_shape[-1])
        except Exception:
            pass

    sc = load_scaler()
    st.write("Scaler loaded:", sc is not None)
    if isinstance(sc, dict):
        # surface which keys exist so you can see what’s inside your saved scaler
        st.write("Scaler keys:", list(sc.keys())[:12])

    ws, info = _get_gsheet()
    if ws is None:
        st.write("Google Sheets: not configured –", info)
    else:
        st.write("Google Sheets: connected ✅  (worksheet:", ws.title, ")")
        st.write("Share this sheet with:", info)


# ---------- Refresh watchlist (ad-hoc) ----------
with st.container():
    if st.button(
        "🔄 Refresh watchlist (live fetch ALL now)",
        help="Fetch fresh candles for every symbol once, ignoring the time windows."
    ):
        if not watchlist:
            st.info("Your watchlist is empty.")
        else:
            with st.spinner("Fetching fresh data…"):
                prime_watchlist_cache(watchlist, days)


# ---------- Watchlist scoring / table ----------
rows: List[Dict] = []

with st.spinner("Scoring watchlist…"):
    for s in (watchlist or []):
        try:
            # Prefer cache / normal path, then (if needed) one-shot live fallback
            d = fetch_history(s, days, allow_live_fallback=False)
            if d is None or d.empty:
                d = fetch_history(s, days, allow_live_fallback=True)
            if d is None or d.empty:
                raise RuntimeError("no history")

            dN = _normalize_ohlcv(d)
            if "close" not in dN.columns:
                raise RuntimeError("missing close")

            last_close = float(dN["close"].iloc[-1])

            model_used_row = "—"
            next_pred = None

            if using_cnn():
                next_pred = predict_next_close_cnnlstm(s, dN, lookback=DEFAULT_LOOKBACK)
                if next_pred is not None:
                    model_used_row = "CNN-LSTM"

            if next_pred is None:
                feat_row = featurize(dN)
                lin_pred = fit_and_predict(feat_row)
                if lin_pred is not None:
                    next_pred = lin_pred
                    model_used_row = "Linear (fallback)"

            if next_pred is None:
                # Last resort so the table always fills (0% change)
                next_pred = last_close
                model_used_row = "Naive"

            ret = 100.0 * (float(next_pred) - last_close) / last_close
            signal = "BUY↑" if ret >= alert_pct else ("SELL↓" if ret <= -alert_pct else "HOLD")

            rows.append({
                "Symbol": s,
                "Last": round(last_close, 2),
                "Predicted": round(float(next_pred), 2),
                "Δ%": round(float(ret), 2),
                "Signal": signal,
                "Model": model_used_row
            })
        except Exception:
            rows.append({
                "Symbol": s,
                "Last": None, "Predicted": None, "Δ%": None,
                "Signal": "ERR", "Model": "—"
            })

st.subheader("Watchlist signals")

if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%", ascending=False, na_position="last")
    st.dataframe(table, use_container_width=True)

    # Optional: log predictions to Google Sheets
    try:
        to_log = []
        for r in rows:
            if r.get("Predicted") is None or r.get("Last") is None:
                continue
            to_log.append({
                "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "ts_pt": now_la().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": r["Symbol"],
                "model": r["Model"],
                "lookback": (DEFAULT_LOOKBACK if "CNN" in (r["Model"] or "") else ""),
                "days_history": days,
                "last_close": r["Last"],
                "predicted": r["Predicted"],
                "pct_change": r["Δ%"],
                "in_window": in_window(),
                "ma10": "", "ma50": "",
                "note": "watchlist",
            })
        if to_log:
            append_prediction_rows(to_log)
    except Exception:
        pass
else:
    st.info("Your watchlist is empty. Add symbols in the sidebar.")
