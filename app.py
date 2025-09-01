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
from sklearn.preprocessing import MinMaxScaler  # noqa: F401

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
    """Return (worksheet, account_email) or (None, message)."""
    if not (gspread and Credentials):
        return None, "gspread / google-auth not installed"

    sheet_id = (
        st.secrets.get("GOOGLE_SHEETS_SHEET_ID")
        if hasattr(st, "secrets") else None
    ) or os.getenv("GOOGLE_SHEETS_SHEET_ID")

    creds_info = None
    if hasattr(st, "secrets"):
        if "gcp_service_account" in st.secrets:
            creds_info = st.secrets["gcp_service_account"]
        elif "GOOGLE_SHEETS_JSON" in st.secrets:
            try:
                creds_info = json.loads(st.secrets["GOOGLE_SHEETS_JSON"])
            except Exception:
                pass
    if creds_info is None and os.getenv("GOOGLE_SHEETS_JSON"):
        try:
            creds_info = json.loads(os.getenv("GOOGLE_SHEETS_JSON"))
        except Exception:
            pass

    if not sheet_id:
        return None, "Missing GOOGLE_SHEETS_SHEET_ID"
    if not creds_info:
        return None, "Missing service account JSON"

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
        values = ws.get_all_values()
        if not values or (values and values[0] != LOG_COLUMNS):
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


def make_cnnlstm_input(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    scaler = load_scaler()
    if scaler is None:
        return None, None
    df_n = _normalize_ohlcv(df)
    if not set(FEATURE_COLS).issubset(df_n.columns):
        return None, None
    if len(df_n) < lookback:
        return None, None
    block = df_n[FEATURE_COLS].tail(lookback).copy()
    last_close = float(block["close"].iloc[-1])
    try:
        scaled = scaler.transform(block.values)
    except Exception:
        return None, None
    X = scaled[np.newaxis, :, :]
    return X, last_close


def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    """
    Predict next close with a CNN-LSTM that may have been trained on either:
      - 5 features (open, high, low, close, volume), or
      - 1 feature (close only)
    We detect the expected feature count from model.input_shape[-1].
    Returns a float or None (caller will fall back to Linear).
    """
    if tf is None:
        return None

    model = load_cnn_lstm(symbol)
    if model is None:
        return None

    # Normalize columns
    d = df.copy()
    if {'o','h','l','close','v'}.issubset(d.columns):
        d = d.rename(columns={'o':'open','h':'high','l':'low','v':'volume'})
    elif {'Open','High','Low','Close'}.issubset(d.columns):
        d = d.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})

    if 'close' not in d.columns or len(d) < lookback:
        return None

    scaler = load_scaler()
    if scaler is None:
        return None

    # How many features does the model expect?
    try:
        n_feats = model.input_shape[-1]
    except Exception:
        n_feats = len(FEATURE_COLS)  # best guess

    try:
        if n_feats == 1:
            # Scale close-only using the close parameters of the 5-feature scaler
            if not hasattr(scaler, "scale_") or not hasattr(scaler, "min_"):
                return None
            a = float(scaler.scale_[CLOSE_IDX])  # multiply
            b = float(scaler.min_[CLOSE_IDX])    # add
            seq = d["close"].tail(lookback).to_numpy().reshape(-1, 1)  # (L,1)
            seq_scaled = seq * a + b                                   # (L,1)
            X = seq_scaled[np.newaxis, :, :]                           # (1,L,1)
        else:
            # Assume 5 features OHLCV
            if not set(FEATURE_COLS).issubset(d.columns):
                return None
            block = d[FEATURE_COLS].tail(lookback).to_numpy()          # (L,5)
            block_scaled = scaler.transform(block)                     # (L,5)
            X = block_scaled[np.newaxis, :, :]                         # (1,L,5)
    except Exception:
        return None

    # Predict (scaled close)
    try:
        y_scaled = float(model.predict(X, verbose=0)[0][0])
    except Exception:
        return None

    # Inverse-scale the close using the 5-feature scaler we loaded
    try:
        dummy = np.zeros((1, len(FEATURE_COLS)), dtype=float)
        dummy[0, CLOSE_IDX] = y_scaled
        y = float(scaler.inverse_transform(dummy)[0, CLOSE_IDX])
        return y
    except Exception:
        return None



def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    if 'close' not in out.columns:
        return pd.DataFrame()
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    return out.dropna()


def fit_and_predict(df_features: pd.DataFrame) -> tuple[LinearRegression, float]:
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    model = LinearRegression().fit(X, y)
    pred = float(model.predict(df_features.iloc[[-1]][["MA10", "MA50"]])[0])
    return model, pred


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
    # Set the flag FIRST so we don't loop even if something below fails
    st.session_state["_session_primed"] = True

    # Prime once per session (ignore windows) so the table can render
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
    st.write("CNN-LSTM model file:", st.session_state.get("cnn_model_path"))
    ws, info = _get_gsheet()
    if ws is None:
        st.write("Google Sheets: not configured –", info)
    else:
        st.write("Google Sheets: connected ✅  (worksheet:", ws.title, ")")

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

            # Choose model per the selector
            model_used_row = "Linear"
            next_pred = None

            if using_cnn():
                next_pred = predict_next_close_cnnlstm(s, dN, lookback=DEFAULT_LOOKBACK)
                if next_pred is not None:
                    model_used_row = "CNN-LSTM"
                else:
                    # fallback to Linear
                    feat_row = featurize(dN)
                    if not feat_row.empty:
                        _, next_pred = fit_and_predict(feat_row)
                        model_used_row = "Linear (fallback)"
            else:
                feat_row = featurize(dN)
                if not feat_row.empty:
                    _, next_pred = fit_and_predict(feat_row)

            if next_pred is None:
                raise RuntimeError("prediction failed")

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
