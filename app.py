# ---------- Imports ----------
import os, sys, json, time, requests
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
import pickle

# --- Optional / safe TF import (only used if you pick the CNN-LSTM model) ---
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # we’ll gracefully fall back to Linear if TF isn’t available

# --- CNN-LSTM / scaling ---
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib

import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")  # 3
DEFAULT_LOOKBACK = 60  # sequence length for CNN-LSTM


# Columns / header for the Google Sheet
LOG_COLUMNS = [
    "ts_utc", "ts_pt", "symbol", "model", "lookback",
    "days_history", "last_close", "predicted", "pct_change",
    "in_window", "ma10", "ma50", "note"
]



# --- logging / google sheets (safe/optional) ---
from typing import Optional, List, Dict
import csv

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:  # libs may not be installed
    gspread = None
    Credentials = None

# --- google sheets helpers ---
@st.cache_resource(show_spinner=False)
def _get_gsheet():
    """
    Returns (worksheet, service_account_email) or (None, msg) if unavailable.
    Looks in st.secrets first, then env vars:
      - st.secrets["gcp_service_account"] (dict)
      - st.secrets["GOOGLE_SHEETS_JSON"] (string JSON)
      - os.environ["GOOGLE_SHEETS_JSON"]
      - st.secrets / env: GOOGLE_SHEETS_SHEET_ID
    """
    if not (gspread and Credentials):
        return None, "gspread / google-auth not installed"

    # Sheet ID
    sheet_id = (
        st.secrets.get("GOOGLE_SHEETS_SHEET_ID")
        if hasattr(st, "secrets") else None
    ) or os.getenv("GOOGLE_SHEETS_SHEET_ID")

    # Service account JSON (as dict)
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
        return None, "Missing service account JSON (st.secrets or env)"

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        # Use 'logs' worksheet if present; else sheet1
        try:
            ws = sh.worksheet("logs")
        except Exception:
            ws = sh.sheet1
        # Ensure header
        values = ws.get_all_values()
        if not values or (values and values[0] != LOG_COLUMNS):
            ws.update("A1", [LOG_COLUMNS])
        return ws, getattr(creds, "service_account_email", "service-account")
    except Exception as e:
        return None, f"Auth/open failed: {e}"


def append_prediction_rows(rows: List[Dict]):
    """
    Append a list of row-dicts to Google Sheet.
    Each row dict can contain keys from LOG_COLUMNS; missing keys are blank.
    Falls back silently if sheet is unavailable.
    """
    ws, info = _get_gsheet()
    if ws is None:
        # Optionally write a local CSV as a fallback
        # with open("pred_logs.csv", "a", newline="") as f:
        #     w = csv.writer(f)
        #     for r in rows:
        #         w.writerow([r.get(k, "") for k in LOG_COLUMNS])
        return  # no-op if sheet not configured

    payload = [[r.get(k, "") for k in LOG_COLUMNS] for r in rows]
    try:
        ws.append_rows(payload, value_input_option="RAW")
    except Exception as e:
        # Silent skip; keep app behavior unchanged
        pass





# ---------- App Config ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption(
    "Education only. Quotes & History: Yahoo (chart JSON). "
    "Model: Linear Regression with MA10/MA50. "
    "Request windows enforced (06:30 & 12:00 PT)."
)

# ---------- Time-window helpers (Pacific Time) ----------
LA = ZoneInfo("America/Los_Angeles")
WINDOWS_PT = [dtime(6, 30), dtime(12, 0)]   # two pulls: market open & midday
WINDOW_WIDTH_MIN = 7                        # each window stays open ~7 minutes


def now_la() -> datetime:
    return datetime.now(LA)


def in_window() -> bool:
    """Return True if current PT time is within any window."""
    t = now_la()
    for w in WINDOWS_PT:
        start = t.replace(hour=w.hour, minute=w.minute, second=0, microsecond=0)
        end = start + timedelta(minutes=WINDOW_WIDTH_MIN)
        if start <= t <= end:
            return True
    return False


def seconds_until_next_window() -> int:
    """Seconds from now (PT) until the next window opens."""
    t = now_la()
    candidates = []
    for w in WINDOWS_PT:
        start = t.replace(hour=w.hour, minute=w.minute, second=0, microsecond=0)
        if start > t:
            candidates.append(start)
    if not candidates:
        # next day's first window
        w = WINDOWS_PT[0]
        candidates.append((t + timedelta(days=1)).replace(hour=w.hour, minute=w.minute, second=0, microsecond=0))
    delta = min(candidates) - t
    return max(0, int(delta.total_seconds()))

# ---------- Yahoo fetchers (quote + candles) ----------
YH_HEADERS = {"User-Agent": "Mozilla/5.0"}  # Yahoo requires a UA header


@st.cache_data(ttl=120)
def get_quote(symbol: str, ignore_windows: bool = False) -> float:
    """
    Return latest 'regularMarketPrice' from Yahoo chart JSON (1d/1d).
    Respects windows unless ignore_windows=True.
    """
    if not ignore_windows and not in_window():
        raise RuntimeError("Outside fetch window")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "1d"}
    r = requests.get(url, params=params, headers=YH_HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json()
    result = (j.get("chart", {}) or {}).get("result") or []
    if not result:
        raise RuntimeError("Yahoo quote: empty result")
    meta = result[0].get("meta", {})
    px = meta.get("regularMarketPrice")
    if px is None:
        raise RuntimeError("Yahoo quote: missing regularMarketPrice")
    return float(px)


@st.cache_data(ttl=600)
def get_candles(symbol: str, days: int = 180, ignore_windows: bool = False) -> pd.DataFrame:
    """
    Return OHLCV indexed by time from Yahoo chart JSON (interval=1d, range=<days>d).
    Respects windows unless ignore_windows=True.
    """
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

# ---------- CNN-LSTM helpers ----------
@st.cache_resource(show_spinner=False)
def load_cnn_lstm(symbol: str | None = None):
    """
    Loads a .keras model from /models. Prefers per-symbol, falls back to generic.
    """
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
    """
    Loads the 5-feature MinMaxScaler trained offline (models/scaler_ALL.pkl).
    Returns None if missing.
    """
    candidates = [
        Path("models") / "scaler_ALL.pkl",
        Path("models") / "scaler.pkl",
        Path("scaler_ALL.pkl"),
        Path("scaler.pkl"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load scaler: {e}")
    return None


def make_cnnlstm_input(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    """
    Build a (1, lookback, 5) tensor from the last `lookback` rows of OHLCV.
    Uses the offline MinMaxScaler (fit on same 5-feature space).
    Returns (X, last_close) or (None, None) if not enough data or scaler missing.
    """
    scaler = load_scaler()
    if scaler is None:
        st.warning("Scaler not found (models/scaler_ALL.pkl).")
        return None, None

    # Normalize column names to open/high/low/close/volume
    df_n = df.copy()
    if {'o', 'h', 'l', 'close', 'v'}.issubset(df_n.columns):
        df_n = df_n.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
    elif {'Open', 'High', 'Low', 'Close'}.issubset(df_n.columns):
        df_n = df_n.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

    if not set(FEATURE_COLS).issubset(df_n.columns):
        st.warning("Dataframe missing one or more OHLCV columns.")
        return None, None

    if len(df_n) < lookback:
        return None, None

    block = df_n[FEATURE_COLS].tail(lookback).copy()
    last_close = float(block["close"].iloc[-1])

    # Scale using the prefit scaler
    try:
        scaled = scaler.transform(block.values)  # shape (lookback, 5)
    except Exception as e:
        st.warning(f"Scaler.transform failed: {e}")
        return None, None

    # Shape for Keras: (1, timesteps, features)
    X = scaled[np.newaxis, :, :]  # (1, lookback, 5)
    return X, last_close


def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    """
    Predict next close using CNN-LSTM trained on 5 features (OHLCV).
    Model outputs a scaled 'close'; we inverse_transform just that slot.
    Returns a float price or None.
    """
    model = load_cnn_lstm(symbol)
    if model is None:
        st.info("CNN-LSTM model unavailable; using fallback.")
        return None

    X, _ = make_cnnlstm_input(df, lookback=lookback)
    if X is None:
        return None

    # Predict scaled close
    try:
        y_scaled = float(model.predict(X, verbose=0)[0][0])
    except Exception as e:
        st.warning(f"CNN-LSTM predict failed: {e}")
        return None

    # Inverse scale the 'close' only
    scaler = load_scaler()
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, CLOSE_IDX] = y_scaled
    try:
        unscaled = scaler.inverse_transform(dummy)[0, CLOSE_IDX]
        return float(unscaled)
    except Exception as e:
        st.warning(f"Scaler.inverse_transform failed: {e}")
        return None

# ---------- Feature engineering + model ----------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Make sure we have a 'close' column
    if 'close' not in out.columns and 'Close' in out.columns:
        out = out.rename(columns={'Close': 'close'})
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    out = out.dropna()
    return out


def fit_and_predict(df_features: pd.DataFrame) -> tuple[LinearRegression, float]:
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    model = LinearRegression().fit(X, y)
    last = df_features.iloc[-1][["MA10", "MA50"]].values.reshape(1, -1)
    pred = float(model.predict(last)[0])
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
    MODEL_CNN    = "CNN-LSTM (pretrained)"
    model_choice = st.selectbox(
        "Choose model",
        (MODEL_LINEAR, MODEL_CNN),
        index=0,
        help="Linear is fast and simple. CNN-LSTM uses a pretrained deep model."
    )
    st.session_state["model_choice"] = model_choice  # keep for use later

    # Helper message about windows
    mins = seconds_until_next_window() // 60
    if in_window():
        st.success("✅ Inside fetch window (Pacific Time). Live pulls allowed.")
    else:
        st.info(f"🕰️ Outside fetch window. Using cached data if available. "
                f"Next window opens in ~{mins} min (Pacific).")

def using_cnn() -> bool:
    return st.session_state.get("model_choice", "Linear (MA10/MA50)") == "CNN-LSTM (pretrained)"

# --- Focus symbol state (driven by watchlist and URL) ---
# If URL contains ?focus=XYZ, honor it
try:
    qp = st.query_params
    url_focus = qp.get("focus")
except Exception:
    url_focus = None

if "focus_symbol" not in st.session_state:
    st.session_state["focus_symbol"] = (watchlist[0] if watchlist else default_watchlist[0])

# If current focus fell out of the watchlist (user changed watchlist), snap to first
if watchlist and st.session_state["focus_symbol"] not in watchlist:
    st.session_state["focus_symbol"] = watchlist[0]

# If a focus is provided in the URL and it’s in the watchlist, use it
if url_focus and isinstance(url_focus, str) and url_focus in (watchlist or []):
    st.session_state["focus_symbol"] = url_focus

# Use this everywhere below instead of 'symbol'
symbol = st.session_state["focus_symbol"]


# ---------- Diagnostics ----------
with st.expander("🛠️ Diagnostics", expanded=False):
    st.write("Python:", sys.version.split()[0])
    st.write("Now (PT):", now_la().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("In window:", in_window())
    st.write("Next window in (min):", seconds_until_next_window() // 60)

    colA, colB = st.columns(2)

    with colA:
        if st.button("Force one-time LIVE fetch & cache for focus symbol"):
            try:
                df_live = get_candles(symbol, days=days, ignore_windows=True)
                st.session_state.setdefault("history_cache", {})[symbol] = df_live
                try:
                    px_live = get_quote(symbol, ignore_windows=True)
                    st.session_state.setdefault("quote_cache", {})[symbol] = px_live
                except Exception as qe:
                    st.warning(f"Quote prime failed (but history cached): {qe}")
                st.success(f"Cached {len(df_live)} rows for {symbol} (Yahoo).")
            except Exception as e:
                st.error(f"Force fetch failed: {e}")

    with colB:
        if st.button("Test history fetch (AAPL, 30d)"):
            try:
                df_debug = get_candles("AAPL", days=30, ignore_windows=True)
                st.success(f"Fetched {len(df_debug)} AAPL rows via Yahoo.")
                st.dataframe(df_debug.tail())
            except Exception as e:
                st.error(f"History test failed: {e}")

# (optional) separate expander just to show model file status
with st.expander("🔎 CNN-LSTM status", expanded=False):
    st.write("Model file:", st.session_state.get("cnn_model_path"))

    # --- google sheets status  ---
    ws, info = _get_gsheet()
    if ws is None:
        st.write("Google Sheets: not configured –", info)
    else:
        st.write("Google Sheets: connected ✅")
        st.write("Worksheet:", ws.title)
        st.write("Service account:", info)

# --- Click a ticker to set focus in watchlist table---
st.markdown("#### Click a symbol to focus")
if watchlist:
    cols = st.columns(min(len(watchlist), 6))
    for i, s in enumerate(watchlist):
        with cols[i % len(cols)]:
            if st.button(s, key=f"focus_{s}", type=("secondary" if s != symbol else "primary"), use_container_width=True):
                st.session_state["focus_symbol"] = s
                # also update URL so refreshes/bookmarks keep your choice
                try:
                    st.query_params["focus"] = s
                except Exception:
                    pass
                st.rerun()
else:
    st.info("Your watchlist is empty. Add symbols in the sidebar.")


# ---------- Focus symbol KPIs & Chart ----------
# 1) Quote
price = None
if "quote_cache" in st.session_state and symbol in st.session_state["quote_cache"]:
    price = st.session_state["quote_cache"][focus]

if price is None:
    try:
        price = get_quote(focus)  # respects window
    except Exception as e:
        st.warning(f"Quote unavailable: {e}. Using last cached quote if any.")
        # keep price None if not cached

title_right = f"{price:,.2f}" if price is not None else "—"
st.subheader(f"{focus} — Live: {title_right}")

# 2) History + chart + forecast
df = None
if "history_cache" in st.session_state and focus in st.session_state["history_cache"]:
    df = st.session_state["history_cache"][focus]

if df is None:
    try:
        df = get_candles(symbol, days=days)  # respects window
    except Exception as e:
        st.error(f"History/forecast error: {e}")

col1, col2 = st.columns([3, 1])

# Prepare prediction variables
next_price = None
pct_change = None
model_used = "—"

if df is not None and not df.empty:
    last_close = float(df["close"].iloc[-1])

    # Choose model for the focus symbol
    if using_cnn():
        npred = predict_next_close_cnnlstm(symbol, df, lookback=DEFAULT_LOOKBACK)
        if npred is not None:
            next_price = float(npred)
            model_used = "CNN-LSTM"
        else:
            # graceful fallback to Linear
            feat = featurize(df)
            if not feat.empty:
                _, next_price = fit_and_predict(feat)
                model_used = "Linear (fallback)"
    else:
        feat = featurize(df)
        if not feat.empty:
            _, next_price = fit_and_predict(feat)
            model_used = "Linear"

    if next_price is not None:
        pct_change = 100.0 * (next_price - last_close) / last_close

with col1:
    if df is not None and not df.empty:
        # ---- Candlestick + MAs (robust to column name schema) ----
        df_plot = df.copy()

        # Normalize to 'open','high','low','close','volume'
        if {'o', 'h', 'l', 'close', 'v'}.issubset(df_plot.columns):
            df_plot = df_plot.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
        elif {'Open', 'High', 'Low', 'Close'}.issubset(df_plot.columns):
            df_plot = df_plot.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

        missing = [c for c in ['open', 'high', 'low', 'close'] if c not in df_plot.columns]
        if missing:
            st.error(f"Data missing columns for candlestick: {missing}. Got {list(df_plot.columns)}")
        else:
            # Compute MAs for overlay from 'close'
            df_plot["MA10"] = df_plot["close"].rolling(10).mean()
            df_plot["MA50"] = df_plot["close"].rolling(50).mean()

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df_plot.index,
                        open=df_plot["open"],
                        high=df_plot["high"],
                        low=df_plot["low"],
                        close=df_plot["close"],
                        name=f"{symbol} OHLC",
                    )
                ]
            )
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MA10"], name="MA10", mode="lines"))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MA50"], name="MA50", mode="lines"))

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history to chart yet. Try Diagnostics → fetch cache.")

# --- KPI / prediction panel ---
with col2:
    if df is not None and not df.empty and next_price is not None:
        # recompute safely (so we don't rely on an earlier pct_change variable)
        last_close = float(df["close"].iloc[-1])
        pct_change = 100.0 * (next_price - last_close) / last_close

        st.metric(
            "Predicted next close",
            f"{next_price:,.2f}",
            f"{pct_change:+.2f}% vs last"
        )
        st.caption(f"Model: {model_used}")

        if pct_change >= alert_pct:
            st.success(f"Potential BUY momentum (≥ {alert_pct}%).")
        elif pct_change <= -alert_pct:
            st.error(f"Potential SELL momentum (≤ -{alert_pct}%).")
        else:
            st.info("No strong signal at your threshold.")
    else:
        # no prediction available yet
        st.metric("Predicted next close", "—")
        # still show which model we tried to use (if any)
        if df is not None and not df.empty:
            st.caption(f"Model: {model_used}")
        else:
            st.caption("Model: —")


    # --- log focus symbol (if we have a prediction) ---
    try:
        if df is not None and not df.empty and "next_price" in locals() and next_price is not None:
            ma10 = df["close"].rolling(10).mean().iloc[-1] if len(df) >= 10 else ""
            ma50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else ""
            append_prediction_rows([
                {
                    "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "ts_pt": now_la().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "model": ("CNN-LSTM" if using_cnn() else "Linear"),
                    "lookback": (DEFAULT_LOOKBACK if using_cnn() else ""),
                    "days_history": days,
                    "last_close": float(df["close"].iloc[-1]),
                    "predicted": float(next_price),
                    "pct_change": float(pct_change),
                    "in_window": in_window(),
                    "ma10": float(ma10) if ma10 != "" else "",
                    "ma50": float(ma50) if ma50 != "" else "",
                    "note": "focus",
                }
            ])
    except Exception:
        pass

# --- One-shot refresh for all symbols (ignores time windows) ---
with st.container():
    if st.button("🔄 Refresh watchlist (live fetch once)", help="Fetch fresh quotes & candles for all symbols now, ignoring the time windows."):
        st.session_state.setdefault("history_cache", {})
        st.session_state.setdefault("quote_cache", {})
        refreshed = []
        for s in watchlist:
            try:
                d = get_candles(s, days=days, ignore_windows=True)
                st.session_state["history_cache"][s] = d
                try:
                    q = get_quote(s, ignore_windows=True)
                    st.session_state["quote_cache"][s] = q
                except Exception:
                    pass
                refreshed.append(s)
            except Exception as e:
                st.warning(f"{s}: {e}")
        if refreshed:
            st.success(f"Refreshed: {', '.join(refreshed)}")
            st.rerun()


# ---------- Watchlist table ----------
rows = []
with st.spinner("Scoring watchlist…"):
    for s in watchlist:
        try:
            d = get_candles(s, days=days)  # cached; respects fetch windows
            if d is None or d.empty:
                raise RuntimeError("no history")

            last_close = float(d["close"].iloc[-1])

            # Choose model per the selector
            model_used_row = "Linear"
            next_pred = None

            if using_cnn():
                next_pred = predict_next_close_cnnlstm(s, d, lookback=DEFAULT_LOOKBACK)
                if next_pred is not None:
                    model_used_row = "CNN-LSTM"
                else:
                    # graceful fallback to Linear for this row
                    feat_row = featurize(d)
                    if not feat_row.empty:
                        _, next_pred = fit_and_predict(feat_row)
                        model_used_row = "Linear (fallback)"
            else:
                feat_row = featurize(d)
                if not feat_row.empty:
                    _, next_pred = fit_and_predict(feat_row)

            # Compute return & signal
            if next_pred is None:
                raise RuntimeError("prediction failed")

            ret = 100.0 * (next_pred - last_close) / last_close
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

if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%", ascending=False)
    st.subheader("Watchlist signals")
    st.dataframe(table, use_container_width=True)

    # --- log watchlist predictions ---
    try:
        to_log = []
        for r in rows:
            if r.get("Predicted") is None or r.get("Last") is None:
                continue
            to_log.append({
                "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "ts_pt": now_la().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": r.get("Symbol", ""),
                "model": r.get("Model", ""),
                "lookback": (DEFAULT_LOOKBACK if "CNN" in (r.get("Model") or "") else ""),
                "days_history": days,
                "last_close": r.get("Last", ""),
                "predicted": r.get("Predicted", ""),
                "pct_change": r.get("Δ%", ""),
                "in_window": in_window(),
                "ma10": "",  # optional: compute if you want
                "ma50": "",
                "note": "watchlist",
            })
        if to_log:
            append_prediction_rows(to_log)
    except Exception:
        pass

# ---- put this once, after you build the watchlist and (maybe) set init focus ----
def get_focus_symbol() -> str:
    # ensure we always have one
    if "focus_symbol" not in st.session_state:
        st.session_state["focus_symbol"] = (watchlist or ["AAPL"])[0]
    return st.session_state["focus_symbol"]

focus = get_focus_symbol()

