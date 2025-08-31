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
from tensorflow.keras.models import load_model

import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import joblib

FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")  # 3
DEFAULT_LOOKBACK = 60 #sequence length for CNN-LSTM

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
    candidates = []
    if symbol:
        candidates.append(Path("models") / f"{symbol.upper()}_cnn_lstm.keras")
    candidates.append(Path("models") / "cnn_lstm_ALL.keras")
    for p in candidates:
        if p.exists():
            try:
                return tf.keras.models.load_model(p)
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

    # Ensure we have the features in the right order & enough rows
    if not set(FEATURE_COLS).issubset(df.columns):
        st.warning("Dataframe missing one or more OHLCV columns.")
        return None, None

    if len(df) < lookback:
        return None, None

    block = df[FEATURE_COLS].tail(lookback).copy()
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
    symbol = st.selectbox("Focus symbol", watchlist or default_watchlist)

with st.sidebar:
    st.subheader("Model")
    model_choice = st.selectbox(
        "Choose model",
        ["Linear (MA10/MA50)", "CNN-LSTM (pretrained)"],
        index=0,
        help="CNN-LSTM requires a pre-trained .h5 in /models."
    )
    
    # Helper message about windows
    mins = seconds_until_next_window() // 60
    if in_window():
        st.success("✅ Inside fetch window (Pacific Time). Live pulls allowed.")
    else:
        st.info(f"🕰️ Outside fetch window. Using cached data if available. Next window opens in ~{mins} min (Pacific).")
# --- Model selector (Sidebar) ---
MODEL_LINEAR = "Linear (MA10/MA50)"
MODEL_CNN    = "CNN-LSTM (pretrained)"

st.markdown("### Model")
model_choice = st.selectbox(
    "Choose model",
    (MODEL_LINEAR, MODEL_CNN),
    index=0,
    key="model_choice",
    help="Linear is fast and simple. CNN-LSTM uses a pretrained deep model."
)

def using_cnn() -> bool:
    return st.session_state.get("model_choice", model_choice) == MODEL_CNN


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
with st.expander("🔎 CNN-LSTM status", expanded=False):
    st.write("Model file:", st.session_state.get("cnn_model_path"))



# ---------- Focus symbol KPIs & Chart ----------
# 1) Quote
price = None
if "quote_cache" in st.session_state and symbol in st.session_state["quote_cache"]:
    price = st.session_state["quote_cache"][symbol]

if price is None:
    try:
        price = get_quote(symbol)  # respects window
    except Exception as e:
        st.warning(f"Quote unavailable: {e}. Using last cached quote if any.")
        # keep price None if not cached

title_right = f"{price:,.2f}" if price is not None else "—"
st.subheader(f"{symbol} — Live: {title_right}")

# 2) History + chart + forecast
df = None
if "history_cache" in st.session_state and symbol in st.session_state["history_cache"]:
    df = st.session_state["history_cache"][symbol]

if df is None:
    try:
        df = get_candles(symbol, days=days)  # respects window
    except Exception as e:
        st.error(f"History/forecast error: {e}")

col1, col2 = st.columns([3, 1])

with col1:
    if df is not None and not df.empty:
        # Compute MAs for overlay
        df_plot = df.copy()
        df_plot["MA10"] = df_plot["close"].rolling(10).mean()
        df_plot["MA50"] = df_plot["close"].rolling(50).mean()

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name=f"{symbol} OHLC"
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["MA10"],
            mode="lines", name="MA10"
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["MA50"],
            mode="lines", name="MA50"
        ))
        fig.update_layout(
            title=f"{symbol} — Candlestick",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history to chart yet. Try Diagnostics → fetch cache.")



with col2:
    if (df is not None and not df.empty and
        "next_price" in locals() and next_price is not None):
        st.metric("Predicted next close", f"{next_price:.2f}", f"{pct_change:+.2f}% vs last")
        if pct_change >= alert_pct:
            st.success(f"Potential BUY momentum (≥ {alert_pct}%).")
        elif pct_change <= -alert_pct:
            st.error(f"Potential SELL momentum (≤ -{alert_pct}%).")
        else:
            st.info("No strong signal at your threshold.")
    else:
        st.metric("Predicted next close", "—")



# ---------- Watchlist table ----------
rows = []
with st.spinner("Scoring watchlist…"):
    for s in watchlist:
        try:
            d = get_candles(s, days=days)  # cached; respects your fetch windows
            if d is None or d.empty:
                raise RuntimeError("no history")

            last_close = float(d["close"].iloc[-1])

            # Choose model per the selector
            model_used = "Linear"
            next_pred = None

            if model_choice.startswith("CNN"):
                next_pred = predict_next_close_cnnlstm(s, d, lookback=60)
                if next_pred is not None:
                    model_used = "CNN-LSTM"
                else:
                    # graceful fallback to Linear for this row
                    feat_row = featurize(d)
                    _, next_pred = fit_and_predict(feat_row)
                    model_used = "Linear (fallback)"
            else:
                feat_row = featurize(d)
                _, next_pred = fit_and_predict(feat_row)

            # Compute return & signal
            ret = 100.0 * (next_pred - last_close) / last_close
            signal = "BUY↑" if ret >= alert_pct else ("SELL↓" if ret <= -alert_pct else "HOLD")

            rows.append({
                "Symbol": s,
                "Last": round(last_close, 2),
                "Predicted": round(float(next_pred), 2),
                "Δ%": round(float(ret), 2),
                "Signal": signal,
                "Model": model_used
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

