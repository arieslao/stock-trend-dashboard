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
from sklearn.linear_model import LinearRegression


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
@st.cache_resource
def load_cnn_lstm(symbol: str):
    """
    Try multiple filenames/locations and support both .keras and .h5.
    Returns a tf.keras.Model or None.
    """
    if tf is None:
        st.warning("TensorFlow is not available in this runtime; using Linear model instead.")
        return None

    from pathlib import Path

    candidates = [
        Path("models") / f"{symbol.upper()}_cnn_lstm.keras",
        Path("models") / f"{symbol.upper()}_cnn_lstm.h5",
        Path("models") / "cnn_lstm_ALL.keras",   # <--- your file
        Path("models") / "cnn_lstm_ALL.h5",
        Path("models") / "cnn_lstm_generic.keras",
        Path("models") / "cnn_lstm_generic.h5",
        Path("cnn_lstm_ALL.keras"),
        Path("cnn_lstm_ALL.h5"),
    ]
    for p in candidates:
        if p.exists():
            try:
                # compile=False is fine for inference; prevents missing loss/optimizer errors
                model = tf.keras.models.load_model(p, compile=False)
                st.session_state["cnn_model_path"] = str(p)
                return model
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load: {e}")
                # Try next candidate

    st.info("CNN-LSTM model file not found; using Linear model instead.")
    return None



@st.cache_resource
def load_scaler(symbol: str):
    """
    Looks for a per-symbol scaler, then global scaler.
    Returns a fitted scaler or None (we’ll create one on the fly if missing).
    """
    candidates = [
        Path("models") / f"scaler_{symbol.upper()}.pkl",
        Path("models") / "scaler_ALL.pkl",
        Path("scaler_ALL.pkl"),       # if at repo root
        Path("models") / "scaler.pkl",
        Path("scaler.pkl"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load scaler: {e}")
    return None


def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = 60):
    """
    Assumes model trained on single feature 'close' scaled via MinMax.
    Returns a float predicted next close, or None on failure.
    """
    model = load_cnn_lstm(symbol)
    if model is None or df is None or len(df) < lookback + 1:
        return None

    series = df["close"].values.reshape(-1, 1)

    scaler = load_scaler(symbol)
    if scaler is None:
        # Fall back: fit a new scaler on the history (not ideal for strict backtests,
        # but fine for live prediction when a saved scaler isn’t provided)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(series)

    try:
        scaled = scaler.transform(series)
    except Exception:
        # If the scaler expects a different shape or range, refit gracefully
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(series)
        scaled = scaler.transform(series)

    X = np.array([scaled[-lookback:]])   # shape (1, lookback, 1)
    y_scaled = model.predict(X, verbose=0)[0, 0]
    next_close = float(scaler.inverse_transform([[y_scaled]])[0, 0])
    return next_close



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
        # Always build features so we can show MA10/MA50 on the chart
        feat = featurize(df)

        # ---- choose model for the focus symbol ----
        next_price = None
        if model_choice.startswith("CNN"):
            next_price = predict_next_close_cnnlstm(symbol, df, lookback=60)
            if next_price is None:
                # graceful fallback
                _, next_price = fit_and_predict(feat)
                st.info("CNN-LSTM unavailable; fell back to Linear model.")
        else:
            _, next_price = fit_and_predict(feat)

        pct_change = 100.0 * (next_price - df["close"].iloc[-1]) / df["close"].iloc[-1]

        # Chart recent closes + MA10/MA50
        line = px.line(df.reset_index(), x="time", y="close", title=f"{symbol} Close Price")
        line.add_scatter(x=feat.index, y=feat["MA10"], mode="lines", name="MA10")
        line.add_scatter(x=feat.index, y=feat["MA50"], mode="lines", name="MA50")
        st.plotly_chart(line, use_container_width=True)
    else:
        st.info("No history to chart yet. Try the Diagnostics → Force one-time fetch button above.")

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

