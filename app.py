# ---------- Imports ----------
import os, json, requests
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict

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
import plotly.graph_objects as go

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
    "Education only. Quotes & History: Yahoo (chart JSON). "
    "Models: CNN-LSTM (default) or Linear MA10/MA50 fallback. "
    "Live fetch windows: 06:30 & 12:00 PT."
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



# --------- Fetch all symbols in one shot -----

def prime_watchlist_cache(symbols: list[str], days: int) -> None:
    """Fetch candles (and quotes) for all symbols at once and cache them."""
    st.session_state.setdefault("history_cache", {})
    st.session_state.setdefault("quote_cache", {})

    progress = st.progress(0)
    status = st.empty()

    ok, fail = [], []
    for i, s in enumerate(symbols):
        status.write(f"Fetching {s} …")
        try:
            # Live fetch once, ignoring the time windows
            d = get_candles(s, days=days, ignore_windows=True)
            if d is None or d.empty:
                raise RuntimeError("empty history")
            st.session_state["history_cache"][s] = d

            try:
                q = get_quote(s, ignore_windows=True)
                st.session_state["quote_cache"][s] = q
            except Exception:
                # quote failure shouldn't block history
                pass

            ok.append(s)
        except Exception as e:
            fail.append(f"{s} ({e})")
        finally:
            progress.progress(int((i + 1) / max(1, len(symbols)) * 100))

    status.empty()
    progress.empty()

    if ok:
        st.success(f"Cached {len(ok)} symbols: {', '.join(ok)}")
    if fail:
        st.warning("Some failed: " + ", ".join(fail))



# ---------- Yahoo fetchers ----------
YH_HEADERS = {"User-Agent": "Mozilla/5.0"}


@st.cache_data(ttl=120)
def get_quote(symbol: str, ignore_windows: bool = False) -> float:
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
    px = (result[0].get("meta", {}) or {}).get("regularMarketPrice")
    if px is None:
        raise RuntimeError("Yahoo quote: missing regularMarketPrice")
    return float(px)


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
def _ohlcv_normalize(df: pd.DataFrame) -> pd.DataFrame:
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

def _get_cache(symbol: str) -> pd.DataFrame | None:
    return st.session_state.get("history_cache", {}).get(symbol)

def fetch_history(symbol: str, days: int, *, allow_live_fallback: bool = True) -> pd.DataFrame | None:
    """
    1) Return from our in-memory cache if present.
    2) Else try the cached @st.cache_data call (respects windows).
    3) Else (optionally) one-time live fetch with ignore_windows=True, then cache it.
    """
    # 1) in-memory cache?
    df = _get_cache(symbol)
    if df is not None and not df.empty:
        return df

    # 2) try normal cached function (respects windows)
    try:
        df2 = get_candles(symbol, days=days)  # may raise if outside window and not cached
        if df2 is not None and not df2.empty:
            return _put_cache(symbol, df2)
    except Exception:
        pass

    # 3) one-time live fallback
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


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {'o', 'h', 'l', 'close', 'v'}.issubset(out.columns):
        out = out.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume'})
    elif {'Open', 'High', 'Low', 'Close'}.issubset(out.columns):
        out = out.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                  'Close': 'close', 'Volume': 'volume'})
    return out


def make_cnnlstm_input(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    scaler = load_scaler()
    if scaler is None:
        st.warning("Scaler not found (models/scaler_ALL.pkl).")
        return None, None
    df_n = _normalize_ohlcv(df)
    if not set(FEATURE_COLS).issubset(df_n.columns):
        st.warning("Dataframe missing one or more OHLCV columns.")
        return None, None
    if len(df_n) < lookback:
        return None, None
    block = df_n[FEATURE_COLS].tail(lookback).copy()
    last_close = float(block["close"].iloc[-1])
    try:
        scaled = scaler.transform(block.values)
    except Exception as e:
        st.warning(f"Scaler.transform failed: {e}")
        return None, None
    X = scaled[np.newaxis, :, :]
    return X, last_close


def predict_next_close_cnnlstm(symbol: str, df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK):
    model = load_cnn_lstm(symbol)
    if model is None:
        return None
    X, _ = make_cnnlstm_input(df, lookback=lookback)
    if X is None:
        return None
    try:
        y_scaled = float(model.predict(X, verbose=0)[0][0])
    except Exception:
        return None
    scaler = load_scaler()
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, CLOSE_IDX] = y_scaled
    try:
        return float(scaler.inverse_transform(dummy)[0, CLOSE_IDX])
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


# ---------- Focus symbol state ----------
if "focus_symbol" not in st.session_state:
    st.session_state["focus_symbol"] = (watchlist[0] if watchlist else default_watchlist[0])
# If focus fell out of current watchlist, snap to first available
if watchlist and st.session_state["focus_symbol"] not in watchlist:
    st.session_state["focus_symbol"] = watchlist[0]
focus = st.session_state["focus_symbol"]



# Auto-prime the cache once when outside the window (only if needed)
if not in_window() and not st.session_state.get("_auto_primed", False):
    need = [s for s in (watchlist or []) 
            if s not in st.session_state.get("history_cache", {})]

    if need:
        with st.expander("Auto-priming cache (outside fetch window)", expanded=False):
            with st.spinner("Fetching initial data for your watchlist…"):
                prime_watchlist_cache(need, days)  # <- helper you added earlier
        st.session_state["_auto_primed"] = True
        st.rerun()
    else:
        # Nothing to do, but mark so we don't re-check on every run
        st.session_state["_auto_primed"] = True




# ---------- Diagnostics ----------
with st.expander("🛠️ Diagnostics", expanded=False):
    st.write("Python:", st.__version__ if hasattr(st, "__version__") else "—")
    st.write("Now (PT):", now_la().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("In window:", in_window())
    st.write("Next window in (min):", seconds_until_next_window() // 60)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Force LIVE fetch for focus"):
            try:
                df_live = get_candles(focus, days=days, ignore_windows=True)
                st.session_state.setdefault("history_cache", {})[focus] = df_live
                try:
                    px_live = get_quote(focus, ignore_windows=True)
                    st.session_state.setdefault("quote_cache", {})[focus] = px_live
                except Exception as qe:
                    st.warning(f"Quote prime failed (but history cached): {qe}")
                st.success(f"Cached {len(df_live)} rows for {focus}.")
            except Exception as e:
                st.error(f"Force fetch failed: {e}")
    with colB:
        if st.button("Test history fetch (AAPL, 30d)"):
            try:
                df_debug = get_candles("AAPL", days=30, ignore_windows=True)
                st.success(f"Fetched {len(df_debug)} AAPL rows.")
                st.dataframe(df_debug.tail())
            except Exception as e:
                st.error(f"History test failed: {e}")

with st.expander("🔎 Integrations status", expanded=False):
    st.write("CNN-LSTM model file:", st.session_state.get("cnn_model_path"))
    ws, info = _get_gsheet()
    if ws is None:
        st.write("Google Sheets: not configured –", info)
    else:
        st.write("Google Sheets: connected ✅")
        st.write("Worksheet:", ws.title)
        st.write("Service account:", info)

# ---------- Focus symbol KPIs & Chart ----------
price = st.session_state.get("quote_cache", {}).get(focus)
if price is None:
    try:
        price = get_quote(focus)
    except Exception as e:
        st.warning(f"Quote unavailable: {e}. Using last cached if any.")

st.subheader(f"{focus} — Live: {price:,.2f}" if price is not None else f"{focus} — Live: —")

df = st.session_state.get("history_cache", {}).get(focus)
if df is None:
    try:
        df = get_candles(focus, days=days)
    except Exception as e:
        df = None
        st.error(f"History/forecast error: {e}")

col1, col2 = st.columns([3, 1])

next_price, pct_change, model_used = None, None, "—"
if df is not None and not df.empty:
    last_close = float(df["close"].iloc[-1])
    if using_cnn():
        npred = predict_next_close_cnnlstm(focus, df, lookback=DEFAULT_LOOKBACK)
        if npred is not None:
            next_price = float(npred); model_used = "CNN-LSTM"
        else:
            feat = featurize(df)
            if not feat.empty:
                _, next_price = fit_and_predict(feat); model_used = "Linear (fallback)"
    else:
        feat = featurize(df)
        if not feat.empty:
            _, next_price = fit_and_predict(feat); model_used = "Linear"
    if next_price is not None:
        pct_change = 100.0 * (next_price - last_close) / last_close

with col1:
    if df is not None and not df.empty:
        d = _normalize_ohlcv(df)
        if {'open', 'high', 'low', 'close'}.issubset(d.columns):
            d["MA10"] = d["close"].rolling(10).mean()
            d["MA50"] = d["close"].rolling(50).mean()
            fig = go.Figure([
                go.Candlestick(
                    x=d.index, open=d["open"], high=d["high"],
                    low=d["low"], close=d["close"], name=f"{focus} OHLC"
                )
            ])
            fig.add_trace(go.Scatter(x=d.index, y=d["MA10"], name="MA10", mode="lines"))
            fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], name="MA50", mode="lines"))
            fig.update_layout(
                xaxis_rangeslider_visible=False, height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Missing columns for candlestick.")
    else:
        st.info("No history to chart yet. Try Diagnostics → fetch cache.")

with col2:
    if df is not None and not df.empty and next_price is not None:
        last_close = float(df["close"].iloc[-1])
        pct_change = 100.0 * (next_price - last_close) / last_close
        st.metric("Predicted next close", f"{next_price:,.2f}", f"{pct_change:+.2f}% vs last")
        st.caption(f"Model: {model_used}")
        if pct_change >= alert_pct:
            st.success(f"Potential BUY momentum (≥ {alert_pct}%).")
        elif pct_change <= -alert_pct:
            st.error(f"Potential SELL momentum (≤ -{alert_pct}%).")
        else:
            st.info("No strong signal at your threshold.")
        # Log focus prediction (optional)
        try:
            ma10 = df["close"].rolling(10).mean().iloc[-1] if len(df) >= 10 else ""
            ma50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else ""
            append_prediction_rows([{
                "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "ts_pt": now_la().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": focus,
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
            }])
        except Exception:
            pass
    else:
        st.metric("Predicted next close", "—")
        st.caption(f"Model: {model_used if df is not None and not df.empty else '—'}")

# ---------- Refresh watchlist (live) ----------
with st.container():
    if st.button(
        "🔄 Refresh watchlist (live fetch ALL now)",
        help="Fetch fresh candles & quotes for every symbol once, ignoring the time windows."
    ):
        if not watchlist:
            st.info("Your watchlist is empty.")
        else:
            with st.spinner("Priming cache for all symbols…"):
                prime_watchlist_cache(watchlist, days)
            st.rerun()



# ---------- Watchlist table ----------
rows = []

with st.spinner("Scoring watchlist…"):
    for s in (watchlist or []):
        try:
            d = fetch_history(s, days, allow_live_fallback=False)  # first try cache / normal path
            if d is None or d.empty:
                # last resort: live fallback once for this symbol in this run
                d = fetch_history(s, days, allow_live_fallback=True)

            if d is None or d.empty:
                raise RuntimeError("no history")

            dN = _ohlcv_normalize(d)
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

if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%", ascending=False, na_position="last")
    st.subheader("Watchlist signals")
    # Make symbols clickable (focus switch) but DON'T open a new tab
    st.dataframe(
        table.assign(_focus=[f"▶ {sym}" for sym in table.index]).reset_index(),
        use_container_width=True,
        hide_index=True
    )

    # Small row of focus buttons under the table (keeps everything in-page)
    click_cols = st.columns(min(6, len(table)))
    for i, sym in enumerate(table.index.tolist()):
        with click_cols[i % len(click_cols)]:
            if st.button(f"Focus {sym}", key=f"focus_btn_{sym}", type=("secondary" if sym != symbol else "primary")):
                st.session_state["focus_symbol"] = sym
                try:
                    st.query_params["focus"] = sym  # update URL without opening a new window
                except Exception:
                    pass
                st.rerun()

    # --- (optional) log watchlist predictions to Google Sheets ---
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


st.subheader("Watchlist signals")

if rows:
    table = pd.DataFrame(rows).sort_values("Δ%", ascending=False)

    # Render a “table-like” layout with a clickable Symbol button per row
    header_cols = st.columns([1.1, 1, 1, 0.8, 0.9, 1.2])
    for col, label in zip(header_cols, ["Symbol", "Last", "Predicted", "Δ%", "Signal", "Model"]):
        col.markdown(f"**{label}**")

    for i, r in table.reset_index(drop=True).iterrows():
        c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1, 1, 0.8, 0.9, 1.2])
        # Symbol as a button (styled like a link via label)
        if c1.button(r["Symbol"], key=f"sym_{r['Symbol']}"):
            st.session_state["focus_symbol"] = r["Symbol"]
            st.rerun()
        c2.write("" if pd.isna(r["Last"]) else f"{r['Last']:.2f}")
        c3.write("" if pd.isna(r["Predicted"]) else f"{r['Predicted']:.2f}")
        c4.write("" if pd.isna(r["Δ%"]) else f"{r['Δ%']:.2f}")
        c5.write(r["Signal"])
        c6.write(r["Model"])

    # Optional: also show a sortable raw grid below
    with st.expander("Show sortable data grid", expanded=False):
        st.dataframe(table, use_container_width=True)

    # Log watchlist predictions
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
                "ma10": "",
                "ma50": "",
                "note": "watchlist",
            })
        if to_log:
            append_prediction_rows(to_log)
    except Exception:
        pass
else:
    st.info("Your watchlist is empty. Add symbols in the sidebar.")
