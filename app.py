# app.py
import os, time, json, traceback
from collections.abc import Mapping
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import streamlit as st
import plotly.express as px

import requests
import yfinance as yf
import joblib

from sklearn.linear_model import LinearRegression

#import tensorflow as tf  # uses tensorflow-cpu in requirements
# TensorFlow is optional; if it can't install, we fall back to a linear model
try:
    import tensorflow as tf
except Exception:
    tf = None

import gspread
from gspread.utils import rowcol_to_a1


# ---------- App Config ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption("Education only. Quotes: Finnhub. History: Finnhub→Yahoo fallback. Model: CNN-LSTM (pretrained) with multi-horizon forecasts.")

# ---------- Keys / Constants ----------
API_KEY = st.secrets.get("FINNHUB_KEY", os.environ.get("FINNHUB_KEY", ""))
if not API_KEY:
    st.error("Missing FINNHUB_KEY. In Streamlit Cloud, set it in Settings → Advanced → Secrets.")
    st.stop()

HORIZONS = (1, 3, 5, 10, 20)  # business-day horizons we predict/log
WINDOW = 60  # days of history used by CNN-LSTM

# ---------- Data Helpers ----------
@st.cache_data(ttl=60)
def get_quote(symbol: str):
    """Live quote (cached 60s to respect rate limits)."""
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        # Fallback to Yahoo last close if Finnhub blocked
        tkr = yf.Ticker(symbol)
        h = tkr.history(period="5d", interval="1d")
        if h.empty:
            raise
        return {"c": float(h["Close"].iloc[-1])}

#--------Replaced Get Candles w/ a Hardened version below--------
@st.cache_data(ttl=120)
def get_candles(symbol: str, days: int = 180, resolution: str = "D") -> pd.DataFrame:
    """Return OHLCV indexed by time with multi-source fallback:
       Finnhub → yfinance(history) → yfinance(download) → Yahoo Chart JSON → Stooq.
    """
    # ---------- 1) Finnhub ----------
    try:
        end = int(time.time())
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": API_KEY},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        # If you’re rate limited, this is usually 403 HTML; the next line will raise for that.
        r.raise_for_status()
        if "application/json" in r.headers.get("content-type", ""):
            data = r.json()
        else:
            raise RuntimeError(f"Finnhub non-JSON response: {r.status_code}")
        if data.get("s") == "ok" and data.get("t"):
            df = pd.DataFrame(
                {"t": data["t"], "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"], "v": data["v"]}
            )
            df["time"] = pd.to_datetime(df["t"], unit="s")
            df.rename(columns={"c": "close"}, inplace=True)
            df.set_index("time", inplace=True)
            return df[["o", "h", "l", "close", "v"]]
    except Exception:
        pass  # fall through to Yahoo

    # ---------- 2) Yahoo / yfinance (two tries, browser UA) ----------
    try:
        import requests as _req
        sess = _req.Session()
        sess.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
        interval = "1d" if resolution == "D" else "60m"
        period_days = min(max(days, 5), 365 * 2)
        tk = yf.Ticker(symbol, session=sess)
        hist = tk.history(period=f"{period_days}d", interval=interval, auto_adjust=False)
        if hist is not None and not hist.empty:
            df = hist.rename(columns={"Close": "close", "Open": "o", "High": "h", "Low": "l", "Volume": "v"})
            df.index.name = "time"
            return df[["o", "h", "l", "close", "v"]]
    except Exception:
        pass

    try:
        import requests as _req
        sess2 = _req.Session()
        sess2.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
        start_dt = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_dt = datetime.now().strftime("%Y-%m-%d")
        dl = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval="1d",
            progress=False,
            group_by="ticker",
            session=sess2,
            auto_adjust=False,
            threads=False,
        )
        if isinstance(dl, pd.DataFrame) and not dl.empty:
            if isinstance(dl.columns, pd.MultiIndex):
                dl = dl.xs(symbol, axis=1, level=0, drop_level=True)
            df = dl.rename(columns={"Close": "close", "Open": "o", "High": "h", "Low": "l", "Volume": "v"})
            df.index.name = "time"
            return df[["o", "h", "l", "close", "v"]]
    except Exception:
        pass  # fall through to Yahoo Chart JSON

    # ---------- 3) Yahoo Chart JSON (often works when yfinance is blocked) ----------
    try:
        import requests as _req
        sess3 = _req.Session()
        sess3.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
        end = int(time.time())
        start = end - int(days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "period1": start,
            "period2": end,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }
        r = sess3.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        res = j["chart"]["result"][0]
        ts = res.get("timestamp", [])
        q = res.get("indicators", {}).get("quote", [{}])[0]
        if ts and "close" in q:
            df = pd.DataFrame(
                {"time": pd.to_datetime(ts, unit="s"),
                 "o": q.get("open", []),
                 "h": q.get("high", []),
                 "l": q.get("low", []),
                 "close": q.get("close", []),
                 "v": q.get("volume", []),}
            )
            df = df.dropna(subset=["time"]).set_index("time").sort_index()
            if not df.empty:
                return df[["o", "h", "l", "close", "v"]]
    except Exception:
        pass  # fall through to Stooq

    # ---------- 4) Stooq CSV fallback (AAPL -> aapl.us) ----------
    try:
        stooq_sym = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        stooq = pd.read_csv(url)
        if not stooq.empty:
            stooq.rename(
                columns={"Date": "time", "Open": "o", "High": "h", "Low": "l",
                         "Close": "close", "Volume": "v"},
                inplace=True,
            )
            stooq["time"] = pd.to_datetime(stooq["time"], errors="coerce", utc=True).dt.tz_localize(None)
            stooq = stooq.dropna(subset=["time"]).set_index("time").sort_index()
            cutoff = (pd.Timestamp.utcnow().tz_localize(None).normalize()
                      - pd.Timedelta(days=days + 5))
            try:
                stooq.index = pd.DatetimeIndex(stooq.index).tz_localize(None)
                stooq = stooq.loc[stooq.index >= cutoff]
            except Exception:
                stooq = stooq.tail(days + 5)
            if not stooq.empty:
                return stooq[["o", "h", "l", "close", "v"]]
    except Exception as e:
        raise RuntimeError(f"No history for {symbol}: Stooq failed: {e}")

    # ---------- If nothing worked ----------
    raise RuntimeError(f"No history for {symbol}: Finnhub/Yahoo/Stooq returned no data")



# ---------- Simple Targets (support/resistance/trendline) ----------
def simple_targets(df, lookback=60):
    d = df.tail(lookback).copy()
    support = float(d["close"].min())
    resistance = float(d["close"].max())
    x = np.arange(len(d)).reshape(-1, 1)
    lr = LinearRegression().fit(x, d["close"].values)
    trend_target = float(lr.predict([[len(d)]])[0])
    return {"support": support, "resistance": resistance, "trend_target": trend_target}

# ---------- Google Sheets: Secrets & Worksheet ----------
def _get_sa_info() -> dict:
    """
    Normalize Streamlit Secrets to a plain dict.
    Accepts AttrDict/Mapping, dict, or JSON string (with or without \\n).
    """
    sa_raw = st.secrets["gsheets"]["service_account"]
    if isinstance(sa_raw, Mapping):
        return dict(sa_raw)
    if isinstance(sa_raw, dict):
        return sa_raw
    s = str(sa_raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s_fixed = s.replace("\r\n", "\n").replace("\n", "\\n")
        return json.loads(s_fixed)

@st.cache_resource
def get_ws():
    """
    Connect/create 'predictions' worksheet. Ensure header covers all fields.
    """
    sa_info = _get_sa_info()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    client = gspread.service_account_from_dict(sa_info, scopes=scopes)
    sh = client.open_by_key(st.secrets["gsheets"]["sheet_id"])
    try:
        ws = sh.worksheet("predictions")
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title="predictions", rows=2000, cols=30)
        ws.append_row([
            "timestamp_utc","symbol","horizon","last_close","predicted",
            "pct_change","signal","scope","model_kind","params_json",
            "train_window","features","actual","err","pct_err","direction_correct"
        ])
        return ws

    wanted = [
        "timestamp_utc","symbol","horizon","last_close","predicted",
        "pct_change","signal","scope","model_kind","params_json",
        "train_window","features","actual","err","pct_err","direction_correct"
    ]
    current = ws.row_values(1) or []
    if current != wanted:
        try:
            need = len(wanted) - ws.col_count
            if need > 0:
                ws.add_cols(need)
        except Exception:
            pass
        ws.update("A1", [wanted])
    return ws


def _ensure_sheet_header(ws):
    """Ensure the sheet has all the columns we expect (add if missing)."""
    wanted = [
        "timestamp_utc", "date_utc", "symbol", "horizon",
        "last_close", "predicted", "pct_change", "signal", "scope",
        "model_kind", "params_json", "train_window", "features",
        "actual", "err", "pct_err", "direction_correct",
        "key"  # unique key per (symbol, horizon, date)
    ]
    header = ws.row_values(1)
    if not header:
        ws.append_row(wanted)
        header = wanted[:]
    else:
        changed = False
        for col in wanted:
            if col not in header:
                header.append(col)
                changed = True
        if changed:
            ws.update("1:1", [header])
    return header


def log_prediction(
    symbol,
    horizon,          # "1d", "3d", "1w", etc.
    last_close,
    predicted,
    pct_change,       # BUY/SELL threshold math you already compute
    signal,           # "BUY" / "SELL" / "HOLD"
    scope,            # "focus" or "watchlist"
    model_kind,
    params,
    train_window,
    features_desc,
):
    """
    Upsert exactly one row per (symbol, horizon, UTC date).
    If today's row exists, update it; else append a new one.
    """
    ws = get_ws()
    header = _ensure_sheet_header(ws)
    col_index = {name: i + 1 for i, name in enumerate(header)}

    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    daily_key = f"{symbol}|{horizon}|{date_str}"

    # Find an existing row by 'key' (includes header in row 1)
    key_vals = ws.col_values(col_index["key"])
    try:
        row_idx = key_vals.index(daily_key) + 1  # 1-based
    except ValueError:
        row_idx = None

    # Build a row mapped to current header order
    row_map = {
        "timestamp_utc": now.isoformat(timespec="seconds") + "Z",
        "date_utc": date_str,
        "symbol": symbol,
        "horizon": horizon,
        "last_close": float(last_close) if last_close is not None else "",
        "predicted": float(predicted) if predicted is not None else "",
        "pct_change": float(pct_change) if pct_change is not None else "",
        "signal": signal,
        "scope": scope,
        "model_kind": model_kind,
        "params_json": json.dumps(params) if isinstance(params, (dict, list)) else str(params),
        "train_window": int(train_window) if train_window is not None else "",
        "features": features_desc,
        "actual": "", "err": "", "pct_err": "", "direction_correct": "",
        "key": daily_key,
    }
    row_list = [row_map.get(col, "") for col in header]

    if row_idx:  # update in place
        end_cell = rowcol_to_a1(row_idx, len(header))
        ws.update(f"A{row_idx}:{end_cell}", [row_list], value_input_option="USER_ENTERED")
    else:        # append new
        ws.append_row(row_list, value_input_option="USER_ENTERED")


# ---------- CNN-LSTM Loader & Predictor ----------
@st.cache_resource
def load_cnn_lstm():
    if tf is None:
        raise RuntimeError("TensorFlow is not installed in this environment.")
    mdl = tf.keras.models.load_model("models/cnn_lstm_ALL.keras", compile=False)
    pack = joblib.load("models/scaler_ALL.pkl")
    return mdl, pack["scaler"], pack["feats"]


def predict_cnn_lstm_for_symbol(df: pd.DataFrame, model, scaler, feats, horizons=HORIZONS, window=WINDOW):
    d = df.copy()
    d["ret1"] = d["close"].pct_change()
    d["ma10"] = d["close"].rolling(10).mean()
    d["ma50"] = d["close"].rolling(50).mean()
    d["vlog"] = np.log1p(d["v"])
    d = d.dropna()
    if len(d) < window + 1:
        return {}
    # align to training feature order
    Xf = d[feats].values
    Xs = scaler.transform(Xf)  # (T,F)
    win = Xs[-window:]
    x = np.expand_dims(win, 0)  # (1,W,F)
    yhat_scaled = model.predict(x, verbose=0)[0]  # len(horizons)

    out = {}
    for i, h in enumerate(horizons):
        dummy = np.zeros((1, len(feats)))
        dummy[0, feats.index("close")] = yhat_scaled[i]
        inv = scaler.inverse_transform(dummy)[0, feats.index("close")]
        out[h] = float(inv)
    return out

# ---------- Linear Baseline (fallback) ----------
def predict_multi_horizon_linear(df: pd.DataFrame, horizons=HORIZONS):
    feat = featurize(df)
    preds = {}
    for h in horizons:
        tmp = feat.copy()
        tmp["target"] = tmp["close"].shift(-h)
        tmp = tmp.dropna()
        if len(tmp) < 10:
            continue
        X = tmp[["MA10","MA50"]]
        y = tmp["target"]
        m = LinearRegression().fit(X, y)
        preds[h] = float(m.predict(feat[["MA10","MA50"]].tail(1))[0])
    return preds

# ---------- Sidebar / Controls ----------
default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA"]
with st.sidebar:
    st.header("Controls")
    watchlist   = st.multiselect("Watchlist", default_watchlist, default=default_watchlist)
    days        = st.slider("History (days)", 120, 730, 365, step=30)  # CNN window 60; give enough history
    alert_pct   = st.slider("Signal threshold (%)", 0.5, 10.0, 2.0, step=0.5)
    symbol      = st.selectbox("Focus symbol", watchlist or default_watchlist)
    st.markdown("**Note:** Free Finnhub keys allow ~60 requests/min.")
    st.divider()

    st.subheader("Model")
    model_kind  = st.selectbox("Choose model", ["CNN-LSTM (pretrained)", "Linear baseline"])
    train_window= st.slider("Baseline training window (days)", 60, 360, 180, 10)  # for logging only (baseline)
    params      = {}  # reserved for future hyperparams

# ---------- Diagnostics ----------
with st.expander("🔧 Diagnostics", expanded=False):
    sheet_id = st.secrets.get("gsheets", {}).get("sheet_id", "(missing)")
    st.write("Google Sheet ID:", sheet_id)
    sa_raw = st.secrets.get("gsheets", {}).get("service_account", None)
    st.write("Service account format detected:", type(sa_raw).__name__)
    try:
        sa_info_dbg = _get_sa_info()
        st.write("Service account email:", sa_info_dbg.get("client_email", "(missing)"))
    except Exception as e:
        st.warning(f"Could not parse service account: {e!r}")

    if st.button("Test write to Google Sheet"):
        try:
            ws = get_ws()
            ws.append_row([
                datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "TEST","1d",100.0,101.0,1.0,"HOLD","diagnostic","DiagModel","{}",120,"diag","","","",""
            ])
            st.success("✅ Test row written to 'predictions'. Check your Google Sheet.")
        except Exception as e:
            st.error(f"❌ Failed to write: {e!r}")
            st.caption("Full traceback:")
            st.code(traceback.format_exc(), language="text")

with st.expander("🛠 Diagnostics"):
    st.write("Python:", __import__("sys").version.split()[0])
    if st.button("Test Finnhub candles (AAPL, 30d)"):
        try:
            _ = get_candles("AAPL", days=30, resolution="D")
            st.success("Finnhub OK")
        except Exception as e:
            st.error(f"Finnhub failed: {e}")

    if st.button("Test Yahoo fallback (AAPL, 30d)"):
        try:
            # call directly without cache to see a fresh request
            st.cache_data.clear()
            _ = get_candles("AAPL", days=30, resolution="D")
            st.success("Yahoo OK")
        except Exception as e:
            st.error(f"Yahoo failed: {e}")

if st.button("Test Stooq fallback (AAPL, 30d)"):
    try:
        st.cache_data.clear()
        _ = get_candles("AAPL", days=30, resolution="D")
        st.success("Stooq OK")
    except Exception as e:
        st.error(f"Stooq failed: {e}")



# ---------- Live KPI ----------
try:
    quote = get_quote(symbol)
    price = float(quote["c"])
    st.subheader(f"{symbol} — Live: {price:,.2f}")
except Exception as e:
    st.error(f"Quote error for {symbol}: {e}")
    st.stop()

# ---------- Chart + Forecast (Focus symbol) ----------
col1, col2 = st.columns([3, 1])
with col1:
    try:
        df = get_candles(symbol, days=days, resolution="D")
        feat = featurize(df)

        # Plot with MAs
        line = px.line(df.reset_index(), x="time", y="close", title=f"{symbol} Close Price")
        if not feat.empty:
            line.add_scatter(x=feat.index, y=feat["MA10"], mode="lines", name="MA10")
            line.add_scatter(x=feat.index, y=feat["MA50"], mode="lines", name="MA50")
        st.plotly_chart(line, use_container_width=True)

        # Targets
        tgt = simple_targets(df, lookback=min(60, len(df)))
        st.caption(f"Targets · Support: {tgt['support']:.2f} · Resistance: {tgt['resistance']:.2f} · Trend: {tgt['trend_target']:.2f}")

        # Predictions (CNN-LSTM preferred)
        mh = {}
        used_model = model_kind
        if model_kind.startswith("CNN"):
            try:
                mdl, scaler, feats = load_cnn_lstm()
                mh = predict_cnn_lstm_for_symbol(df, mdl, scaler, feats, horizons=HORIZONS, window=WINDOW)
            except Exception as e:
                st.warning(f"Deep model unavailable, falling back to baseline: {e}")
                used_model = "Linear baseline"
        if not mh:  # fallback or if user chose baseline
            mh = predict_multi_horizon_linear(df, horizons=HORIZONS)

        last_close = float(df["close"].iloc[-1])

        # Log focus symbol for all horizons
        for h, pred in mh.items():
            pct = 100.0 * (pred - last_close) / last_close
            sig = "BUY" if pct >= alert_pct else ("SELL" if pct <= -alert_pct else "HOLD")
            log_prediction(
                symbol=symbol,
                horizon=f"{h}d",
                last_close=last_close,
                predicted=pred,
                pct_change=pct,
                signal=sig,
                scope="focus",
                model_kind=used_model,
                params=params,
                train_window=train_window,
                features_desc="MA10,MA50 (+ret1,vlog in CNN)",
            )

    except Exception as e:
        st.error(f"History/forecast error: {e}")

with col2:
    # Show 1-day horizon on the KPI card
    try:
        # recompute mh here if needed (for KPI only)
        # To avoid recompute, we could cache mh; for simplicity, recompute fast baseline if missing.
        if 'mh' not in locals() or 1 not in mh:
            mh = mh if 'mh' in locals() else {}
            if 1 not in mh:
                mh = predict_multi_horizon_linear(df, horizons=HORIZONS)
        next_1d = mh.get(1, np.nan)
        pct_1d = 100.0 * (next_1d - df["close"].iloc[-1]) / df["close"].iloc[-1]
        st.metric(
            "Predicted next close (1d)",
            f"{next_1d:,.2f}" if np.isfinite(next_1d) else "—",
            f"{pct_1d:+.2f}% vs last" if np.isfinite(pct_1d) else None
        )
        if np.isfinite(pct_1d):
            if pct_1d >= alert_pct:
                st.success(f"Potential BUY (≥ {alert_pct}%)")
            elif pct_1d <= -alert_pct:
                st.error(f"Potential SELL (≤ -{alert_pct}%)")
            else:
                st.info("HOLD at current threshold.")
    except Exception:
        st.metric("Predicted next close (1d)", "—")

# ---------- Watchlist Table + Logging ----------
rows = []
for s in watchlist:
    try:
        q = get_quote(s); p = float(q["c"])
        d = get_candles(s, days=days)
        # Predict
        preds = {}
        if model_kind.startswith("CNN"):
            try:
                mdl, scaler, feats = load_cnn_lstm()
                preds = predict_cnn_lstm_for_symbol(d, mdl, scaler, feats, horizons=HORIZONS, window=WINDOW)
            except Exception:
                preds = predict_multi_horizon_linear(d, horizons=HORIZONS)
        else:
            preds = predict_multi_horizon_linear(d, horizons=HORIZONS)

        last_c = float(d["close"].iloc[-1])
        # Use 1-day horizon for table ranking
        nxt = preds.get(1, last_c)
        ret = 100.0 * (nxt - last_c) / last_c
        signal = "BUY" if ret >= alert_pct else ("SELL" if ret <= -alert_pct else "HOLD")
        rows.append({"Symbol": s, "Last": round(p,2), "Predicted(1d)": round(nxt,2), "Δ%(1d)": round(ret,2), "Signal": signal})

        # Log all horizons for watchlist symbols
        for h, pred in preds.items():
            pct = 100.0 * (pred - last_c) / last_c
            sig = "BUY" if pct >= alert_pct else ("SELL" if pct <= -alert_pct else "HOLD")
            log_prediction(
                symbol=s,
                horizon=f"{h}d",
                last_close=last_c,
                predicted=pred,
                pct_change=pct,
                signal=sig,
                scope="watchlist",
                model_kind=model_kind,
                params=params,
                train_window=train_window,
                features_desc="MA10,MA50 (+ret1,vlog in CNN)",
            )

    except Exception:
        rows.append({"Symbol": s, "Last": None, "Predicted(1d)": None, "Δ%(1d)": None, "Signal": "ERR"})

if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%(1d)", ascending=False)
    st.subheader("Watchlist signals (1-day)")
    st.dataframe(table, use_container_width=True)

# ---------- Reconcile & Accuracy ----------
def reconcile_predictions():
    """Fill actual outcomes for rows whose horizon date has passed (business days)."""
    import math
    ws = get_ws()
    rows = ws.get_all_records()
    if not rows:
        return "No rows to reconcile."
    header = ws.row_values(1)
    cols = {name: i+1 for i, name in enumerate(header)}
    updated = 0
    for idx, r in enumerate(rows, start=2):
        if r.get("actual"):
            continue
        try:
            ts = pd.to_datetime(r["timestamp_utc"])
            h = int(str(r["horizon"]).lower().strip("d"))
        except Exception:
            continue
        due = (ts + BDay(h)).normalize()
        if pd.Timestamp.utcnow().normalize() < due:
            continue  # not due yet
        sym = r["symbol"]
        try:
            tkr = yf.Ticker(sym)
            hist = tkr.history(start=due.tz_localize(None).date(), end=(due + BDay(5)).date())
            if hist.empty:
                continue
            actual = float(hist["Close"].iloc[0])
            predicted = float(r["predicted"])
            last_close = float(r["last_close"])
            err = actual - predicted
            pct_err = (err / actual) * 100.0 if actual else None
            dir_ok = int(np.sign(predicted - last_close) == np.sign(actual - last_close))
            ws.update_cell(idx, cols["actual"], actual)
            ws.update_cell(idx, cols["err"], err)
            ws.update_cell(idx, cols["pct_err"], pct_err)
            ws.update_cell(idx, cols["direction_correct"], dir_ok)
            updated += 1
        except Exception:
            continue
    return f"Reconciled {updated} row(s)."

with st.expander("📊 Model accuracy"):
    if st.button("Reconcile predictions now"):
        msg = reconcile_predictions()
        st.success(msg)
    try:
        ws = get_ws()
        dfp = pd.DataFrame(ws.get_all_records())
        if not dfp.empty and "actual" in dfp.columns:
            dfp = dfp[dfp["actual"] != ""].copy()
            if not dfp.empty:
                # coerce types
                for c in ["actual","predicted"]:
                    dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
                dfp["err_abs"] = (dfp["actual"] - dfp["predicted"]).abs()
                dfp["mape"] = (dfp["err_abs"] / dfp["actual"].replace(0, np.nan)).abs() * 100
                dfp["dir_ok"] = pd.to_numeric(dfp["direction_correct"], errors="coerce")
                summary = (dfp.groupby("horizon")
                              .agg(n=("symbol","count"),
                                   MAE=("err_abs","mean"),
                                   MAPE=("mape","mean"),
                                   DirectionalAcc=("dir_ok","mean"))
                              .sort_index())
                summary["MAE"] = summary["MAE"].round(3)
                summary["MAPE"] = summary["MAPE"].round(2)
                summary["DirectionalAcc"] = (summary["DirectionalAcc"]*100).round(1).astype(str) + "%"
                st.dataframe(summary, use_container_width=True)
            else:
                st.info("No reconciled rows yet — click the button after horizons have passed.")
        else:
            st.info("Sheet empty or not reconciled yet.")
    except Exception as e:
        st.warning(f"Could not load accuracy summary: {e}")
