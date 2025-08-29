import os, time, requests, json
from datetime import datetime, timedelta
from collections.abc import Mapping

from collections.abc import Mapping
import traceback


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
import yfinance as yf
import gspread

# ---------- App Config ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption("Education only. Data: Finnhub (quotes) + Yahoo fallback (history). Forecasts: scikit-learn (Linear Regression).")

# ---------- API Key ----------
API_KEY = st.secrets.get("FINNHUB_KEY", os.environ.get("FINNHUB_KEY", ""))
if not API_KEY:
    st.error("Missing FINNHUB_KEY. In Streamlit Cloud, set it in 'Advanced settings → Secrets'.")
    st.stop()

# ---------- Data Helpers ----------
@st.cache_data(ttl=60)
def get_quote(symbol: str):
    """Live quote for a symbol. Cached for 60s to respect rate limits."""
    r = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": API_KEY},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120)
def get_candles(symbol: str, days: int = 180, resolution="D"):
    """Historical candles: try Finnhub; on error, fall back to Yahoo Finance."""
    # --- Try Finnhub first
    try:
        end = int(time.time())
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("s") == "ok":
            df = pd.DataFrame({
                "t": data["t"], "o": data["o"], "h": data["h"],
                "l": data["l"], "c": data["c"], "v": data["v"]
            })
            df["time"] = pd.to_datetime(df["t"], unit="s")
            df.rename(columns={"c": "close"}, inplace=True)
            df.set_index("time", inplace=True)
            return df
        # else fall through to Yahoo
    except Exception:
        pass

    # --- Yahoo fallback
    interval = "1d" if resolution == "D" else "60m"
    period_days = min(max(days, 5), 365 * 2)
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{period_days}d", interval=interval)
    if hist.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
    df = hist.rename(columns={
        "Close": "close", "Open": "o", "High": "h", "Low": "l", "Volume": "v"
    })
    df.index.name = "time"
    return df[["o", "h", "l", "close", "v"]]

def featurize(df: pd.DataFrame):
    out = df.copy()
    out["MA10"]  = out["close"].rolling(10).mean()
    out["MA50"]  = out["close"].rolling(50).mean()
    out = out.dropna()
    return out

def fit_and_predict(df_features: pd.DataFrame):
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    model = LinearRegression().fit(X, y)
    last = df_features.iloc[-1][["MA10", "MA50"]].values.reshape(1, -1)
    next_price = float(model.predict(last)[0])
    return model, next_price

# Added this section to normalize the service-account info
def _get_sa_info() -> dict:
    """
    Normalize Streamlit Secrets to a plain dict.
    Accepts AttrDict/Mapping, dict, or JSON string (with or without \\n).
    """
    sa_raw = st.secrets["gsheets"]["service_account"]

    if isinstance(sa_raw, Mapping):          # AttrDict from TOML table
        return dict(sa_raw)
    if isinstance(sa_raw, dict):             # plain dict
        return sa_raw

    s = str(sa_raw)
    try:
        return json.loads(s)                  # JSON string with '\n' in private_key
    except json.JSONDecodeError:
        s_fixed = s.replace("\r\n", "\n").replace("\n", "\\n")
        return json.loads(s_fixed)

# ---------- Google Sheets: robust connector ----------
@st.cache_resource
def get_ws():
    """
    Connect to (or create) the 'predictions' worksheet.
    Robust to Secrets formatting and includes Drive scope (safer for open_by_key).
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
        ws = sh.add_worksheet(title="predictions", rows=1000, cols=30)
        ws.append_row([
            "timestamp_utc","symbol","horizon","last_close","predicted",
            "model_kind","params_json","train_window","features",
            "actual","err","pct_err","direction_correct"
        ])
    return ws


def log_prediction(symbol, last_close, predicted, model_kind, params, train_window, features_desc, horizon="next_close"):
    ws = get_ws()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ws.append_row([
        ts, symbol, horizon, last_close, predicted,
        model_kind, json.dumps(params), int(train_window), features_desc,
        "", "", "", ""  # placeholders for later reconciliation
    ])

# ---------- Sidebar / Controls ----------
default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA"]
with st.sidebar:
    st.header("Controls")
    watchlist   = st.multiselect("Watchlist", default_watchlist, default=default_watchlist)
    days        = st.slider("History (days)", 90, 365, 180, step=10)
    alert_pct   = st.slider("Alert threshold (%)", 0.5, 5.0, 2.0, step=0.5)
    symbol      = st.selectbox("Focus symbol", watchlist or default_watchlist)
    st.markdown("**Note:** Free Finnhub keys allow ~60 requests/min.")
    st.divider()

    st.subheader("Model")
    model_kind  = st.selectbox("Choose model", ["Linear"])
    params      = {}
    train_window= st.slider("Training window (days)", 60, 300, 120, 10)



# ---------- Diagnostics (test Sheets connection) ----------
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
                "TEST","next_close",100.0,101.0,"Linear","{}",120,"diagnostic","","","",""
            ])
            st.success("✅ Test row written to 'predictions'. Check your Google Sheet.")
        except Exception as e:
            st.error(f"❌ Failed to write: {e!r}")
            st.caption("Full traceback:")
            st.code(traceback.format_exc(), language="text")
            st.info("Most common fixes:\n"
                    "• Share the sheet with the service account email above (Editor).\n"
                    "• Double-check the Sheet ID (between /d/ and /edit in the URL).\n"
                    "• Ensure Secrets saved with no TOML error.")



# ---------- Live KPI ----------
try:
    quote = get_quote(symbol)
    price = quote["c"]
    st.subheader(f"{symbol} — Live: {price:,.2f}")
except Exception as e:
    st.error(f"Quote error for {symbol}: {e}")
    st.stop()

# ---------- Chart + Forecast ----------
col1, col2 = st.columns([3, 1])
with col1:
    try:
        df   = get_candles(symbol, days=days, resolution="D")
        feat = featurize(df)
        model, next_price = fit_and_predict(feat)
        pct_change = 100.0 * (next_price - df["close"].iloc[-1]) / df["close"].iloc[-1]

        line = px.line(df.reset_index(), x="time", y="close", title=f"{symbol} Close Price")
        line.add_scatter(x=feat.index, y=feat["MA10"], mode="lines", name="MA10")
        line.add_scatter(x=feat.index, y=feat["MA50"], mode="lines", name="MA50")
        st.plotly_chart(line, use_container_width=True)

        # --- Log the prediction only if it succeeded ---
        features_desc = "MA10,MA50"
        try:
            log_prediction(
                symbol=symbol,
                last_close=float(df["close"].iloc[-1]),
                predicted=float(next_price),
                model_kind=model_kind,
                params=params,
                train_window=train_window,
                features_desc=features_desc
            )
        except Exception as e:
            st.warning(f"Logged locally but failed to write to Google Sheet: {e}")

    except Exception as e:
        st.error(f"History/forecast error: {e}")

with col2:
    st.metric(
        "Predicted next close",
        f"{next_price:,.2f}" if "next_price" in locals() else "—",
        f"{pct_change:+.2f}% vs last" if "pct_change" in locals() else None
    )
    if "pct_change" in locals():
        if pct_change >= alert_pct:
            st.success(f"Potential BUY momentum (>{alert_pct}% ↑).")
        elif pct_change <= -alert_pct:
            st.error(f"Potential SELL momentum / caution (<-{alert_pct}% ↓).")
        else:
            st.info("No strong signal at your threshold.")

# ---------- Watchlist Table ----------
rows = []
for s in watchlist:
    try:
        q = get_quote(s); p = q["c"]
        d = get_candles(s, days=days)
        f = featurize(d)
        _, nxt = fit_and_predict(f)
        ret = 100.0 * (nxt - d["close"].iloc[-1]) / d["close"].iloc[-1]
        signal = "BUY↑" if ret >= alert_pct else ("SELL↓" if ret <= -alert_pct else "HOLD")
        rows.append({"Symbol": s, "Last": round(p,2), "Predicted": round(nxt,2), "Δ%": round(ret,2), "Signal": signal})
    except Exception:
        rows.append({"Symbol": s, "Last": None, "Predicted": None, "Δ%": None, "Signal": "ERR"})

if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%", ascending=False)
    st.subheader("Watchlist signals")
    st.dataframe(table, use_container_width=True)
