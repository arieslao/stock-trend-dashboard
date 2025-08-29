import os, time, requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px
import yfinance as yf

# ---------- App Config ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption("Education only. Data: Finnhub. Forecasts: scikit-learn (Linear Regression).")

# Read API key from Streamlit secrets or environment
API_KEY = st.secrets.get("FINNHUB_KEY", os.environ.get("FINNHUB_KEY", ""))
if not API_KEY:
    st.error("Missing FINNHUB_KEY. In Streamlit Cloud, set it in 'Advanced settings → Secrets'.")
    st.stop()

# ---------- Helpers ----------
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
    """Historical candles: try Finnhub; on 403 or error, fall back to Yahoo Finance."""
    # --- First, try Finnhub
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
            df = pd.DataFrame({"t": data["t"], "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"], "v": data["v"]})
            df["time"] = pd.to_datetime(df["t"], unit="s")
            df.rename(columns={"c": "close"}, inplace=True)
            df.set_index("time", inplace=True)
            return df
        else:
            # Fall through to Yahoo if Finnhub responds with error status
            raise RuntimeError(f"Finnhub candles status: {data.get('s')}")
    except Exception as e:
        # --- Fallback: Yahoo Finance
        # Map daily to 1d; intraday could map resolution '60' -> interval='60m'
        interval = "1d" if resolution == "D" else "60m"
        period_days = min(max(days, 5), 365*2)  # safety bounds
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{period_days}d", interval=interval)
        if hist.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {symbol}") from e
        df = hist.rename(columns={"Close": "close", "Open":"o", "High":"h", "Low":"l", "Volume":"v"})
        df.index.name = "time"
        return df[["o","h","l","close","v"]]


def featurize(df: pd.DataFrame):
    out = df.copy()
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    out = out.dropna()
    return out

def fit_and_predict(df_features: pd.DataFrame):
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    model = LinearRegression().fit(X, y)
    last = df_features.iloc[-1][["MA10", "MA50"]].values.reshape(1, -1)
    next_price = float(model.predict(last)[0])
    return model, next_price

# ---------- Sidebar / Controls ----------
default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA"]
with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect("Watchlist", default_watchlist, default=default_watchlist)
    days = st.slider("History (days)", 90, 365, 180, step=10)
    alert_pct = st.slider("Alert threshold (%)", 0.5, 5.0, 2.0, step=0.5)
    symbol = st.selectbox("Focus symbol", watchlist or default_watchlist)
    st.markdown("**Note:** Free Finnhub keys allow ~60 requests/min.")

# ---------- Focus symbol KPIs ----------
try:
    quote = get_quote(symbol)
    price = quote["c"]
    st.subheader(f"{symbol} — Live: {price:,.2f}")
except Exception as e:
    st.error(f"Quote error for {symbol}: {e}")
    st.stop()

# ---------- Chart + Forecast ----------
col1, col2 = st.columns([3,1])
with col1:
    try:
        df = get_candles(symbol, days=days, resolution="D")
        feat = featurize(df)
        model, next_price = fit_and_predict(feat)
        pct_change = 100.0 * (next_price - df["close"].iloc[-1]) / df["close"].iloc[-1]

        line = px.line(df.reset_index(), x="time", y="close", title=f"{symbol} Close Price")
        line.add_scatter(x=feat.index, y=feat["MA10"], mode="lines", name="MA10")
        line.add_scatter(x=feat.index, y=feat["MA50"], mode="lines", name="MA50")
        st.plotly_chart(line, use_container_width=True)
    except Exception as e:
        st.error(f"History/forecast error: {e}")

with col2:
    st.metric("Predicted next close", f"{next_price:,.2f}" if 'next_price' in locals() else "—",
              f"{pct_change:+.2f}% vs last" if 'pct_change' in locals() else None)
    if 'pct_change' in locals():
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
        q = get_quote(s)
        p = q["c"]
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
