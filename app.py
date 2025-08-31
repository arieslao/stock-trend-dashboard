# app.py
import os, time, json, requests
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread


# ---------------- Page setup ----------------
st.set_page_config(page_title="AI-Powered Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard")
st.caption("Education only. Quotes: Finnhub. History: Finnhub→Yahoo (Chart JSON)→Stooq with mirrors. "
           "Model: Linear Regression. Request windows enforced (06:30 & 12:00 PT).")

# ---------------- Secrets / config ----------------
API_KEY = st.secrets.get("FINNHUB_KEY") or os.environ.get("FINNHUB_KEY", "")
if not API_KEY:
    st.warning("Missing FINNHUB_KEY. Set it in Streamlit → Settings → Secrets.")
GS_SECRETS = st.secrets.get("gsheets", {})
SHEET_ID = GS_SECRETS.get("sheet_id")
SA_INFO = GS_SECRETS.get("service_account")  # JSON string (from Google Cloud service account)

# ---------------- Timezone helpers (Pacific windows) ----------------
try:
    from zoneinfo import ZoneInfo             # Python 3.9+
except Exception:                             # pragma: no cover
    from pytz import timezone as ZoneInfo

LA = ZoneInfo("America/Los_Angeles")

# Two pull times: 06:30 and 12:00 PT. Allow a small window (minutes) after each.
WINDOWS_PT = [dtime(6, 30), dtime(12, 0)]
WINDOW_WIDTH_MIN = 7  # fetches allowed for 7 minutes after each start time

def now_la() -> datetime:
    return datetime.now(LA)

def in_window(now: datetime | None = None) -> bool:
    now = now or now_la()
    for t in WINDOWS_PT:
        start = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        end = start + timedelta(minutes=WINDOW_WIDTH_MIN)
        if start <= now <= end:
            return True
    return False

def seconds_until_next_window(now: datetime | None = None) -> int:
    now = now or now_la()
    waits = []
    for t in WINDOWS_PT:
        start = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        if start <= now:
            start += timedelta(days=1)
        waits.append((start - now).total_seconds())
    return int(min(waits))

# ---------------- Last-good caches (survive reruns) ----------------
@st.cache_resource
def _last_good_cache():
    # structure: {"quotes": {sym: price}, "candles": {sym: df}}
    return {"quotes": {}, "candles": {}}

# ---------------- Robust live fetchers (no yfinance) ----------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index.name = "time"
    return df[["o", "h", "l", "close", "v"]]

def _get_quote_live(symbol: str) -> float:
    """Finnhub real-time quote."""
    r = requests.get("https://finnhub.io/api/v1/quote",
                     params={"symbol": symbol, "token": API_KEY},
                     headers={"Accept": "application/json"},
                     timeout=10)
    r.raise_for_status()
    j = r.json()
    return float(j["c"])

def _get_candles_live(symbol: str, days: int = 180, resolution: str = "D") -> pd.DataFrame:
    """OHLCV by time. Order: Finnhub → Yahoo Chart JSON (multiple hosts + mirrors) → Stooq (direct + mirror)."""
    # 1) Finnhub
    try:
        end = int(time.time())
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={"symbol": symbol, "resolution": resolution, "from": start, "to": end, "token": API_KEY},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        if "json" not in r.headers.get("content-type", ""):
            raise RuntimeError("Finnhub did not return JSON")
        data = r.json()
        if data.get("s") == "ok" and data.get("t"):
            df = pd.DataFrame(
                {"t": data["t"], "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"], "v": data["v"]}
            )
            df["time"] = pd.to_datetime(df["t"], unit="s")
            df.rename(columns={"c": "close"}, inplace=True)
            df.set_index("time", inplace=True)
            return _normalize_ohlcv(df)
    except Exception:
        pass

    # 2) Yahoo Chart JSON (multi-host + mirror)
    def _fetch_yahoo_chart(base_url: str):
        end = int(time.time())
        start = end - int(days * 86400)
        params = {
            "period1": start, "period2": end, "interval": "1d",
            "includePrePost": "false", "events": "div,splits",
        }
        s = requests.Session()
        s.headers["User-Agent"] = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
        r = s.get(base_url + symbol, params=params, timeout=15)
        r.raise_for_status()
        try:
            j = r.json()
        except Exception:
            j = json.loads(r.text)
        res = j["chart"]["result"][0]
        ts = res.get("timestamp", [])
        q = res.get("indicators", {}).get("quote", [{}])[0]
        if ts and q.get("close"):
            df = pd.DataFrame({
                "time": pd.to_datetime(ts, unit="s"),
                "o": q.get("open", []), "h": q.get("high", []),
                "l": q.get("low", []), "close": q.get("close", []),
                "v": q.get("volume", []),
            })
            df = df.dropna(subset=["time"]).set_index("time").sort_index()
            if not df.empty:
                return _normalize_ohlcv(df)
        return None

    YAHOO_HOSTS = [
        "https://query2.finance.yahoo.com/v8/finance/chart/",
        "https://query1.finance.yahoo.com/v8/finance/chart/",
        "https://r.jina.ai/http://query2.finance.yahoo.com/v8/finance/chart/",
        "https://r.jina.ai/http://query1.finance.yahoo.com/v8/finance/chart/",
    ]
    for base in YAHOO_HOSTS:
        try:
            out = _fetch_yahoo_chart(base)
            if out is not None:
                return out
        except Exception:
            continue

    # 3) Stooq (direct + mirror)
    def _stooq_read(text: str):
        stooq = pd.read_csv(StringIO(text))
        if stooq.empty:
            return None
        stooq.rename(columns={"Date": "time", "Open": "o", "High": "h", "Low": "l",
                              "Close": "close", "Volume": "v"}, inplace=True)
        stooq["time"] = pd.to_datetime(stooq["time"], errors="coerce", utc=True).dt.tz_localize(None)
        stooq = stooq.dropna(subset=["time"]).set_index("time").sort_index()
        cutoff = (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=days + 5))
        try:
            stooq.index = pd.DatetimeIndex(stooq.index).tz_localize(None)
            stooq = stooq.loc[stooq.index >= cutoff]
        except Exception:
            stooq = stooq.tail(days + 5)
        if stooq.empty:
            return None
        return _normalize_ohlcv(stooq)

    stooq_sym = f"{symbol.lower()}.us"
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d",
                         timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = _stooq_read(r.text)
        if df is not None:
            return df
    except Exception:
        pass

    try:
        r = requests.get("https://r.jina.ai/http://stooq.com/q/d/l/",
                         params={"s": stooq_sym, "i": "d"},
                         timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = _stooq_read(r.text)
        if df is not None:
            return df
    except Exception:
        pass

    raise RuntimeError(f"No history for {symbol}: all sources blocked/empty")

# ---------------- Gated wrappers (respect time windows) ----------------
def get_quote(symbol: str) -> float:
    cache = _last_good_cache()["quotes"]
    if in_window():
        try:
            p = _get_quote_live(symbol)
            cache[symbol] = p
            return p
        except Exception:
            pass
    if symbol in cache:
        return cache[symbol]
    raise RuntimeError("Outside fetch window and no cached quote yet.")

def get_candles(symbol: str, days: int = 180, resolution: str = "D") -> pd.DataFrame:
    cache = _last_good_cache()["candles"]
    if in_window():
        df = _get_candles_live(symbol, days=days, resolution=resolution)
        cache[symbol] = df
        return df
    if symbol in cache:
        return cache[symbol]
    raise RuntimeError("Outside fetch window and no cached history yet.")

# ---------------- Simple features + model ----------------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA10"] = out["close"].rolling(10).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    return out.dropna()

def fit_and_predict(df_features: pd.DataFrame):
    X = df_features[["MA10", "MA50"]]
    y = df_features["close"]
    model = LinearRegression().fit(X, y)
    last = df_features.iloc[-1][["MA10", "MA50"]].values.reshape(1, -1)
    next_price = float(model.predict(last)[0])
    return model, next_price

# ---------------- Google Sheets logging (1 row per day) ----------------
@st.cache_resource
def get_ws():
    """Connect to 'predictions' worksheet, create if needed."""
    if not (SHEET_ID and SA_INFO):
        raise RuntimeError("Google Sheets not configured in secrets.")
    sa = SA_INFO
    if isinstance(sa, str):
        sa = json.loads(sa)
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("predictions")
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title="predictions", rows=2000, cols=20)
        ws.append_row([
            "timestamp_utc", "date_utc", "symbol", "horizon",
            "last_close", "predicted", "pct_change", "signal",
            "model_kind", "params_json", "train_window", "features",
            "actual", "err", "pct_err", "direction_correct"
        ])
    return ws

def log_prediction_once(symbol: str, horizon: str, last_close: float, predicted: float,
                        pct_change: float, signal: str, model_kind: str,
                        params: dict, train_window: int, features_desc: str):
    """Append only one row per (symbol, horizon, UTC date)."""
    try:
        ws = get_ws()
    except Exception as e:
        st.warning(f"Logged locally but Sheets not configured/available: {e}")
        return

    date_utc = datetime.utcnow().date().isoformat()  # YYYY-MM-DD
    # Check recent rows only (cheap): last 300 rows
    try:
        values = ws.get_values("A2:P2000")[-300:]
    except Exception:
        values = []
    # columns: 0 timestamp, 1 date_utc, 2 symbol, 3 horizon
    for row in values:
        if len(row) >= 4 and row[1] == date_utc and row[2] == symbol and row[3] == horizon:
            return  # already logged today

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ws.append_row([
        ts, date_utc, symbol, horizon,
        float(last_close), float(predicted), float(pct_change), signal,
        model_kind, json.dumps(params), int(train_window), features_desc,
        "", "", "", ""
    ])

# ---------------- Sidebar controls ----------------
default_watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA"]
with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect("Watchlist", default_watchlist, default_watchlist)
    days = st.slider("History (days)", 90, 365, 180, step=10)
    alert_pct = st.slider("Alert threshold (%)", 0.5, 5.0, 2.0, step=0.5)
    focus = st.selectbox("Focus symbol", watchlist or default_watchlist)

    # Window status
    if not in_window():
        mins = seconds_until_next_window() // 60
        st.info(f"⏳ Outside fetch window. Using last cached data. "
                f"Next window opens in ~{mins} min (Pacific).")
    else:
        st.success("✅ Inside fetch window — live data will refresh.")

# ---------------- Diagnostics ----------------
with st.expander("🧪 Diagnostics", expanded=False):
    st.write("Python:", sys.version.split()[0])
    st.write("Now (PT):", now_la().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("In window:", in_window())
    st.write("Next window in (min):", seconds_until_next_window() // 60)

    if SHEET_ID:
        st.write("Google Sheet ID:", SHEET_ID)
    if SA_INFO:
        try:
            sa_json = json.loads(SA_INFO) if isinstance(SA_INFO, str) else SA_INFO
            st.write("Service account email:", sa_json.get("client_email", "(unknown)"))
        except Exception:
            st.write("Service account present.")

# inside the Diagnostics expander for a "Force Fetch Now" capability outside of the 2 windows
if st.button("Force one-time LIVE fetch & cache for focus symbol"):
    try:
        df_live = _get_candles_live(focus, days=days)   # bypasses window
        _last_good_cache()["candles"][focus] = df_live
        price_live = _get_quote_live(focus)
        _last_good_cache()["quotes"][focus] = price_live
        st.success("Fetched & cached live data. Reload the page.")
    except Exception as e:
        st.error(f"Force fetch failed: {e}")
# continuation of original code
           
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Reload the app.")

    with c2:
        if st.button("Test write to Google Sheet"):
            try:
                log_prediction_once("TEST", "1d", 100.0, 101.0, 1.0, "TEST",
                                    "Linear", {}, 100, "MA10,MA50")
                st.success("Wrote a test row to 'predictions'.")
            except Exception as e:
                st.error(f"Failed to write: {e}")

#--------- commenting this section out to print out the HTML status for troubleshooting-----
   # with c3:
    #    if st.button("Test history fetch (AAPL, 30d)"):
     #       try:
      #          df_test = _get_candles_live("AAPL", days=30)
       #         st.success(f"Fetched {len(df_test)} rows via live pipeline.")
        #    except Exception as e:
         #       st.error(str(e))
#---------- Replacing it with the below script for testing, might consider reinstating the above after fixing---------

if st.button("Test history fetch (AAPL, 30d)"):
    import requests, time
    symbol = "AAPL"
    days_to_get = 30

    st.write("### Live history debug (bypasses windows)")
    st.caption("Shows HTTP status and first ~160 chars of the body to diagnose rate limits / blocks.")

    # --- FINNHUB ---
    try:
        end = int(time.time())
        start = end - days_to_get * 86400
        finnhub_url = "https://finnhub.io/api/v1/stock/candle"
        finnhub_params = {
            "symbol": symbol,
            "resolution": "D",
            "from": start,
            "to": end,
            "token": API_KEY,   # assumes you've defined API_KEY already
        }
        r = requests.get(finnhub_url, params=finnhub_params, timeout=15)
        st.write("**Finnhub** status:", r.status_code)
        st.write("Finnhub head:", r.text[:160])
        r.raise_for_status()
        j = r.json()
        if j.get("s") == "ok":
            st.success(f"Finnhub OK — bars: {len(j.get('t', []))}")
        else:
            st.warning(f"Finnhub returned s='{j.get('s')}'")
    except Exception as e:
        st.error(f"Finnhub error: {e}")

    # --- YAHOO CHART ---
    try:
        yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        yahoo_params = {"interval": "1d", "range": f"{days_to_get}d"}
        # User-Agent helps reduce HTML/CF blocks
        r = requests.get(yahoo_url, params=yahoo_params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        st.write("**Yahoo** status:", r.status_code)
        st.write("Yahoo head:", r.text[:160])
        r.raise_for_status()
        j = r.json()
        result = j.get("chart", {}).get("result") or []
        st.success(f"Yahoo OK — result objects: {len(result)}")
    except Exception as e:
        st.error(f"Yahoo error: {e}")

    # --- STOOQ CSV ---
    try:
        stooq_url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        r = requests.get(stooq_url, timeout=15)
        st.write("**Stooq** status:", r.status_code)
        st.write("Stooq head:", r.text[:160])
        r.raise_for_status()
        # if needed: parse CSV with pandas
        # df = pd.read_csv(io.StringIO(r.text))
        st.success("Stooq OK (CSV returned).")
    except Exception as e:
        st.error(f"Stooq error: {e}")




# ---------------- Focus symbol area ----------------
col1, col2 = st.columns([3, 1])
with col1:
    # PRICE
    try:
        price = get_quote(focus)
        st.subheader(f"{focus} — Live: {price:,.2f}")
    except Exception as e:
        st.subheader(f"{focus} — Live: —")
        st.warning(f"Quote unavailable: {e}")

    # CHART + FORECAST
    try:
        df = get_candles(focus, days=days, resolution="D")
        feat = featurize(df)
        model, next_price = fit_and_predict(feat)
        pct_change = 100.0 * (next_price - df['close'].iloc[-1]) / df['close'].iloc[-1]

        line = px.line(df.reset_index(), x="time", y="close", title=f"{focus} Close Price")
        line.add_scatter(x=feat.index, y=feat["MA10"], mode="lines", name="MA10")
        line.add_scatter(x=feat.index, y=feat["MA50"], mode="lines", name="MA50")
        st.plotly_chart(line, use_container_width=True)

        # Log once/day to Google Sheet
        log_prediction_once(
            symbol=focus, horizon="1d",
            last_close=float(df["close"].iloc[-1]),
            predicted=float(next_price),
            pct_change=float(pct_change),
            signal=("BUY↑" if pct_change >= alert_pct else ("SELL↓" if pct_change <= -alert_pct else "HOLD")),
            model_kind="Linear", params={"features": ["MA10", "MA50"]},
            train_window=days, features_desc="MA10,MA50"
        )

    except Exception as e:
        st.error(f"History/forecast error: {e}")
        next_price = None
        pct_change = None

with col2:
    st.metric(
        "Predicted next close",
        f"{next_price:,.2f}" if next_price is not None else "—",
        f"{pct_change:+.2f}% vs last" if pct_change is not None else None
    )
    if pct_change is not None:
        if pct_change >= alert_pct:
            st.success(f"Potential BUY momentum (≥ {alert_pct}%).")
        elif pct_change <= -alert_pct:
            st.error(f"Potential SELL/caution (≤ -{alert_pct}%).")
        else:
            st.info("No strong signal at your threshold.")

# ---------------- Watchlist table ----------------
rows = []
for sym in watchlist:
    try:
        p = get_quote(sym)
    except Exception:
        p = None
    try:
        d = get_candles(sym, days=days)
        f = featurize(d)
        _, nxt = fit_and_predict(f)
        ret = 100.0 * (nxt - d["close"].iloc[-1]) / d["close"].iloc[-1]
        sig = "BUY↑" if ret >= alert_pct else ("SELL↓" if ret <= -alert_pct else "HOLD")
        rows.append({"Symbol": sym, "Last": round(p, 2) if p else None,
                     "Predicted(1d)": round(nxt, 2), "Δ%(1d)": round(ret, 2), "Signal": sig})
    except Exception:
        rows.append({"Symbol": sym, "Last": round(p, 2) if p else None,
                     "Predicted(1d)": None, "Δ%(1d)": None, "Signal": "ERR"})

if rows:
    st.subheader("Watchlist signals (1-day)")
    st.dataframe(pd.DataFrame(rows).set_index("Symbol"), use_container_width=True)
