# app.py — Stock Trend Dashboard (Sheets-first, with predictions writer)
# - Primary data source: Google Sheets 'prices' (case-insensitive headers)
# - CNN-LSTM multi-horizon predictions using the same scaler + feature list as training
# - Linear MA10/MA50 fallback for the UI table (optional)
# - Button to append per-horizon predictions to 'predictions' sheet

import os, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Optional / safe TF import (only used if CNN-LSTM is present)
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # allow UI to run without TF

# ---------- Google Sheets (service account) ----------
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

# ---------- Constants ----------
DEFAULT_LOOKBACK = 60          # must match training WINDOW
HORIZONS_DEFAULT = [1, 3, 5, 10, 20]
PRICES_TAB = "prices"
WATCHLIST_TAB = "watchlist_cnnlstm"
WATCHLIST_COL = "Ticker"
PREDICTIONS_TAB = "predictions"

LOG_COLUMNS = [
    "ts_utc", "symbol", "model", "lookback",
    "days_history", "last_close", "predicted", "pct_change",
    "note"
]

# ---------- Streamlit page ----------
st.set_page_config(page_title="AI Stock Trend Dashboard", layout="wide")
st.title("📈 AI-Powered Stock Trend Dashboard (Google Sheets)")
st.caption(
    "Education only — not financial advice. "
    "Source of truth: Google Sheets → 'prices'. "
    "Models: CNN-LSTM (multi-horizon) with Linear MA10/MA50 fallback for the table."
)

# ---------- Google Sheets helpers ----------
@st.cache_resource(show_spinner=False)
def _open_sheet() -> Tuple[Optional[object], Optional[object], Optional[str]]:
    """Return (client, spreadsheet, sheet_id) or (None, None, msg) if misconfigured."""
    if not (gspread and Credentials):
        return None, None, "gspread / google-auth not installed"

    # Sheet ID
    sheet_id = None
    if hasattr(st, "secrets"):
        sheet_id = (
            st.secrets.get("GOOGLE_SHEETS_SHEET_ID")
            or (st.secrets.get("gsheets", {}) or {}).get("sheet_id")
        )
    sheet_id = sheet_id or os.getenv("GOOGLE_SHEETS_SHEET_ID")
    if not sheet_id:
        return None, None, "Missing GOOGLE_SHEETS_SHEET_ID"

    # Credentials
    creds_info = None
    if hasattr(st, "secrets"):
        if "GOOGLE_SHEETS_JSON" in st.secrets:  # full JSON string
            try:
                creds_info = json.loads(st.secrets["GOOGLE_SHEETS_JSON"])
            except json.JSONDecodeError:
                return None, None, "GOOGLE_SHEETS_JSON is not valid JSON"
        elif "gcp_service_account" in st.secrets:  # TOML table
            creds_info = dict(st.secrets["gcp_service_account"])
    if creds_info is None and os.getenv("GOOGLE_SHEETS_JSON"):
        try:
            creds_info = json.loads(os.getenv("GOOGLE_SHEETS_JSON"))
        except json.JSONDecodeError:
            return None, None, "Env GOOGLE_SHEETS_JSON is not valid JSON"

    if not creds_info:
        return None, None, "Missing service account JSON (GOOGLE_SHEETS_JSON or [gcp_service_account])"

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
        return client, sh, sheet_id
    except Exception as e:
        return None, None, f"Auth/open failed: {e}"

@st.cache_data(ttl=300)
def _prices_df_from_sheet(prices_tab: str = PRICES_TAB) -> pd.DataFrame:
    """Read 'prices' → canonical df with: date, symbol, close, (vol optional). Case-insensitive."""
    _, sh, _ = _open_sheet()
    if sh is None:
        return pd.DataFrame()
    try:
        ws = sh.worksheet(prices_tab)
    except Exception:
        return pd.DataFrame()

    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(vals[1:], columns=[h.strip() for h in vals[0]])
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common variants
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})

    required = {"date", "symbol", "close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["close"]  = pd.to_numeric(df["close"], errors="coerce")
    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return df[["date", "symbol", "close"] + (["vol"] if "vol" in df.columns else [])]

def _read_watchlist_symbols(worksheet: str = WATCHLIST_TAB, column: str = WATCHLIST_COL) -> List[str]:
    _, sh, _ = _open_sheet()
    if sh is None:
        return []
    try:
        ws = sh.worksheet(worksheet)
    except Exception:
        return []
    header = [h.strip() for h in ws.row_values(1)]
    try:
        idx = [h.lower() for h in header].index(column.lower()) + 1
    except ValueError:
        return []
    syms = [s.strip().upper() for s in ws.col_values(idx)[1:] if s.strip()]
    # Deduplicate
    out, seen = [], set()
    for t in syms:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def _ensure_logs_header(ws):
    vals = ws.get_all_values()
    if not vals or (vals and vals[0] != LOG_COLUMNS):
        ws.update("A1", [LOG_COLUMNS])

def _append_logs(rows: List[Dict]):
    _, sh, _ = _open_sheet()
    if sh is None: return
    try:
        try:
            ws = sh.worksheet("logs")
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title="logs", rows=1000, cols=20)
        _ensure_logs_header(ws)
        payload = [[r.get(k, "") for k in LOG_COLUMNS] for r in rows]
        if payload:
            ws.append_rows(payload, value_input_option="RAW")
    except Exception:
        pass

# ---------- Model loaders ----------
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
                m = tf.keras.models.load_model(p)
                st.session_state["cnn_model_path"] = str(p)
                return m
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_scaler_and_feats() -> Optional[Tuple[object, List[str]]]:
    """Return (scaler, feats_order) saved during training, or None if missing."""
    for p in [
        Path("models") / "scaler_ALL.pkl",
        Path("models") / "scaler.pkl",
        Path("scaler_ALL.pkl"),
        Path("scaler.pkl"),
    ]:
        if p.exists():
            try:
                obj = joblib.load(p)
                if isinstance(obj, dict) and "scaler" in obj and "feats" in obj:
                    return obj["scaler"], list(obj["feats"])
                # Fallbacks if stored differently
                if isinstance(obj, dict) and "feats" in obj:
                    scaler = obj.get("scaler") or obj.get("minmax") or obj.get("mm")
                    return scaler, list(obj["feats"])
                if hasattr(obj, "transform"):
                    # Unknown feats; assume ['close'] to avoid crash
                    return obj, ["close"]
            except Exception as e:
                st.warning(f"Failed to load scaler/features from {p.name}: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_linear_model():
    for p in [Path("models") / "linear_model.pkl", Path("linear_model.pkl")]:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.warning(f"Failed to load linear model from {p.name}: {e}")
                return None
    return None

# ---------- Training feature engineering (matches train.py) ----------
def _build_training_features_from_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns (lower-case): date(index or col), symbol(optional), close, (vol optional).
    Output: adds ret1, ma10, ma50, vlog -> drops NaNs.
    """
    df = df_prices.copy()
    if "vol" not in df.columns:
        df["vol"] = 0.0
    df["ret1"] = df["close"].pct_change()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def _inverse_close_only(pred_scaled: np.ndarray, scaler, feats_order: List[str]) -> np.ndarray:
    """Invert MinMax scaling for the 'close' feature."""
    # scikit-learn MinMaxScaler:
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
        close_idx = feats_order.index("close")
        data_min = scaler.data_min_[close_idx]
        data_rng = scaler.data_range_[close_idx]
        return pred_scaled * data_rng + data_min
    # If custom scaler dict with scale_/min_
    if hasattr(scaler, "scale_") and hasattr(scaler, "min_"):
        # y_scaled = y*scale + min -> y = (y_scaled - min)/scale
        close_idx = feats_order.index("close")
        return (pred_scaled - scaler.min_[close_idx]) / scaler.scale_[close_idx]
    return pred_scaled  # last resort (no-op)

# ---------- Price access (Sheets) ----------
@st.cache_data(ttl=300)
def fetch_history(symbol: str, days: int, prices_tab: str = PRICES_TAB) -> Optional[pd.DataFrame]:
    """
    Read from Google Sheets 'prices' tab, filter to symbol & last N days.
    Returns columns: close, (vol), indexed by date.
    """
    df_all = _prices_df_from_sheet(prices_tab)
    if df_all is None or df_all.empty:
        return None
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    d = df_all[(df_all["symbol"] == symbol.upper()) & (df_all["date"] >= cutoff)].copy()
    if d.empty:
        return None
    d = d.set_index("date").sort_index()  # keep lower-case columns: close, vol?
    return d  # columns: close, (vol)

# ---------- Predictors ----------
def predict_multi_cnnlstm(symbol: str, df_prices: pd.DataFrame,
                          lookback: int = DEFAULT_LOOKBACK) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """
    Return (predicted_closes[horizons], last_close, feats_order) using the real training features & scaler.
    df_prices: must contain 'close' (and ideally 'vol'); index is datetime ascending.
    """
    if tf is None:
        return None
    model = load_cnn_lstm(symbol)
    sf = load_scaler_and_feats()
    if model is None or sf is None:
        return None
    scaler, feats_order = sf

    # Prepare features exactly like training
    feat = _build_training_features_from_prices(
        pd.DataFrame({
            "close": df_prices["close"].astype(float).values,
            "vol": (df_prices["vol"].astype(float).values if "vol" in df_prices.columns else np.zeros(len(df_prices))),
        }, index=df_prices.index)
    )

    if len(feat) < lookback:
        return None

    # Scale in the same order used during training
    try:
        feat_scaled = feat.copy()
        feat_scaled[feats_order] = scaler.transform(feat_scaled[feats_order].values)
    except Exception:
        # If scaler is dict-like, try best-effort
        try:
            feat_scaled = feat.copy()
            feat_scaled[feats_order] = scaler.transform(feat_scaled[feats_order])
        except Exception:
            return None

    X = feat_scaled[feats_order].tail(lookback).to_numpy(dtype=np.float32)[np.newaxis, :, :]  # (1,L,F)
    if X.shape[1] != lookback:
        return None

    try:
        y_scaled = model.predict(X, verbose=0)[0]  # shape (n_horizons,)
    except Exception:
        return None

    preds = _inverse_close_only(np.asarray(y_scaled, dtype=float), scaler, feats_order)
    last_close = float(feat["close"].iloc[-1])
    return preds, last_close, feats_order

def predict_next_close_linear(df_prices: pd.DataFrame) -> Optional[float]:
    """Simple fallback using MA10/MA50 and a pre-trained LinearRegression."""
    mdl = load_linear_model()
    if mdl is None or df_prices is None or df_prices.empty or "close" not in df_prices.columns:
        return None
    tmp = df_prices.copy()
    tmp["MA10"] = tmp["close"].rolling(10).mean()
    tmp["MA50"] = tmp["close"].rolling(50).mean()
    tmp = tmp.dropna()
    if tmp.empty:
        return None
    try:
        X_last = tmp.iloc[[-1]][["MA10","MA50"]].to_numpy(dtype=float)
        return float(mdl.predict(X_last)[0])
    except Exception:
        return None

# ---------- Predictions sheet writer ----------
def _ensure_predictions_header(ws):
    header = [
        "timestamp_utc","symbol","horizon","last_close","predicted",
        "pct_change","signal","scope","model_kind","params_json",
        "train_window","features","actual"
    ]
    vals = ws.get_all_values()
    if not vals:
        ws.update("A1", [header])

def append_predictions_rows(rows: List[List]):
    """Append rows to predictions tab, creating it (and header) if needed."""
    _, sh, _ = _open_sheet()
    if sh is None: return
    try:
        try:
            ws = sh.worksheet(PREDICTIONS_TAB)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=PREDICTIONS_TAB, rows=2000, cols=20)
            _ensure_predictions_header(ws)
        _ensure_predictions_header(ws)
        for i in range(0, len(rows), 800):
            ws.append_rows(rows[i:i+800], value_input_option="USER_ENTERED")
    except Exception:
        pass

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")

    # Try to use watchlist from Google Sheet
    sheet_syms = _read_watchlist_symbols(WATCHLIST_TAB, WATCHLIST_COL)
    default_watchlist = sheet_syms if sheet_syms else ["AAPL", "MSFT", "GOOGL", "TSLA"]

    manual = st.text_input(
        "Watchlist (comma-separated, overrides sheet)",
        value=",".join(default_watchlist)
    )
    watchlist = [s.strip().upper() for s in manual.split(",") if s.strip()]

    days = st.slider("History window (days)", 60, 3650, 365, step=5)
    alert_pct = st.slider("Alert threshold (%)", 0.5, 10.0, 2.0, step=0.25)

    MODEL_CNN = "CNN-LSTM (multi-horizon)"
    MODEL_LINEAR = "Linear (MA10/MA50)"
    model_choice = st.selectbox(
        "Table model",
        (MODEL_CNN, MODEL_LINEAR),
        index=0,
        help="CNN-LSTM uses your trained deep model; Linear is a simple fallback for the table."
    )
    st.session_state["model_choice"] = model_choice

    if st.button("🔄 Reload prices cache from Google Sheets"):
        _prices_df_from_sheet.clear()
        fetch_history.clear()
        st.success("Reloaded prices cache.")

def using_cnn() -> bool:
    return st.session_state.get("model_choice") == "CNN-LSTM (multi-horizon)"

# ---------- Score watchlist (UI table) ----------
rows: List[Dict] = []
logs: List[Dict] = []

with st.spinner("Scoring watchlist from Google Sheets…"):
    for sym in (watchlist or []):
        try:
            d = fetch_history(sym, days)  # columns: close,(vol), index=date
            if d is None or d.empty or "close" not in d.columns:
                raise RuntimeError("No price rows in 'prices' for symbol")

            last_close = float(d["close"].iloc[-1])
            model_used = "—"
            next_pred = None

            if using_cnn():
                res = predict_multi_cnnlstm(sym, d, lookback=DEFAULT_LOOKBACK)
                if res is not None:
                    preds_vec, _, _ = res
                    if preds_vec.size > 0:
                        next_pred = float(preds_vec[0])  # 1-day ahead for the table
                        model_used = "CNN-LSTM"

            if next_pred is None:
                lin = predict_next_close_linear(d)
                if lin is not None:
                    next_pred = lin
                    model_used = "Linear (fallback)"

            if next_pred is None:  # Naive
                next_pred = last_close
                model_used = "Naive"

            ret = 100.0 * (float(next_pred) - last_close) / last_close
            signal = "BUY↑" if ret >= alert_pct else ("SELL↓" if ret <= -alert_pct else "HOLD")

            rows.append({
                "Symbol": sym,
                "Last": round(last_close, 2),
                "Predicted": round(float(next_pred), 2),
                "Δ%": round(float(ret), 2),
                "Signal": signal,
                "Model": model_used
            })

            logs.append({
                "ts_utc": pd.Timestamp.utcnow().replace(microsecond=0).isoformat() + "Z",
                "symbol": sym,
                "model": model_used,
                "lookback": DEFAULT_LOOKBACK if "CNN" in model_used else "",
                "days_history": days,
                "last_close": round(last_close, 6),
                "predicted": round(float(next_pred), 6),
                "pct_change": round(float(ret), 4),
                "note": "watchlist"
            })

        except Exception:
            rows.append({
                "Symbol": sym,
                "Last": None, "Predicted": None, "Δ%": None,
                "Signal": "ERR", "Model": "—"
            })

st.subheader("Watchlist signals")
if rows:
    table = pd.DataFrame(rows).set_index("Symbol").sort_values("Δ%", ascending=False, na_position="last")
    st.dataframe(table, use_container_width=True)
else:
    st.info("Your watchlist is empty. Add symbols in the sidebar.")

# Optional: append table logs (non-fatal)
if logs:
    _append_logs(logs)

# ---------- Button: write per-horizon predictions to 'predictions' ----------
st.markdown("---")
st.subheader("✍️ Generate multi-horizon predictions → Google Sheets")

colA, colB = st.columns([1, 3])
with colA:
    write_btn = st.button("Write predictions (1,3,5,10,20 days)")

with colB:
    st.caption(
        "Writes rows to the **'predictions'** worksheet using your **CNN-LSTM** model, "
        "matching the nightly workflow format (timestamp_utc, symbol, horizon, last_close, predicted, …). "
        "If CNN-LSTM or scaler/feats are missing, nothing will be written."
    )

if write_btn:
    wrote = 0
    missing = []
    for sym in (watchlist or []):
        try:
            d = fetch_history(sym, days)
            if d is None or d.empty or "close" not in d.columns:
                missing.append(sym); continue

            res = predict_multi_cnnlstm(sym, d, lookback=DEFAULT_LOOKBACK)
            if res is None:
                missing.append(sym); continue

            preds_vec, last_close, feats_order = res
            now_iso = pd.Timestamp.utcnow().replace(microsecond=0).isoformat() + "Z"
            rows_out: List[List] = []
            for h, p in zip(HORIZONS_DEFAULT, preds_vec[:len(HORIZONS_DEFAULT)]):
                p = float(p)
                pct = 0.0 if last_close == 0 else (p - last_close) / last_close * 100.0
                signal = 1 if p > last_close else (-1 if p < last_close else 0)
                params_json = json.dumps({"horizons": HORIZONS_DEFAULT})
                rows_out.append([
                    now_iso, sym, h, round(last_close, 6), round(p, 6),
                    round(pct, 4), signal, "APP", "CNN-LSTM",
                    params_json, DEFAULT_LOOKBACK, ",".join(feats_order), ""
                ])
            if rows_out:
                append_predictions_rows(rows_out)
                wrote += len(rows_out)
        except Exception:
            missing.append(sym)

    if wrote > 0:
        st.success(f"Wrote {wrote} prediction rows to '{PREDICTIONS_TAB}'.")
    if missing:
        st.warning(f"Skipped (no data or model): {', '.join(missing[:12])}{'…' if len(missing)>12 else ''}")

# ---------- Diagnostics ----------
with st.expander("🔧 Diagnostics", expanded=False):
    client, sh, sid = _open_sheet()
    if sh is not None:
        st.write("Google Sheets: connected ✅")
        st.write("Spreadsheet ID:", sid)
        try:
            ws_names = [ws.title for ws in sh.worksheets()]
            st.write("Worksheets:", ", ".join(ws_names))
        except Exception:
            pass
    else:
        st.write("Google Sheets: not connected — check secrets (GOOGLE_SHEETS_JSON) and sheet id.")

    mdl = load_cnn_lstm()
    st.write("CNN-LSTM loaded:", bool(mdl))
    if mdl:
        try:
            st.write("Model input shape:", mdl.input_shape)
        except Exception:
            pass

    sf = load_scaler_and_feats()
    st.write("Scaler+feats loaded:", bool(sf))
    if sf:
        s, feats = sf
        st.write("Training feature order:", feats)

    lin = load_linear_model()
    st.write("Linear model loaded:", bool(lin))

    prices_sample = _prices_df_from_sheet().head(5)
    st.write("Prices sample (from sheet):")
    st.dataframe(prices_sample if not prices_sample.empty else pd.DataFrame({"status":["(empty)"]}))
