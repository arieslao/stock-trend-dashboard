# predict_cnn_lstm.py — load the trained CNN-LSTM and write predictions to Google Sheets
import os, json, argparse, datetime as dt
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import gspread
import joblib
import tensorflow as tf

WINDOW = 60  # must match training


# ---------------------- Helpers ----------------------

def period_to_days(p: str) -> int:
    p = str(p).strip().lower()
    if p.endswith("y"):   return int(p[:-1]) * 365
    if p.endswith("mo"):  return int(p[:-2]) * 30
    if p.endswith("m"):   return int(p[:-1]) * 30
    if p.endswith("d"):   return int(p[:-1])
    return 365


def load_prices_from_sheet(
    gc,
    sheet_id: str,
    prices_worksheet: str,
    symbols: Optional[List[str]],
    period: str,
) -> pd.DataFrame:
    """
    Load prices from Google Sheet, normalize headers, filter by symbols/period,
    and return a tidy DataFrame with columns: date, symbol, close [, vol].
    """
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_worksheet)

    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        raise RuntimeError(f"'{prices_worksheet}' is empty or missing headers")

    # 1) Create df, then normalize headers
    df = pd.DataFrame(vals[1:], columns=[h.strip() for h in vals[0]])
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) Map common variants → expected names
    rename_map = {}
    if "adj close" in df.columns and "close" not in df.columns:
        rename_map["adj close"] = "close"
    if "adj_close" in df.columns and "close" not in df.columns:
        rename_map["adj_close"] = "close"
    if "volume" in df.columns and "vol" not in df.columns:
        rename_map["volume"] = "vol"
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) Validate required columns
    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"'{prices_worksheet}' missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # 4) Types & cleaning
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")

    df = df.dropna(subset=["date", "close"])

    # 5) Filter by symbols (if provided)
    if symbols:
        want = {s.strip().upper() for s in symbols if s and str(s).strip()}
        df = df[df["symbol"].isin(want)]

    # 6) Filter by period: e.g., "3y", "18m", "max"
    per = (period or "").strip().lower()
    if per and per != "max":
        now = pd.Timestamp.utcnow().tz_localize(None)
        cutoff = None
        if per.endswith("y"):
            cutoff = now - pd.DateOffset(years=int(per[:-1] or "1"))
        elif per.endswith("m"):
            cutoff = now - pd.DateOffset(months=int(per[:-1] or "1"))
        elif per.endswith("d"):
            cutoff = now - pd.Timedelta(days=int(per[:-1] or "1"))
        if cutoff is not None:
            df = df[df["date"] >= cutoff]

    # 7) Canonical order & sort
    cols = ["date", "symbol", "close"] + (["vol"] if "vol" in df.columns else [])
    df = df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def make_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same features used in training: close, ret1, ma10, ma50, vlog.
    All column names here are lower-case: symbol, close, vol.
    """
    df = df_prices.copy()

    # If 'vol' missing, fabricate zeros so 'vlog' is defined (keeps pipeline stable)
    if "vol" not in df.columns:
        df["vol"] = 0.0

    # Group by lower-case 'symbol'
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])

    return df.dropna()


def last_window_tensor(df_feat: pd.DataFrame, feats_order: List[str], window: int = WINDOW):
    if len(df_feat) < window:
        return None, None
    X = df_feat.iloc[-window:][feats_order].values.astype(np.float32)
    last_close_raw = float(df_feat.iloc[-1]["close_raw"])
    return X, last_close_raw


def inverse_close_only(pred_scaled: np.ndarray, scaler, feats_order: List[str]) -> np.ndarray:
    """
    Invert MinMax scaling for the 'close' feature only.
    """
    close_idx = feats_order.index("close")
    data_min = scaler.data_min_[close_idx]
    data_rng = scaler.data_range_[close_idx]
    return pred_scaled * data_rng + data_min


def read_calibration(gc, sheet_id: str, tab: str) -> Dict[str, float]:
    """
    Read optional per-symbol calibration (mean error_pct) from a sheet.
    Accepts either 'Symbol' or 'symbol' and 'bias_pct' (case-insensitive).
    """
    try:
        ws = gc.open_by_key(sheet_id).worksheet(tab)
    except Exception:
        return {}
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return {}
    header = [h.strip() for h in vals[0]]
    hlow = [h.lower() for h in header]
    need = ["symbol", "bias_pct"]
    if any(c not in hlow for c in need):
        return {}
    i_symbol = hlow.index("symbol")
    i_bias = hlow.index("bias_pct")
    out: Dict[str, float] = {}
    for r in vals[1:]:
        try:
            sym = str(r[i_symbol]).strip().upper()
            bias = float(r[i_bias])
            if sym:
                out[sym] = bias
        except Exception:
            continue
    return out


def sheet_append_rows(gc, sheet_id: str, tab: str, rows: List[List]):
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab)
    except Exception:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=26)
        ws.append_row(
            [
                "timestamp_utc", "symbol", "horizon", "last_close", "predicted",
                "pct_change", "signal", "scope", "model_kind", "params_json",
                "train_window", "features", "actual",
            ]
        )
    for i in range(0, len(rows), 1000):
        ws.append_rows(rows[i:i + 1000], value_input_option="USER_ENTERED")


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--worksheet", required=True, help="watchlist tab with a 'Ticker' column")
    ap.add_argument("--symbol-column", default="Ticker")
    ap.add_argument("--prices-worksheet", default="prices")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--model-path", default="models/cnn_lstm_ALL.keras")
    ap.add_argument("--scaler-path", default="models/scaler_ALL.pkl")
    ap.add_argument("--predictions-tab", default="predictions")
    ap.add_argument("--scope", default="PROD")
    ap.add_argument("--calibration-worksheet", default="model_calibration")
    ap.add_argument("--apply-calibration", action="store_true")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]

    # Google auth (env var set by workflow)
    gc = gspread.service_account(filename=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    # 1) Watchlist symbols
    sh = gc.open_by_key(args.sheet_id)
    wl = sh.worksheet(args.worksheet)
    hdr = [h.strip() for h in wl.row_values(1)]
    try:
        cidx = hdr.index(args.symbol_column) + 1
    except ValueError:
        raise SystemExit(f"Column '{args.symbol_column}' not found in watchlist '{args.worksheet}'")
    symbols = [s.strip().upper() for s in wl.col_values(cidx)[1:] if s.strip()]
    if not symbols:
        print("No symbols in watchlist — nothing to predict.")
        return

    # 2) Load prices & build features
    raw = load_prices_from_sheet(gc, args.sheet_id, args.prices_worksheet, symbols, args.period)
    if raw.empty:
        raise SystemExit("No rows in 'prices' for requested symbols/period")
    raw["close_raw"] = raw["close"]
    feat = make_features(raw)

    # 3) Load artifacts
    model = tf.keras.models.load_model(args.model_path)
    scalerd = joblib.load(args.scaler_path)
    scaler, feats_order = scalerd["scaler"], scalerd["feats"]

    # 4) Scale features (same order as training)
    feat_scaled = feat.copy()
    feat_scaled[feats_order] = scaler.transform(feat_scaled[feats_order].values)

    # 5) Optional calibration
    calib = read_calibration(gc, args.sheet_id, args.calibration_worksheet) if args.apply_calibration else {}

    # 6) Predict per symbol
    out_rows: List[List] = []
    now_iso = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for sym, g_scaled in feat_scaled.groupby("symbol"):
        # align raw closes with the scaled feature rows for last window inference
        g_raw = feat[feat["symbol"] == sym].copy()
        g_raw["close_raw"] = raw[raw["symbol"] == sym]["close"].values[-len(g_raw):]

        X, last_close = last_window_tensor(
            pd.concat([g_scaled.reset_index(drop=True), g_raw[["close_raw"]].reset_index(drop=True)], axis=1),
            feats_order,
            WINDOW,
        )
        if X is None:
            continue

        y_scaled = model.predict(X[None, ...], verbose=0)[0]
        preds = inverse_close_only(np.array(y_scaled), scaler, feats_order)

        bias_pct = float(calib.get(sym, 0.0))
        adj_factor = (1.0 + bias_pct / 100.0)

        for h, p in zip(horizons, preds[: len(horizons)]):
            p_adj = float(p) * adj_factor
            pct = float((p_adj - last_close) / last_close * 100.0)
            signal = 1 if p_adj > last_close else (-1 if p_adj < last_close else 0)
            params = {"horizons": horizons}
            if args.apply_calibration:
                params["bias_pct"] = bias_pct
            out_rows.append(
                [
                    now_iso,
                    sym,
                    h,
                    round(last_close, 6),
                    round(p_adj, 6),
                    round(pct, 4),
                    signal,
                    args.scope,
                    "CNN-LSTM",
                    json.dumps(params),
                    WINDOW,
                    ",".join(feats_order),
                    "",
                ]
            )

    if not out_rows:
        print("No predictions produced.")
        return

    # 7) Write to sheet
    sheet_append_rows(gc, args.sheet_id, args.predictions_tab, out_rows)
    print(f"Wrote {len(out_rows)} predictions to '{args.predictions_tab}'")


if __name__ == "__main__":
    main()
