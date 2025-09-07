# train.py — train a multi-horizon CNN-LSTM that predicts future CLOSE at H horizons
import os, argparse, json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import gspread

WINDOW = 60
FEATS_ORDER = ["close", "ret1", "ma10", "ma50", "vlog"]  # must match predict

def parse_horizons(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]

def load_prices_from_sheet(gc, sheet_id: str, prices_worksheet: str,
                           symbols: Optional[List[str]], period: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_worksheet)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        raise RuntimeError(f"'{prices_worksheet}' is empty or missing headers")

    df = pd.DataFrame(vals[1:], columns=[h.strip() for h in vals[0]])
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize common variants
    rename = {}
    if "adj close" in df.columns and "close" not in df.columns:
        rename["adj close"] = "close"
    if "adj_close" in df.columns and "close" not in df.columns:
        rename["adj_close"] = "close"
    if "volume" in df.columns and "vol" not in df.columns:
        rename["volume"] = "vol"
    if rename:
        df = df.rename(columns=rename)

    need = {"date", "symbol", "close"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"'{prices_worksheet}' missing required columns: {miss}. Found: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    # optional time filter (like "10y", "3y", "6m", "90d")
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

    cols = ["date", "symbol", "close"] + (["vol"] if "vol" in df.columns else [])
    return df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)

def make_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    df = df_prices.copy()
    if "vol" not in df.columns:
        df["vol"] = 0.0
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def build_windows_multihorizon(
    feat_scaled: pd.DataFrame,
    raw_feat: pd.DataFrame,
    horizons: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      - feat_scaled: features already scaled on FEATS_ORDER
      - raw_feat: same rows as feat_scaled (unscaled), used to read future CLOSE for targets
    Returns:
      X: (N, WINDOW, n_feats), y: (N, H) where H=len(horizons) and each y is CLOSE at +h days (scaled using close-scaler)
    """
    H = len(horizons)
    X_list, y_list = [], []
    max_h = max(horizons)
    # we assume feat_scaled and raw_feat are aligned (same index) per symbol
    for sym, g_sc in feat_scaled.groupby("symbol", sort=False):
        g_raw = raw_feat[raw_feat["symbol"] == sym]
        g_sc = g_sc.reset_index(drop=True)
        g_raw = g_raw.reset_index(drop=True)

        if len(g_sc) < WINDOW + max_h:
            continue

        # for each end index t (inclusive) that allows all horizons
        for t in range(WINDOW - 1, len(g_sc) - max_h):
            X = g_sc.loc[t - WINDOW + 1 : t, FEATS_ORDER].values.astype(np.float32)
            fut_idx = [t + h for h in horizons]
            closes_future = g_raw.loc[fut_idx, "close"].values.astype(np.float32)
            X_list.append(X)
            y_list.append(closes_future)

    if not X_list:
        return np.zeros((0, WINDOW, len(FEATS_ORDER)), dtype=np.float32), np.zeros((0, len(horizons)), dtype=np.float32)

    X = np.stack(X_list, axis=0)
    y_close = np.stack(y_list, axis=0)  # real CLOSE, not scaled (yet) — we’ll scale with the close scaler below
    return X, y_close

def scale_targets_close_only(y_close: np.ndarray, scaler: MinMaxScaler, feats_order: List[str]) -> np.ndarray:
    """Scale CLOSE targets with the scaler's 'close' params."""
    close_idx = feats_order.index("close")
    dmin = scaler.data_min_[close_idx]
    drng = scaler.data_range_[close_idx]
    return (y_close - dmin) / drng

def make_model(n_feats: int, H: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(WINDOW, n_feats))
    x = tf.keras.layers.Conv1D(32, 3, activation="relu")(inp)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(H, activation="linear")(x)  # outputs scaled CLOSE at H horizons
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def main():
    ap = argparse.ArgumentParser(description="Train stock models from Google Sheets watchlists.")
    ap.add_argument("--model", default="cnn-lstm", help="cnn-lstm, linear")
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--worksheet", required=True)
    ap.add_argument("--symbol-column", default="Ticker")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--interval", default="1d")  # kept for compatibility; unused when reading prices sheet
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--use-prices-tab", action="store_true", help="Read from 'prices' worksheet instead of downloading")
    ap.add_argument("--prices-worksheet", default="prices", help="Name of the prices worksheet")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--model-path", default="models/cnn_lstm_ALL.keras")
    ap.add_argument("--scaler-path", default="models/scaler_ALL.pkl")
    args = ap.parse_args()

    horizons = parse_horizons(args.horizons)
    H = len(horizons)
    assert H >= 1, "At least one horizon required"

    print(
        f"Args: model={args.model}, sheet-id={args.sheet_id}, worksheet={args.worksheet}, "
        f"symbol-column={args.symbol_column}, period={args.period}, interval={args.interval}, "
        f"epochs={args.epochs}, batch={args.batch_size}, patience={args.patience}, "
        f"use_prices_tab={args.use_prices_tab}, prices_ws={args.prices_worksheet}, horizons={horizons}"
    )

    # ---- Watchlist ----
    print("Authenticating with Google Sheets...")
    gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    sh = gc.open_by_key(args.sheet_id)
    wl = sh.worksheet(args.worksheet)
    hdr = [h.strip() for h in wl.row_values(1)]
    try:
        cidx = [h.lower() for h in hdr].index(args.symbol_column.lower()) + 1
    except ValueError:
        raise SystemExit(f"Column '{args.symbol_column}' not found in watchlist '{args.worksheet}'")
    symbols = [s.strip().upper() for s in wl.col_values(cidx)[1:] if s.strip()]
    print(f"Using tickers: {symbols}")

    # ---- Prices & features ----
    if not args.use_prices_tab:
        raise SystemExit("This training script expects --use-prices-tab; prices download not wired here.")
    print(f"Loading data from prices tab '{args.prices_worksheet}'...")
    raw = load_prices_from_sheet(gc, args.sheet_id, args.prices_worksheet, symbols, args.period)
    if raw.empty:
        raise SystemExit("No rows in 'prices' for requested symbols/period")
    feat_raw = make_features(raw)

    # fit scaler on FEATS_ORDER columns over entire dataset
    scaler = MinMaxScaler()
    feat_scaled = feat_raw.copy()
    feat_scaled[FEATS_ORDER] = scaler.fit_transform(feat_raw[FEATS_ORDER].values)

    # build multi-horizon windows
    X, y_close = build_windows_multihorizon(feat_scaled, feat_raw, horizons)
    y = scale_targets_close_only(y_close, scaler, FEATS_ORDER)

    print(f"Created training dataset with shape X: {X.shape}, y: {y.shape}")

    if X.shape[0] == 0:
        raise SystemExit("No training samples (not enough history per symbol for given WINDOW/horizons).")

    # ---- Model ----
    if args.model == "cnn-lstm":
        model = make_model(n_feats=len(FEATS_ORDER), H=y.shape[1])
    elif args.model == "linear":
        # baseline linear head
        inp = tf.keras.layers.Input(shape=(WINDOW, len(FEATS_ORDER)))
        x = tf.keras.layers.Flatten()(inp)
        out = tf.keras.layers.Dense(y.shape[1], activation="linear")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    else:
        raise SystemExit(f"Unknown --model '{args.model}'")

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
        callbacks=cbs,
    )

    # ---- Save artifacts ----
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)

    os.makedirs(os.path.dirname(args.scaler_path), exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "feats": FEATS_ORDER, "horizons": horizons},
        args.scaler_path
    )

    print(f"✅ Saved {args.model_path} and {args.scaler_path}")

if __name__ == "__main__":
    main()
