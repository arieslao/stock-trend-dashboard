# Models folder

Put your pretrained files here:

- `cnn_lstm_ALL.keras`  → Keras model (CNN-LSTM).
  - Input shape must be `(lookback, n_features)`.
  - `n_features` should be either:
    - 5 for OHLCV (o, h, l, close, v), or
    - 1 for Close-only.

- `scaler_ALL.joblib`   → `sklearn` scaler used at train time.
  - If your model is 5-feature, the scaler must have been fit on OHLCV columns in that order: `[o,h,l,close,v]`.
  - Use `.joblib` (preferred). `.pkl` also supported.

Optionally, you may provide symbol-specific files:
- `AAPL_cnn_lstm.keras`, `scaler_AAPL.joblib`, etc.

This app automatically prefers symbol-specific files; otherwise it falls back to `*_ALL.*`.
