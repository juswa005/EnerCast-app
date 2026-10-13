"""Hybrid ARIMA + Random Forest power forecasting for EnerCast.

ARIMA models the linear trend of the power series; a Random Forest then learns
the nonlinear pattern left in ARIMA's residuals. The final forecast is
ARIMA forecast + RF-predicted residual.
"""
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

N_LAGS = 6               # past values the RF looks at
MAX_GAP_MINUTES = 10     # longer outages are not interpolated across
ARIMA_ORDERS = [(p, 1, q) for p in range(3) for q in range(3)]
MIN_POINTS = 60
TIME_FEATURE_MIN_DAYS = 2  # hour-of-day only means something once days repeat


def prepare_series(df, max_gap=MAX_GAP_MINUTES):
    """Resample power to 1-minute means and return the latest continuous stretch.

    Short gaps (dropped uploads) are interpolated; long gaps (device off) are not,
    because drawing a straight line across hours of missing data teaches the
    models a pattern that never happened.
    """
    ts = df.set_index('timestamp')['power'].sort_index().resample('min').mean()
    ts = ts.interpolate(limit=max_gap, limit_area='inside')
    missing = ts.isna()
    if missing.any():
        ts = ts[missing[::-1].cumsum()[::-1] == 0]  # keep only what follows the last gap
    return ts.dropna()


def _features(values, index, use_time):
    """Feature table for predicting values[t] from values before t."""
    s = pd.Series(values, index=index)
    feats = pd.DataFrame(index=index)
    for i in range(1, N_LAGS + 1):
        feats[f'lag_{i}'] = s.shift(i)
    window = s.shift(1).rolling(N_LAGS)
    feats['roll_mean'] = window.mean()
    feats['roll_std'] = window.std()
    if use_time:
        hours = index.hour + index.minute / 60.0
        feats['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    return feats


def _use_time(index):
    return (index[-1] - index[0]) >= timedelta(days=TIME_FEATURE_MIN_DAYS)


def _new_rf():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1
    )


def _fit_rf(values, index):
    rf = _new_rf()
    rf.use_time_ = _use_time(index)
    feats = _features(values, index, rf.use_time_)
    mask = feats.notna().all(axis=1)
    rf.fit(feats[mask], np.asarray(values)[mask.values])
    return rf


def _rf_recursive(rf, history, index, steps):
    """Predict `steps` values ahead, feeding each prediction back in as a lag."""
    hist = list(history[-(N_LAGS + 1):])
    idx = list(index[-(N_LAGS + 1):])
    preds = []
    for _ in range(steps):
        idx.append(idx[-1] + timedelta(minutes=1))
        hist.append(np.nan)  # placeholder for the value being predicted
        row = _features(hist, pd.DatetimeIndex(idx), rf.use_time_).iloc[[-1]]
        pred = float(rf.predict(row)[0])
        hist[-1] = pred
        preds.append(pred)
        hist, idx = hist[1:], idx[1:]
    return np.array(preds)


def _fit_arima(ts):
    """Pick the ARIMA order with the lowest AIC from a small grid."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for order in ARIMA_ORDERS:
            try:
                res = ARIMA(ts, order=order).fit()
            except Exception:
                continue
            if best is None or res.aic < best.aic:
                best = res
    if best is None:
        raise ValueError('ARIMA could not be fitted to this data.')
    return best


def _fit_hybrid(ts):
    arima = _fit_arima(ts)
    resid = ts - arima.fittedvalues
    # The first fitted value of a differenced model is 0, so its residual is the
    # whole reading; drop the start so the RF doesn't learn from that artifact.
    burn = arima.model.k_diff + max(arima.model.k_ar, 1)
    resid = resid.iloc[burn:]
    rf = _fit_rf(resid.values, resid.index)
    return arima, rf, resid


MODELS = ['Hybrid ARIMA-RF', 'ARIMA', 'Random Forest']


def forecast(ts, model='Hybrid ARIMA-RF', steps=30, alpha=0.2):
    """Forecast the next `steps` minutes with the chosen model.

    Returns (DataFrame with forecast/lower/upper columns, description string).
    The Random Forest alone has no statistical interval, so lower/upper are NaN.
    """
    if len(ts) < MIN_POINTS:
        raise ValueError(f'Need at least {MIN_POINTS} minutes of continuous data.')
    index = pd.date_range(ts.index[-1] + timedelta(minutes=1), periods=steps, freq='min')

    if model == 'Random Forest':
        rf = _fit_rf(ts.values, ts.index)
        pred = np.clip(_rf_recursive(rf, ts.values, ts.index, steps), 0, None)
        out = pd.DataFrame({'forecast': pred, 'lower': np.nan, 'upper': np.nan}, index=index)
        return out, 'Random Forest'

    if model == 'ARIMA':
        arima = _fit_arima(ts)
        correction = np.zeros(steps)
    else:
        arima, rf, resid = _fit_hybrid(ts)
        correction = _rf_recursive(rf, resid.values, resid.index, steps)

    fc = arima.get_forecast(steps=steps)
    ci = fc.conf_int(alpha=alpha)
    ci = ci.values if hasattr(ci, 'values') else np.asarray(ci)
    out = pd.DataFrame({
        'forecast': np.clip(np.asarray(fc.predicted_mean) + correction, 0, None),
        'lower': np.clip(ci[:, 0] + correction, 0, None),
        'upper': np.clip(ci[:, 1] + correction, 0, None),
    }, index=index)
    return out, f'{model}, ARIMA order {arima.model.order}'


def _scores(actual, pred):
    return {
        'RMSE (W)': float(np.sqrt(mean_squared_error(actual, pred))),
        'MAE (W)': float(mean_absolute_error(actual, pred)),
        'R²': float(r2_score(actual, pred)),
    }


def backtest(ts, steps=30):
    """Hold out the last `steps` minutes, forecast them, and score each model.

    Returns a DataFrame with one row per model (ARIMA, Random Forest, Hybrid).
    """
    if len(ts) < MIN_POINTS + steps:
        raise ValueError('Not enough data to hold out a test period.')
    train, test = ts.iloc[:-steps], ts.iloc[-steps:]

    arima, rf_resid_model, resid = _fit_hybrid(train)
    arima_pred = arima.forecast(steps=steps).values
    hybrid_pred = arima_pred + _rf_recursive(rf_resid_model, resid.values, resid.index, steps)

    rf_only = _fit_rf(train.values, train.index)
    rf_pred = _rf_recursive(rf_only, train.values, train.index, steps)

    rows = {
        'Hybrid ARIMA-RF': _scores(test.values, np.clip(hybrid_pred, 0, None)),
        'ARIMA': _scores(test.values, np.clip(arima_pred, 0, None)),
        'Random Forest': _scores(test.values, np.clip(rf_pred, 0, None)),
    }
    return pd.DataFrame(rows).T
