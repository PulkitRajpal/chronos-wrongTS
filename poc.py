from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from chronos import Chronos2Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

AAPL_FILE = DATA_DIR / 'apple_historical.csv'
SPX_FILE = DATA_DIR / 'S&P 500 Historical Data.csv'
VIX_FILE = DATA_DIR / 'VIXCLS.csv'

CHRONOS_MODEL = 'amazon/chronos-2'


def resolve_device():
    if not torch.cuda.is_available():
        print('CUDA is not available; falling back to CPU.')
        return 'cpu'

    try:
        torch.randn(1, device='cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
        return 'cuda'
    except Exception:
        print(
            'CUDA is present but not usable on this machine (kernel/image mismatch or incompatible driver). '
            'Falling back to CPU.'
        )
        return 'cpu'


DEVICE = resolve_device()
HORIZONS = [1, 5, 10, 20]
MIN_HISTORY = 500
N_WINDOWS = 25
ML_LOOKBACK = 252
CHRONOS_CONTEXT = 512
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]
SPIKE_SIGMA = 2.0


def clean_number(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s in {"", "-", "—", "N/A", "NA", "null", "None"}:
        return np.nan

    s = (
        s.replace("$", "")
         .replace(",", "")
         .replace("%", "")
         .strip()
    )

    # Handle accounting-style negative values such as (123.45).
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        return float(s)
    except ValueError:
        return np.nan


def find_column(df, candidates, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise ValueError(f'Could not find one of {candidates}. Columns: {list(df.columns)}')
    return None


def load_aapl(path):
    raw = pd.read_csv(path)
    d = find_column(raw, ['Date'])
    c = find_column(raw, ['Close/Last', 'Close'])
    v = find_column(raw, ['Volume'])
    o = find_column(raw, ['Open'])
    h = find_column(raw, ['High'])
    l = find_column(raw, ['Low'])
    df = pd.DataFrame({
        'Date': pd.to_datetime(raw[d], errors='coerce'),
        'AAPL_Close': raw[c].map(clean_number),
        'AAPL_Open': raw[o].map(clean_number),
        'AAPL_High': raw[h].map(clean_number),
        'AAPL_Low': raw[l].map(clean_number),
        'AAPL_Volume': raw[v].map(clean_number),
    })
    return df.dropna(subset=['Date', 'AAPL_Close']).sort_values('Date').drop_duplicates('Date').reset_index(drop=True)


def load_price_series(path, name):
    raw = pd.read_csv(path)
    d = find_column(raw, ['Date', 'DATE', 'observation_date', 'Time'])
    c = find_column(raw, ['Close/Last', 'Close', 'Last', 'Price', 'VIXCLS', name])
    df = pd.DataFrame({
        'Date': pd.to_datetime(raw[d], errors='coerce'),
        f'{name}_Close': raw[c].map(clean_number),
    })
    return df.dropna(subset=['Date', f'{name}_Close']).sort_values('Date').drop_duplicates('Date').reset_index(drop=True)


def load_data():
    aapl = load_aapl(AAPL_FILE)
    spx = load_price_series(SPX_FILE, 'SPX')
    vix = load_price_series(VIX_FILE, 'VIX')
    aligned = aapl.merge(spx, on='Date', how='inner').merge(vix, on='Date', how='inner')
    aligned = aligned.sort_values('Date').reset_index(drop=True)
    print('\nDATA CHECK')
    print('=' * 80)
    print(f'AAPL rows: {len(aapl):,}')
    print(f'Aligned AAPL/SPX/VIX rows: {len(aligned):,}')
    print(f'AAPL range: {aapl.Date.min().date()} -> {aapl.Date.max().date()}')
    print(f'Aligned range: {aligned.Date.min().date()} -> {aligned.Date.max().date()}')
    print('Aligned columns:', list(aligned.columns))

    print("\nAAPL dtypes:")
    print(aapl.dtypes.to_string())

    print("\nMissing AAPL values:")
    print(
        aapl[
            [
                "AAPL_Close",
                "AAPL_Open",
                "AAPL_High",
                "AAPL_Low",
                "AAPL_Volume",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    return aapl, aligned


def add_regular_timestamp(df):
    out = df.copy()
    # One synthetic timestamp per observed trading session. This avoids
    # Chronos frequency inference failing on weekends/market holidays.
    out['ChronosTime'] = pd.date_range('2000-01-01', periods=len(out), freq='D')
    return out


def chronos_context(history, covariates=None):
    covariates = covariates or []
    h = add_regular_timestamp(history.tail(CHRONOS_CONTEXT).copy())
    out = pd.DataFrame({
        'id': ['AAPL'] * len(h),
        'timestamp': h['ChronosTime'].values,
        'target': h['AAPL_Close'].astype(float).values,
    })
    for col in covariates:
        out[col] = h[col].astype(float).values
    return out


def load_chronos():
    print('\nLoading Chronos-2...')
    pipe = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=DEVICE)
    print(f'Chronos-2 loaded on {DEVICE.upper()}.')
    return pipe


def chronos_predict(pipe, history, horizon, covariates=None):
    ctx = chronos_context(history, covariates)
    pred = pipe.predict_df(
        ctx,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column='id',
        timestamp_column='timestamp',
        target='target',
    )
    return pred


# -----------------------------
# XGBoost features
# -----------------------------

def build_features(history, extra_series=None):
    extra_series = extra_series or []
    out = history.copy()
    close = out['AAPL_Close']
    out['ret_1'] = close.pct_change(1)
    out['ret_2'] = close.pct_change(2)
    out['ret_3'] = close.pct_change(3)
    out['ret_5'] = close.pct_change(5)
    out['ret_10'] = close.pct_change(10)
    out['ret_20'] = close.pct_change(20)
    out['hl_range'] = (out['AAPL_High'] - out['AAPL_Low']) / close
    out['oc_return'] = (out['AAPL_Close'] - out['AAPL_Open']) / out['AAPL_Open']
    out['volume_return'] = out['AAPL_Volume'].pct_change()
    out['vol_5'] = out['ret_1'].rolling(5).std()
    out['vol_20'] = out['ret_1'].rolling(20).std()
    out['vol_60'] = out['ret_1'].rolling(60).std()
    out['mom_5'] = close / close.shift(5) - 1
    out['mom_20'] = close / close.shift(20) - 1
    out['mom_60'] = close / close.shift(60) - 1
    for col in extra_series:
        p = col.replace('_Close', '').lower()
        out[f'{p}_ret_1'] = out[col].pct_change(1)
        out[f'{p}_ret_5'] = out[col].pct_change(5)
        out[f'{p}_ret_20'] = out[col].pct_change(20)
    return out


def feature_columns(extra_series=None):
    extra_series = extra_series or []
    cols = [
        'ret_1','ret_2','ret_3','ret_5','ret_10','ret_20',
        'hl_range','oc_return','volume_return',
        'vol_5','vol_20','vol_60',
        'mom_5','mom_20','mom_60',
    ]
    for col in extra_series:
        p = col.replace('_Close', '').lower()
        cols += [f'{p}_ret_1', f'{p}_ret_5', f'{p}_ret_20']
    return cols


def train_xgb_direct(history, horizon, extra_series=None):
    extra_series = extra_series or []
    data = build_features(history, extra_series)
    feats = feature_columns(extra_series)
    data['target_return'] = data['AAPL_Close'].shift(-horizon) / data['AAPL_Close'] - 1
    data = data.dropna(subset=feats + ['target_return']).tail(ML_LOOKBACK)
    if len(data) < 100:
        raise ValueError(f'Not enough XGBoost training rows: {len(data)}')
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )
    X = (
        data[feats]
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
    )
    y = pd.to_numeric(
        data["target_return"],
        errors="coerce",
    ).astype(np.float64)

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 100:
        raise ValueError(
            f"Not enough clean XGBoost training rows: {len(X)}"
        )

    model.fit(X, y)
    return model


def xgb_endpoint(history, horizon, model, extra_series=None):
    extra_series = extra_series or []
    data = build_features(history, extra_series)
    feats = feature_columns(extra_series)
    latest = data.iloc[-1]

    # Pandas can return an object-dtype ndarray when the Series contains
    # mixed dtypes. XGBoost/numpy.isfinite require a numeric array here.
    x = (
        pd.to_numeric(
            latest[feats],
            errors="coerce",
        )
        .to_numpy(dtype=np.float64)
        .reshape(1, -1)
    )

    if not np.isfinite(x).all():
        bad = {
            feat: latest[feat]
            for feat in feats
            if not np.isfinite(
                pd.to_numeric(
                    latest[feat],
                    errors="coerce",
                )
            )
        }
        raise ValueError(
            "Latest XGBoost feature row contains NaN/inf/non-numeric values: "
            f"{bad}"
        )

    ret = float(model.predict(x)[0])
    return float(history['AAPL_Close'].iloc[-1]) * (1 + ret)


def cutoffs(n_rows):
    need = MIN_HISTORY + max(HORIZONS)
    if n_rows <= need:
        raise ValueError(f'Not enough data. Need > {need} rows, got {n_rows}.')
    possible = list(range(MIN_HISTORY, n_rows - max(HORIZONS)))
    if len(possible) <= N_WINDOWS:
        return possible
    idx = np.linspace(0, len(possible)-1, N_WINDOWS, dtype=int)
    return [possible[i] for i in idx]


def endpoint_return(last_price, price):
    return price / last_price - 1


def hit_direction(last_price, actual, predicted):
    return float(np.sign(endpoint_return(last_price, actual)) == np.sign(endpoint_return(last_price, predicted)))


# -----------------------------
# Experiment runner
# -----------------------------

def run_forecast_experiment(df, pipe, covariates, filename, title):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)
    rows = []
    cs = cutoffs(len(df))
    for i, cutoff in enumerate(cs, 1):
        history = df.iloc[:cutoff].copy()
        last_price = float(history.AAPL_Close.iloc[-1])
        print(f'  window {i}/{len(cs)} (cutoff={history.Date.iloc[-1].date()})')
        for horizon in HORIZONS:
            future = df.iloc[cutoff:cutoff+horizon]
            actual = float(future.AAPL_Close.iloc[-1])

            xgb = train_xgb_direct(history, horizon, extra_series=[] if covariates else [])
            ml = xgb_endpoint(history, horizon, xgb, extra_series=[])

            cp = chronos_predict(pipe, history, horizon, covariates=covariates)
            p10 = float(cp['0.1'].iloc[-1])
            p50 = float(cp['0.5'].iloc[-1])
            p90 = float(cp['0.9'].iloc[-1])

            rows.append({
                'cutoff_date': history.Date.iloc[-1],
                'horizon': horizon,
                'last_price': last_price,
                'actual_final': actual,
                'ml_pred': ml,
                'chronos_p10': p10,
                'chronos_p50': p50,
                'chronos_p90': p90,
                'ml_mae': abs(ml - actual),
                'chronos_mae': abs(p50 - actual),
                'ml_direction': hit_direction(last_price, actual, ml),
                'chronos_direction': hit_direction(last_price, actual, p50),
                'actual_return': endpoint_return(last_price, actual),
                'chronos_p10_return': endpoint_return(last_price, p10),
                'chronos_p50_return': endpoint_return(last_price, p50),
                'chronos_p90_return': endpoint_return(last_price, p90),
            })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / filename, index=False)
    return result


def summarize(result, prefix):
    s = result.groupby('horizon').agg(
        ml_mae=('ml_mae','mean'),
        chronos_mae=('chronos_mae','mean'),
        ml_direction=('ml_direction','mean'),
        chronos_direction=('chronos_direction','mean'),
    ).reset_index()
    s['chronos_mae_improvement_pct'] = (s.ml_mae - s.chronos_mae) / s.ml_mae * 100
    s['chronos_direction_advantage_pp'] = (s.chronos_direction - s.ml_direction) * 100
    s.to_csv(RESULTS_DIR / f'{prefix}_summary.csv', index=False)
    print('\n' + s.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    return s


def coverage(result, prefix):
    x = result.copy()
    x['p10_breach'] = x.actual_return < x.chronos_p10_return
    x['p90_breach'] = x.actual_return > x.chronos_p90_return
    c = x.groupby('horizon').agg(
        p10_breach_rate=('p10_breach','mean'),
        p90_breach_rate=('p90_breach','mean'),
    ).reset_index()
    c.to_csv(RESULTS_DIR / f'{prefix}_coverage.csv', index=False)
    print('\nChronos coverage (ideal P10/P90 breach rate is about 10%):')
    print(c.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return c


# -----------------------------
# Spike experiment
# -----------------------------

def run_spike_experiment(aapl, pipe):
    print('\n' + '=' * 80)
    print('PRICE SPIKE EXPERIMENT')
    print('=' * 80)
    d = aapl.copy()
    d['ret'] = d.AAPL_Close.pct_change()
    d['vol20'] = d.ret.rolling(20).std()
    rows = []
    for cutoff in cutoffs(len(d)):
        hist = d.iloc[:cutoff].copy()
        future = d.iloc[cutoff]
        last = float(hist.AAPL_Close.iloc[-1])
        actual = float(future.AAPL_Close)
        r = actual / last - 1
        v = float(hist.vol20.iloc[-1])
        if not np.isfinite(v) or v <= 0:
            continue
        threshold = SPIKE_SIGMA * v
        actual_spike = abs(r) > threshold

        xgb = train_xgb_direct(hist, 1, [])
        ml_price = xgb_endpoint(hist, 1, xgb, [])
        ml_ret = ml_price / last - 1
        ml_warning = abs(ml_ret) > threshold

        pred = chronos_predict(pipe, hist, 1, [])
        p10 = float(pred['0.1'].iloc[-1]) / last - 1
        p50 = float(pred['0.5'].iloc[-1]) / last - 1
        p90 = float(pred['0.9'].iloc[-1]) / last - 1
        chronos_warning = p10 < -threshold or p90 > threshold

        rows.append({
            'date': future.Date,
            'actual_return': r,
            'rolling_vol': v,
            'threshold': threshold,
            'actual_spike': actual_spike,
            'ml_return': ml_ret,
            'ml_spike_warning': ml_warning,
            'chronos_p10_return': p10,
            'chronos_p50_return': p50,
            'chronos_p90_return': p90,
            'chronos_spike_warning': chronos_warning,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'spike_results.csv', index=False)
    return out


def classification_metrics(actual, predicted):
    a = actual.astype(bool).to_numpy()
    p = predicted.astype(bool).to_numpy()
    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return precision, recall, tp, fp, fn


def summarize_spikes(spikes):
    m1 = classification_metrics(spikes.actual_spike, spikes.ml_spike_warning)
    m2 = classification_metrics(spikes.actual_spike, spikes.chronos_spike_warning)
    s = pd.DataFrame([
        {'model':'XGBoost','precision':m1[0],'recall':m1[1],'tp':m1[2],'fp':m1[3],'fn':m1[4]},
        {'model':'Chronos-2','precision':m2[0],'recall':m2[1],'tp':m2[2],'fp':m2[3],'fn':m2[4]},
    ])
    s.to_csv(RESULTS_DIR / 'spike_summary.csv', index=False)
    print('\n' + s.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return s


# -----------------------------
# Plot
# -----------------------------

def make_plot(aapl, pipe):
    horizon = 20
    cutoff = len(aapl) - horizon
    history = aapl.iloc[:cutoff].copy()
    future = aapl.iloc[cutoff:].copy()

    chronos = chronos_predict(pipe, history, horizon, [])
    p10 = chronos['0.1'].values
    p50 = chronos['0.5'].values
    p90 = chronos['0.9'].values

    xgb_predictions = []
    for h in range(1, horizon + 1):
        model = train_xgb_direct(history, h, [])
        xgb_predictions.append(xgb_endpoint(history, h, model, []))

    plt.figure(figsize=(14,7))
    plt.plot(history.Date.tail(100), history.AAPL_Close.tail(100), label='Known history')
    plt.plot(future.Date, future.AAPL_Close, label='Actual future', linewidth=3)
    plt.plot(future.Date, xgb_predictions, '--', label='XGBoost')
    plt.plot(future.Date, p50, '--', label='Chronos P50')
    plt.fill_between(future.Date, p10, p90, alpha=0.2, label='Chronos P10-P90')
    plt.axvline(history.Date.iloc[-1], linestyle=':', label='Forecast start')
    plt.title('AAPL — Chronos-2 vs Statistical ML')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'forecast_example.png', dpi=150)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('CHRONOS-2 vs STATISTICAL ML — PROXY POC')
    print('=' * 80)

    aapl, aligned = load_data()
    pipe = load_chronos()

    # Experiment 1: Close only.
    e1 = run_forecast_experiment(
        aapl,
        pipe,
        covariates=[],
        filename='experiment_1_univariate.csv',
        title='EXPERIMENT 1 — UNIVARIATE (AAPL CLOSE)',
    )
    s1 = summarize(e1, 'experiment_1')
    coverage(e1, 'experiment_1')

    # Experiment 2: OHLCV as historical/past covariates for Chronos.
    e2 = run_forecast_experiment(
        aapl,
        pipe,
        covariates=['AAPL_Open','AAPL_High','AAPL_Low','AAPL_Volume'],
        filename='experiment_2_ohlcv.csv',
        title='EXPERIMENT 2 — OHLCV',
    )
    s2 = summarize(e2, 'experiment_2')
    coverage(e2, 'experiment_2')

    # Experiment 3: AAPL target, SPX + VIX as past covariates.
    e3 = run_forecast_experiment(
        aligned,
        pipe,
        covariates=['SPX_Close','VIX_Close'],
        filename='experiment_3_cross_market.csv',
        title='EXPERIMENT 3 — CROSS-MARKET (SPX + VIX)',
    )
    s3 = summarize(e3, 'experiment_3')
    coverage(e3, 'experiment_3')

    spikes = run_spike_experiment(aapl, pipe)
    summarize_spikes(spikes)

    make_plot(aapl, pipe)

    # Compact final table: 5-session horizon.
    rows = []
    for name, summary in [('univariate', s1), ('ohlcv', s2), ('cross_market', s3)]:
        r = summary[summary.horizon == 5]
        if not r.empty:
            r = r.iloc[0]
            rows.append({
                'experiment': name,
                '5d_ml_mae': r.ml_mae,
                '5d_chronos_mae': r.chronos_mae,
                '5d_chronos_improvement_pct': r.chronos_mae_improvement_pct,
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / 'final_summary.csv', index=False)

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    print(f'Results saved to: {RESULTS_DIR.resolve()}')


if __name__ == '__main__':
    main()
