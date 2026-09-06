
"""
CHRONOS-2 vs STATISTICAL MODELS — FINANCIAL FORECASTING BENCHMARK
=================================================================

Models
------
1. Naive Random Walk
2. ARIMA
3. ETS / Exponential Smoothing
4. GARCH(1,1)
5. EGARCH(1,1)          [optional, if arch is installed]
6. GJR-GARCH(1,1)       [optional, if arch is installed]
7. VAR                  [cross-market only]
8. XGBoost              [modern ML baseline]
9. LightGBM             [optional, if installed]
10. Chronos-2           [TSFM]

Experiments
-----------
A. Univariate Close
B. OHLCV
C. Cross-market: stock + SPX + VIX
D. Spike / tail evaluation
E. Limited-history (cold-start) evaluation

Expected directory
------------------
project/
├── poc.py
├── data/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── NVDA.csv
│   ├── AMZN.csv
│   ├── ENPH.csv
│   ├── SMCI.csv
│   ├── CVNA.csv
│   ├── PLUG.csv
│   ├── RKLB.csv
│   ├── IONQ.csv
│   ├── SPX.csv        # optional for cross-market
│   └── VIX.csv        # optional for cross-market
└── results/

Stock CSV format
----------------
date,open,high,low,close,volume

Notes
-----
- All evaluation is walk-forward.
- Models only see data at or before the cutoff.
- Forecast horizons are trading observations, not calendar days.
- Chronos receives a synthetic regular timeline because market dates contain
  weekends/holidays.
- Metrics are calculated both in price space and normalized return space.
- The goal is model comparison, not trading-strategy evaluation.

Install
-------
pip install -U pandas numpy scipy scikit-learn statsmodels xgboost torch \
    "chronos-forecasting>=2.0" arch matplotlib

Optional:
pip install lightgbm
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR

from xgboost import XGBRegressor

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from chronos import Chronos2Pipeline


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "ENPH",
    "SMCI",
    "CVNA",
    "PLUG",
    "RKLB",
    "IONQ",
]

SPX_FILE = DATA_DIR / "SPX.csv"
VIX_FILE = DATA_DIR / "VIX.csv"

CHRONOS_MODEL = "amazon/chronos-2"

REQUESTED_DEVICE = "cuda"

HORIZONS = [1, 5, 10, 20]

# Use 50 initially. Set to None to use every possible cutoff.
N_WINDOWS = 50

# Minimum data required for a benchmark.
MIN_HISTORY = 300

# Training window for ML models.
ML_LOOKBACK = 252

# Chronos context.
CHRONOS_CONTEXT = 512

# Cold-start history lengths.
COLD_START_LENGTHS = [30, 60, 90, 180]

# Spike threshold.
SPIKE_SIGMA = 2.0

# Probabilistic levels.
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


# ============================================================
# DATA LOADING
# ============================================================

def clean_number(value) -> float:
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s in {"", "-", "—", "N/A", "NA", "null", "None"}:
        return np.nan

    s = (
        s.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        return float(s)
    except ValueError:
        return np.nan


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_stock(
    ticker: str,
) -> pd.DataFrame:

    path = DATA_DIR / f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        ["date", "datetime", "timestamp"],
    )

    close_col = find_column(
        raw,
        ["close", "close/last", "last", "price"],
    )

    open_col = find_column(raw, ["open"])
    high_col = find_column(raw, ["high"])
    low_col = find_column(raw, ["low"])
    volume_col = find_column(raw, ["volume"])

    if date_col is None or close_col is None:
        raise ValueError(
            f"{ticker}: date/close not found. "
            f"Columns={list(raw.columns)}"
        )

    data = pd.DataFrame({
        "Date": pd.to_datetime(
            raw[date_col],
            errors="coerce",
        ),
        f"{ticker}_Close": raw[
            close_col
        ].map(clean_number),
    })

    # Optional OHLCV.
    for output_name, source_col in [
        (f"{ticker}_Open", open_col),
        (f"{ticker}_High", high_col),
        (f"{ticker}_Low", low_col),
        (f"{ticker}_Volume", volume_col),
    ]:
        if source_col is not None:
            data[output_name] = raw[
                source_col
            ].map(clean_number)
        else:
            data[output_name] = np.nan

    data = (
        data
        .dropna(subset=[
            "Date",
            f"{ticker}_Close",
        ])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return data


def load_macro(
    path: Path,
    name: str,
) -> pd.DataFrame:

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        ["date", "datetime", "timestamp", "observation_date"],
    )

    value_col = find_column(
        raw,
        [
            "close",
            "close/last",
            "last",
            "price",
            "value",
            name,
        ],
    )

    if date_col is None or value_col is None:
        raise ValueError(
            f"{name}: date/value not found. "
            f"Columns={list(raw.columns)}"
        )

    out = pd.DataFrame({
        "Date": pd.to_datetime(
            raw[date_col],
            errors="coerce",
        ),
        f"{name}_Close": raw[
            value_col
        ].map(clean_number),
    })

    return (
        out
        .dropna(subset=[
            "Date",
            f"{name}_Close",
        ])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def align_cross_market(
    stock: pd.DataFrame,
) -> Optional[pd.DataFrame]:

    if not SPX_FILE.exists() or not VIX_FILE.exists():
        return None

    try:
        spx = load_macro(
            SPX_FILE,
            "SPX",
        )

        vix = load_macro(
            VIX_FILE,
            "VIX",
        )

        out = stock.merge(
            spx,
            on="Date",
            how="inner",
        )

        out = out.merge(
            vix,
            on="Date",
            how="inner",
        )

        return (
            out
            .sort_values("Date")
            .reset_index(drop=True)
        )

    except Exception as exc:
        print(
            f"    Cross-market alignment failed: {exc}"
        )
        return None


# ============================================================
# CHRONOS
# ============================================================

def load_chronos():

    print(
        f"\nLoading {CHRONOS_MODEL}..."
    )

    try:

        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL,
            device_map=REQUESTED_DEVICE,
        )

        print(
            f"Chronos-2 loaded on {REQUESTED_DEVICE}"
        )

        return pipeline

    except Exception as exc:

        if REQUESTED_DEVICE != "cuda":
            raise

        print(
            "CUDA unavailable; falling back to CPU."
        )

        print(
            f"CUDA error: {exc}"
        )

        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL,
            device_map="cpu",
        )

        print(
            "Chronos-2 loaded on CPU"
        )

        return pipeline


def chronos_context(
    history: pd.DataFrame,
    ticker: str,
    covariates: Optional[list[str]] = None,
) -> pd.DataFrame:

    if covariates is None:
        covariates = []

    history = history.tail(
        CHRONOS_CONTEXT
    ).copy()

    # Regular synthetic time index.
    synthetic_time = pd.date_range(
        start="2000-01-01",
        periods=len(history),
        freq="D",
    )

    context = pd.DataFrame({
        "id": [ticker] * len(history),
        "timestamp": synthetic_time,
        "target": history[
            f"{ticker}_Close"
        ].astype(float).values,
    })

    for col in covariates:
        context[col] = (
            history[col]
            .astype(float)
            .values
        )

    return context


def chronos_forecast(
    pipeline,
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    covariates: Optional[list[str]] = None,
) -> dict:

    context = chronos_context(
        history,
        ticker,
        covariates,
    )

    pred = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    return {
        "p05": float(pred["0.05"].iloc[-1]),
        "p10": float(pred["0.1"].iloc[-1]),
        "p50": float(pred["0.5"].iloc[-1]),
        "p90": float(pred["0.9"].iloc[-1]),
        "p95": float(pred["0.95"].iloc[-1]),
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    history: pd.DataFrame,
    ticker: str,
    external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    if external_columns is None:
        external_columns = []

    out = history.copy()

    close = out[f"{ticker}_Close"]

    out["ret_1"] = close.pct_change(1)
    out["ret_2"] = close.pct_change(2)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)

    out["hl_range"] = (
        out[f"{ticker}_High"]
        -
        out[f"{ticker}_Low"]
    ) / close

    out["oc_return"] = (
        out[f"{ticker}_Close"]
        -
        out[f"{ticker}_Open"]
    ) / out[f"{ticker}_Open"]

    out["volume_return"] = (
        out[f"{ticker}_Volume"].pct_change()
    )

    out["vol_5"] = (
        out["ret_1"].rolling(5).std()
    )

    out["vol_20"] = (
        out["ret_1"].rolling(20).std()
    )

    out["vol_60"] = (
        out["ret_1"].rolling(60).std()
    )

    out["mom_5"] = (
        close / close.shift(5) - 1
    )

    out["mom_20"] = (
        close / close.shift(20) - 1
    )

    out["mom_60"] = (
        close / close.shift(60) - 1
    )

    for col in external_columns:

        prefix = (
            col.replace("_Close", "")
            .lower()
        )

        out[f"{prefix}_ret_1"] = (
            out[col].pct_change(1)
        )

        out[f"{prefix}_ret_5"] = (
            out[col].pct_change(5)
        )

        out[f"{prefix}_ret_20"] = (
            out[col].pct_change(20)
        )

    return out


def feature_columns(
    external_columns: Optional[list[str]] = None,
) -> list[str]:

    if external_columns is None:
        external_columns = []

    features = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "hl_range",
        "oc_return",
        "volume_return",
        "vol_5",
        "vol_20",
        "vol_60",
        "mom_5",
        "mom_20",
        "mom_60",
    ]

    for col in external_columns:

        prefix = (
            col.replace("_Close", "")
            .lower()
        )

        features.extend([
            f"{prefix}_ret_1",
            f"{prefix}_ret_5",
            f"{prefix}_ret_20",
        ])

    return features


# ============================================================
# NAIVE MODEL
# ============================================================

def naive_forecast(
    history: pd.DataFrame,
    ticker: str,
) -> float:

    return float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )


# ============================================================
# ARIMA
# ============================================================

def arima_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> float:

    close = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
    )

    # ARIMA on log prices.
    log_price = np.log(close)

    model = ARIMA(
        log_price,
        order=(1, 1, 1),
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=horizon
    )

    return float(
        np.exp(forecast.iloc[-1])
    )


# ============================================================
# ETS
# ============================================================

def ets_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> float:

    close = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
    )

    # Multiplicative trend can fail on some series;
    # use additive trend for robustness.
    model = ExponentialSmoothing(
        close,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    )

    fitted = model.fit(
        optimized=True
    )

    forecast = fitted.forecast(
        horizon
    )

    return float(
        forecast.iloc[-1]
    )


# ============================================================
# GARCH FAMILY
# ============================================================

def garch_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    variant: str = "GARCH",
) -> tuple[float, float]:

    if not ARCH_AVAILABLE:
        raise RuntimeError(
            "arch package not installed"
        )

    close = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
    )

    returns = (
        np.log(close)
        .diff()
        .dropna()
        * 100.0
    )

    if len(returns) < 100:
        raise ValueError(
            "Not enough returns for GARCH."
        )

    if variant == "GARCH":
        model = arch_model(
            returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t",
            rescale=False,
        )

    elif variant == "EGARCH":
        model = arch_model(
            returns,
            mean="Constant",
            vol="EGARCH",
            p=1,
            o=1,
            q=1,
            dist="t",
            rescale=False,
        )

    elif variant == "GJR-GARCH":
        model = arch_model(
            returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="t",
            rescale=False,
        )

    else:
        raise ValueError(variant)

    fitted = model.fit(
        disp="off",
    )

    # Mean return forecast.
    mean_forecast = (
        fitted
        .forecast(
            horizon=horizon,
            reindex=False,
        )
        .mean
        .iloc[-1]
        .to_numpy()
    )

    cumulative_return = (
        np.sum(mean_forecast)
        / 100.0
    )

    last_price = float(
        close.iloc[-1]
    )

    price_forecast = (
        last_price
        * np.exp(cumulative_return)
    )

    # Average forecast volatility across horizon.
    variance = (
        fitted
        .forecast(
            horizon=horizon,
            reindex=False,
        )
        .variance
        .iloc[-1]
        .to_numpy()
    )

    avg_vol = float(
        np.sqrt(
            np.maximum(
                variance,
                0,
            )
        ).mean()
        / 100.0
    )

    return (
        float(price_forecast),
        avg_vol,
    )


# ============================================================
# XGBOOST
# ============================================================

def train_xgb(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
) -> XGBRegressor:

    if external_columns is None:
        external_columns = []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    feats = feature_columns(
        external_columns
    )

    data["target_return"] = (
        data[
            f"{ticker}_Close"
        ].shift(-horizon)
        /
        data[
            f"{ticker}_Close"
        ]
        - 1.0
    )

    data = data.dropna(
        subset=feats + ["target_return"]
    )

    data = data.tail(
        min(ML_LOOKBACK, len(data))
    )

    if len(data) < 100:
        raise ValueError(
            "Not enough data for XGBoost."
        )

    X = (
        data[feats]
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
    )

    y = (
        pd.to_numeric(
            data["target_return"],
            errors="coerce",
        )
        .astype(np.float64)
    )

    valid = (
        X.notna().all(axis=1)
        &
        y.notna()
        &
        np.isfinite(y)
    )

    X = X.loc[valid]
    y = y.loc[valid]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        y,
    )

    return model


def xgb_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    model: XGBRegressor,
    external_columns: Optional[list[str]] = None,
) -> float:

    if external_columns is None:
        external_columns = []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    feats = feature_columns(
        external_columns
    )

    latest = data.iloc[-1]

    x = (
        pd.to_numeric(
            latest[feats],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
        .reshape(1, -1)
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Latest XGBoost features invalid."
        )

    ret = float(
        model.predict(x)[0]
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return (
        last_price
        * (1.0 + ret)
    )


# ============================================================
# LIGHTGBM
# ============================================================

def train_lgbm(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
):

    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(
            "lightgbm package not installed"
        )

    if external_columns is None:
        external_columns = []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    feats = feature_columns(
        external_columns
    )

    data["target_return"] = (
        data[
            f"{ticker}_Close"
        ].shift(-horizon)
        /
        data[
            f"{ticker}_Close"
        ]
        - 1.0
    )

    data = data.dropna(
        subset=feats + ["target_return"]
    ).tail(ML_LOOKBACK)

    X = data[feats].astype(float)
    y = data["target_return"].astype(float)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X,
        y,
    )

    return model


def lgbm_forecast(
    history: pd.DataFrame,
    ticker: str,
    model,
    external_columns: Optional[list[str]] = None,
) -> float:

    if external_columns is None:
        external_columns = []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    feats = feature_columns(
        external_columns
    )

    x = (
        data.iloc[-1][feats]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    ret = float(
        model.predict(x)[0]
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return last_price * (
        1.0 + ret
    )


# ============================================================
# VAR
# ============================================================

def var_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> float:

    columns = [
        f"{ticker}_Close",
        "SPX_Close",
        "VIX_Close",
    ]

    if not all(
        col in history.columns
        for col in columns
    ):
        raise ValueError(
            "VAR requires stock + SPX + VIX."
        )

    returns = (
        np.log(
            history[columns]
        )
        .diff()
        .dropna()
    )

    # Keep VAR manageable.
    returns = returns.tail(
        ML_LOOKBACK
    )

    if len(returns) < 100:
        raise ValueError(
            "Not enough data for VAR."
        )

    # Choose lag 5 but cap by available observations.
    lag = min(5, max(1, len(returns) // 20))

    model = VAR(returns)

    fitted = model.fit(lag)

    fc = fitted.forecast(
        returns.values[-fitted.k_ar:],
        steps=horizon,
    )

    cumulative_stock_return = (
        fc[:, 0].sum()
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return (
        last_price
        * np.exp(
            cumulative_stock_return
        )
    )


# ============================================================
# METRICS
# ============================================================

def safe_mae(
    actual: float,
    predicted: float,
) -> float:

    return abs(
        actual - predicted
    )


def safe_return_error(
    last_price: float,
    actual: float,
    predicted: float,
) -> float:

    actual_return = (
        actual / last_price - 1.0
    )

    predicted_return = (
        predicted / last_price - 1.0
    )

    return abs(
        actual_return
        -
        predicted_return
    )


def evaluate_forecast_predictions(
    records: list[dict],
) -> pd.DataFrame:

    df = pd.DataFrame(records)

    # Normalized error.
    df["ml_return_error"] = (
        df["ml_mae"]
        /
        df["last_price"]
    )

    df["chronos_return_error"] = (
        df["chronos_mae"]
        /
        df["last_price"]
    )

    return df


# ============================================================
# FORECAST BENCHMARK
# ============================================================

def run_stock_benchmark(
    stock: pd.DataFrame,
    ticker: str,
    pipeline,
    experiment_name: str,
    chronos_covariates: Optional[list[str]] = None,
    ml_external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    if chronos_covariates is None:
        chronos_covariates = []

    if ml_external_columns is None:
        ml_external_columns = []

    print("\n")
    print("=" * 80)
    print(
        f"{experiment_name} — {ticker}"
    )
    print("=" * 80)

    cutoffs = get_cutoffs(
        len(stock)
    )

    rows = []

    for i, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        if (
            i == 1
            or i == len(cutoffs)
            or i % 10 == 0
        ):
            print(
                f"  window {i}/{len(cutoffs)} "
                f"(cutoff="
                f"{stock['Date'].iloc[cutoff - 1].date()})"
            )

        history = stock.iloc[
            :cutoff
        ].copy()

        last_price = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        for horizon in HORIZONS:

            future = stock.iloc[
                cutoff:
                cutoff + horizon
            ]

            actual = float(
                future[
                    f"{ticker}_Close"
                ].iloc[-1]
            )

            # --------------------------
            # Naive
            # --------------------------

            naive = naive_forecast(
                history,
                ticker,
            )

            # --------------------------
            # ARIMA
            # --------------------------

            try:
                arima = arima_forecast(
                    history,
                    ticker,
                    horizon,
                )
            except Exception:
                arima = np.nan

            # --------------------------
            # ETS
            # --------------------------

            try:
                ets = ets_forecast(
                    history,
                    ticker,
                    horizon,
                )
            except Exception:
                ets = np.nan

            # --------------------------
            # GARCH
            # --------------------------

            garch_price = np.nan
            garch_vol = np.nan

            if ARCH_AVAILABLE:
                try:
                    (
                        garch_price,
                        garch_vol,
                    ) = garch_forecast(
                        history,
                        ticker,
                        horizon,
                        "GARCH",
                    )
                except Exception:
                    pass

            # --------------------------
            # EGARCH
            # --------------------------

            egarch_price = np.nan

            if ARCH_AVAILABLE:
                try:
                    egarch_price, _ = (
                        garch_forecast(
                            history,
                            ticker,
                            horizon,
                            "EGARCH",
                        )
                    )
                except Exception:
                    pass

            # --------------------------
            # GJR-GARCH
            # --------------------------

            gjr_price = np.nan

            if ARCH_AVAILABLE:
                try:
                    gjr_price, _ = (
                        garch_forecast(
                            history,
                            ticker,
                            horizon,
                            "GJR-GARCH",
                        )
                    )
                except Exception:
                    pass

            # --------------------------
            # XGBoost
            # --------------------------

            try:

                xgb_model = train_xgb(
                    history,
                    ticker,
                    horizon,
                    ml_external_columns,
                )

                xgb = xgb_forecast(
                    history,
                    ticker,
                    horizon,
                    xgb_model,
                    ml_external_columns,
                )

            except Exception:

                xgb = np.nan

            # --------------------------
            # LightGBM
            # --------------------------

            lgbm = np.nan

            if LIGHTGBM_AVAILABLE:
                try:

                    lgbm_model = train_lgbm(
                        history,
                        ticker,
                        horizon,
                        ml_external_columns,
                    )

                    lgbm = lgbm_forecast(
                        history,
                        ticker,
                        lgbm_model,
                        ml_external_columns,
                    )

                except Exception:
                    pass

            # --------------------------
            # VAR
            # --------------------------

            var = np.nan

            if "SPX_Close" in history.columns:
                try:
                    var = var_forecast(
                        history,
                        ticker,
                        horizon,
                    )
                except Exception:
                    pass

            # --------------------------
            # Chronos
            # --------------------------

            try:

                cp = chronos_forecast(
                    pipeline,
                    history,
                    ticker,
                    horizon,
                    chronos_covariates,
                )

            except Exception:

                cp = {
                    "p05": np.nan,
                    "p10": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                    "p95": np.nan,
                }

            rows.append({
                "ticker": ticker,
                "experiment": experiment_name,
                "cutoff_date":
                    history["Date"].iloc[-1],
                "horizon": horizon,

                "last_price": last_price,
                "actual": actual,

                "naive": naive,

                "arima": arima,
                "ets": ets,

                "garch": garch_price,
                "egarch": egarch_price,
                "gjr_garch": gjr_price,

                "var": var,

                "xgboost": xgb,
                "lightgbm": lgbm,

                "chronos_p05": cp["p05"],
                "chronos_p10": cp["p10"],
                "chronos_p50": cp["p50"],
                "chronos_p90": cp["p90"],
                "chronos_p95": cp["p95"],
            })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# CONVERT TO LONG FORMAT / SCORE
# ============================================================

FORECAST_COLUMNS = [
    "naive",
    "arima",
    "ets",
    "garch",
    "egarch",
    "gjr_garch",
    "var",
    "xgboost",
    "lightgbm",
    "chronos_p50",
]


def score_forecasts(
    result: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in result.iterrows():

        actual = row["actual"]
        last = row["last_price"]

        for model_name in FORECAST_COLUMNS:

            predicted = row[model_name]

            if pd.isna(predicted):
                continue

            rows.append({
                "ticker":
                    row["ticker"],

                "experiment":
                    row["experiment"],

                "cutoff_date":
                    row["cutoff_date"],

                "horizon":
                    row["horizon"],

                "model":
                    model_name,

                "actual":
                    actual,

                "predicted":
                    predicted,

                "price_mae":
                    abs(
                        actual
                        - predicted
                    ),

                "return_abs_error":
                    safe_return_error(
                        last,
                        actual,
                        predicted,
                    ),

                "direction_hit":
                    float(
                        np.sign(
                            actual / last - 1
                        )
                        ==
                        np.sign(
                            predicted / last - 1
                        )
                    ),
            })

    return pd.DataFrame(rows)


def aggregate_scores(
    scored: pd.DataFrame,
) -> pd.DataFrame:

    return (
        scored
        .groupby([
            "ticker",
            "experiment",
            "horizon",
            "model",
        ])
        .agg(
            observations=(
                "price_mae",
                "size",
            ),

            mean_price_mae=(
                "price_mae",
                "mean",
            ),

            median_price_mae=(
                "price_mae",
                "median",
            ),

            mean_return_abs_error=(
                "return_abs_error",
                "mean",
            ),

            directional_accuracy=(
                "direction_hit",
                "mean",
            ),
        )
        .reset_index()
    )


# ============================================================
# CHRONOS CALIBRATION
# ============================================================

def evaluate_chronos_quantiles(
    result: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        ticker,
        experiment,
        horizon,
    ), group in result.groupby([
        "ticker",
        "experiment",
        "horizon",
    ]):

        actual = group["actual"].to_numpy()

        for q, col in [
            (0.10, "chronos_p10"),
            (0.90, "chronos_p90"),
        ]:

            pred = group[col].to_numpy()

            valid = (
                np.isfinite(actual)
                &
                np.isfinite(pred)
            )

            actual_v = actual[valid]
            pred_v = pred[valid]

            if len(actual_v) == 0:
                continue

            if q == 0.10:

                breach = (
                    actual_v < pred_v
                )

            else:

                breach = (
                    actual_v > pred_v
                )

            rows.append({
                "ticker": ticker,
                "experiment": experiment,
                "horizon": horizon,
                "quantile": q,
                "expected_breach_rate": 1 - q,
                "actual_breach_rate":
                    float(breach.mean()),
                "n":
                    len(actual_v),
            })

    return pd.DataFrame(rows)


# ============================================================
# SPIKES
# ============================================================

def run_spike_test(
    stock: pd.DataFrame,
    ticker: str,
    pipeline,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        f"SPIKE TEST — {ticker}"
    )
    print("=" * 80)

    cutoffs = get_cutoffs(
        len(stock)
    )

    rows = []

    for cutoff in cutoffs:

        history = stock.iloc[
            :cutoff
        ].copy()

        future = stock.iloc[
            cutoff
        ]

        last = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        actual = float(
            future[
                f"{ticker}_Close"
            ]
        )

        actual_return = (
            actual / last - 1
        )

        returns = (
            history[
                f"{ticker}_Close"
            ]
            .pct_change()
            .dropna()
        )

        vol = float(
            returns.tail(20).std()
        )

        if not np.isfinite(vol) or vol <= 0:
            continue

        threshold = (
            SPIKE_SIGMA * vol
        )

        actual_spike = (
            abs(actual_return)
            >
            threshold
        )

        # XGBoost
        xgb_return = np.nan

        try:

            model = train_xgb(
                history,
                ticker,
                horizon=1,
            )

            pred = xgb_forecast(
                history,
                ticker,
                horizon=1,
                model=model,
            )

            xgb_return = (
                pred / last - 1
            )

        except Exception:
            pass

        # Chronos
        try:

            cp = chronos_forecast(
                pipeline,
                history,
                ticker,
                horizon=1,
            )

            p10_return = (
                cp["p10"] / last - 1
            )

            p90_return = (
                cp["p90"] / last - 1
            )

            chronos_warning = (
                p10_return < -threshold
                or
                p90_return > threshold
            )

        except Exception:

            p10_return = np.nan
            p90_return = np.nan
            chronos_warning = False

        xgb_warning = (
            np.isfinite(xgb_return)
            and
            abs(xgb_return) > threshold
        )

        rows.append({
            "ticker": ticker,
            "date":
                future["Date"],
            "actual_return":
                actual_return,
            "rolling_vol":
                vol,
            "threshold":
                threshold,
            "actual_spike":
                actual_spike,
            "xgb_warning":
                xgb_warning,
            "chronos_warning":
                chronos_warning,
        })

    return pd.DataFrame(rows)


def precision_recall(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, float, int, int, int]:

    actual = (
        actual
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    predicted = (
        predicted
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    tp = int(
        np.sum(
            actual & predicted
        )
    )

    fp = int(
        np.sum(
            ~actual & predicted
        )
    )

    fn = int(
        np.sum(
            actual & ~predicted
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    return (
        precision,
        recall,
        tp,
        fp,
        fn,
    )


def summarize_spikes(
    all_spikes: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for ticker, group in all_spikes.groupby(
        "ticker"
    ):

        for model, column in [
            ("XGBoost", "xgb_warning"),
            ("Chronos-2", "chronos_warning"),
        ]:

            (
                precision,
                recall,
                tp,
                fp,
                fn,
            ) = precision_recall(
                group["actual_spike"],
                group[column],
            )

            rows.append({
                "ticker": ticker,
                "model": model,
                "actual_spikes":
                    int(
                        group[
                            "actual_spike"
                        ].sum()
                    ),
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

    return pd.DataFrame(rows)


# ============================================================
# COLD START EXPERIMENT
# ============================================================

def run_cold_start(
    stock: pd.DataFrame,
    ticker: str,
    pipeline,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        f"COLD START — {ticker}"
    )
    print("=" * 80)

    rows = []

    for history_length in COLD_START_LENGTHS:

        if len(stock) <= (
            history_length
            +
            max(HORIZONS)
        ):
            continue

        # Use many launch dates, sampled across the series.
        possible = np.arange(
            history_length,
            len(stock)
            -
            max(HORIZONS),
        )

        if len(possible) == 0:
            continue

        if N_WINDOWS is not None:
            n = min(
                N_WINDOWS,
                len(possible),
            )

            positions = np.linspace(
                0,
                len(possible) - 1,
                n,
                dtype=int,
            )

            cutoffs = [
                int(possible[i])
                for i in positions
            ]

        else:
            cutoffs = [
                int(x)
                for x in possible
            ]

        for cutoff in cutoffs:

            history = stock.iloc[
                :cutoff
            ].copy()

            last = float(
                history[
                    f"{ticker}_Close"
                ].iloc[-1]
            )

            for horizon in HORIZONS:

                future = stock.iloc[
                    cutoff:
                    cutoff + horizon
                ]

                actual = float(
                    future[
                        f"{ticker}_Close"
                    ].iloc[-1]
                )

                # ----------------------
                # Naive
                # ----------------------

                naive = last

                # ----------------------
                # ARIMA
                # ----------------------

                try:
                    arima = arima_forecast(
                        history,
                        ticker,
                        horizon,
                    )
                except Exception:
                    arima = np.nan

                # ----------------------
                # XGBoost
                # ----------------------

                try:

                    model = train_xgb(
                        history,
                        ticker,
                        horizon,
                    )

                    xgb = xgb_forecast(
                        history,
                        ticker,
                        horizon,
                        model,
                    )

                except Exception:

                    xgb = np.nan

                # ----------------------
                # Chronos
                # ----------------------

                try:

                    cp = chronos_forecast(
                        pipeline,
                        history,
                        ticker,
                        horizon,
                    )

                    chronos = cp["p50"]

                except Exception:

                    chronos = np.nan

                for model_name, predicted in [
                    ("Naive", naive),
                    ("ARIMA", arima),
                    ("XGBoost", xgb),
                    ("Chronos-2", chronos),
                ]:

                    if pd.isna(predicted):
                        continue

                    rows.append({
                        "ticker":
                            ticker,

                        "history_length":
                            history_length,

                        "cutoff_date":
                            history["Date"].iloc[-1],

                        "horizon":
                            horizon,

                        "model":
                            model_name,

                        "last_price":
                            last,

                        "actual":
                            actual,

                        "predicted":
                            predicted,

                        "price_mae":
                            abs(
                                actual
                                - predicted
                            ),

                        "return_abs_error":
                            safe_return_error(
                                last,
                                actual,
                                predicted,
                            ),
                    })

    return pd.DataFrame(rows)


# ============================================================
# CUT OFF GENERATOR
# ============================================================

def get_cutoffs(
    n_rows: int,
) -> list[int]:

    required = (
        MIN_HISTORY
        +
        max(HORIZONS)
    )

    if n_rows <= required:
        raise ValueError(
            f"{n_rows} rows available; "
            f"need > {required}"
        )

    possible = np.arange(
        MIN_HISTORY,
        n_rows - max(HORIZONS),
    )

    if N_WINDOWS is None:
        return [
            int(x)
            for x in possible
        ]

    if len(possible) <= N_WINDOWS:
        return [
            int(x)
            for x in possible
        ]

    positions = np.linspace(
        0,
        len(possible) - 1,
        N_WINDOWS,
        dtype=int,
    )

    return [
        int(possible[i])
        for i in positions
    ]


# ============================================================
# VISUALIZATION
# ============================================================

def plot_example(
    stock: pd.DataFrame,
    ticker: str,
    pipeline,
) -> None:

    horizon = 20

    if len(stock) <= (
        MIN_HISTORY + horizon
    ):
        return

    cutoff = len(stock) - horizon

    history = stock.iloc[
        :cutoff
    ].copy()

    future = stock.iloc[
        cutoff:
    ].copy()

    try:

        xgb_model = train_xgb(
            history,
            ticker,
            horizon,
        )

        xgb = xgb_forecast(
            history,
            ticker,
            horizon,
            xgb_model,
        )

        cp = chronos_forecast(
            pipeline,
            history,
            ticker,
            horizon,
        )

    except Exception as exc:

        print(
            f"Plot failed for {ticker}: {exc}"
        )
        return

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        history["Date"].tail(100),
        history[
            f"{ticker}_Close"
        ].tail(100),
        label="Known history",
    )

    plt.plot(
        future["Date"],
        future[
            f"{ticker}_Close"
        ],
        linewidth=3,
        label="Actual future",
    )

    plt.plot(
        future["Date"],
        np.repeat(
            xgb,
            len(future),
        ),
        linestyle="--",
        label="XGBoost 20D endpoint",
    )

    plt.plot(
        future["Date"],
        np.repeat(
            cp["p50"],
            len(future),
        ),
        linestyle="--",
        label="Chronos P50 endpoint",
    )

    plt.axhline(
        cp["p10"],
        linestyle=":",
        label="Chronos P10",
    )

    plt.axhline(
        cp["p90"],
        linestyle=":",
        label="Chronos P90",
    )

    plt.axvline(
        history["Date"].iloc[-1],
        linestyle=":",
        label="Forecast start",
    )

    plt.title(
        f"{ticker} — Chronos-2 vs XGBoost"
    )

    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / f"{ticker}_forecast_example.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CHRONOS-2 vs STATISTICAL MODELS")
    print("10-STOCK FINANCIAL FORECASTING POC")
    print("=" * 80)

    print(
        f"\nARCH package available: "
        f"{ARCH_AVAILABLE}"
    )

    print(
        f"LightGBM available: "
        f"{LIGHTGBM_AVAILABLE}"
    )

    pipeline = load_chronos()

    all_results = []
    all_spikes = []
    all_cold_start = []

    evaluated = []

    for ticker in TICKERS:

        path = (
            DATA_DIR
            /
            f"{ticker}.csv"
        )

        if not path.exists():

            print(
                f"\nSkipping {ticker}: "
                f"{path} not found."
            )

            continue

        try:

            stock = load_stock(
                ticker
            )

        except Exception as exc:

            print(
                f"\nSkipping {ticker}: {exc}"
            )

            continue

        print("\n")
        print("-" * 80)
        print(
            f"{ticker}: "
            f"{len(stock):,} rows | "
            f"{stock['Date'].min().date()} -> "
            f"{stock['Date'].max().date()}"
        )
        print("-" * 80)

        if len(stock) <= (
            MIN_HISTORY
            +
            max(HORIZONS)
        ):

            print(
                "  Not enough history; skipping."
            )

            continue

        evaluated.append(ticker)

        # ----------------------------------------------------
        # Experiment 1
        # ----------------------------------------------------

        e1 = run_stock_benchmark(
            stock,
            ticker,
            pipeline,
            "Experiment 1 — Univariate Close",
        )

        all_results.append(e1)

        # ----------------------------------------------------
        # Experiment 2
        # ----------------------------------------------------

        e2 = run_stock_benchmark(
            stock,
            ticker,
            pipeline,
            "Experiment 2 — OHLCV",
            chronos_covariates=[
                f"{ticker}_Open",
                f"{ticker}_High",
                f"{ticker}_Low",
                f"{ticker}_Volume",
            ],
        )

        all_results.append(e2)

        # ----------------------------------------------------
        # Experiment 3
        # ----------------------------------------------------

        cross = align_cross_market(
            stock
        )

        if (
            cross is not None
            and len(cross) > (
                MIN_HISTORY
                +
                max(HORIZONS)
            )
        ):

            e3 = run_stock_benchmark(
                cross,
                ticker,
                pipeline,
                "Experiment 3 — Cross Market",
                chronos_covariates=[
                    "SPX_Close",
                    "VIX_Close",
                ],
                ml_external_columns=[
                    "SPX_Close",
                    "VIX_Close",
                ],
            )

            all_results.append(e3)

        # ----------------------------------------------------
        # Spike test
        # ----------------------------------------------------

        spike = run_spike_test(
            stock,
            ticker,
            pipeline,
        )

        all_spikes.append(
            spike
        )

        # ----------------------------------------------------
        # Cold-start
        # ----------------------------------------------------

        cold = run_cold_start(
            stock,
            ticker,
            pipeline,
        )

        all_cold_start.append(
            cold
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_example(
            stock,
            ticker,
            pipeline,
        )

    # ========================================================
    # SAVE FORECAST RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    if all_results:

        result = pd.concat(
            all_results,
            ignore_index=True,
        )

        result.to_csv(
            RESULTS_DIR
            / "all_forecasts.csv",
            index=False,
        )

        scored = score_forecasts(
            result
        )

        scored.to_csv(
            RESULTS_DIR
            / "all_scored_forecasts.csv",
            index=False,
        )

        aggregate = aggregate_scores(
            scored
        )

        aggregate.to_csv(
            RESULTS_DIR
            / "model_comparison.csv",
            index=False,
        )

        coverage = evaluate_chronos_quantiles(
            result
        )

        coverage.to_csv(
            RESULTS_DIR
            / "chronos_coverage.csv",
            index=False,
        )

    # ========================================================
    # SPIKES
    # ========================================================

    if all_spikes:

        spikes = pd.concat(
            all_spikes,
            ignore_index=True,
        )

        spikes.to_csv(
            RESULTS_DIR
            / "spikes_all_stocks.csv",
            index=False,
        )

        spike_summary = summarize_spikes(
            spikes
        )

        spike_summary.to_csv(
            RESULTS_DIR
            / "spike_summary.csv",
            index=False,
        )

        print("\nSPIKE SUMMARY")
        print(
            spike_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ========================================================
    # COLD START
    # ========================================================

    if all_cold_start:

        cold_start = pd.concat(
            all_cold_start,
            ignore_index=True,
        )

        cold_start.to_csv(
            RESULTS_DIR
            / "cold_start_results.csv",
            index=False,
        )

        cold_summary = (
            cold_start
            .groupby([
                "ticker",
                "history_length",
                "horizon",
                "model",
            ])
            .agg(
                observations=(
                    "price_mae",
                    "size",
                ),
                mean_price_mae=(
                    "price_mae",
                    "mean",
                ),
                mean_return_abs_error=(
                    "return_abs_error",
                    "mean",
                ),
            )
            .reset_index()
        )

        cold_summary.to_csv(
            RESULTS_DIR
            / "cold_start_summary.csv",
            index=False,
        )

    # ========================================================
    # MAIN COMPARISON TABLE
    # ========================================================

    if all_results:

        print("\n")
        print("=" * 80)
        print("OVERALL MODEL COMPARISON")
        print("=" * 80)

        overall = (
            scored
            .groupby([
                "experiment",
                "horizon",
                "model",
            ])
            .agg(
                observations=(
                    "price_mae",
                    "size",
                ),

                mean_price_mae=(
                    "price_mae",
                    "mean",
                ),

                median_price_mae=(
                    "price_mae",
                    "median",
                ),

                mean_return_abs_error=(
                    "return_abs_error",
                    "mean",
                ),

                directional_accuracy=(
                    "direction_hit",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(
            overall.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        overall.to_csv(
            RESULTS_DIR
            / "overall_model_comparison.csv",
            index=False,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n")
    print("=" * 80)
    print("POC COMPLETE")
    print("=" * 80)

    print(
        f"Successfully evaluated: "
        f"{len(evaluated)} / {len(TICKERS)}"
    )

    print(
        "Stocks:",
        ", ".join(evaluated)
    )

    print(
        f"\nResults: "
        f"{RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
