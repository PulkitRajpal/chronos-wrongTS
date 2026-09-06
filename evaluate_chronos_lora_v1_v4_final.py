"""
FINAL CHRONOS-2 LoRA V1/V2/V3/V4 EVALUATION
============================================

Purpose
-------
Evaluate four financially fine-tuned Chronos-2 LoRA adapters against the
same statistical / ML baselines on the same 10 held-out stocks.

LoRA models
-----------
V1: models/chronos2-finance-lora/finetuned-ckpt
V2: models/chronos2-finance-lora-v2/finetuned-ckpt
V3: models/chronos2-finance-lora-v3-broad/finetuned-ckpt
V4: models/chronos2-finance-lora-v4-equity-rates-vol/finetuned-ckpt

V1:
    79 financial series

V2:
    79 financial series
    improved LR/scheduler

V3:
    152 broader financial series

V4:
    132 equities + rates/bonds + volatility

IMPORTANT DATA REPRESENTATION
-----------------------------
The LoRA models were trained on DAILY LOG RETURNS.

Therefore evaluation for the LoRA models is:

    price history
        -> daily log returns
        -> Chronos forecast of future daily log returns
        -> cumulative median log return
        -> endpoint price

Probabilistic evaluation
------------------------
Chronos-2 in this installation exposes quantile forecasts, not independent
Monte-Carlo trajectory samples.

We therefore do NOT incorrectly sum per-step quantiles to claim a joint
80% endpoint interval.

Instead, this script evaluates proper HORIZON-BY-HORIZON MARGINAL calibration:

    predicted q10/q90 daily return at lead h
                    vs
    realized daily log return at lead h

This produces:
    - coverage by forecast lead (1..20)
    - average coverage across leads 1..H
    - lower/upper tail breach rates
    - average interval width

This is a valid marginal calibration test. It is NOT a joint 20-day path
coverage claim.

Statistical model robustness
-----------------------------
ARIMA:
    - retry with stationarity/invertibility constraints relaxed
    - increased optimizer iterations
    - failures are explicitly recorded

GARCH / EGARCH / GJR-GARCH:
    - retry with Student-t
    - retry with Normal
    - increased optimizer iterations
    - convergence success/failure is recorded

The script never silently converts a failed model fit into a zero forecast.

Win-rate analysis
-----------------
For every ticker and horizon, models are compared on price MAE.

Outputs include:
    - pairwise win rate vs GARCH
    - number of stock wins/losses/ties
    - overall model win counts
    - fit-success rates

Evaluation
----------
Stocks:
    AAPL, MSFT, NVDA, AMZN, ENPH, SMCI, CVNA, PLUG, RKLB, IONQ

Horizons:
    1, 5, 10, 20 trading sessions

Walk-forward windows:
    50 per stock

Expected stock CSV:
    date,open,high,low,close,volume

Optional:
    data/SPX.csv
    data/VIX.csv
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline
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


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR = (
    BASE_DIR
    / "results_lora_final"
)

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

HORIZONS = [
    1,
    5,
    10,
    20,
]

N_WINDOWS = 50

MIN_HISTORY = 300

ML_LOOKBACK = 252

CHRONOS_CONTEXT = 256

# Nominal 80% interval.
LOWER_Q = 0.10
MEDIAN_Q = 0.50
UPPER_Q = 0.90

# Extreme-return threshold for spike test.
SPIKE_SIGMA = 2.0

# LoRA checkpoints.
LORA_MODELS = {
    "Chronos-LoRA-V1":
        BASE_DIR
        / "models"
        / "chronos2-finance-lora"
        / "finetuned-ckpt",

    "Chronos-LoRA-V2":
        BASE_DIR
        / "models"
        / "chronos2-finance-lora-v2"
        / "finetuned-ckpt",

    "Chronos-LoRA-V3":
        BASE_DIR
        / "models"
        / "chronos2-finance-lora-v3-broad"
        / "finetuned-ckpt",

    "Chronos-LoRA-V4":
        BASE_DIR
        / "models"
        / "chronos2-finance-lora-v4-equity-rates-vol"
        / "finetuned-ckpt",
}

STAT_MODELS = [
    "Naive",
    "ARIMA",
    "ETS",
    "GARCH",
    "EGARCH",
    "GJR-GARCH",
    "VAR",
    "XGBoost",
    "LightGBM",
]


# ============================================================
# DEVICE
# ============================================================

def resolve_device() -> str:

    if not torch.cuda.is_available():

        print(
            "CUDA unavailable -> CPU"
        )

        return "cpu"

    try:

        torch.randn(
            1,
            device="cuda",
        )

        print(
            "Using CUDA device:",
            torch.cuda.get_device_name(0),
        )

        return "cuda"

    except Exception as exc:

        print(
            "CUDA tensor test failed -> CPU"
        )

        print(exc)

        return "cpu"


DEVICE = resolve_device()


# ============================================================
# DATA HELPERS
# ============================================================

def clean_number(
    value,
) -> float:

    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s in {
        "",
        "-",
        "—",
        "N/A",
        "NA",
        "null",
        "None",
    }:
        return np.nan

    s = (
        s
        .replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    if (
        s.startswith("(")
        and
        s.endswith(")")
    ):
        s = "-" + s[1:-1]

    try:
        return float(s)

    except ValueError:
        return np.nan


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> Optional[str]:

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]

    if required:

        raise ValueError(
            f"Could not find {candidates}. "
            f"Available columns: "
            f"{list(df.columns)}"
        )

    return None


def load_stock(
    ticker: str,
) -> pd.DataFrame:

    path = (
        DATA_DIR
        / f"{ticker}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        [
            "date",
            "datetime",
            "timestamp",
        ],
    )

    close_col = find_column(
        raw,
        [
            "close/last",
            "close",
            "last",
            "price",
        ],
    )

    open_col = find_column(
        raw,
        ["open"],
        required=False,
    )

    high_col = find_column(
        raw,
        ["high"],
        required=False,
    )

    low_col = find_column(
        raw,
        ["low"],
        required=False,
    )

    volume_col = find_column(
        raw,
        ["volume"],
        required=False,
    )

    out = pd.DataFrame({

        "Date":
            pd.to_datetime(
                raw[date_col],
                errors="coerce",
            ),

        f"{ticker}_Close":
            raw[close_col]
            .map(clean_number),
    })

    for name, source in [
        (
            f"{ticker}_Open",
            open_col,
        ),
        (
            f"{ticker}_High",
            high_col,
        ),
        (
            f"{ticker}_Low",
            low_col,
        ),
        (
            f"{ticker}_Volume",
            volume_col,
        ),
    ]:

        if source is None:

            out[name] = np.nan

        else:

            out[name] = (
                raw[source]
                .map(clean_number)
            )

    out = (
        out
        .dropna(
            subset=[
                "Date",
                f"{ticker}_Close",
            ]
        )
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return out


def load_macro(
    path: Path,
    name: str,
) -> pd.DataFrame:

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        [
            "date",
            "datetime",
            "timestamp",
            "observation_date",
        ],
    )

    value_col = find_column(
        raw,
        [
            "close/last",
            "close",
            "last",
            "price",
            "value",
            name,
        ],
    )

    out = pd.DataFrame({
        "Date":
            pd.to_datetime(
                raw[date_col],
                errors="coerce",
            ),

        f"{name}_Close":
            raw[value_col]
            .map(clean_number),
    })

    return (
        out
        .dropna(
            subset=[
                "Date",
                f"{name}_Close",
            ]
        )
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def align_cross_market(
    stock: pd.DataFrame,
) -> Optional[pd.DataFrame]:

    spx_path = DATA_DIR / "SPX.csv"
    vix_path = DATA_DIR / "VIX.csv"

    if (
        not spx_path.exists()
        or
        not vix_path.exists()
    ):
        return None

    try:

        spx = load_macro(
            spx_path,
            "SPX",
        )

        vix = load_macro(
            vix_path,
            "VIX",
        )

        return (
            stock
            .merge(
                spx,
                on="Date",
                how="inner",
            )
            .merge(
                vix,
                on="Date",
                how="inner",
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

    except Exception as exc:

        print(
            "Cross-market alignment failed:",
            exc,
        )

        return None


# ============================================================
# WALK-FORWARD CUT-OFFS
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
            f"Need > {required} rows; "
            f"got {n_rows}"
        )

    possible = np.arange(
        MIN_HISTORY,
        n_rows - max(HORIZONS),
    )

    if (
        N_WINDOWS is None
        or
        len(possible) <= N_WINDOWS
    ):

        return [
            int(x)
            for x in possible
        ]

    indices = np.linspace(
        0,
        len(possible) - 1,
        N_WINDOWS,
        dtype=int,
    )

    return [
        int(possible[i])
        for i in indices
    ]


# ============================================================
# STATISTICAL MODELS
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


def fit_arima(
    history: pd.DataFrame,
    ticker: str,
):
    """
    Robust ARIMA fit.

    We fit once per cutoff and forecast the maximum horizon.
    """

    close = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
    )

    log_price = np.log(close)

    attempts = [
        {
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        {
            "enforce_stationarity": False,
            "enforce_invertibility": False,
        },
    ]

    last_error = None

    for options in attempts:

        try:

            model = ARIMA(
                log_price,
                order=(1, 1, 1),
                **options,
            )

            fitted = model.fit(
                method_kwargs={
                    "maxiter": 500,
                }
            )

            return (
                fitted,
                "success",
            )

        except Exception as exc:

            last_error = str(exc)

    return (
        None,
        f"failed: {last_error}",
    )


def arima_forecast(
    fitted,
    horizon: int,
) -> float:

    forecast = fitted.forecast(
        steps=horizon
    )

    return float(
        np.exp(
            forecast.iloc[-1]
        )
    )


def fit_ets(
    history: pd.DataFrame,
    ticker: str,
):

    close = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
    )

    try:

        model = ExponentialSmoothing(
            close,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        )

        fitted = model.fit(
            optimized=True,
        )

        return (
            fitted,
            "success",
        )

    except Exception as exc:

        return (
            None,
            f"failed: {exc}",
        )


def ets_forecast(
    fitted,
    horizon: int,
) -> float:

    forecast = fitted.forecast(
        horizon
    )

    return float(
        forecast.iloc[-1]
    )


def fit_garch(
    history: pd.DataFrame,
    ticker: str,
    variant: str,
):
    """
    Robust GARCH-family fit with retries.

    Returns:
        fitted model, status
    """

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

    distributions = [
        "t",
        "normal",
    ]

    last_error = None

    for distribution in distributions:

        try:

            if variant == "GARCH":

                model = arch_model(
                    returns,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    o=0,
                    q=1,
                    dist=distribution,
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
                    dist=distribution,
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
                    dist=distribution,
                    rescale=False,
                )

            else:

                raise ValueError(
                    variant
                )

            fitted = model.fit(
                disp="off",
                update_freq=0,
                options={
                    "maxiter": 2000,
                },
            )

            converged = getattr(
                fitted,
                "convergence_flag",
                0,
            )

            if converged == 0:

                return (
                    fitted,
                    "success",
                )

            # Even if the optimizer flags non-convergence,
            # do not immediately discard the fit. Return it as
            # a flagged result so the benchmark can report the
            # condition explicitly.
            return (
                fitted,
                f"nonconverged_flag={converged}",
            )

        except Exception as exc:

            last_error = str(exc)

    return (
        None,
        f"failed: {last_error}",
    )


def garch_forecast(
    fitted,
    horizon: int,
    history: pd.DataFrame,
    ticker: str,
) -> float:

    forecast = fitted.forecast(
        horizon=horizon,
        reindex=False,
    )

    mean_path = (
        forecast.mean
        .iloc[-1]
        .to_numpy()
    )

    cumulative_return = (
        mean_path.sum()
        /
        100.0
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return float(
        last_price
        *
        np.exp(cumulative_return)
    )


# ============================================================
# ML FEATURES
# ============================================================

def build_features(
    history: pd.DataFrame,
    ticker: str,
    external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    external_columns = (
        external_columns or []
    )

    data = history.copy()

    close = data[
        f"{ticker}_Close"
    ]

    data["ret_1"] = (
        close.pct_change(1)
    )

    data["ret_2"] = (
        close.pct_change(2)
    )

    data["ret_3"] = (
        close.pct_change(3)
    )

    data["ret_5"] = (
        close.pct_change(5)
    )

    data["ret_10"] = (
        close.pct_change(10)
    )

    data["ret_20"] = (
        close.pct_change(20)
    )

    data["hl_range"] = (
        (
            data[
                f"{ticker}_High"
            ]
            -
            data[
                f"{ticker}_Low"
            ]
        )
        /
        close
    )

    data["oc_return"] = (
        (
            data[
                f"{ticker}_Close"
            ]
            -
            data[
                f"{ticker}_Open"
            ]
        )
        /
        data[
            f"{ticker}_Open"
        ]
    )

    data["volume_return"] = (
        data[
            f"{ticker}_Volume"
        ].pct_change()
    )

    data["vol_5"] = (
        data["ret_1"]
        .rolling(5)
        .std()
    )

    data["vol_20"] = (
        data["ret_1"]
        .rolling(20)
        .std()
    )

    data["vol_60"] = (
        data["ret_1"]
        .rolling(60)
        .std()
    )

    data["mom_5"] = (
        close / close.shift(5)
        - 1.0
    )

    data["mom_20"] = (
        close / close.shift(20)
        - 1.0
    )

    data["mom_60"] = (
        close / close.shift(60)
        - 1.0
    )

    for col in external_columns:

        prefix = (
            col
            .replace("_Close", "")
            .lower()
        )

        data[
            f"{prefix}_ret_1"
        ] = data[col].pct_change(1)

        data[
            f"{prefix}_ret_5"
        ] = data[col].pct_change(5)

        data[
            f"{prefix}_ret_20"
        ] = data[col].pct_change(20)

    return data


def feature_names(
    external_columns: Optional[list[str]] = None,
) -> list[str]:

    external_columns = (
        external_columns or []
    )

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
            col
            .replace("_Close", "")
            .lower()
        )

        features.extend([
            f"{prefix}_ret_1",
            f"{prefix}_ret_5",
            f"{prefix}_ret_20",
        ])

    return features


def train_xgb(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
):

    external_columns = (
        external_columns or []
    )

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = feature_names(
        external_columns
    )

    target = (
        f"{ticker}_Close"
    )

    data["target_return"] = (
        data[target].shift(-horizon)
        /
        data[target]
        - 1.0
    )

    data = (
        data
        .dropna(
            subset=features + [
                "target_return"
            ]
        )
        .tail(ML_LOOKBACK)
    )

    X = data[
        features
    ].astype(float)

    y = data[
        "target_return"
    ].astype(float)

    valid = (
        np.isfinite(X)
        .all(axis=1)
        &
        np.isfinite(y)
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 100:

        raise ValueError(
            f"Only {len(X)} "
            "valid ML observations."
        )

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
    model,
    external_columns: Optional[list[str]] = None,
) -> float:

    external_columns = (
        external_columns or []
    )

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = feature_names(
        external_columns
    )

    X = (
        data.iloc[-1][features]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Latest XGBoost features "
            "contain NaN/inf."
        )

    predicted_return = float(
        model.predict(X)[0]
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return float(
        last_price
        *
        (
            1.0
            +
            predicted_return
        )
    )


def train_lgbm(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
):

    if not LIGHTGBM_AVAILABLE:

        raise RuntimeError(
            "LightGBM not installed."
        )

    external_columns = (
        external_columns or []
    )

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = feature_names(
        external_columns
    )

    target = (
        f"{ticker}_Close"
    )

    data["target_return"] = (
        data[target].shift(-horizon)
        /
        data[target]
        - 1.0
    )

    data = (
        data
        .dropna(
            subset=features + [
                "target_return"
            ]
        )
        .tail(ML_LOOKBACK)
    )

    X = data[
        features
    ].astype(float)

    y = data[
        "target_return"
    ].astype(float)

    valid = (
        np.isfinite(X)
        .all(axis=1)
        &
        np.isfinite(y)
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 100:

        raise ValueError(
            "Not enough LightGBM data."
        )

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
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

    external_columns = (
        external_columns or []
    )

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = feature_names(
        external_columns
    )

    X = (
        data.iloc[-1][features]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Latest LightGBM features "
            "contain NaN/inf."
        )

    predicted_return = float(
        model.predict(X)[0]
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return float(
        last_price
        *
        (
            1.0
            +
            predicted_return
        )
    )


def var_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> float:

    cols = [
        f"{ticker}_Close",
        "SPX_Close",
        "VIX_Close",
    ]

    if not all(
        c in history.columns
        for c in cols
    ):

        raise ValueError(
            "VAR requires stock + SPX + VIX."
        )

    returns = (
        np.log(
            history[cols]
        )
        .diff()
        .dropna()
        .tail(ML_LOOKBACK)
    )

    if len(returns) < 100:

        raise ValueError(
            "Not enough VAR observations."
        )

    fitted = VAR(
        returns
    ).fit(5)

    forecast = fitted.forecast(
        returns.values[
            -fitted.k_ar:
        ],
        steps=horizon,
    )

    cumulative_stock_return = (
        forecast[:, 0].sum()
    )

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return float(
        last_price
        *
        np.exp(
            cumulative_stock_return
        )
    )


# ============================================================
# CHRONOS LoRA RETURN FORECAST
# ============================================================

def lora_predict(
    pipeline,
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> dict:

    prices = (
        history[
            f"{ticker}_Close"
        ]
        .astype(float)
        .to_numpy()
    )

    daily_log_returns = np.diff(
        np.log(prices)
    )

    daily_log_returns = (
        daily_log_returns[
            np.isfinite(
                daily_log_returns
            )
        ]
    )

    if len(daily_log_returns) < 20:

        raise ValueError(
            "Not enough return history."
        )

    context_returns = (
        daily_log_returns[
            -CHRONOS_CONTEXT:
        ]
        .astype(np.float32)
    )

    context = pd.DataFrame({
        "id":
            [ticker]
            *
            len(context_returns),

        "timestamp":
            pd.date_range(
                "2000-01-01",
                periods=len(
                    context_returns
                ),
                freq="D",
            ),

        "target":
            context_returns,
    })

    pred = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=[
            LOWER_Q,
            MEDIAN_Q,
            UPPER_Q,
        ],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        validate_inputs=True,
    )

    q10 = (
        pred[str(LOWER_Q)]
        .to_numpy(float)
    )

    q50 = (
        pred[str(MEDIAN_Q)]
        .to_numpy(float)
    )

    q90 = (
        pred[str(UPPER_Q)]
        .to_numpy(float)
    )

    if not (
        len(q10)
        ==
        len(q50)
        ==
        len(q90)
        ==
        horizon
    ):

        raise RuntimeError(
            "Unexpected Chronos output length."
        )

    last_price = float(
        prices[-1]
    )

    # Point forecast:
    # cumulative median daily log returns.
    median_endpoint = (
        last_price
        *
        np.exp(
            np.cumsum(q50)
        )
    )

    # IMPORTANT:
    # q10/q90 are NOT summed to form a joint endpoint interval.
    # They are retained per lead for marginal calibration below.
    return {
        "q10":
            q10,

        "q50":
            q50,

        "q90":
            q90,

        "median_endpoint":
            float(
                median_endpoint[-1]
            ),

        "last_price":
            last_price,
    }


# ============================================================
# METRICS
# ============================================================

def score_point_forecast(
    last_price: float,
    actual_price: float,
    predicted_price: float,
) -> tuple[float, float, float]:

    if not np.isfinite(
        predicted_price
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    price_mae = abs(
        actual_price
        -
        predicted_price
    )

    actual_return = (
        actual_price
        /
        last_price
        - 1.0
    )

    predicted_return = (
        predicted_price
        /
        last_price
        - 1.0
    )

    return_error = abs(
        actual_return
        -
        predicted_return
    )

    if predicted_return == 0:

        direction = np.nan

    else:

        direction = float(
            np.sign(actual_return)
            ==
            np.sign(predicted_return)
        )

    return (
        price_mae,
        return_error,
        direction,
    )


# ============================================================
# EVALUATE ONE STOCK
# ============================================================

def evaluate_stock(
    stock: pd.DataFrame,
    ticker: str,
    lora_name: str,
    pipeline,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    point_rows = []
    interval_rows = []
    fit_rows = []

    cutoffs = get_cutoffs(
        len(stock)
    )

    print(
        f"\n{ticker} / {lora_name}: "
        f"{len(cutoffs)} windows"
    )

    for window_index, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = stock.iloc[
            :cutoff
        ].copy()

        if (
            window_index == 1
            or
            window_index == len(cutoffs)
            or
            window_index % 10 == 0
        ):

            print(
                f"  window "
                f"{window_index}/"
                f"{len(cutoffs)} "
                f"cutoff="
                f"{history['Date'].iloc[-1].date()}"
            )

        last_price = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        # ----------------------------------------------------
        # Fit statistical models ONCE per cutoff.
        # Then forecast all required horizons.
        # ----------------------------------------------------

        arima_model, arima_status = (
            fit_arima(
                history,
                ticker,
            )
        )

        ets_model, ets_status = (
            fit_ets(
                history,
                ticker,
            )
        )

        garch_fits = {}

        for variant in [
            "GARCH",
            "EGARCH",
            "GJR-GARCH",
        ]:

            model, status = fit_garch(
                history,
                ticker,
                variant,
            )

            garch_fits[variant] = (
                model,
                status,
            )

        # Store fit diagnostics.
        fit_rows.extend([
            {
                "ticker": ticker,
                "cutoff_date":
                    history[
                        "Date"
                    ].iloc[-1],
                "model": "ARIMA",
                "status": arima_status,
            },
            {
                "ticker": ticker,
                "cutoff_date":
                    history[
                        "Date"
                    ].iloc[-1],
                "model": "ETS",
                "status": ets_status,
            },
            *[
                {
                    "ticker": ticker,
                    "cutoff_date":
                        history[
                            "Date"
                        ].iloc[-1],
                    "model": variant,
                    "status": status,
                }
                for variant, (
                    _model,
                    status,
                )
                in garch_fits.items()
            ],
        ])

        for horizon in HORIZONS:

            future = stock.iloc[
                cutoff:
                cutoff + horizon
            ]

            if len(future) < horizon:
                continue

            actual_price = float(
                future[
                    f"{ticker}_Close"
                ].iloc[-1]
            )

            predictions = {}

            # ------------------------------------------------
            # Naive
            # ------------------------------------------------

            predictions[
                "Naive"
            ] = naive_forecast(
                history,
                ticker,
            )

            # ------------------------------------------------
            # ARIMA
            # ------------------------------------------------

            if arima_model is not None:

                try:

                    predictions[
                        "ARIMA"
                    ] = arima_forecast(
                        arima_model,
                        horizon,
                    )

                except Exception:

                    predictions[
                        "ARIMA"
                    ] = np.nan

            else:

                predictions[
                    "ARIMA"
                ] = np.nan

            # ------------------------------------------------
            # ETS
            # ------------------------------------------------

            if ets_model is not None:

                try:

                    predictions[
                        "ETS"
                    ] = ets_forecast(
                        ets_model,
                        horizon,
                    )

                except Exception:

                    predictions[
                        "ETS"
                    ] = np.nan

            else:

                predictions[
                    "ETS"
                ] = np.nan

            # ------------------------------------------------
            # GARCH family
            # ------------------------------------------------

            for variant in [
                "GARCH",
                "EGARCH",
                "GJR-GARCH",
            ]:

                model, status = (
                    garch_fits[
                        variant
                    ]
                )

                if model is None:

                    predictions[
                        variant
                    ] = np.nan

                    continue

                try:

                    predictions[
                        variant
                    ] = garch_forecast(
                        model,
                        horizon,
                        history,
                        ticker,
                    )

                except Exception:

                    predictions[
                        variant
                    ] = np.nan

            # ------------------------------------------------
            # VAR
            # ------------------------------------------------

            predictions[
                "VAR"
            ] = np.nan

            # VAR is evaluated only if cross-market files exist.
            cross_market = (
                align_cross_market(
                    history
                )
            )

            if (
                cross_market is not None
                and
                len(cross_market)
                > MIN_HISTORY
            ):

                try:

                    predictions[
                        "VAR"
                    ] = var_forecast(
                        cross_market,
                        ticker,
                        horizon,
                    )

                except Exception:

                    predictions[
                        "VAR"
                    ] = np.nan

            # ------------------------------------------------
            # XGBoost
            # ------------------------------------------------

            try:

                xgb_model = train_xgb(
                    history,
                    ticker,
                    horizon,
                )

                predictions[
                    "XGBoost"
                ] = xgb_forecast(
                    history,
                    ticker,
                    xgb_model,
                )

            except Exception:

                predictions[
                    "XGBoost"
                ] = np.nan

            # ------------------------------------------------
            # LightGBM
            # ------------------------------------------------

            predictions[
                "LightGBM"
            ] = np.nan

            if LIGHTGBM_AVAILABLE:

                try:

                    lgbm_model = train_lgbm(
                        history,
                        ticker,
                        horizon,
                    )

                    predictions[
                        "LightGBM"
                    ] = lgbm_forecast(
                        history,
                        ticker,
                        lgbm_model,
                    )

                except Exception:

                    pass

            # ------------------------------------------------
            # Save statistical scores
            # ------------------------------------------------

            for model_name, prediction in (
                predictions.items()
            ):

                mae, ret_error, direction = (
                    score_point_forecast(
                        last_price,
                        actual_price,
                        prediction,
                    )
                )

                point_rows.append({
                    "ticker":
                        ticker,
                    "cutoff_date":
                        history[
                            "Date"
                        ].iloc[-1],
                    "horizon":
                        horizon,
                    "model":
                        model_name,
                    "model_type":
                        "statistical_ml",
                    "actual":
                        actual_price,
                    "last_price":
                        last_price,
                    "prediction":
                        prediction,
                    "price_mae":
                        mae,
                    "return_abs_error":
                        ret_error,
                    "direction_hit":
                        direction,
                    "fit_status":
                        (
                            "not_applicable"
                            if model_name
                            in {
                                "Naive",
                                "XGBoost",
                                "LightGBM",
                                "VAR",
                            }
                            else ""
                        ),
                })

            # ------------------------------------------------
            # LoRA Chronos
            # ------------------------------------------------

            try:

                chrono = lora_predict(
                    pipeline,
                    history,
                    ticker,
                    horizon,
                )

                predicted_price = (
                    chrono[
                        "median_endpoint"
                    ]
                )

                mae, ret_error, direction = (
                    score_point_forecast(
                        last_price,
                        actual_price,
                        predicted_price,
                    )
                )

                point_rows.append({
                    "ticker":
                        ticker,
                    "cutoff_date":
                        history[
                            "Date"
                        ].iloc[-1],
                    "horizon":
                        horizon,
                    "model":
                        lora_name,
                    "model_type":
                        "chronos_lora",
                    "actual":
                        actual_price,
                    "last_price":
                        last_price,
                    "prediction":
                        predicted_price,
                    "price_mae":
                        mae,
                    "return_abs_error":
                        ret_error,
                    "direction_hit":
                        direction,
                    "fit_status":
                        "success",
                })

                # --------------------------------------------
                # PROPER MARGINAL INTERVAL CALIBRATION
                #
                # Compare each predicted daily-return quantile
                # at lead h to the realized daily return at lead h.
                # --------------------------------------------

                hist_prices = (
                    history[
                        f"{ticker}_Close"
                    ]
                    .astype(float)
                    .to_numpy()
                )

                future_prices = (
                    future[
                        f"{ticker}_Close"
                    ]
                    .astype(float)
                    .to_numpy()
                )

                all_prices = np.concatenate([
                    hist_prices[
                        -1:
                    ],
                    future_prices,
                ])

                realized_daily_log_returns = (
                    np.diff(
                        np.log(
                            all_prices
                        )
                    )
                )

                # Should contain exactly horizon values.
                realized_daily_log_returns = (
                    realized_daily_log_returns[
                        :horizon
                    ]
                )

                for lead in range(
                    1,
                    horizon + 1,
                ):

                    idx = lead - 1

                    realized_return = float(
                        realized_daily_log_returns[
                            idx
                        ]
                    )

                    q10 = float(
                        chrono[
                            "q10"
                        ][idx]
                    )

                    q50 = float(
                        chrono[
                            "q50"
                        ][idx]
                    )

                    q90 = float(
                        chrono[
                            "q90"
                        ][idx]
                    )

                    lower_breach = float(
                        realized_return
                        <
                        q10
                    )

                    upper_breach = float(
                        realized_return
                        >
                        q90
                    )

                    inside = float(
                        (
                            realized_return
                            >=
                            q10
                        )
                        and
                        (
                            realized_return
                            <=
                            q90
                        )
                    )

                    interval_rows.append({
                        "ticker":
                            ticker,
                        "cutoff_date":
                            history[
                                "Date"
                            ].iloc[-1],
                        "model":
                            lora_name,
                        "lead":
                            lead,
                        "realized_daily_log_return":
                            realized_return,
                        "q10":
                            q10,
                        "q50":
                            q50,
                        "q90":
                            q90,
                        "inside_80_interval":
                            inside,
                        "lower_breach":
                            lower_breach,
                        "upper_breach":
                            upper_breach,
                        "interval_width":
                            q90 - q10,
                    })

            except Exception as exc:

                point_rows.append({
                    "ticker":
                        ticker,
                    "cutoff_date":
                        history[
                            "Date"
                        ].iloc[-1],
                    "horizon":
                        horizon,
                    "model":
                        lora_name,
                    "model_type":
                        "chronos_lora",
                    "actual":
                        actual_price,
                    "last_price":
                        last_price,
                    "prediction":
                        np.nan,
                    "price_mae":
                        np.nan,
                    "return_abs_error":
                        np.nan,
                    "direction_hit":
                        np.nan,
                    "fit_status":
                        f"failed: {exc}",
                })

    return (
        pd.DataFrame(point_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(fit_rows),
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_point_results(
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    summary_by_stock = (
        points
        .groupby([
            "ticker",
            "horizon",
            "model",
            "model_type",
        ])
        .agg(
            observations=(
                "price_mae",
                "count",
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

    overall = (
        summary_by_stock
        .groupby([
            "horizon",
            "model",
            "model_type",
        ])
        .agg(
            stocks=(
                "ticker",
                "nunique",
            ),
            observations=(
                "observations",
                "sum",
            ),
            avg_price_mae=(
                "mean_price_mae",
                "mean",
            ),
            median_stock_mae=(
                "mean_price_mae",
                "median",
            ),
            avg_return_abs_error=(
                "mean_return_abs_error",
                "mean",
            ),
            avg_directional_accuracy=(
                "directional_accuracy",
                "mean",
            ),
        )
        .reset_index()
    )

    return (
        summary_by_stock,
        overall,
    )


# ============================================================
# WIN RATES
# ============================================================

def calculate_pairwise_win_rates(
    points: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compare model MAE at the ticker+horizon level.

    Lower MAE wins.

    Ties use a tolerance of 1e-12.
    """

    base = (
        points[
            [
                "ticker",
                "horizon",
                "model",
                "price_mae",
            ]
        ]
        .dropna(subset=["price_mae"])
    )

    pivot = (
        base
        .pivot_table(
            index=[
                "ticker",
                "horizon",
            ],
            columns="model",
            values="price_mae",
            aggfunc="mean",
        )
    )

    rows = []

    models = [
        c
        for c in pivot.columns
    ]

    for challenger in models:

        for baseline in models:

            if challenger == baseline:
                continue

            if (
                baseline not in pivot.columns
                or
                challenger not in pivot.columns
            ):
                continue

            valid = pivot[
                [
                    challenger,
                    baseline,
                ]
            ].dropna()

            if valid.empty:
                continue

            challenger_mae = (
                valid[
                    challenger
                ]
            )

            baseline_mae = (
                valid[
                    baseline
                ]
            )

            wins = (
                challenger_mae
                <
                baseline_mae - 1e-12
            ).sum()

            losses = (
                challenger_mae
                >
                baseline_mae + 1e-12
            ).sum()

            ties = (
                len(valid)
                -
                wins
                -
                losses
            )

            rows.append({
                "challenger":
                    challenger,
                "baseline":
                    baseline,
                "comparisons":
                    len(valid),
                "wins":
                    int(wins),
                "losses":
                    int(losses),
                "ties":
                    int(ties),
                "win_rate_excluding_ties":
                    (
                        wins
                        /
                        (wins + losses)
                        if wins + losses
                        else np.nan
                    ),
            })

    return pd.DataFrame(rows)


def calculate_win_rates_vs_garch(
    points: pd.DataFrame,
) -> pd.DataFrame:

    data = (
        points[
            [
                "ticker",
                "horizon",
                "model",
                "price_mae",
            ]
        ]
        .dropna(subset=["price_mae"])
    )

    pivot = (
        data
        .pivot_table(
            index=[
                "ticker",
                "horizon",
            ],
            columns="model",
            values="price_mae",
            aggfunc="mean",
        )
    )

    if "GARCH" not in pivot.columns:
        return pd.DataFrame()

    rows = []

    for model in [
        c
        for c in pivot.columns
        if c != "GARCH"
    ]:

        valid = pivot[
            [
                model,
                "GARCH",
            ]
        ].dropna()

        if valid.empty:
            continue

        challenger = valid[
            model
        ]

        baseline = valid[
            "GARCH"
        ]

        wins = (
            challenger
            <
            baseline - 1e-12
        ).sum()

        losses = (
            challenger
            >
            baseline + 1e-12
        ).sum()

        ties = (
            len(valid)
            -
            wins
            -
            losses
        )

        rows.append({
            "model":
                model,
            "comparisons":
                len(valid),
            "wins_vs_garch":
                int(wins),
            "losses_vs_garch":
                int(losses),
            "ties_vs_garch":
                int(ties),
            "win_rate_excluding_ties":
                (
                    wins
                    /
                    (wins + losses)
                    if wins + losses
                    else np.nan
                ),
        })

    return pd.DataFrame(rows)


def calculate_best_model_counts(
    points: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = (
        points[
            [
                "ticker",
                "horizon",
                "model",
                "price_mae",
            ]
        ]
        .dropna(subset=["price_mae"])
        .groupby([
            "ticker",
            "horizon",
        ])
    )

    for (
        ticker,
        horizon,
    ), group in grouped:

        best = group[
            "price_mae"
        ].min()

        winners = group[
            np.isclose(
                group[
                    "price_mae"
                ],
                best,
                rtol=0,
                atol=1e-12,
            )
        ]

        for model in winners[
            "model"
        ]:

            rows.append({
                "ticker":
                    ticker,
                "horizon":
                    horizon,
                "winner":
                    model,
            })

    if not rows:
        return pd.DataFrame()

    best = pd.DataFrame(rows)

    return (
        best
        .groupby("winner")
        .agg(
            wins=(
                "winner",
                "size",
            )
        )
        .reset_index()
        .sort_values(
            "wins",
            ascending=False,
        )
    )


# ============================================================
# PROBABILISTIC CALIBRATION
# ============================================================

def summarize_interval_calibration(
    intervals: pd.DataFrame,
) -> pd.DataFrame:

    if intervals.empty:
        return pd.DataFrame()

    # Lead-level calibration.
    lead_summary = (
        intervals
        .groupby([
            "model",
            "lead",
        ])
        .agg(
            observations=(
                "inside_80_interval",
                "size",
            ),
            empirical_coverage=(
                "inside_80_interval",
                "mean",
            ),
            lower_breach_rate=(
                "lower_breach",
                "mean",
            ),
            upper_breach_rate=(
                "upper_breach",
                "mean",
            ),
            mean_interval_width=(
                "interval_width",
                "mean",
            ),
        )
        .reset_index()
    )

    # Add target distance from nominal 80%.
    lead_summary[
        "coverage_error_vs_80pct"
    ] = (
        lead_summary[
            "empirical_coverage"
        ]
        -
        0.80
    )

    return lead_summary


def summarize_horizon_calibration(
    intervals: pd.DataFrame,
) -> pd.DataFrame:

    if intervals.empty:
        return pd.DataFrame()

    # These are averages of daily marginal coverage across
    # the first H forecast leads.
    rows = []

    for model, group in intervals.groupby(
        "model"
    ):

        for H in HORIZONS:

            subset = group[
                group["lead"] <= H
            ]

            if subset.empty:
                continue

            rows.append({
                "model":
                    model,
                "horizon":
                    H,
                "observations":
                    len(subset),
                "mean_marginal_coverage":
                    subset[
                        "inside_80_interval"
                    ].mean(),
                "lower_breach_rate":
                    subset[
                        "lower_breach"
                    ].mean(),
                "upper_breach_rate":
                    subset[
                        "upper_breach"
                    ].mean(),
                "mean_interval_width":
                    subset[
                        "interval_width"
                    ].mean(),
                "coverage_error_vs_80pct":
                    (
                        subset[
                            "inside_80_interval"
                        ].mean()
                        -
                        0.80
                    ),
            })

    return pd.DataFrame(rows)


# ============================================================
# SPIKE EVALUATION
# ============================================================

def calculate_spike_results(
    points: pd.DataFrame,
) -> pd.DataFrame:

    """
    Uses 1-step forecasts only.

    A Chronos warning is generated when its 1-step q10/q90
    return interval reaches beyond the recent 2-sigma threshold.

    The interval data is joined separately below.
    """

    return pd.DataFrame()


def evaluate_spikes_for_model(
    stock: pd.DataFrame,
    ticker: str,
    pipeline,
    model_name: str,
) -> pd.DataFrame:

    rows = []

    if pipeline is None:
        return pd.DataFrame()

    d = stock.copy()

    close = d[
        f"{ticker}_Close"
    ].astype(float)

    d["daily_return"] = (
        close.pct_change()
    )

    d["vol20"] = (
        d["daily_return"]
        .rolling(20)
        .std()
    )

    for cutoff in get_cutoffs(
        len(d)
    ):

        history = d.iloc[
            :cutoff
        ].copy()

        future = d.iloc[
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
            np.log(actual / last)
        )

        vol = float(
            history[
                "vol20"
            ].iloc[-1]
        )

        if (
            not np.isfinite(vol)
            or vol <= 0
        ):
            continue

        threshold = (
            SPIKE_SIGMA * vol
        )

        actual_spike = (
            abs(
                actual_return
            )
            >
            threshold
        )

        try:

            chrono = lora_predict(
                pipeline,
                history,
                ticker,
                horizon=1,
            )

            q10 = float(
                chrono["q10"][0]
            )

            q90 = float(
                chrono["q90"][0]
            )

            warning = (
                q10 < -threshold
                or
                q90 > threshold
            )

        except Exception as exc:

            warning = False
            q10 = np.nan
            q90 = np.nan

        rows.append({
            "ticker":
                ticker,
            "cutoff_date":
                history[
                    "Date"
                ].iloc[-1],
            "model":
                model_name,
            "actual_return":
                actual_return,
            "threshold":
                threshold,
            "actual_spike":
                actual_spike,
            "warning":
                warning,
            "q10":
                q10,
            "q90":
                q90,
        })

    return pd.DataFrame(rows)


def summarize_spikes(
    spike_df: pd.DataFrame,
) -> pd.DataFrame:

    if spike_df.empty:
        return pd.DataFrame()

    rows = []

    for (
        ticker,
        model,
    ), group in spike_df.groupby([
        "ticker",
        "model",
    ]):

        actual = (
            group[
                "actual_spike"
            ]
            .astype(bool)
            .to_numpy()
        )

        predicted = (
            group[
                "warning"
            ]
            .astype(bool)
            .to_numpy()
        )

        tp = int(
            np.sum(
                actual
                &
                predicted
            )
        )

        fp = int(
            np.sum(
                ~actual
                &
                predicted
            )
        )

        fn = int(
            np.sum(
                actual
                &
                ~predicted
            )
        )

        precision = (
            tp
            /
            (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp
            /
            (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
            if precision + recall
            else 0.0
        )

        rows.append({
            "ticker":
                ticker,
            "model":
                model,
            "actual_spikes":
                int(actual.sum()),
            "tp":
                tp,
            "fp":
                fp,
            "fn":
                fn,
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                f1,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "FINAL CHRONOS-2 LoRA V1/V2/V3/V4"
    )
    print(
        "VS STATISTICAL / ML MODELS"
    )
    print(
        "10-STOCK HELD-OUT EVALUATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Check checkpoints.
    # --------------------------------------------------------

    for name, path in (
        LORA_MODELS.items()
    ):

        print(
            f"{name}: {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing checkpoint:\n"
                f"{path.resolve()}"
            )

    all_points = []
    all_intervals = []
    all_fits = []
    all_spikes = []

    # --------------------------------------------------------
    # Evaluate each LoRA sequentially.
    #
    # We deliberately load only ONE adapter at a time to avoid
    # duplicating the Chronos base model four times in VRAM.
    # --------------------------------------------------------

    for lora_name, checkpoint in (
        LORA_MODELS.items()
    ):

        print()
        print("=" * 80)
        print(
            f"EVALUATING {lora_name}"
        )
        print("=" * 80)

        pipeline = (
            Chronos2Pipeline
            .from_pretrained(
                checkpoint,
                device_map=DEVICE,
                import_allowlist=[
                    "chronos.chronos2.model"
                ],
            )
        )

        for ticker in TICKERS:

            path = (
                DATA_DIR
                / f"{ticker}.csv"
            )

            if not path.exists():

                print(
                    f"Skipping {ticker}: "
                    f"{path} missing."
                )

                continue

            try:

                stock = load_stock(
                    ticker
                )

            except Exception as exc:

                print(
                    f"Skipping {ticker}: "
                    f"{exc}"
                )

                continue

            if len(stock) <= (
                MIN_HISTORY
                +
                max(HORIZONS)
            ):

                print(
                    f"Skipping {ticker}: "
                    "not enough history."
                )

                continue

            points, intervals, fits = (
                evaluate_stock(
                    stock,
                    ticker,
                    lora_name,
                    pipeline,
                )
            )

            if not points.empty:
                all_points.append(
                    points
                )

            if not intervals.empty:
                all_intervals.append(
                    intervals
                )

            if not fits.empty:
                all_fits.append(
                    fits
                )

            # Spike test.
            spike = evaluate_spikes_for_model(
                stock,
                ticker,
                pipeline,
                lora_name,
            )

            if not spike.empty:
                all_spikes.append(
                    spike
                )

        # Free adapter/base model before loading next one.
        del pipeline

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Combine.
    # --------------------------------------------------------

    if not all_points:

        raise RuntimeError(
            "No evaluations completed."
        )

    points = pd.concat(
        all_points,
        ignore_index=True,
    )

    points.to_csv(
        RESULTS_DIR
        /
        "all_point_forecasts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    by_stock, overall = (
        summarize_point_results(
            points
        )
    )

    by_stock.to_csv(
        RESULTS_DIR
        /
        "model_comparison_by_stock.csv",
        index=False,
    )

    overall.to_csv(
        RESULTS_DIR
        /
        "model_comparison_overall.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Win rates.
    # --------------------------------------------------------

    pairwise = (
        calculate_pairwise_win_rates(
            points
        )
    )

    pairwise.to_csv(
        RESULTS_DIR
        /
        "pairwise_win_rates.csv",
        index=False,
    )

    vs_garch = (
        calculate_win_rates_vs_garch(
            points
        )
    )

    vs_garch.to_csv(
        RESULTS_DIR
        /
        "win_rates_vs_garch.csv",
        index=False,
    )

    best_counts = (
        calculate_best_model_counts(
            points
        )
    )

    best_counts.to_csv(
        RESULTS_DIR
        /
        "best_model_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Interval calibration.
    # --------------------------------------------------------

    if all_intervals:

        intervals = pd.concat(
            all_intervals,
            ignore_index=True,
        )

        intervals.to_csv(
            RESULTS_DIR
            /
            "chronos_lora_intervals_raw.csv",
            index=False,
        )

        lead_calibration = (
            summarize_interval_calibration(
                intervals
            )
        )

        lead_calibration.to_csv(
            RESULTS_DIR
            /
            "chronos_lora_calibration_by_lead.csv",
            index=False,
        )

        horizon_calibration = (
            summarize_horizon_calibration(
                intervals
            )
        )

        horizon_calibration.to_csv(
            RESULTS_DIR
            /
            "chronos_lora_calibration_by_horizon.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Fit diagnostics.
    # --------------------------------------------------------

    if all_fits:

        fits = pd.concat(
            all_fits,
            ignore_index=True,
        )

        fits.to_csv(
            RESULTS_DIR
            /
            "statistical_model_fit_diagnostics.csv",
            index=False,
        )

        fit_summary = (
            fits
            .assign(
                success_flag=
                    fits["status"]
                    .astype(str)
                    .str.startswith(
                        "success"
                    )
            )
            .groupby("model")
            .agg(
                cutoffs=(
                    "success_flag",
                    "size",
                ),
                successful_fits=(
                    "success_flag",
                    "sum",
                ),
            )
            .reset_index()
        )

        fit_summary[
            "success_rate"
        ] = (
            fit_summary[
                "successful_fits"
            ]
            /
            fit_summary[
                "cutoffs"
            ]
        )

        fit_summary.to_csv(
            RESULTS_DIR
            /
            "statistical_model_fit_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Spike results.
    # --------------------------------------------------------

    if all_spikes:

        spikes = pd.concat(
            all_spikes,
            ignore_index=True,
        )

        spikes.to_csv(
            RESULTS_DIR
            /
            "lora_spike_results.csv",
            index=False,
        )

        spike_summary = summarize_spikes(
            spikes
        )

        spike_summary.to_csv(
            RESULTS_DIR
            /
            "lora_spike_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Console report.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FINAL OVERALL RESULTS"
    )
    print("=" * 80)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print()
    print("=" * 80)
    print(
        "WIN RATES VS GARCH"
    )
    print("=" * 80)

    print(
        vs_garch.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    if all_intervals:

        print()
        print("=" * 80)
        print(
            "CHRONOS LoRA MARGINAL INTERVAL CALIBRATION"
        )
        print("=" * 80)

        print(
            horizon_calibration.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

    if all_fits:

        print()
        print("=" * 80)
        print(
            "STATISTICAL MODEL FIT SUCCESS"
        )
        print("=" * 80)

        print(
            fit_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Results saved to:",
        RESULTS_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
