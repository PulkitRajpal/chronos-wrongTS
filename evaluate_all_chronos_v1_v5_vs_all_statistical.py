from __future__ import annotations

"""
FINAL CHRONOS vs STATISTICAL / ML BENCHMARK
===========================================

Chronos:
    - Chronos-2 pretrained / zero-shot
    - Chronos-LoRA-V1
    - Chronos-LoRA-V2
    - Chronos-LoRA-V3
    - Chronos-LoRA-V4
    - Chronos-LoRA-V5 (multivariate with past-only covariates)

Statistical / ML:
    - Naive
    - ARIMA
    - ETS
    - GARCH
    - EGARCH
    - GJR-GARCH
    - VAR
    - XGBoost
    - LightGBM

Held-out stocks:
    AAPL, MSFT, NVDA, AMZN, ENPH, SMCI, CVNA, PLUG, RKLB, IONQ

Horizons:
    1 / 5 / 10 / 20 trading sessions

Evaluation:
    - endpoint price MAE
    - return absolute error
    - directional accuracy
    - Chronos q10-q90 marginal calibration
    - model win rates vs GARCH
    - pairwise win rates
    - best-model counts
    - statistical-model fit success

IMPORTANT V5 DETAIL
-------------------
V5 was trained with:
    target = stock daily log return
    past-only covariates =
        SPY_RET
        QQQ_RET
        VIX_RET
        US5Y_CHG
        US10Y_CHG
        realized_vol_20
        volume_z

For V5 inference we pass those columns directly in `context_df` and
leave `future_df=None`. Chronos-2 treats remaining context columns as
past-only covariates.

To avoid irregular trading-day timestamps causing validation problems,
the V5 context uses a synthetic regular daily timestamp sequence while
preserving the original observation order.

The return-only Chronos models use the same synthetic regular timestamps.

This script is designed around your main POC question:

    "Does Chronos add value versus established statistical / ML models?"

It is NOT intended to prove that Chronos replaces any production risk
model.
"""

import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from chronos import Chronos2Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ETS_AVAILABLE = True
except ImportError:
    ETS_AVAILABLE = False

try:
    from statsmodels.tsa.api import VAR
    VAR_AVAILABLE = True
except ImportError:
    VAR_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    / "results_final_chronos_vs_statistical"
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

# Use 256 to replicate the return-only V1-V4 evaluation setup.
# V5 itself was trained with 512. We evaluate V5 with 512.
RETURN_ONLY_CONTEXT = 256
V5_CONTEXT = 512
# Minimum valid V5 context. Early walk-forward windows can be shorter than
# 512 after aligning stock and reference-market calendars; those windows use
# all available aligned history rather than being discarded.
V5_MIN_CONTEXT = 10

LOWER_Q = 0.10
MEDIAN_Q = 0.50
UPPER_Q = 0.90

SPIKE_SIGMA = 2.0
ML_LOOKBACK = 252

PRETRAINED_MODEL = "amazon/chronos-2"

CHECKPOINTS = {
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

    "Chronos-LoRA-V5":
        BASE_DIR
        / "models"
        / "chronos2-finance-lora-v5-multivariate"
        / "finetuned-ckpt",
}

EVAL_TICKERS = set(TICKERS)

V5_COVARIATES = [
    "SPY_RET",
    "QQQ_RET",
    "VIX_RET",
    "US5Y_CHG",
    "US10Y_CHG",
    "realized_vol_20",
    "volume_z",
]

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
        print("CUDA unavailable -> CPU")
        return "cpu"

    try:
        torch.randn(
            1,
            device="cuda",
        )

        print(
            "Using CUDA:",
            torch.cuda.get_device_name(0),
        )

        props = torch.cuda.get_device_properties(0)

        print(
            "GPU memory:",
            round(
                props.total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
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
# BASIC DATA HELPERS
# ============================================================

def clean_number(value) -> float:

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


def find_col(
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
            f"Missing one of {candidates}; "
            f"available={list(df.columns)}"
        )

    return None


def load_stock(
    ticker: str,
) -> pd.DataFrame:

    path = (
        DATA_DIR
        /
        f"{ticker}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    date_col = find_col(
        raw,
        [
            "date",
            "datetime",
            "timestamp",
        ],
    )

    close_col = find_col(
        raw,
        [
            "close/last",
            "close",
            "last",
            "price",
        ],
    )

    open_col = find_col(
        raw,
        ["open"],
        required=False,
    )

    high_col = find_col(
        raw,
        ["high"],
        required=False,
    )

    low_col = find_col(
        raw,
        ["low"],
        required=False,
    )

    volume_col = find_col(
        raw,
        ["volume"],
        required=False,
    )

    result = pd.DataFrame({

        "Date":
            pd.to_datetime(
                raw[date_col],
                errors="coerce",
            ),

        "Close":
            raw[close_col].map(
                clean_number
            ),

        "Open":
            (
                raw[open_col].map(
                    clean_number
                )
                if open_col
                else np.nan
            ),

        "High":
            (
                raw[high_col].map(
                    clean_number
                )
                if high_col
                else np.nan
            ),

        "Low":
            (
                raw[low_col].map(
                    clean_number
                )
                if low_col
                else np.nan
            ),

        "Volume":
            (
                raw[volume_col].map(
                    clean_number
                )
                if volume_col
                else np.nan
            ),
    })

    return (
        result
        .dropna(
            subset=[
                "Date",
                "Close",
            ]
        )
        .sort_values("Date")
        .drop_duplicates(
            "Date",
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# REFERENCE DATA FOR V5 / VAR
# ============================================================

def load_reference(
    name: str,
    ticker: str,
) -> pd.DataFrame:

    for path in [
        DATA_DIR / f"{name}.csv",
        DATA_DIR / f"{ticker}.csv",
    ]:

        if not path.exists():
            continue

        try:

            raw = pd.read_csv(path)

            date_col = find_col(
                raw,
                [
                    "date",
                    "datetime",
                    "timestamp",
                ],
            )

            value_col = find_col(
                raw,
                [
                    "close/last",
                    "close",
                    "last",
                    "price",
                    "value",
                ],
            )

            out = pd.DataFrame({
                "Date":
                    pd.to_datetime(
                        raw[date_col],
                        errors="coerce",
                    ),

                "Value":
                    raw[value_col].map(
                        clean_number
                    ),
            })

            out = out.reset_index(
                drop=True
            )

            return (
                out
                .dropna(
                    subset=[
                        "Date",
                        "Value",
                    ]
                )
                .sort_values(
                    by="Date"
                )
                .drop_duplicates(
                    subset=["Date"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

        except Exception:
            continue

    raw = yf.download(
        ticker,
        start="2010-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"No reference data for {name}"
        )

    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):
        raw.columns = (
            raw.columns
            .get_level_values(0)
        )

    close_col = find_col(
        raw,
        ["close"],
    )

    out = pd.DataFrame({
        "Date":
            pd.to_datetime(
                raw.index
            ),

        "Value":
            pd.to_numeric(
                raw[close_col],
                errors="coerce",
            ),
    })

    out = out.reset_index(
        drop=True
    )

    return (
        out
        .dropna(
            subset=[
                "Date",
                "Value",
            ]
        )
        .sort_values(
            by="Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_references() -> pd.DataFrame:

    specs = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "VIX": "^VIX",
        "US5Y": "^FVX",
        "US10Y": "^TNX",
    }

    pieces = []

    for name, ticker in specs.items():

        print(
            f"Loading reference "
            f"{name} ({ticker})"
        )

        x = load_reference(
            name,
            ticker,
        )

        pieces.append(
            x.rename(
                columns={
                    "Value":
                        name
                }
            )
        )

    out = pieces[0]

    for x in pieces[1:]:
        out = out.merge(
            x,
            on="Date",
            how="inner",
        )

    return (
        out
        .sort_values("Date")
        .reset_index(drop=True)
    )


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
            f"Not enough history: "
            f"{n_rows} <= {required}"
        )

    possible = np.arange(
        MIN_HISTORY,
        n_rows - max(HORIZONS),
    )

    if len(possible) <= N_WINDOWS:

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
) -> float:

    return float(
        history["Close"].iloc[-1]
    )


def fit_arima(
    history: pd.DataFrame,
):

    if not ARIMA_AVAILABLE:
        return None, "unavailable"

    log_price = np.log(
        history["Close"].astype(float)
    )

    last_error = ""

    for opts in [
        {
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        {
            "enforce_stationarity": False,
            "enforce_invertibility": False,
        },
    ]:

        try:

            model = ARIMA(
                log_price,
                order=(1, 1, 1),
                **opts,
            )

            fitted = model.fit(
                method_kwargs={
                    "maxiter": 500,
                }
            )

            return fitted, "success"

        except Exception as exc:
            last_error = str(exc)

    return (
        None,
        f"failed:{last_error}",
    )


def arima_forecast(
    fitted,
    horizon: int,
) -> float:

    fc = fitted.forecast(
        steps=horizon
    )

    return float(
        np.exp(
            fc.iloc[-1]
        )
    )


def fit_ets(
    history: pd.DataFrame,
):

    if not ETS_AVAILABLE:
        return None, "unavailable"

    try:

        model = ExponentialSmoothing(
            history["Close"].astype(float),
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        )

        fitted = model.fit(
            optimized=True
        )

        return fitted, "success"

    except Exception as exc:

        return (
            None,
            f"failed:{exc}",
        )


def ets_forecast(
    fitted,
    horizon: int,
) -> float:

    fc = fitted.forecast(
        horizon
    )

    return float(
        fc.iloc[-1]
    )


def fit_arch(
    history: pd.DataFrame,
    variant: str,
):

    if not ARCH_AVAILABLE:
        return None, "unavailable"

    returns = (
        np.log(
            history["Close"]
        )
        .diff()
        .dropna()
        * 100.0
    )

    last_error = ""

    for dist in [
        "t",
        "normal",
    ]:

        try:

            if variant == "GARCH":

                model = arch_model(
                    returns,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    o=0,
                    q=1,
                    dist=dist,
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
                    dist=dist,
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
                    dist=dist,
                    rescale=False,
                )

            else:
                raise ValueError(variant)

            fitted = model.fit(
                disp="off",
                update_freq=0,
                options={
                    "maxiter": 2000,
                },
            )

            flag = getattr(
                fitted,
                "convergence_flag",
                0,
            )

            if flag == 0:
                return fitted, "success"

            last_error = (
                f"nonconverged:{flag}"
            )

        except Exception as exc:

            last_error = str(exc)

    return (
        None,
        f"failed:{last_error}",
    )


def arch_forecast(
    fitted,
    history: pd.DataFrame,
    horizon: int,
) -> float:

    fc = fitted.forecast(
        horizon=horizon,
        reindex=False,
    )

    mean_path = (
        fc.mean
        .iloc[-1]
        .to_numpy()
    )

    cumulative_log_return = (
        mean_path.sum()
        /
        100.0
    )

    last_price = float(
        history["Close"].iloc[-1]
    )

    return float(
        last_price
        *
        np.exp(
            cumulative_log_return
        )
    )


# ------------------------------------------------------------
# ML FEATURES
# ------------------------------------------------------------

ML_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "hl_range",
    "oc_return",
    "volume_z",
    "vol_5",
    "vol_20",
    "vol_60",
    "mom_5",
    "mom_20",
    "mom_60",
]


def build_ml_features(
    history: pd.DataFrame,
) -> pd.DataFrame:

    d = history.copy()

    close = d["Close"].astype(float)

    d["ret_1"] = close.pct_change(1)
    d["ret_2"] = close.pct_change(2)
    d["ret_3"] = close.pct_change(3)
    d["ret_5"] = close.pct_change(5)
    d["ret_10"] = close.pct_change(10)
    d["ret_20"] = close.pct_change(20)

    d["hl_range"] = (
        (
            d["High"]
            -
            d["Low"]
        )
        /
        close
    )

    d["oc_return"] = (
        (
            d["Close"]
            -
            d["Open"]
        )
        /
        d["Open"]
    )

    log_volume = np.log1p(
        d["Volume"].clip(lower=0)
    )

    d["volume_z"] = (
        (
            log_volume
            -
            log_volume.rolling(20).mean()
        )
        /
        log_volume.rolling(20).std()
    )

    d["vol_5"] = (
        d["ret_1"]
        .rolling(5)
        .std()
    )

    d["vol_20"] = (
        d["ret_1"]
        .rolling(20)
        .std()
    )

    d["vol_60"] = (
        d["ret_1"]
        .rolling(60)
        .std()
    )

    d["mom_5"] = (
        close / close.shift(5)
        - 1.0
    )

    d["mom_20"] = (
        close / close.shift(20)
        - 1.0
    )

    d["mom_60"] = (
        close / close.shift(60)
        - 1.0
    )

    return d


def train_xgb(
    history: pd.DataFrame,
    horizon: int,
):

    if not XGB_AVAILABLE:
        raise RuntimeError(
            "xgboost unavailable"
        )

    d = build_ml_features(
        history
    )

    d["target_return"] = (
        d["Close"].shift(-horizon)
        /
        d["Close"]
        -
        1.0
    )

    d = (
        d
        .dropna(
            subset=ML_FEATURES
            + ["target_return"]
        )
        .tail(ML_LOOKBACK)
    )

    X = d[
        ML_FEATURES
    ].astype(float)

    y = d[
        "target_return"
    ].astype(float)

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

    model.fit(X, y)

    return model


def xgb_forecast(
    history: pd.DataFrame,
    model,
) -> float:

    d = build_ml_features(
        history
    )

    X = (
        d.iloc[-1][ML_FEATURES]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Invalid XGBoost features"
        )

    pred_return = float(
        model.predict(X)[0]
    )

    return float(
        history["Close"].iloc[-1]
        *
        (1.0 + pred_return)
    )


def train_lgbm(
    history: pd.DataFrame,
    horizon: int,
):

    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(
            "lightgbm unavailable"
        )

    d = build_ml_features(
        history
    )

    d["target_return"] = (
        d["Close"].shift(-horizon)
        /
        d["Close"]
        -
        1.0
    )

    d = (
        d
        .dropna(
            subset=ML_FEATURES
            + ["target_return"]
        )
        .tail(ML_LOOKBACK)
    )

    X = d[
        ML_FEATURES
    ].astype(float)

    y = d[
        "target_return"
    ].astype(float)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X, y)

    return model


def lgbm_forecast(
    history: pd.DataFrame,
    model,
) -> float:

    d = build_ml_features(
        history
    )

    X = (
        d.iloc[-1][ML_FEATURES]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Invalid LightGBM features"
        )

    pred_return = float(
        model.predict(X)[0]
    )

    return float(
        history["Close"].iloc[-1]
        *
        (1.0 + pred_return)
    )


def var_forecast(
    history: pd.DataFrame,
    horizon: int,
) -> float:

    if not VAR_AVAILABLE:

        raise RuntimeError(
            "VAR unavailable"
        )

    required = [
        "SPX",
        "VIX",
    ]

    if not all(
        c in history.columns
        for c in required
    ):

        raise RuntimeError(
            "SPX/VIX unavailable for VAR"
        )

    d = history[
        [
            "Close",
            "SPX",
            "VIX",
        ]
    ].copy()

    returns = (
        np.log(d)
        .diff()
        .dropna()
        .tail(ML_LOOKBACK)
    )

    if len(returns) < 100:

        raise ValueError(
            "Not enough VAR history"
        )

    fitted = VAR(
        returns
    ).fit(5)

    fc = fitted.forecast(
        returns.values[
            -fitted.k_ar:
        ],
        steps=horizon,
    )

    return float(
        history["Close"].iloc[-1]
        *
        np.exp(
            fc[:, 0].sum()
        )
    )


# ============================================================
# CHRONOS RETURN-ONLY
# ============================================================

def make_return_input(
    history: pd.DataFrame,
):
    """
    Build the low-level Chronos-2 input for return-only inference.

    We intentionally use predict_quantiles() rather than predict_df().
    This removes timestamp/frequency inference from the evaluation path.
    """

    returns = (
        np.log(
            history["Close"]
        )
        .diff()
        .dropna()
        .to_numpy(
            dtype=np.float32
        )
    )

    returns = returns[
        -RETURN_ONLY_CONTEXT:
    ]

    if len(returns) < 10:
        raise ValueError(
            f"Only {len(returns)} returns available."
        )

    return [
        {
            "target": returns
        }
    ]


def return_only_predict(
    pipeline,
    history: pd.DataFrame,
    horizon: int,
) -> dict:

    inputs = make_return_input(
        history
    )

    quantiles, mean = (
        pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=[
                LOWER_Q,
                MEDIAN_Q,
                UPPER_Q,
            ],
            context_length=RETURN_ONLY_CONTEXT,
            limit_prediction_length=False,
        )
    )

    q = (
        quantiles[0]
        .detach()
        .cpu()
        .numpy()
    )

    # Shape: [n_variates=1, horizon, n_quantiles]
    q = q[0]

    return {
        "q10":
            q[:, 0].astype(float),

        "q50":
            q[:, 1].astype(float),

        "q90":
            q[:, 2].astype(float),
    }


# ============================================================
# V5 COVARIATE INPUT
# ============================================================

V5_COVARIATES = [
    "SPY_RET",
    "QQQ_RET",
    "VIX_RET",
    "US5Y_CHG",
    "US10Y_CHG",
    "realized_vol_20",
    "volume_z",
]


def build_v5_features(
    history: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:

    d = history.copy()

    d["target"] = (
        np.log(
            d["Close"]
        )
        .diff()
    )

    d["realized_vol_20"] = (
        d["target"]
        .rolling(20)
        .std()
    )

    log_volume = np.log1p(
        d["Volume"].clip(
            lower=0
        )
    )

    d["volume_z"] = (
        (
            log_volume
            -
            log_volume.rolling(20).mean()
        )
        /
        log_volume.rolling(20).std()
    )

    d = (
        d[
            [
                "Date",
                "target",
                "realized_vol_20",
                "volume_z",
            ]
        ]
        .merge(
            references,
            on="Date",
            how="inner",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    d["SPY_RET"] = (
        np.log(d["SPY"])
        .diff()
    )

    d["QQQ_RET"] = (
        np.log(d["QQQ"])
        .diff()
    )

    d["VIX_RET"] = (
        np.log(d["VIX"])
        .diff()
    )

    d["US5Y_CHG"] = (
        d["US5Y"].diff()
    )

    d["US10Y_CHG"] = (
        d["US10Y"].diff()
    )

    return (
        d[
            [
                "Date",
                "target",
                *V5_COVARIATES,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )


def make_v5_input(
    history: pd.DataFrame,
    references: pd.DataFrame,
    ticker: str,
) -> dict:

    d = build_v5_features(
        history,
        references,
    )

    if len(d) < V5_MIN_CONTEXT:
        raise ValueError(
            f"{ticker}: only {len(d)} aligned observations; "
            f"minimum is {V5_MIN_CONTEXT}."
        )

    # IMPORTANT:
    # Do not require 512 observations.
    # Chronos-2 supports variable history lengths. We cap at the
    # training context length but use a shorter valid context for
    # early walk-forward windows where the stock and references have
    # not yet overlapped for 512 observations.
    d = d.tail(V5_CONTEXT).copy()

    target = (
        d["target"]
        .to_numpy(
            dtype=np.float32
        )
    )

    past_covariates = {
        column:
            d[column]
            .to_numpy(
                dtype=np.float32
            )
        for column in V5_COVARIATES
    }

    return {
        "target":
            target,

        "past_covariates":
            past_covariates,
    }


def v5_predict(
    pipeline,
    history: pd.DataFrame,
    references: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> dict:

    inputs = [
        make_v5_input(
            history,
            references,
            ticker,
        )
    ]

    quantiles, mean = (
        pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=[
                LOWER_Q,
                MEDIAN_Q,
                UPPER_Q,
            ],
            # Let the model use the entire supplied context,
            # capped at its supported V5 training context.
            context_length=V5_CONTEXT,
            limit_prediction_length=False,
        )
    )

    q = (
        quantiles[0]
        .detach()
        .cpu()
        .numpy()
    )

    # Shape [1, horizon, 3] for one target.
    q = q[0]

    return {
        "q10":
            q[:, 0].astype(float),

        "q50":
            q[:, 1].astype(float),

        "q90":
            q[:, 2].astype(float),
    }


# ============================================================
# SCORE
# ============================================================

def score(
    actual_price: float,
    last_price: float,
    predicted_price: float,
):

    if not np.isfinite(
        predicted_price
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mae = abs(
        actual_price
        -
        predicted_price
    )

    actual_return = (
        actual_price
        /
        last_price
        -
        1.0
    )

    predicted_return = (
        predicted_price
        /
        last_price
        -
        1.0
    )

    return (
        mae,
        abs(
            actual_return
            -
            predicted_return
        ),
        float(
            np.sign(actual_return)
            ==
            np.sign(predicted_return)
        ),
    )


# ============================================================
# STATISTICAL WALK-FORWARD
# ============================================================

def evaluate_statistical_stock(
    stock: pd.DataFrame,
    ticker: str,
    references: pd.DataFrame,
):

    rows = []
    fit_rows = []

    # Add reference series for VAR.
    merged = (
        stock
        .merge(
            references,
            on="Date",
            how="left",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    cutoffs = get_cutoffs(
        len(merged)
    )

    print(
        f"\n{ticker} / statistical: "
        f"{len(cutoffs)} windows"
    )

    for w, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = merged.iloc[
            :cutoff
        ].copy()

        if (
            w == 1
            or
            w == len(cutoffs)
            or
            w % 10 == 0
        ):

            print(
                f"  {w}/{len(cutoffs)}"
            )

        last_price = float(
            history["Close"].iloc[-1]
        )

        # Fit once for each cutoff.
        arima_model, arima_status = (
            fit_arima(history)
        )

        ets_model, ets_status = (
            fit_ets(history)
        )

        arch_models = {}

        for variant in [
            "GARCH",
            "EGARCH",
            "GJR-GARCH",
        ]:

            model, status = fit_arch(
                history,
                variant,
            )

            arch_models[
                variant
            ] = (
                model,
                status,
            )

        fit_rows.extend([
            {
                "ticker":
                    ticker,
                "cutoff_date":
                    history[
                        "Date"
                    ].iloc[-1],
                "model":
                    "ARIMA",
                "status":
                    arima_status,
            },
            {
                "ticker":
                    ticker,
                "cutoff_date":
                    history[
                        "Date"
                    ].iloc[-1],
                "model":
                    "ETS",
                "status":
                    ets_status,
            },
            *[
                {
                    "ticker":
                        ticker,
                    "cutoff_date":
                        history[
                            "Date"
                        ].iloc[-1],
                    "model":
                        variant,
                    "status":
                        status,
                }
                for variant, (
                    _model,
                    status,
                )
                in arch_models.items()
            ],
        ])

        for horizon in HORIZONS:

            future = merged.iloc[
                cutoff:
                cutoff + horizon
            ]

            if len(future) < horizon:
                continue

            actual = float(
                future["Close"].iloc[-1]
            )

            predictions = {}

            # Naive
            predictions[
                "Naive"
            ] = naive_forecast(
                history
            )

            # ARIMA
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

            # ETS
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

            # GARCH / EGARCH / GJR
            for variant in [
                "GARCH",
                "EGARCH",
                "GJR-GARCH",
            ]:

                model, _ = arch_models[
                    variant
                ]

                if model is None:

                    predictions[
                        variant
                    ] = np.nan

                    continue

                try:

                    predictions[
                        variant
                    ] = arch_forecast(
                        model,
                        history,
                        horizon,
                    )

                except Exception:

                    predictions[
                        variant
                    ] = np.nan

            # VAR
            predictions[
                "VAR"
            ] = np.nan

            try:

                predictions[
                    "VAR"
                ] = var_forecast(
                    history,
                    horizon,
                )

            except Exception:

                pass

            # XGBoost
            predictions[
                "XGBoost"
            ] = np.nan

            try:

                xgb_model = train_xgb(
                    history,
                    horizon,
                )

                predictions[
                    "XGBoost"
                ] = xgb_forecast(
                    history,
                    xgb_model,
                )

            except Exception:

                pass

            # LightGBM
            predictions[
                "LightGBM"
            ] = np.nan

            try:

                lgb_model = train_lgbm(
                    history,
                    horizon,
                )

                predictions[
                    "LightGBM"
                ] = lgbm_forecast(
                    history,
                    lgb_model,
                )

            except Exception:

                pass

            for model_name, prediction in (
                predictions.items()
            ):

                mae, ret_err, direction = (
                    score(
                        actual,
                        last_price,
                        prediction,
                    )
                )

                rows.append({
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
                        actual,
                    "last_price":
                        last_price,
                    "prediction":
                        prediction,
                    "price_mae":
                        mae,
                    "return_abs_error":
                        ret_err,
                    "direction_hit":
                        direction,
                })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(fit_rows),
    )


# ============================================================
# CHRONOS WALK-FORWARD
# ============================================================

def evaluate_chronos_stock(
    model_name: str,
    pipeline,
    stock: pd.DataFrame,
    ticker: str,
    references: pd.DataFrame,
):

    rows = []
    interval_rows = []

    cutoffs = get_cutoffs(
        len(stock)
    )

    print(
        f"\n{ticker} / {model_name}: "
        f"{len(cutoffs)} windows"
    )

    for w, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = stock.iloc[
            :cutoff
        ].copy()

        if (
            w == 1
            or
            w == len(cutoffs)
            or
            w % 10 == 0
        ):

            print(
                f"  {w}/{len(cutoffs)}"
            )

        last_price = float(
            history["Close"].iloc[-1]
        )

        for horizon in HORIZONS:

            future = stock.iloc[
                cutoff:
                cutoff + horizon
            ]

            if len(future) < horizon:
                continue

            actual = float(
                future["Close"].iloc[-1]
            )

            try:

                if model_name == (
                    "Chronos-LoRA-V5"
                ):

                    cp = v5_predict(
                        pipeline,
                        history,
                        references,
                        ticker,
                        horizon,
                    )

                else:

                    cp = return_only_predict(
                        pipeline,
                        history,
                        horizon,
                    )

                predicted = float(
                    last_price
                    *
                    np.exp(
                        cp["q50"].sum()
                    )
                )

                mae, ret_err, direction = (
                    score(
                        actual,
                        last_price,
                        predicted,
                    )
                )

                model_type = (
                    "chronos_pretrained"
                    if model_name
                    == "Chronos-2"
                    else "chronos_lora"
                )

                rows.append({
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
                        model_type,
                    "actual":
                        actual,
                    "last_price":
                        last_price,
                    "prediction":
                        predicted,
                    "price_mae":
                        mae,
                    "return_abs_error":
                        ret_err,
                    "direction_hit":
                        direction,
                })

                # Lead-by-lead marginal calibration.
                combined = np.concatenate([
                    np.array(
                        [
                            last_price
                        ]
                    ),
                    future[
                        "Close"
                    ]
                    .astype(float)
                    .to_numpy(),
                ])

                realized = np.diff(
                    np.log(combined)
                )

                for lead in range(
                    1,
                    horizon + 1,
                ):

                    j = lead - 1

                    actual_return = float(
                        realized[j]
                    )

                    q10 = float(
                        cp["q10"][j]
                    )

                    q50 = float(
                        cp["q50"][j]
                    )

                    q90 = float(
                        cp["q90"][j]
                    )

                    interval_rows.append({
                        "ticker":
                            ticker,
                        "cutoff_date":
                            history[
                                "Date"
                            ].iloc[-1],
                        "model":
                            model_name,
                        "horizon":
                            horizon,
                        "lead":
                            lead,
                        "realized_return":
                            actual_return,
                        "q10":
                            q10,
                        "q50":
                            q50,
                        "q90":
                            q90,
                        "inside_80":
                            float(
                                q10
                                <=
                                actual_return
                                <=
                                q90
                            ),
                        "lower_breach":
                            float(
                                actual_return
                                <
                                q10
                            ),
                        "upper_breach":
                            float(
                                actual_return
                                >
                                q90
                            ),
                        "interval_width":
                            q90 - q10,
                    })

            except Exception as exc:

                model_type = (
                    "chronos_pretrained"
                    if model_name
                    == "Chronos-2"
                    else "chronos_lora"
                )

                rows.append({
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
                        model_type,
                    "actual":
                        actual,
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
                })

                print(
                    f"    WARNING "
                    f"{model_name} "
                    f"h={horizon}: "
                    f"{exc}"
                )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(interval_rows),
    )


# ============================================================
# SUMMARIES
# ============================================================

def summarize_points(
    points: pd.DataFrame,
):

    by_stock = (
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
        by_stock
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
        by_stock,
        overall,
    )


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
        .dropna(
            subset=[
                "price_mae"
            ]
        )
    )

    rows = []

    for (
        model,
        horizon,
    ), group in data.groupby([
        "model",
        "horizon",
    ]):

        if model == "GARCH":
            continue

        baseline = data[
            (
                data["model"]
                == "GARCH"
            )
            &
            (
                data["horizon"]
                == horizon
            )
        ][
            [
                "ticker",
                "price_mae",
            ]
        ]

        merged = group[
            [
                "ticker",
                "price_mae",
            ]
        ].merge(
            baseline,
            on="ticker",
            suffixes=(
                "_model",
                "_garch",
            ),
        )

        if merged.empty:
            continue

        wins = (
            merged["price_mae_model"]
            <
            merged["price_mae_garch"]
        ).sum()

        losses = (
            merged["price_mae_model"]
            >
            merged["price_mae_garch"]
        ).sum()

        ties = (
            len(merged)
            -
            wins
            -
            losses
        )

        rows.append({
            "model":
                model,
            "horizon":
                horizon,
            "comparisons":
                len(merged),
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


def calculate_pairwise_win_rates(
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
        .dropna(
            subset=[
                "price_mae"
            ]
        )
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

    rows = []

    models = list(
        pivot.columns
    )

    for i, model_a in enumerate(
        models
    ):

        for model_b in models[
            i + 1:
        ]:

            valid = pivot[
                [
                    model_a,
                    model_b,
                ]
            ].dropna()

            if valid.empty:
                continue

            a_wins = (
                valid[model_a]
                <
                valid[model_b]
            ).sum()

            b_wins = (
                valid[model_b]
                <
                valid[model_a]
            ).sum()

            ties = (
                len(valid)
                -
                a_wins
                -
                b_wins
            )

            rows.append({
                "model_a":
                    model_a,
                "model_b":
                    model_b,
                "comparisons":
                    len(valid),
                "model_a_wins":
                    int(a_wins),
                "model_b_wins":
                    int(b_wins),
                "ties":
                    int(ties),
            })

    return pd.DataFrame(rows)


def calculate_best_model_counts(
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
        .dropna(
            subset=[
                "price_mae"
            ]
        )
        .groupby([
            "ticker",
            "horizon",
        ])
    )

    winners = []

    for (
        ticker,
        horizon,
    ), group in data:

        minimum = group[
            "price_mae"
        ].min()

        winning = group[
            np.isclose(
                group[
                    "price_mae"
                ],
                minimum,
                rtol=0,
                atol=1e-12,
            )
        ]

        for model in winning[
            "model"
        ]:

            winners.append({
                "ticker":
                    ticker,
                "horizon":
                    horizon,
                "winner":
                    model,
            })

    if not winners:
        return pd.DataFrame()

    return (
        pd.DataFrame(winners)
        .groupby("winner")
        .size()
        .reset_index(
            name="wins"
        )
        .sort_values(
            "wins",
            ascending=False,
        )
    )


def summarize_calibration(
    intervals: pd.DataFrame,
):

    if intervals.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    by_lead = (
        intervals
        .groupby([
            "model",
            "lead",
        ])
        .agg(
            observations=(
                "inside_80",
                "size",
            ),
            empirical_coverage=(
                "inside_80",
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

    by_lead[
        "coverage_error_vs_80pct"
    ] = (
        by_lead[
            "empirical_coverage"
        ]
        -
        0.80
    )

    horizon_rows = []

    for model, group in (
        intervals.groupby("model")
    ):

        for horizon in HORIZONS:

            x = group[
                group["lead"]
                <=
                horizon
            ]

            if x.empty:
                continue

            horizon_rows.append({
                "model":
                    model,
                "horizon":
                    horizon,
                "mean_marginal_coverage":
                    x[
                        "inside_80"
                    ].mean(),
                "lower_breach_rate":
                    x[
                        "lower_breach"
                    ].mean(),
                "upper_breach_rate":
                    x[
                        "upper_breach"
                    ].mean(),
                "mean_interval_width":
                    x[
                        "interval_width"
                    ].mean(),
                "coverage_error_vs_80pct":
                    (
                        x[
                            "inside_80"
                        ].mean()
                        -
                        0.80
                    ),
            })

    return (
        by_lead,
        pd.DataFrame(
            horizon_rows
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "FINAL CHRONOS vs STATISTICAL / ML BENCHMARK"
    )
    print("=" * 90)

    print()
    print("Chronos:")
    print("  Chronos-2")
    for model_name in CHECKPOINTS:
        print(f"  {model_name}")

    print()
    print("Statistical / ML:")
    for model_name in STAT_MODELS:
        print(f"  {model_name}")

    print()
    print(
        f"Held-out stocks: {len(TICKERS)}"
    )
    print(
        f"Horizons: {HORIZONS}"
    )
    print(
        f"Walk-forward windows: {N_WINDOWS}"
    )

    # --------------------------------------------------------
    # Check checkpoints.
    # --------------------------------------------------------

    for model_name, path in (
        CHECKPOINTS.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"{model_name} checkpoint "
                f"not found:\n"
                f"{path.resolve()}"
            )

    # --------------------------------------------------------
    # Reference data.
    # --------------------------------------------------------

    print()
    print(
        "=" * 90
    )
    print(
        "LOADING REFERENCE SERIES"
    )
    print(
        "=" * 90
    )

    references = load_references()

    print(
        f"Reference rows: "
        f"{len(references):,}"
    )

    all_points = []
    all_intervals = []
    all_fits = []

    # --------------------------------------------------------
    # Chronos models.
    # --------------------------------------------------------

    chronos_models = [
        (
            "Chronos-2",
            PRETRAINED_MODEL,
        ),
        *CHECKPOINTS.items(),
    ]

    for model_name, model_path in (
        chronos_models
    ):

        print()
        print(
            "=" * 90
        )
        print(
            f"LOADING {model_name}"
        )
        print(
            "=" * 90
        )

        if model_name == "Chronos-2":

            pipeline = (
                Chronos2Pipeline
                .from_pretrained(
                    model_path,
                    device_map=DEVICE,
                )
            )

        else:

            pipeline = (
                Chronos2Pipeline
                .from_pretrained(
                    model_path,
                    device_map=DEVICE,
                    import_allowlist=[
                        "chronos.chronos2.model"
                    ],
                )
            )

        # V5 preflight: verify the covariate API before spending hundreds
        # of walk-forward calls. This deliberately uses low-level
        # predict_quantiles() and therefore avoids timestamp frequency
        # inference.
        if model_name == "Chronos-LoRA-V5":
            try:
                test_ticker = TICKERS[0]
                test_stock = load_stock(test_ticker)
                test_history = test_stock.iloc[:min(
                    len(test_stock),
                    400,
                )].copy()

                _ = v5_predict(
                    pipeline,
                    test_history,
                    references,
                    test_ticker,
                    1,
                )

                print(
                    "V5 covariate preflight: PASS"
                )

            except Exception as exc:
                del pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "V5 inference preflight failed. "
                    "No walk-forward evaluation was started.\n"
                    f"Reason: {exc}"
                ) from exc

        for ticker in TICKERS:

            try:

                stock = load_stock(
                    ticker
                )

                if len(stock) <= (
                    MIN_HISTORY
                    +
                    max(HORIZONS)
                ):
                    print(
                        f"[SKIP] {ticker}: "
                        "not enough history"
                    )
                    continue

                points, intervals = (
                    evaluate_chronos_stock(
                        model_name,
                        pipeline,
                        stock,
                        ticker,
                        references,
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

            except Exception as exc:

                print(
                    f"[ERROR] {ticker} / "
                    f"{model_name}: "
                    f"{exc}"
                )

        # Prevent 7+ GB of VRAM from being duplicated.
        del pipeline

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Statistical models.
    # --------------------------------------------------------

    print()
    print(
        "=" * 90
    )
    print(
        "EVALUATING STATISTICAL / ML MODELS"
    )
    print(
        "=" * 90
    )

    for ticker in TICKERS:

        try:

            stock = load_stock(
                ticker
            )

            if len(stock) <= (
                MIN_HISTORY
                +
                max(HORIZONS)
            ):
                continue

            points, fits = (
                evaluate_statistical_stock(
                    stock,
                    ticker,
                    references,
                )
            )

            if not points.empty:
                all_points.append(points)

            if not fits.empty:
                all_fits.append(fits)

        except Exception as exc:

            print(
                f"[ERROR] {ticker} / "
                f"statistical: {exc}"
            )

    # --------------------------------------------------------
    # Combine.
    # --------------------------------------------------------

    if not all_points:

        raise RuntimeError(
            "No forecasts were generated."
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

    by_stock, overall = (
        summarize_points(
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

    vs_garch = (
        calculate_win_rates_vs_garch(
            points
        )
    )

    vs_garch.to_csv(
        RESULTS_DIR
        /
        "win_rates_vs_garch_by_horizon.csv",
        index=False,
    )

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
    # Calibration.
    # --------------------------------------------------------

    if all_intervals:

        intervals = pd.concat(
            all_intervals,
            ignore_index=True,
        )

        intervals.to_csv(
            RESULTS_DIR
            /
            "chronos_intervals_raw.csv",
            index=False,
        )

        cal_lead, cal_horizon = (
            summarize_calibration(
                intervals
            )
        )

        cal_lead.to_csv(
            RESULTS_DIR
            /
            "chronos_calibration_by_lead.csv",
            index=False,
        )

        cal_horizon.to_csv(
            RESULTS_DIR
            /
            "chronos_calibration_by_horizon.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Statistical fit success.
    # --------------------------------------------------------

    if all_fits:

        fits = pd.concat(
            all_fits,
            ignore_index=True,
        )

        fits.to_csv(
            RESULTS_DIR
            /
            "statistical_fit_diagnostics.csv",
            index=False,
        )

        fit_summary = (
            fits
            .assign(
                success_flag=(
                    fits["status"]
                    ==
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
            "statistical_fit_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Console report.
    # --------------------------------------------------------

    print()
    print(
        "=" * 90
    )
    print(
        "OVERALL RESULTS"
    )
    print(
        "=" * 90
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "WIN RATES VS GARCH"
    )
    print(
        "=" * 90
    )

    if not vs_garch.empty:

        print(
            vs_garch.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
        )

    print()
    print(
        "=" * 90
    )
    print(
        "BEST-MODEL COUNTS"
    )
    print(
        "=" * 90
    )

    if not best_counts.empty:

        print(
            best_counts.to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 90
    )
    print(
        "COMPLETE"
    )
    print(
        "=" * 90
    )

    print(
        "Results saved to:",
        RESULTS_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
