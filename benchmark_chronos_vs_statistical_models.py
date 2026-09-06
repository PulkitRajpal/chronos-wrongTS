
"""
FINAL CHRONOS-2 vs STATISTICAL / ML BENCHMARK
=============================================

10 stocks:
    AAPL, MSFT, NVDA, AMZN, ENPH, SMCI, CVNA, PLUG, RKLB, IONQ

Models:
    Naive
    ARIMA
    ETS
    GARCH
    EGARCH
    GJR-GARCH
    XGBoost
    LightGBM
    VAR (cross-market only)
    Chronos-2

Experiments:
    1. Univariate Close
    2. OHLCV
    3. Cross-market: Stock + SPX + VIX
    4. Spike/tail test
    5. Cold-start history-length test

IMPORTANT:
    This version preserves the Chronos implementation pattern from the
    known-working AAPL code:
      - resolve/check device before loading
      - regular synthetic timestamps
      - Chronos2Pipeline.predict_df(...)
      - target="target"
      - explicit quantiles

For debugging:
    FORCE_CPU = True
    Set to False only after the CPU run is confirmed.

Expected stock CSV format:
    date,open,high,low,close,volume

Macro CSV:
    SPX.csv
    VIX.csv
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import mean_absolute_error
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

# True = safest for validating the pipeline.
# False = use CUDA if the installed PyTorch/driver/model combination works.
FORCE_CPU = False

HORIZONS = [1, 5, 10, 20]

# Number of walk-forward cutoffs per stock.
# Set to None for every eligible cutoff.
N_WINDOWS = 50

MIN_HISTORY = 300
ML_LOOKBACK = 252
CHRONOS_CONTEXT = 512

# Cold-start experiment.
COLD_START_LENGTHS = [30, 60, 90, 180]

# Spike definition.
SPIKE_SIGMA = 2.0

# Chronos probability levels.
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


# ============================================================
# DEVICE
# ============================================================

def resolve_device() -> str:
    """
    Preserve the working-code pattern:
    check actual CUDA tensor allocation, not only
    torch.cuda.is_available().
    """

    if FORCE_CPU:
        print("FORCE_CPU=True -> using CPU.")
        return "cpu"

    if not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return "cpu"

    try:
        torch.randn(1, device="cuda")
        name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {name}")
        return "cuda"

    except Exception as exc:
        print(
            "CUDA is present but not usable "
            "(kernel/image mismatch or incompatible driver). "
            "Falling back to CPU."
        )
        print(f"CUDA check error: {exc}")
        return "cpu"


DEVICE = resolve_device()


# ============================================================
# DATA HELPERS
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
    required: bool = True,
) -> Optional[str]:

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Columns: {list(df.columns)}"
        )

    return None


def load_stock(
    ticker: str,
) -> pd.DataFrame:

    path = DATA_DIR / f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    d = find_column(
        raw,
        ["date", "datetime", "timestamp"],
    )

    c = find_column(
        raw,
        ["close/last", "close", "last", "price"],
    )

    o = find_column(
        raw,
        ["open"],
        required=False,
    )

    h = find_column(
        raw,
        ["high"],
        required=False,
    )

    l = find_column(
        raw,
        ["low"],
        required=False,
    )

    v = find_column(
        raw,
        ["volume"],
        required=False,
    )

    out = pd.DataFrame({
        "Date": pd.to_datetime(
            raw[d],
            errors="coerce",
        ),
        f"{ticker}_Close": raw[c].map(clean_number),
    })

    for name, source in [
        (f"{ticker}_Open", o),
        (f"{ticker}_High", h),
        (f"{ticker}_Low", l),
        (f"{ticker}_Volume", v),
    ]:
        if source is None:
            out[name] = np.nan
        else:
            out[name] = raw[source].map(clean_number)

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

    d = find_column(
        raw,
        [
            "date",
            "datetime",
            "timestamp",
            "observation_date",
        ],
    )

    c = find_column(
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
        "Date": pd.to_datetime(
            raw[d],
            errors="coerce",
        ),
        f"{name}_Close": raw[c].map(clean_number),
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

        out = (
            stock
            .merge(spx, on="Date", how="inner")
            .merge(vix, on="Date", how="inner")
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return out

    except Exception as exc:
        print(
            f"Cross-market alignment failed: {exc}"
        )
        return None


# ============================================================
# CHRONOS
# ============================================================

def load_chronos():

    print(
        f"\nLoading {CHRONOS_MODEL}..."
    )

    pipe = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map=DEVICE,
    )

    print(
        f"Chronos-2 loaded on {DEVICE.upper()}."
    )

    return pipe


def add_regular_timestamp(
    history: pd.DataFrame,
) -> pd.DataFrame:

    out = history.copy()

    # Exactly one regular step per observed market session.
    # This avoids frequency inference failures caused by weekends/holidays.
    out["ChronosTime"] = pd.date_range(
        "2000-01-01",
        periods=len(out),
        freq="D",
    )

    return out


def chronos_context(
    history: pd.DataFrame,
    ticker: str,
    covariates: Optional[list[str]] = None,
) -> pd.DataFrame:

    covariates = covariates or []

    h = add_regular_timestamp(
        history
        .tail(CHRONOS_CONTEXT)
        .copy()
    )

    target = f"{ticker}_Close"

    out = pd.DataFrame({
        "id": [ticker] * len(h),
        "timestamp": h["ChronosTime"].values,
        "target": h[target].astype(float).values,
    })

    for col in covariates:
        out[col] = h[col].astype(float).values

    return out


def chronos_predict(
    pipe,
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    covariates: Optional[list[str]] = None,
) -> pd.DataFrame:

    ctx = chronos_context(
        history,
        ticker,
        covariates,
    )

    # This is intentionally kept in the same form as the known-working code.
    pred = pipe.predict_df(
        ctx,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    required = [
        "0.05",
        "0.1",
        "0.5",
        "0.9",
        "0.95",
    ]

    missing = [
        c for c in required
        if c not in pred.columns
    ]

    if missing:
        raise RuntimeError(
            "Chronos returned unexpected columns. "
            f"Missing {missing}; got {list(pred.columns)}"
        )

    return pred


# ============================================================
# FEATURES FOR ML
# ============================================================

def build_features(
    history: pd.DataFrame,
    ticker: str,
    external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    external_columns = external_columns or []

    out = history.copy()
    close = out[f"{ticker}_Close"]

    # Return features.
    out["ret_1"] = close.pct_change(1)
    out["ret_2"] = close.pct_change(2)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)

    # OHLCV features.
    out["hl_range"] = (
        out[f"{ticker}_High"]
        - out[f"{ticker}_Low"]
    ) / close

    out["oc_return"] = (
        out[f"{ticker}_Close"]
        - out[f"{ticker}_Open"]
    ) / out[f"{ticker}_Open"]

    out["volume_return"] = (
        out[f"{ticker}_Volume"].pct_change()
    )

    # Volatility.
    out["vol_5"] = (
        out["ret_1"].rolling(5).std()
    )
    out["vol_20"] = (
        out["ret_1"].rolling(20).std()
    )
    out["vol_60"] = (
        out["ret_1"].rolling(60).std()
    )

    # Momentum.
    out["mom_5"] = (
        close / close.shift(5) - 1
    )
    out["mom_20"] = (
        close / close.shift(20) - 1
    )
    out["mom_60"] = (
        close / close.shift(60) - 1
    )

    # External market returns.
    for col in external_columns:

        prefix = (
            col
            .replace("_Close", "")
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


def get_ml_features(
    external_columns: Optional[list[str]] = None,
) -> list[str]:

    external_columns = external_columns or []

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


# ============================================================
# STATISTICAL / ML MODELS
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

    # Work on log prices with one difference.
    # This is still the classical ARIMA benchmark used in this POC.
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


def garch_forecast(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    variant: str,
) -> tuple[float, float]:

    if not ARCH_AVAILABLE:
        raise RuntimeError(
            "arch package is not installed."
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
        disp="off"
    )

    fc = fitted.forecast(
        horizon=horizon,
        reindex=False,
    )

    mean_forecast = (
        fc.mean
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

    forecast_price = (
        last_price
        * np.exp(cumulative_return)
    )

    variance = (
        fc.variance
        .iloc[-1]
        .to_numpy()
    )

    avg_vol = float(
        np.sqrt(
            np.maximum(
                variance,
                0.0,
            )
        ).mean()
        / 100.0
    )

    return (
        float(forecast_price),
        avg_vol,
    )


def train_xgb(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
):

    external_columns = external_columns or []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = get_ml_features(
        external_columns
    )

    target = f"{ticker}_Close"

    data["target_return"] = (
        data[target].shift(-horizon)
        /
        data[target]
        - 1.0
    )

    data = data.dropna(
        subset=features + ["target_return"]
    ).tail(ML_LOOKBACK)

    if len(data) < 100:
        raise ValueError(
            f"Only {len(data)} clean rows for XGBoost."
        )

    X = (
        data[features]
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
        & y.notna()
        & np.isfinite(y)
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
    model,
    external_columns: Optional[list[str]] = None,
) -> float:

    external_columns = external_columns or []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = get_ml_features(
        external_columns
    )

    x = (
        pd.to_numeric(
            data.iloc[-1][features],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
        .reshape(1, -1)
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Latest XGBoost features contain "
            "NaN/inf/non-numeric values."
        )

    ret = float(
        model.predict(x)[0]
    )

    last = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return (
        last
        * (1.0 + ret)
    )


def train_lgbm(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_columns: Optional[list[str]] = None,
):

    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(
            "lightgbm package is not installed."
        )

    external_columns = external_columns or []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = get_ml_features(
        external_columns
    )

    target = f"{ticker}_Close"

    data["target_return"] = (
        data[target].shift(-horizon)
        /
        data[target]
        - 1.0
    )

    data = (
        data
        .dropna(
            subset=features + ["target_return"]
        )
        .tail(ML_LOOKBACK)
    )

    X = data[features].astype(float)
    y = data["target_return"].astype(float)

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

    external_columns = external_columns or []

    data = build_features(
        history,
        ticker,
        external_columns,
    )

    features = get_ml_features(
        external_columns
    )

    x = (
        data.iloc[-1][features]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    ret = float(
        model.predict(x)[0]
    )

    last = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return last * (
        1.0 + ret
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
            "VAR requires stock, SPX and VIX."
        )

    returns = (
        np.log(history[cols])
        .diff()
        .dropna()
        .tail(ML_LOOKBACK)
    )

    if len(returns) < 100:
        raise ValueError(
            "Not enough observations for VAR."
        )

    fitted = VAR(returns).fit(5)

    fc = fitted.forecast(
        returns.values[-fitted.k_ar:],
        steps=horizon,
    )

    cumulative_stock_return = (
        fc[:, 0].sum()
    )

    last = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    return (
        last
        * np.exp(
            cumulative_stock_return
        )
    )


# ============================================================
# CUT-OFFS
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
            f"got {n_rows}."
        )

    possible = list(
        range(
            MIN_HISTORY,
            n_rows - max(HORIZONS),
        )
    )

    if N_WINDOWS is None:
        return possible

    if len(possible) <= N_WINDOWS:
        return possible

    idx = np.linspace(
        0,
        len(possible) - 1,
        N_WINDOWS,
        dtype=int,
    )

    return [
        possible[i]
        for i in idx
    ]


# ============================================================
# SINGLE FORECAST BENCHMARK
# ============================================================

def run_benchmark(
    df: pd.DataFrame,
    ticker: str,
    pipe,
    experiment: str,
    chronos_covariates: Optional[list[str]] = None,
    ml_external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    chronos_covariates = (
        chronos_covariates or []
    )

    ml_external_columns = (
        ml_external_columns or []
    )

    print("\n")
    print("=" * 80)
    print(
        f"{experiment} — {ticker}"
    )
    print("=" * 80)

    rows = []

    cutoffs = get_cutoffs(
        len(df)
    )

    for i, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = df.iloc[
            :cutoff
        ].copy()

        if (
            i == 1
            or i == len(cutoffs)
            or i % 10 == 0
        ):

            print(
                f"  window {i}/{len(cutoffs)} "
                f"(cutoff="
                f"{history['Date'].iloc[-1].date()})"
            )

        last = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        for horizon in HORIZONS:

            future = df.iloc[
                cutoff:
                cutoff + horizon
            ]

            actual = float(
                future[
                    f"{ticker}_Close"
                ].iloc[-1]
            )

            predictions = {}

            # ----------------------
            # Naive
            # ----------------------

            predictions["Naive"] = (
                naive_forecast(
                    history,
                    ticker,
                )
            )

            # ----------------------
            # ARIMA
            # ----------------------

            try:
                predictions["ARIMA"] = (
                    arima_forecast(
                        history,
                        ticker,
                        horizon,
                    )
                )
            except Exception as exc:
                predictions["ARIMA"] = np.nan

            # ----------------------
            # ETS
            # ----------------------

            try:
                predictions["ETS"] = (
                    ets_forecast(
                        history,
                        ticker,
                        horizon,
                    )
                )
            except Exception:
                predictions["ETS"] = np.nan

            # ----------------------
            # GARCH family
            # ----------------------

            for name in [
                "GARCH",
                "EGARCH",
                "GJR-GARCH",
            ]:

                predictions[name] = np.nan

                if ARCH_AVAILABLE:

                    try:

                        p, _ = garch_forecast(
                            history,
                            ticker,
                            horizon,
                            name,
                        )

                        predictions[name] = p

                    except Exception:
                        pass

            # ----------------------
            # XGBoost
            # ----------------------

            try:

                model = train_xgb(
                    history,
                    ticker,
                    horizon,
                    ml_external_columns,
                )

                predictions["XGBoost"] = (
                    xgb_forecast(
                        history,
                        ticker,
                        model,
                        ml_external_columns,
                    )
                )

            except Exception:
                predictions["XGBoost"] = np.nan

            # ----------------------
            # LightGBM
            # ----------------------

            predictions["LightGBM"] = np.nan

            if LIGHTGBM_AVAILABLE:

                try:

                    model = train_lgbm(
                        history,
                        ticker,
                        horizon,
                        ml_external_columns,
                    )

                    predictions["LightGBM"] = (
                        lgbm_forecast(
                            history,
                            ticker,
                            model,
                            ml_external_columns,
                        )
                    )

                except Exception:
                    pass

            # ----------------------
            # VAR
            # ----------------------

            predictions["VAR"] = np.nan

            if "SPX_Close" in history.columns:

                try:

                    predictions["VAR"] = (
                        var_forecast(
                            history,
                            ticker,
                            horizon,
                        )
                    )

                except Exception:
                    pass

            # ----------------------
            # Chronos-2
            # ----------------------

            # DO NOT swallow the Chronos exception.
            # During benchmark validation, a Chronos failure should stop
            # the run and show the actual root cause.
            cp = chronos_predict(
                pipe,
                history,
                ticker,
                horizon,
                chronos_covariates,
            )

            p10 = float(
                cp["0.1"].iloc[-1]
            )

            p50 = float(
                cp["0.5"].iloc[-1]
            )

            p90 = float(
                cp["0.9"].iloc[-1]
            )

            predictions["Chronos-2"] = p50

            row = {
                "ticker": ticker,
                "experiment": experiment,
                "cutoff_date":
                    history["Date"].iloc[-1],
                "horizon": horizon,
                "last_price": last,
                "actual": actual,
                **predictions,
                "Chronos_P10": p10,
                "Chronos_P90": p90,
            }

            # Every model gets the same metrics.
            for model_name, prediction in predictions.items():

                if pd.isna(prediction):
                    row[f"{model_name}_price_mae"] = np.nan
                    row[f"{model_name}_return_abs_error"] = np.nan
                    row[f"{model_name}_direction_hit"] = np.nan
                    continue

                row[f"{model_name}_price_mae"] = (
                    abs(
                        actual - prediction
                    )
                )

                actual_return = (
                    actual / last - 1.0
                )

                predicted_return = (
                    prediction / last - 1.0
                )

                row[f"{model_name}_return_abs_error"] = (
                    abs(
                        actual_return
                        -
                        predicted_return
                    )
                )

                row[f"{model_name}_direction_hit"] = (
                    float(
                        np.sign(actual_return)
                        ==
                        np.sign(predicted_return)
                    )
                )

            # Chronos coverage.
            actual_return = (
                actual / last - 1
            )

            p10_return = (
                p10 / last - 1
            )

            p90_return = (
                p90 / last - 1
            )

            row["Chronos_P10_breach"] = (
                actual_return < p10_return
            )

            row["Chronos_P90_breach"] = (
                actual_return > p90_return
            )

            rows.append(row)

    result = pd.DataFrame(rows)

    return result


# ============================================================
# SPIKES
# ============================================================

def run_spike_test(
    stock: pd.DataFrame,
    ticker: str,
    pipe,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        f"SPIKE TEST — {ticker}"
    )
    print("=" * 80)

    d = stock.copy()

    d["ret"] = (
        d[
            f"{ticker}_Close"
        ].pct_change()
    )

    d["vol20"] = (
        d["ret"].rolling(20).std()
    )

    rows = []

    for cutoff in get_cutoffs(
        len(d)
    ):

        hist = d.iloc[
            :cutoff
        ].copy()

        future = d.iloc[
            cutoff
        ]

        last = float(
            hist[
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

        vol = float(
            hist["vol20"].iloc[-1]
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

        # XGBoost warning.
        try:

            model = train_xgb(
                hist,
                ticker,
                horizon=1,
            )

            pred = xgb_forecast(
                hist,
                ticker,
                model,
            )

            xgb_return = (
                pred / last - 1
            )

            xgb_warning = (
                abs(xgb_return)
                >
                threshold
            )

        except Exception:

            xgb_return = np.nan
            xgb_warning = False

        # Chronos warning.
        cp = chronos_predict(
            pipe,
            hist,
            ticker,
            horizon=1,
        )

        p10_return = (
            float(
                cp["0.1"].iloc[-1]
            )
            / last
            - 1
        )

        p90_return = (
            float(
                cp["0.9"].iloc[-1]
            )
            / last
            - 1
        )

        chronos_warning = (
            p10_return < -threshold
            or
            p90_return > threshold
        )

        rows.append({
            "ticker": ticker,
            "date":
                future["Date"],
            "actual_return":
                actual_return,
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


def spike_metrics(
    actual: pd.Series,
    pred: pd.Series,
) -> dict:

    a = (
        actual
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    p = (
        pred
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))

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

    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# COLD START
# ============================================================

def run_cold_start(
    stock: pd.DataFrame,
    ticker: str,
    pipe,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        f"COLD START — {ticker}"
    )
    print("=" * 80)

    rows = []

    for history_length in COLD_START_LENGTHS:

        max_required = (
            history_length
            +
            max(HORIZONS)
        )

        if len(stock) <= max_required:
            continue

        possible = np.arange(
            history_length,
            len(stock)
            -
            max(HORIZONS),
        )

        if len(possible) == 0:
            continue

        if N_WINDOWS is None:

            cutoffs = [
                int(x)
                for x in possible
            ]

        else:

            n = min(
                N_WINDOWS,
                len(possible),
            )

            idx = np.linspace(
                0,
                len(possible) - 1,
                n,
                dtype=int,
            )

            cutoffs = [
                int(possible[i])
                for i in idx
            ]

        for cutoff in cutoffs:

            history = stock.iloc[
                :cutoff
            ].copy()

            # Restrict model context to the requested cold-start history.
            history = history.tail(
                history_length
            ).copy()

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

                # Naive.
                naive = last

                # ARIMA.
                try:
                    arima = arima_forecast(
                        history,
                        ticker,
                        horizon,
                    )
                except Exception:
                    arima = np.nan

                # XGBoost.
                try:

                    xgb_model = train_xgb(
                        history,
                        ticker,
                        horizon,
                    )

                    xgb = xgb_forecast(
                        history,
                        ticker,
                        xgb_model,
                    )

                except Exception:

                    xgb = np.nan

                # Chronos.
                cp = chronos_predict(
                    pipe,
                    history,
                    ticker,
                    horizon,
                )

                chronos = float(
                    cp["0.5"].iloc[-1]
                )

                for model_name, prediction in [
                    ("Naive", naive),
                    ("ARIMA", arima),
                    ("XGBoost", xgb),
                    ("Chronos-2", chronos),
                ]:

                    if pd.isna(prediction):
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
                            prediction,
                        "price_mae":
                            abs(
                                actual
                                -
                                prediction
                            ),
                        "return_abs_error":
                            safe_return_error(
                                last,
                                actual,
                                prediction,
                            ),
                    })

    return pd.DataFrame(rows)


def safe_return_error(
    last_price: float,
    actual: float,
    predicted: float,
) -> float:

    actual_return = (
        actual / last_price - 1
    )

    predicted_return = (
        predicted / last_price - 1
    )

    return abs(
        actual_return
        -
        predicted_return
    )


# ============================================================
# SUMMARY
# ============================================================

MODEL_NAMES = [
    "Naive",
    "ARIMA",
    "ETS",
    "GARCH",
    "EGARCH",
    "GJR-GARCH",
    "VAR",
    "XGBoost",
    "LightGBM",
    "Chronos-2",
]


def long_score_table(
    all_forecasts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in all_forecasts.iterrows():

        for model in MODEL_NAMES:

            value = row.get(
                model,
                np.nan,
            )

            if pd.isna(value):
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
                    model,
                "actual":
                    row["actual"],
                "prediction":
                    value,
                "price_mae":
                    row.get(
                        f"{model}_price_mae",
                        np.nan,
                    ),
                "return_abs_error":
                    row.get(
                        f"{model}_return_abs_error",
                        np.nan,
                    ),
                "direction_hit":
                    row.get(
                        f"{model}_direction_hit",
                        np.nan,
                    ),
            })

    return pd.DataFrame(rows)


def summarize(
    scored: pd.DataFrame,
) -> pd.DataFrame:

    return (
        scored
        .groupby([
            "experiment",
            "ticker",
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


def summarize_overall(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    return (
        summary
        .groupby([
            "experiment",
            "horizon",
            "model",
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
            median_price_mae=(
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


def chronos_coverage(
    all_forecasts: pd.DataFrame,
) -> pd.DataFrame:

    valid = all_forecasts.copy()

    return (
        valid
        .groupby([
            "experiment",
            "ticker",
            "horizon",
        ])
        .agg(
            observations=(
                "actual",
                "size",
            ),
            p10_breach_rate=(
                "Chronos_P10_breach",
                "mean",
            ),
            p90_breach_rate=(
                "Chronos_P90_breach",
                "mean",
            ),
        )
        .reset_index()
    )


# ============================================================
# PLOT
# ============================================================

def plot_aapl_example(
    stock: pd.DataFrame,
    pipe,
) -> None:

    if len(stock) <= (
        MIN_HISTORY + 20
    ):
        return

    horizon = 20

    cutoff = len(stock) - horizon

    history = stock.iloc[
        :cutoff
    ].copy()

    future = stock.iloc[
        cutoff:
    ].copy()

    cp = chronos_predict(
        pipe,
        history,
        "AAPL",
        horizon,
    )

    chronos_p10 = cp["0.1"].values
    chronos_p50 = cp["0.5"].values
    chronos_p90 = cp["0.9"].values

    xgb_values = []

    for h in range(
        1,
        horizon + 1,
    ):

        model = train_xgb(
            history,
            "AAPL",
            h,
        )

        xgb_values.append(
            xgb_forecast(
                history,
                "AAPL",
                model,
            )
        )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        history["Date"].tail(100),
        history[
            "AAPL_Close"
        ].tail(100),
        label="Known history",
    )

    plt.plot(
        future["Date"],
        future["AAPL_Close"],
        linewidth=3,
        label="Actual future",
    )

    plt.plot(
        future["Date"],
        xgb_values,
        linestyle="--",
        label="XGBoost direct horizon",
    )

    plt.plot(
        future["Date"],
        chronos_p50,
        linestyle="--",
        label="Chronos P50",
    )

    plt.fill_between(
        future["Date"],
        chronos_p10,
        chronos_p90,
        alpha=0.20,
        label="Chronos P10-P90",
    )

    plt.axvline(
        history["Date"].iloc[-1],
        linestyle=":",
        label="Forecast start",
    )

    plt.title(
        "AAPL — Chronos-2 vs Statistical / ML Models"
    )

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        /
        "AAPL_forecast_example.png",
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
    print(
        "CHRONOS-2 vs STATISTICAL / ML MODELS"
    )
    print(
        "10-STOCK FINANCIAL FORECASTING POC"
    )
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"ARCH available: "
        f"{ARCH_AVAILABLE}"
    )

    print(
        f"LightGBM available: "
        f"{LIGHTGBM_AVAILABLE}"
    )

    pipe = load_chronos()

    all_forecasts = []
    all_spikes = []
    all_cold = []

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
                f"{path} missing."
            )
            continue

        stock = load_stock(
            ticker
        )

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
                "  Not enough history."
            )
            continue

        evaluated.append(
            ticker
        )

        # ----------------------------------------------------
        # EXPERIMENT 1
        # ----------------------------------------------------

        e1 = run_benchmark(
            stock,
            ticker,
            pipe,
            "Experiment 1 — Univariate Close",
        )

        all_forecasts.append(
            e1
        )

        # ----------------------------------------------------
        # EXPERIMENT 2
        # ----------------------------------------------------

        e2 = run_benchmark(
            stock,
            ticker,
            pipe,
            "Experiment 2 — OHLCV",
            chronos_covariates=[
                f"{ticker}_Open",
                f"{ticker}_High",
                f"{ticker}_Low",
                f"{ticker}_Volume",
            ],
        )

        all_forecasts.append(
            e2
        )

        # ----------------------------------------------------
        # EXPERIMENT 3
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

            e3 = run_benchmark(
                cross,
                ticker,
                pipe,
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

            all_forecasts.append(
                e3
            )

        # ----------------------------------------------------
        # SPIKE
        # ----------------------------------------------------

        spikes = run_spike_test(
            stock,
            ticker,
            pipe,
        )

        all_spikes.append(
            spikes
        )

        # ----------------------------------------------------
        # COLD START
        # ----------------------------------------------------

        cold = run_cold_start(
            stock,
            ticker,
            pipe,
        )

        all_cold.append(
            cold
        )

        # ----------------------------------------------------
        # AAPL plot only
        # ----------------------------------------------------

        if ticker == "AAPL":
            plot_aapl_example(
                stock,
                pipe,
            )

    # ========================================================
    # FORECAST RESULTS
    # ========================================================

    if all_forecasts:

        forecasts = pd.concat(
            all_forecasts,
            ignore_index=True,
        )

        forecasts.to_csv(
            RESULTS_DIR
            /
            "all_forecasts.csv",
            index=False,
        )

        scored = long_score_table(
            forecasts
        )

        scored.to_csv(
            RESULTS_DIR
            /
            "all_scored_forecasts.csv",
            index=False,
        )

        summary = summarize(
            scored
        )

        summary.to_csv(
            RESULTS_DIR
            /
            "model_comparison_by_stock.csv",
            index=False,
        )

        overall = summarize_overall(
            summary
        )

        overall.to_csv(
            RESULTS_DIR
            /
            "model_comparison_overall.csv",
            index=False,
        )

        coverage = chronos_coverage(
            forecasts
        )

        coverage.to_csv(
            RESULTS_DIR
            /
            "chronos_coverage.csv",
            index=False,
        )

    else:

        forecasts = pd.DataFrame()
        summary = pd.DataFrame()
        overall = pd.DataFrame()

    # ========================================================
    # SPIKE RESULTS
    # ========================================================

    if all_spikes:

        spikes = pd.concat(
            all_spikes,
            ignore_index=True,
        )

        spikes.to_csv(
            RESULTS_DIR
            /
            "spikes_all_stocks.csv",
            index=False,
        )

        spike_rows = []

        for ticker, group in spikes.groupby(
            "ticker"
        ):

            for model, column in [
                (
                    "XGBoost",
                    "xgb_warning",
                ),
                (
                    "Chronos-2",
                    "chronos_warning",
                ),
            ]:

                m = spike_metrics(
                    group["actual_spike"],
                    group[column],
                )

                spike_rows.append({
                    "ticker":
                        ticker,
                    "model":
                        model,
                    "actual_spikes":
                        int(
                            group[
                                "actual_spike"
                            ].sum()
                        ),
                    **m,
                })

        spike_summary = pd.DataFrame(
            spike_rows
        )

        spike_summary.to_csv(
            RESULTS_DIR
            /
            "spike_summary.csv",
            index=False,
        )

        print("\n")
        print("=" * 80)
        print("SPIKE SUMMARY")
        print("=" * 80)
        print(
            spike_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ========================================================
    # COLD START
    # ========================================================

    if all_cold:

        cold = pd.concat(
            all_cold,
            ignore_index=True,
        )

        cold.to_csv(
            RESULTS_DIR
            /
            "cold_start_results.csv",
            index=False,
        )

        cold_summary = (
            cold
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
            /
            "cold_start_summary.csv",
            index=False,
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Successfully evaluated: "
        f"{len(evaluated)} / {len(TICKERS)}"
    )

    print(
        "Stocks:",
        ", ".join(evaluated)
    )

    if not overall.empty:

        for experiment in overall[
            "experiment"
        ].unique():

            print("\n")
            print(
                experiment
            )

            subset = overall[
                overall["experiment"]
                == experiment
            ].copy()

            print(
                subset[
                    [
                        "horizon",
                        "model",
                        "avg_price_mae",
                        "avg_return_abs_error",
                        "avg_directional_accuracy",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Results saved to: "
        f"{RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
