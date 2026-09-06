
"""
CHRONOS-2 vs XGBoost — 10 Stock Comparison POC

Experiments per stock
---------------------
1. Univariate:
   Close -> Chronos-2
   Close -> XGBoost

2. OHLCV:
   Close + past Open/High/Low/Volume -> Chronos-2
   OHLCV-derived lagged features -> XGBoost

3. Cross-market:
   Target stock + SPX + VIX past covariates -> Chronos-2
   Target stock + SPX/VIX lagged features -> XGBoost

4. Spike detection:
   Compare Chronos-2 vs XGBoost on large moves.

5. Missing/future-block proxy test:
   Walk-forward forecasting inherently hides the future block after cutoff.
   Evaluate 1/5/10/20 trading sessions.

The script loops over all stock CSVs in ./data and writes per-stock
and aggregate results to ./results.

Expected stock files:
    data/AAPL.csv
    data/MSFT.csv
    data/NVDA.csv
    data/AMZN.csv
    data/ENPH.csv
    data/SMCI.csv
    data/CVNA.csv
    data/PLUG.csv
    data/RKLB.csv
    data/IONQ.csv

Optional macro files:
    data/SPX.csv
    data/VIX.csv

If SPX/VIX are missing, Experiment 3 is skipped.
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
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


def resolve_device():
    if not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return "cpu"

    try:
        torch.randn(1, device="cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except Exception:
        print(
            "CUDA is present but not usable on this machine (kernel/image mismatch or incompatible driver). "
            "Falling back to CPU."
        )
        return "cpu"


DEVICE = resolve_device()

HORIZONS = [1, 5, 10, 20]

MIN_HISTORY = 300

# Number of equally spaced historical cutoffs per stock.
# Increase to None to use every possible cutoff.
N_WINDOWS = 50

# XGBoost training history.
ML_LOOKBACK = 252

# Chronos context.
CHRONOS_CONTEXT = 512

# Spike definition.
SPIKE_SIGMA = 2.0

QUANTILES = [
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
]


# ============================================================
# DATA UTILITIES
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
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


def load_stock_csv(
    path: Path,
    ticker: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing stock file: {path}"
        )

    raw = pd.read_csv(path)

    date_col = find_column(
        raw,
        ["date", "datetime", "timestamp"],
    )

    close_col = find_column(
        raw,
        ["close", "close/last", "last", "price"],
    )

    open_col = find_column(
        raw,
        ["open"],
    )

    high_col = find_column(
        raw,
        ["high"],
    )

    low_col = find_column(
        raw,
        ["low"],
    )

    volume_col = find_column(
        raw,
        ["volume"],
    )

    if date_col is None or close_col is None:
        raise ValueError(
            f"{ticker}: Could not locate date/close columns. "
            f"Columns = {list(raw.columns)}"
        )

    data = {
        "Date": pd.to_datetime(
            raw[date_col],
            errors="coerce",
        ),
        f"{ticker}_Close": raw[
            close_col
        ].map(clean_number),
    }

    if open_col is not None:
        data[f"{ticker}_Open"] = raw[
            open_col
        ].map(clean_number)

    if high_col is not None:
        data[f"{ticker}_High"] = raw[
            high_col
        ].map(clean_number)

    if low_col is not None:
        data[f"{ticker}_Low"] = raw[
            low_col
        ].map(clean_number)

    if volume_col is not None:
        data[f"{ticker}_Volume"] = raw[
            volume_col
        ].map(clean_number)

    df = pd.DataFrame(data)

    # Some sources may not have OHLCV.
    # Fill absent columns so the rest of the pipeline can operate.
    for col in [
        f"{ticker}_Open",
        f"{ticker}_High",
        f"{ticker}_Low",
        f"{ticker}_Volume",
    ]:
        if col not in df:
            df[col] = np.nan

    df = (
        df
        .dropna(subset=["Date", f"{ticker}_Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return df


def load_macro_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

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
            f"{name}: could not locate date/value. "
            f"Columns = {list(raw.columns)}"
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

    out = (
        out
        .dropna(subset=[
            "Date",
            f"{name}_Close",
        ])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return out


def align_with_macro(
    stock: pd.DataFrame,
    macro_files: dict[str, Path],
) -> pd.DataFrame:

    result = stock.copy()

    for name, path in macro_files.items():
        try:
            macro = load_macro_csv(
                path,
                name,
            )
        except Exception as exc:
            print(
                f"  {name} unavailable: {exc}"
            )
            return result

        result = result.merge(
            macro,
            on="Date",
            how="inner",
        )

    return (
        result
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# DEVICE
# ============================================================

def load_chronos():

    print(
        f"\nLoading {CHRONOS_MODEL}..."
    )

    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL,
            device_map=DEVICE,
        )
        print(f"Chronos-2 loaded on {DEVICE.upper()}.")
        return pipeline
    except Exception as exc:
        if DEVICE != "cuda":
            raise

        print(
            "\nCUDA is present but not usable on this machine; falling back to CPU."
        )
        print(f"CUDA error: {exc}")

        pipeline = Chronos2Pipeline.from_pretrained(
            CHRONOS_MODEL,
            device_map="cpu",
        )
        print("Chronos-2 loaded on CPU.")
        return pipeline


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def get_cutoffs(
    n_rows: int,
) -> list[int]:

    max_horizon = max(HORIZONS)

    required = (
        MIN_HISTORY
        +
        max_horizon
    )

    if n_rows <= required:
        raise ValueError(
            f"Not enough data: {n_rows} rows. "
            f"Need > {required}."
        )

    possible = np.arange(
        MIN_HISTORY,
        n_rows - max_horizon,
    )

    if N_WINDOWS is None:
        return possible.tolist()

    if len(possible) <= N_WINDOWS:
        return possible.tolist()

    idx = np.linspace(
        0,
        len(possible) - 1,
        N_WINDOWS,
        dtype=int,
    )

    return [
        int(possible[i])
        for i in idx
    ]


# ============================================================
# XGBOOST FEATURES
# ============================================================

def build_features(
    df: pd.DataFrame,
    ticker: str,
    external_close_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    if external_close_columns is None:
        external_close_columns = []

    out = df.copy()

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
        out[f"{ticker}_Volume"]
        .pct_change()
    )

    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["vol_60"] = out["ret_1"].rolling(60).std()

    out["mom_5"] = (
        close / close.shift(5) - 1
    )

    out["mom_20"] = (
        close / close.shift(20) - 1
    )

    out["mom_60"] = (
        close / close.shift(60) - 1
    )

    for col in external_close_columns:

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


def feature_list(
    ticker: str,
    external_close_columns: Optional[list[str]] = None,
) -> list[str]:

    if external_close_columns is None:
        external_close_columns = []

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

    for col in external_close_columns:

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
# XGBOOST TRAINING
# ============================================================

def train_xgb(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    external_close_columns: Optional[list[str]] = None,
) -> XGBRegressor:

    if external_close_columns is None:
        external_close_columns = []

    data = build_features(
        history,
        ticker,
        external_close_columns,
    )

    feats = feature_list(
        ticker,
        external_close_columns,
    )

    target_col = "target_return"

    data[target_col] = (
        data[f"{ticker}_Close"]
        .shift(-horizon)
        /
        data[f"{ticker}_Close"]
        - 1.0
    )

    data = data.dropna(
        subset=feats + [target_col]
    )

    data = data.tail(
        min(ML_LOOKBACK, len(data))
    )

    if len(data) < 100:
        raise ValueError(
            f"{ticker}: only {len(data)} clean "
            f"training rows for horizon {horizon}"
        )

    X = (
        data[feats]
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
    )

    y = (
        pd.to_numeric(
            data[target_col],
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


def xgb_endpoint(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
    model: XGBRegressor,
    external_close_columns: Optional[list[str]] = None,
) -> float:

    if external_close_columns is None:
        external_close_columns = []

    data = build_features(
        history,
        ticker,
        external_close_columns,
    )

    feats = feature_list(
        ticker,
        external_close_columns,
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
            f"{ticker}: XGBoost latest features "
            "contain NaN/inf."
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
# CHRONOS
# ============================================================

def chronos_context(
    history: pd.DataFrame,
    ticker: str,
    covariate_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    if covariate_columns is None:
        covariate_columns = []

    history = history.tail(
        CHRONOS_CONTEXT
    ).copy()

    # Regular synthetic timestamps:
    # one row = one observed trading session.
    synthetic_dates = pd.date_range(
        start="2000-01-01",
        periods=len(history),
        freq="D",
    )

    context = pd.DataFrame({
        "id": ticker,
        "timestamp": synthetic_dates,
        "target": (
            history[
                f"{ticker}_Close"
            ]
            .astype(float)
            .values
        ),
    })

    for col in covariate_columns:
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
    covariate_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    context = chronos_context(
        history,
        ticker,
        covariate_columns,
    )

    pred = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    return pred


# ============================================================
# FORECAST EXPERIMENT
# ============================================================

def run_forecast_experiment(
    df: pd.DataFrame,
    ticker: str,
    pipeline,
    experiment_name: str,
    chronos_covariates: Optional[list[str]] = None,
    xgb_external_columns: Optional[list[str]] = None,
) -> pd.DataFrame:

    if chronos_covariates is None:
        chronos_covariates = []

    if xgb_external_columns is None:
        xgb_external_columns = []

    print("\n")
    print("=" * 80)
    print(
        f"{experiment_name} ({ticker})"
    )
    print("=" * 80)

    cutoffs = get_cutoffs(
        len(df)
    )

    rows = []

    for i, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = df.iloc[
            :cutoff
        ].copy()

        if i == 1 or i == len(cutoffs) or i % 10 == 0:
            print(
                f"  window {i}/{len(cutoffs)} "
                f"(cutoff="
                f"{history['Date'].iloc[-1].date()})"
            )

        last_price = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        for horizon in HORIZONS:

            future = df.iloc[
                cutoff:
                cutoff + horizon
            ]

            actual_prices = (
                future[
                    f"{ticker}_Close"
                ]
                .astype(float)
                .values
            )

            actual_final = float(
                actual_prices[-1]
            )

            # --------------------------
            # XGBoost
            # --------------------------

            try:
                xgb = train_xgb(
                    history,
                    ticker,
                    horizon,
                    xgb_external_columns,
                )

                ml_final = xgb_endpoint(
                    history,
                    ticker,
                    horizon,
                    xgb,
                    xgb_external_columns,
                )

                ml_ok = True

            except Exception as exc:

                print(
                    f"    XGBoost failed: {exc}"
                )

                ml_final = np.nan
                ml_ok = False

            # --------------------------
            # Chronos
            # --------------------------

            try:
                pred = chronos_forecast(
                    pipeline,
                    history,
                    ticker,
                    horizon,
                    chronos_covariates,
                )

                p10 = float(
                    pred["0.1"].iloc[-1]
                )

                p50 = float(
                    pred["0.5"].iloc[-1]
                )

                p90 = float(
                    pred["0.9"].iloc[-1]
                )

                chronos_ok = True

            except Exception as exc:

                print(
                    f"    Chronos failed: {exc}"
                )

                p10 = np.nan
                p50 = np.nan
                p90 = np.nan
                chronos_ok = False

            # --------------------------
            # Metrics
            # --------------------------

            actual_return = (
                actual_final
                /
                last_price
                - 1.0
            )

            ml_return = (
                ml_final
                /
                last_price
                - 1.0
            ) if ml_ok else np.nan

            chronos_p10_return = (
                p10
                /
                last_price
                - 1.0
            ) if chronos_ok else np.nan

            chronos_p50_return = (
                p50
                /
                last_price
                - 1.0
            ) if chronos_ok else np.nan

            chronos_p90_return = (
                p90
                /
                last_price
                - 1.0
            ) if chronos_ok else np.nan

            rows.append({
                "ticker": ticker,
                "experiment": experiment_name,
                "cutoff_date":
                    history["Date"].iloc[-1],
                "horizon": horizon,

                "last_price": last_price,
                "actual_final": actual_final,

                "ml_final": ml_final,

                "chronos_p10": p10,
                "chronos_p50": p50,
                "chronos_p90": p90,

                "ml_mae":
                    abs(
                        ml_final
                        - actual_final
                    ),

                "chronos_mae":
                    abs(
                        p50
                        - actual_final
                    ),

                "ml_direction":
                    float(
                        np.sign(ml_return)
                        ==
                        np.sign(actual_return)
                    )
                    if ml_ok else np.nan,

                "chronos_direction":
                    float(
                        np.sign(
                            chronos_p50_return
                        )
                        ==
                        np.sign(actual_return)
                    )
                    if chronos_ok else np.nan,

                "actual_return":
                    actual_return,

                "ml_return":
                    ml_return,

                "chronos_p10_return":
                    chronos_p10_return,

                "chronos_p50_return":
                    chronos_p50_return,

                "chronos_p90_return":
                    chronos_p90_return,

                "chronos_p10_breach":
                    (
                        actual_return
                        <
                        chronos_p10_return
                    )
                    if chronos_ok else np.nan,

                "chronos_p90_breach":
                    (
                        actual_return
                        >
                        chronos_p90_return
                    )
                    if chronos_ok else np.nan,
            })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# SPIKE EXPERIMENT
# ============================================================

def run_spike_experiment(
    df: pd.DataFrame,
    ticker: str,
    pipeline,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        f"PRICE SPIKE EXPERIMENT ({ticker})"
    )
    print("=" * 80)

    cutoffs = get_cutoffs(
        len(df)
    )

    rows = []

    for i, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = df.iloc[
            :cutoff
        ].copy()

        future = df.iloc[
            cutoff
        ]

        last_price = float(
            history[
                f"{ticker}_Close"
            ].iloc[-1]
        )

        actual_price = float(
            future[
                f"{ticker}_Close"
            ]
        )

        actual_return = (
            actual_price
            /
            last_price
            - 1.0
        )

        hist_returns = (
            history[
                f"{ticker}_Close"
            ]
            .pct_change()
            .dropna()
        )

        vol = float(
            hist_returns.tail(20).std()
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

        # --------------------------
        # XGBoost
        # --------------------------

        try:

            xgb = train_xgb(
                history,
                ticker,
                horizon=1,
            )

            ml_price = xgb_endpoint(
                history,
                ticker,
                horizon=1,
                model=xgb,
            )

            ml_return = (
                ml_price
                /
                last_price
                - 1.0
            )

            ml_warning = (
                abs(ml_return)
                >
                threshold
            )

        except Exception:

            ml_return = np.nan
            ml_warning = False

        # --------------------------
        # Chronos
        # --------------------------

        try:

            pred = chronos_forecast(
                pipeline,
                history,
                ticker,
                horizon=1,
                covariate_columns=[],
            )

            p10 = float(
                pred["0.1"].iloc[-1]
            )

            p90 = float(
                pred["0.9"].iloc[-1]
            )

            p10_return = (
                p10 / last_price - 1
            )

            p90_return = (
                p90 / last_price - 1
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

        rows.append({
            "ticker": ticker,
            "date": future["Date"],
            "actual_return": actual_return,
            "rolling_vol": vol,
            "threshold": threshold,
            "actual_spike": actual_spike,

            "ml_return": ml_return,
            "ml_spike_warning": ml_warning,

            "chronos_p10_return":
                p10_return,

            "chronos_p90_return":
                p90_return,

            "chronos_spike_warning":
                chronos_warning,
        })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:

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
        np.sum(actual & predicted)
    )

    fp = int(
        np.sum(~actual & predicted)
    )

    fn = int(
        np.sum(actual & ~predicted)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_forecasts(
    result: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        result
        .groupby([
            "ticker",
            "experiment",
            "horizon",
        ])
        .agg(
            ml_mae=("ml_mae", "mean"),
            chronos_mae=(
                "chronos_mae",
                "mean",
            ),

            ml_direction=(
                "ml_direction",
                "mean",
            ),

            chronos_direction=(
                "chronos_direction",
                "mean",
            ),

            p10_breach_rate=(
                "chronos_p10_breach",
                "mean",
            ),

            p90_breach_rate=(
                "chronos_p90_breach",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["chronos_mae_improvement_pct"] = (
        (
            summary["ml_mae"]
            -
            summary["chronos_mae"]
        )
        /
        summary["ml_mae"]
        * 100
    )

    return summary


def summarize_cross_stock(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    return (
        summary
        .groupby([
            "experiment",
            "horizon",
        ])
        .agg(
            stocks=("ticker", "nunique"),
            avg_ml_mae=("ml_mae", "mean"),
            avg_chronos_mae=(
                "chronos_mae",
                "mean",
            ),
            median_ml_mae=(
                "ml_mae",
                "median",
            ),
            median_chronos_mae=(
                "chronos_mae",
                "median",
            ),
            avg_chronos_improvement=(
                "chronos_mae_improvement_pct",
                "mean",
            ),
            avg_ml_direction=(
                "ml_direction",
                "mean",
            ),
            avg_chronos_direction=(
                "chronos_direction",
                "mean",
            ),
        )
        .reset_index()
    )


# ============================================================
# PLOT PER STOCK
# ============================================================

def plot_example(
    df: pd.DataFrame,
    ticker: str,
    pipeline,
) -> None:

    horizon = 20

    if len(df) <= MIN_HISTORY + horizon:
        return

    cutoff = len(df) - horizon

    history = df.iloc[
        :cutoff
    ].copy()

    future = df.iloc[
        cutoff:
    ].copy()

    try:

        xgb = train_xgb(
            history,
            ticker,
            horizon,
        )

        xgb_pred = xgb_endpoint(
            history,
            ticker,
            horizon,
            xgb,
        )

        # This is an endpoint forecast, so plot it as a horizontal point.
        xgb_values = np.repeat(
            xgb_pred,
            horizon,
        )

        pred = chronos_forecast(
            pipeline,
            history,
            ticker,
            horizon,
        )

        p10 = pred["0.1"].values
        p50 = pred["0.5"].values
        p90 = pred["0.9"].values

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
        p50,
        linestyle="--",
        label="Chronos P50",
    )

    plt.fill_between(
        future["Date"],
        p10,
        p90,
        alpha=0.2,
        label="Chronos P10-P90",
    )

    # XGBoost endpoint is one H-session prediction.
    plt.axhline(
        xgb_pred,
        linestyle="--",
        label=f"XGBoost {horizon}D endpoint",
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
    print("CHRONOS-2 vs XGBOOST")
    print("10-STOCK PROXY / FORECASTING POC")
    print("=" * 80)

    # --------------------------------------------------------
    # Load Chronos once.
    # --------------------------------------------------------

    pipeline = load_chronos()

    all_exp1 = []
    all_exp2 = []
    all_exp3 = []
    all_spikes = []

    successful_tickers = []

    # --------------------------------------------------------
    # Loop through stocks.
    # --------------------------------------------------------

    for ticker in TICKERS:

        path = DATA_DIR / f"{ticker}.csv"

        if not path.exists():

            print(
                f"\nSkipping {ticker}: "
                f"{path} not found."
            )

            continue

        try:

            stock = load_stock_csv(
                path,
                ticker,
            )

        except Exception as exc:

            print(
                f"\nSkipping {ticker}: "
                f"{exc}"
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
            MIN_HISTORY + max(HORIZONS)
        ):

            print(
                f"Skipping {ticker}: "
                f"not enough history."
            )

            continue

        successful_tickers.append(
            ticker
        )

        # ----------------------------------------------------
        # Experiment 1: Close only
        # ----------------------------------------------------

        e1 = run_forecast_experiment(
            stock,
            ticker,
            pipeline,
            "Experiment 1 — Univariate Close",
        )

        all_exp1.append(e1)

        # ----------------------------------------------------
        # Experiment 2: OHLCV
        # ----------------------------------------------------

        e2 = run_forecast_experiment(
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

        all_exp2.append(e2)

        # ----------------------------------------------------
        # Spike experiment
        # ----------------------------------------------------

        spike = run_spike_experiment(
            stock,
            ticker,
            pipeline,
        )

        all_spikes.append(spike)

        # ----------------------------------------------------
        # Experiment 3: Cross market
        # ----------------------------------------------------

        if SPX_FILE.exists() and VIX_FILE.exists():

            try:

                cross = align_with_macro(
                    stock,
                    {
                        "SPX": SPX_FILE,
                        "VIX": VIX_FILE,
                    },
                )

                if len(cross) > (
                    MIN_HISTORY
                    +
                    max(HORIZONS)
                ):

                    e3 = run_forecast_experiment(
                        cross,
                        ticker,
                        pipeline,
                        "Experiment 3 — Cross Market",
                        chronos_covariates=[
                            "SPX_Close",
                            "VIX_Close",
                        ],
                        xgb_external_columns=[
                            "SPX_Close",
                            "VIX_Close",
                        ],
                    )

                    all_exp3.append(
                        e3
                    )

                else:

                    print(
                        f"  {ticker}: "
                        "not enough aligned SPX/VIX data "
                        "for Experiment 3."
                    )

            except Exception as exc:

                print(
                    f"  {ticker}: "
                    f"Experiment 3 failed: {exc}"
                )

        # ----------------------------------------------------
        # Example chart
        # ----------------------------------------------------

        plot_example(
            stock,
            ticker,
            pipeline,
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)

    if all_exp1:

        exp1 = pd.concat(
            all_exp1,
            ignore_index=True,
        )

        exp1.to_csv(
            RESULTS_DIR
            / "experiment_1_all_stocks.csv",
            index=False,
        )

        exp1_summary = summarize_forecasts(
            exp1
        )

        exp1_summary.to_csv(
            RESULTS_DIR
            / "experiment_1_summary_all_stocks.csv",
            index=False,
        )

    else:

        exp1 = pd.DataFrame()
        exp1_summary = pd.DataFrame()

    if all_exp2:

        exp2 = pd.concat(
            all_exp2,
            ignore_index=True,
        )

        exp2.to_csv(
            RESULTS_DIR
            / "experiment_2_all_stocks.csv",
            index=False,
        )

        exp2_summary = summarize_forecasts(
            exp2
        )

        exp2_summary.to_csv(
            RESULTS_DIR
            / "experiment_2_summary_all_stocks.csv",
            index=False,
        )

    else:

        exp2 = pd.DataFrame()
        exp2_summary = pd.DataFrame()

    if all_exp3:

        exp3 = pd.concat(
            all_exp3,
            ignore_index=True,
        )

        exp3.to_csv(
            RESULTS_DIR
            / "experiment_3_all_stocks.csv",
            index=False,
        )

        exp3_summary = summarize_forecasts(
            exp3
        )

        exp3_summary.to_csv(
            RESULTS_DIR
            / "experiment_3_summary_all_stocks.csv",
            index=False,
        )

    else:

        exp3 = pd.DataFrame()
        exp3_summary = pd.DataFrame()

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
            / "spike_results_all_stocks.csv",
            index=False,
        )

        spike_rows = []

        for ticker, group in spikes.groupby(
            "ticker"
        ):

            chronos = classification_metrics(
                group["actual_spike"],
                group[
                    "chronos_spike_warning"
                ],
            )

            ml = classification_metrics(
                group["actual_spike"],
                group[
                    "ml_spike_warning"
                ],
            )

            spike_rows.append({
                "ticker": ticker,
                "actual_spikes":
                    int(
                        group[
                            "actual_spike"
                        ]
                        .sum()
                    ),

                "xgb_precision":
                    ml["precision"],

                "xgb_recall":
                    ml["recall"],

                "chronos_precision":
                    chronos["precision"],

                "chronos_recall":
                    chronos["recall"],
            })

        spike_summary = pd.DataFrame(
            spike_rows
        )

        spike_summary.to_csv(
            RESULTS_DIR
            / "spike_summary_all_stocks.csv",
            index=False,
        )

    else:

        spike_summary = pd.DataFrame()

    # ========================================================
    # CROSS-STOCK SUMMARY
    # ========================================================

    summaries = []

    for s in [
        exp1_summary,
        exp2_summary,
        exp3_summary,
    ]:

        if not s.empty:
            summaries.append(s)

    if summaries:

        combined_summary = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined_summary.to_csv(
            RESULTS_DIR
            / "all_experiment_summaries.csv",
            index=False,
        )

        aggregate = summarize_cross_stock(
            combined_summary
        )

        aggregate.to_csv(
            RESULTS_DIR
            / "cross_stock_aggregate.csv",
            index=False,
        )

    # ========================================================
    # FINAL REPORT TO TERMINAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL STOCK COVERAGE")
    print("=" * 80)

    print(
        f"Successfully evaluated: "
        f"{len(successful_tickers)} / {len(TICKERS)}"
    )

    print(
        "Stocks:",
        ", ".join(successful_tickers)
    )

    if not exp1_summary.empty:

        print("\n")
        print(
            "EXPERIMENT 1 — UNIVARIATE"
        )

        print(
            exp1_summary[
                [
                    "ticker",
                    "horizon",
                    "ml_mae",
                    "chronos_mae",
                    "chronos_mae_improvement_pct",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    if not exp2_summary.empty:

        print("\n")
        print(
            "EXPERIMENT 2 — OHLCV"
        )

        print(
            exp2_summary[
                [
                    "ticker",
                    "horizon",
                    "ml_mae",
                    "chronos_mae",
                    "chronos_mae_improvement_pct",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    if not exp3_summary.empty:

        print("\n")
        print(
            "EXPERIMENT 3 — CROSS MARKET"
        )

        print(
            exp3_summary[
                [
                    "ticker",
                    "horizon",
                    "ml_mae",
                    "chronos_mae",
                    "chronos_mae_improvement_pct",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    if not spike_summary.empty:

        print("\n")
        print(
            "SPIKE RESULTS"
        )

        print(
            spike_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print("\n")
    print("=" * 80)
    print("POC COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to: "
        f"{RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
