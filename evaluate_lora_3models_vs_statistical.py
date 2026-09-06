"""
Chronos-2 LoRA V1/V2/V3 vs Statistical Models
================================================

Purpose
-------
Evaluate the three financially fine-tuned Chronos-2 LoRA adapters against
the same statistical / ML baselines on the same 10 held-out stocks.

IMPORTANT
---------
The LoRA models were fine-tuned on DAILY LOG RETURNS, not raw prices.

Therefore this evaluator:
    1. supplies log-return history to the LoRA Chronos models
    2. forecasts future log returns
    3. converts the cumulative forecast back to an endpoint price
    4. evaluates the resulting endpoint price against the actual price

This is essential for a fair evaluation of the fine-tuned adapters.

Adapters
--------
V1:
    models/chronos2-finance-lora/finetuned-ckpt

V2:
    models/chronos2-finance-lora-v2/finetuned-ckpt

V3:
    models/chronos2-finance-lora-v3-broad/finetuned-ckpt

Statistical / ML baselines
--------------------------
    Naive
    ARIMA
    ETS
    GARCH
    EGARCH
    GJR-GARCH
    XGBoost
    LightGBM
    VAR (when SPX/VIX are available)

Evaluation
----------
10 held-out stocks
Horizons: 1 / 5 / 10 / 20 trading sessions
Walk-forward evaluation
50 cutoffs per stock

Outputs
-------
results_lora_3models/
    all_scored_forecasts.csv
    model_comparison_by_stock.csv
    model_comparison_overall.csv
    chronos_lora_coverage.csv
    spike_results.csv
    spike_summary.csv
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

# Reuse the already-tested statistical-model implementation.
from benchmark_chronos_vs_statistical_models import (
    DATA_DIR,
    TICKERS,
    HORIZONS,
    MIN_HISTORY,
    N_WINDOWS,
    ML_LOOKBACK,
    CHRONOS_CONTEXT,
    SPIKE_SIGMA,
    ARCH_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    load_stock,
    align_cross_market,
    naive_forecast,
    arima_forecast,
    ets_forecast,
    garch_forecast,
    train_xgb,
    xgb_forecast,
    train_lgbm,
    lgbm_forecast,
    var_forecast,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_lora_3models"

LORA_MODELS = {
    "Chronos-LoRA-V1": BASE_DIR
    / "models"
    / "chronos2-finance-lora"
    / "finetuned-ckpt",

    "Chronos-LoRA-V2": BASE_DIR
    / "models"
    / "chronos2-finance-lora-v2"
    / "finetuned-ckpt",

    "Chronos-LoRA-V3": BASE_DIR
    / "models"
    / "chronos2-finance-lora-v3-broad"
    / "finetuned-ckpt",
}

STAT_MODELS = [
    "Naive",
    "ARIMA",
    "ETS",
    "GARCH",
    "EGARCH",
    "GJR-GARCH",
    "XGBoost",
    "LightGBM",
    "VAR",
]

# Use only the univariate return objective for the LoRA adapters.
# This is the cleanest apples-to-apples evaluation because the adapters
# were fine-tuned on one target: daily log returns.
USE_UNIVARIATE_ONLY = True

# Spike definition:
# one-day absolute return > 2 x recent 20-day return volatility
SPIKE_SIGMA = SPIKE_SIGMA


# ============================================================
# DEVICE
# ============================================================

def resolve_device() -> str:
    if not torch.cuda.is_available():
        print("CUDA unavailable -> CPU")
        return "cpu"

    try:
        torch.randn(1, device="cuda")
        print(
            "Using:",
            torch.cuda.get_device_name(0),
        )
        return "cuda"

    except Exception as exc:
        print(
            "CUDA detected but tensor test failed; "
            "falling back to CPU."
        )
        print(exc)
        return "cpu"


DEVICE = resolve_device()


# ============================================================
# LOAD LoRA MODELS
# ============================================================

def load_lora_pipelines() -> dict[str, Chronos2Pipeline]:

    pipelines = {}

    print()
    print("=" * 80)
    print("LOADING LoRA ADAPTERS")
    print("=" * 80)

    for model_name, path in LORA_MODELS.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{model_name} checkpoint not found:\n"
                f"{path.resolve()}"
            )

        print()
        print(
            f"Loading {model_name}:"
        )
        print(path)

        # Chronos-2 fine-tuning saves a loadable checkpoint.
        pipeline = Chronos2Pipeline.from_pretrained(
            path,
            device_map=DEVICE,
            import_allowlist=[
                "chronos.chronos2.model"
            ],
        )

        pipelines[model_name] = pipeline

        print(
            f"{model_name} loaded."
        )

    return pipelines


# ============================================================
# RETURN DATA
# ============================================================

def log_returns(
    history: pd.DataFrame,
    ticker: str,
) -> np.ndarray:

    prices = (
        history[f"{ticker}_Close"]
        .astype(float)
        .to_numpy()
    )

    if len(prices) < 2:
        raise ValueError(
            "Need at least two prices."
        )

    returns = np.diff(
        np.log(prices)
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) == 0:
        raise ValueError(
            "No valid log returns."
        )

    return returns.astype(
        np.float32
    )


# ============================================================
# LoRA FORECAST
# ============================================================

def lora_return_forecast(
    pipeline,
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> dict:

    """
    Forecast future log returns with a LoRA-adapted Chronos-2 model.

    The adapter was trained on log returns.

    Returns:
        p10_price
        p50_price
        p90_price
        p10_return
        p50_return
        p90_return
    """

    returns = log_returns(
        history,
        ticker,
    )

    # Keep the most recent model context.
    returns = returns[-CHRONOS_CONTEXT:]

    last_price = float(
        history[
            f"{ticker}_Close"
        ].iloc[-1]
    )

    # IMPORTANT:
    # The model was trained on a return series, so target is return,
    # not price.
    context = pd.DataFrame({
        "id": [ticker] * len(returns),
        "timestamp": pd.date_range(
            "2000-01-01",
            periods=len(returns),
            freq="D",
        ),
        "target": returns,
    })

    pred = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=[
            0.1,
            0.5,
            0.9,
        ],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Chronos predicts a path of log returns.
    #
    # Endpoint log return is approximated by the cumulative sum
    # of each quantile path.
    p10_log_return = float(
        pred["0.1"].to_numpy().sum()
    )

    p50_log_return = float(
        pred["0.5"].to_numpy().sum()
    )

    p90_log_return = float(
        pred["0.9"].to_numpy().sum()
    )

    # Convert cumulative log returns back into price.
    p10_price = (
        last_price
        * np.exp(
            p10_log_return
        )
    )

    p50_price = (
        last_price
        * np.exp(
            p50_log_return
        )
    )

    p90_price = (
        last_price
        * np.exp(
            p90_log_return
        )
    )

    return {
        "p10_price": p10_price,
        "p50_price": p50_price,
        "p90_price": p90_price,
        "p10_return": np.exp(
            p10_log_return
        ) - 1.0,
        "p50_return": np.exp(
            p50_log_return
        ) - 1.0,
        "p90_return": np.exp(
            p90_log_return
        ) - 1.0,
    }


# ============================================================
# CUT-OFFS
# ============================================================

def get_cutoffs(
    n_rows: int,
) -> list[int]:

    required = (
        MIN_HISTORY
        + max(HORIZONS)
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
# STATISTICAL FORECASTS
# ============================================================

def statistical_forecasts(
    history: pd.DataFrame,
    ticker: str,
    horizon: int,
) -> dict:

    predictions = {}

    # Naive
    predictions["Naive"] = (
        naive_forecast(
            history,
            ticker,
        )
    )

    # ARIMA
    try:
        predictions["ARIMA"] = (
            arima_forecast(
                history,
                ticker,
                horizon,
            )
        )
    except Exception:
        predictions["ARIMA"] = np.nan

    # ETS
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

    # GARCH family
    for name in [
        "GARCH",
        "EGARCH",
        "GJR-GARCH",
    ]:

        predictions[name] = np.nan

        if not ARCH_AVAILABLE:
            continue

        try:

            forecast, _ = garch_forecast(
                history,
                ticker,
                horizon,
                name,
            )

            predictions[name] = forecast

        except Exception:
            pass

    # XGBoost
    try:

        xgb_model = train_xgb(
            history,
            ticker,
            horizon,
        )

        predictions["XGBoost"] = (
            xgb_forecast(
                history,
                ticker,
                xgb_model,
            )
        )

    except Exception:
        predictions["XGBoost"] = np.nan

    # LightGBM
    predictions["LightGBM"] = np.nan

    if LIGHTGBM_AVAILABLE:

        try:

            lgbm_model = train_lgbm(
                history,
                ticker,
                horizon,
            )

            predictions["LightGBM"] = (
                lgbm_forecast(
                    history,
                    ticker,
                    lgbm_model,
                )
            )

        except Exception:
            pass

    # VAR is not available for univariate stock input.
    predictions["VAR"] = np.nan

    return predictions


# ============================================================
# SCORE HELPER
# ============================================================

def model_metrics(
    last_price: float,
    actual: float,
    prediction: float,
) -> tuple[float, float, float]:

    if not np.isfinite(prediction):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    price_mae = abs(
        actual - prediction
    )

    actual_return = (
        actual / last_price - 1.0
    )

    predicted_return = (
        prediction / last_price - 1.0
    )

    return_error = abs(
        actual_return
        -
        predicted_return
    )

    # Standard sign comparison.
    # For Naive, predicted return is zero, so direction is undefined.
    # We explicitly mark it NaN rather than incorrectly calling it wrong.
    if predicted_return == 0:
        direction_hit = np.nan
    else:
        direction_hit = float(
            np.sign(actual_return)
            ==
            np.sign(predicted_return)
        )

    return (
        price_mae,
        return_error,
        direction_hit,
    )


# ============================================================
# MAIN WALK-FORWARD BENCHMARK
# ============================================================

def run_stock(
    stock: pd.DataFrame,
    ticker: str,
    pipelines: dict[str, Chronos2Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    coverage_rows = []

    cutoffs = get_cutoffs(
        len(stock)
    )

    print()
    print(
        f"{ticker}: "
        f"{len(cutoffs)} walk-forward cutoffs"
    )

    for window_idx, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = stock.iloc[
            :cutoff
        ].copy()

        if (
            window_idx == 1
            or
            window_idx == len(cutoffs)
            or
            window_idx % 10 == 0
        ):

            print(
                f"  window "
                f"{window_idx}/"
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
        # Forecast each horizon.
        # ----------------------------------------------------

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

            # ================================================
            # Statistical models
            # ================================================

            stats = statistical_forecasts(
                history,
                ticker,
                horizon,
            )

            for model_name, prediction in stats.items():

                mae, ret_err, direction = (
                    model_metrics(
                        last_price,
                        actual,
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

                    "model_type":
                        "statistical_ml",
                })

            # ================================================
            # LoRA Chronos models
            # ================================================

            for model_name, pipeline in pipelines.items():

                try:

                    cp = lora_return_forecast(
                        pipeline,
                        history,
                        ticker,
                        horizon,
                    )

                    prediction = (
                        cp["p50_price"]
                    )

                    mae, ret_err, direction = (
                        model_metrics(
                            last_price,
                            actual,
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

                        "model_type":
                            "chronos_lora",
                    })

                    # ----------------------------------------
                    # Coverage
                    #
                    # These are derived endpoint bands from
                    # cumulative predicted log-return paths.
                    # ----------------------------------------

                    actual_return = (
                        actual
                        /
                        last_price
                        - 1.0
                    )

                    coverage_rows.append({
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

                        "actual":
                            actual,

                        "p10":
                            cp["p10_price"],

                        "p50":
                            cp["p50_price"],

                        "p90":
                            cp["p90_price"],

                        "p10_breach":
                            float(
                                actual_return
                                <
                                cp["p10_return"]
                            ),

                        "p90_breach":
                            float(
                                actual_return
                                >
                                cp["p90_return"]
                            ),
                    })

                except Exception as exc:

                    print(
                        f"    WARNING "
                        f"{model_name} "
                        f"{ticker} "
                        f"h={horizon}: "
                        f"{exc}"
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

                        "model_type":
                            "chronos_lora",
                    })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(coverage_rows),
    )


# ============================================================
# SPIKE EVALUATION
# ============================================================

def run_spike_eval(
    stock: pd.DataFrame,
    ticker: str,
    pipelines: dict[str, Chronos2Pipeline],
) -> pd.DataFrame:

    rows = []

    d = stock.copy()

    d["return"] = (
        d[
            f"{ticker}_Close"
        ].pct_change()
    )

    d["vol20"] = (
        d["return"]
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
            actual / last - 1.0
        )

        vol = float(
            history[
                "vol20"
            ].iloc[-1]
        )

        if (
            not np.isfinite(vol)
            or
            vol <= 0
        ):
            continue

        threshold = (
            SPIKE_SIGMA * vol
        )

        actual_spike = (
            abs(actual_return)
            >
            threshold
        )

        # -----------------------------
        # Statistical XGBoost warning
        # -----------------------------

        try:

            xgb_model = train_xgb(
                history,
                ticker,
                horizon=1,
            )

            xgb_prediction = xgb_forecast(
                history,
                ticker,
                xgb_model,
            )

            xgb_return = (
                xgb_prediction / last
                - 1.0
            )

            xgb_warning = (
                abs(xgb_return)
                >
                threshold
            )

        except Exception:

            xgb_warning = False

        # -----------------------------
        # LoRA warning
        # -----------------------------

        for model_name, pipeline in pipelines.items():

            try:

                cp = lora_return_forecast(
                    pipeline,
                    history,
                    ticker,
                    horizon=1,
                )

                # Large positive / negative tail.
                p10_return = (
                    cp["p10_return"]
                )

                p90_return = (
                    cp["p90_return"]
                )

                warning = (
                    p10_return < -threshold
                    or
                    p90_return > threshold
                )

            except Exception:

                warning = False

            rows.append({
                "ticker":
                    ticker,

                "date":
                    future["Date"],

                "actual_return":
                    actual_return,

                "threshold":
                    threshold,

                "actual_spike":
                    actual_spike,

                "model":
                    model_name,

                "spike_warning":
                    warning,
            })

        # XGBoost row.
        rows.append({
            "ticker":
                ticker,

            "date":
                future["Date"],

            "actual_return":
                actual_return,

            "threshold":
                threshold,

            "actual_spike":
                actual_spike,

            "model":
                "XGBoost",

            "spike_warning":
                xgb_warning,
        })

    return pd.DataFrame(rows)


# ============================================================
# SPIKE METRICS
# ============================================================

def calculate_spike_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    output = []

    for (
        ticker,
        model,
    ), group in df.groupby(
        [
            "ticker",
            "model",
        ]
    ):

        actual = (
            group[
                "actual_spike"
            ]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        pred = (
            group[
                "spike_warning"
            ]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        tp = int(
            np.sum(actual & pred)
        )

        fp = int(
            np.sum(~actual & pred)
        )

        fn = int(
            np.sum(actual & ~pred)
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

        f1 = (
            2 * precision * recall
            /
            (precision + recall)
            if precision + recall
            else 0.0
        )

        output.append({
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

    return pd.DataFrame(output)


# ============================================================
# SUMMARY TABLES
# ============================================================

def summarize_by_stock(
    scored: pd.DataFrame,
) -> pd.DataFrame:

    return (
        scored
        .groupby([
            "ticker",
            "horizon",
            "model",
            "model_type",
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


def summarize_coverage(
    coverage: pd.DataFrame,
) -> pd.DataFrame:

    if coverage.empty:
        return pd.DataFrame()

    out = (
        coverage
        .groupby([
            "horizon",
            "model",
        ])
        .agg(
            observations=(
                "p50",
                "size",
            ),

            p10_breach_rate=(
                "p10_breach",
                "mean",
            ),

            p90_breach_rate=(
                "p90_breach",
                "mean",
            ),

            interval_coverage=(
                "p10_breach",
                lambda x: 1.0 - (
                    x.mean()
                    +
                    coverage.loc[
                        x.index,
                        "p90_breach",
                    ].mean()
                ),
            ),

            mean_interval_width=(
                "p90",
                lambda x: np.nan,
            ),
        )
        .reset_index()
    )

    # Calculate actual interval width separately.
    widths = (
        coverage
        .assign(
            interval_width=(
                coverage["p90"]
                -
                coverage["p10"]
            )
        )
        .groupby([
            "horizon",
            "model",
        ])["interval_width"]
        .mean()
        .reset_index(
            name="mean_interval_width"
        )
    )

    out = out.drop(
        columns=["mean_interval_width"],
        errors="ignore",
    )

    out = out.merge(
        widths,
        on=[
            "horizon",
            "model",
        ],
        how="left",
    )

    return out


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
        "CHRONOS-2 LoRA V1/V2/V3 "
        "VS STATISTICAL MODELS"
    )
    print(
        "10-STOCK HELD-OUT EVALUATION"
    )
    print("=" * 80)

    print()
    print("LoRA checkpoints:")

    for name, path in LORA_MODELS.items():

        print(
            f"  {name}: {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing checkpoint: "
                f"{path.resolve()}"
            )

    pipelines = load_lora_pipelines()

    all_scored = []
    all_coverage = []
    all_spikes = []

    evaluated = []

    # --------------------------------------------------------
    # Stocks
    # --------------------------------------------------------

    for ticker in TICKERS:

        path = DATA_DIR / f"{ticker}.csv"

        if not path.exists():

            print(
                f"Skipping {ticker}: "
                f"{path} not found."
            )

            continue

        try:

            stock = load_stock(
                ticker
            )

        except Exception as exc:

            print(
                f"Skipping {ticker}: {exc}"
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

        evaluated.append(ticker)

        scored, coverage = run_stock(
            stock,
            ticker,
            pipelines,
        )

        all_scored.append(
            scored
        )

        all_coverage.append(
            coverage
        )

        # Spike test
        spikes = run_spike_eval(
            stock,
            ticker,
            pipelines,
        )

        all_spikes.append(
            spikes
        )

    # --------------------------------------------------------
    # Combine forecasts
    # --------------------------------------------------------

    if not all_scored:

        raise RuntimeError(
            "No stock evaluations completed."
        )

    scored = pd.concat(
        all_scored,
        ignore_index=True,
    )

    scored.to_csv(
        RESULTS_DIR
        /
        "all_scored_forecasts.csv",
        index=False,
    )

    summary = summarize_by_stock(
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

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if all_coverage:

        coverage = pd.concat(
            all_coverage,
            ignore_index=True,
        )

        coverage.to_csv(
            RESULTS_DIR
            /
            "chronos_lora_coverage_raw.csv",
            index=False,
        )

        coverage_summary = summarize_coverage(
            coverage
        )

        coverage_summary.to_csv(
            RESULTS_DIR
            /
            "chronos_lora_coverage.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Spikes
    # --------------------------------------------------------

    if all_spikes:

        spikes = pd.concat(
            all_spikes,
            ignore_index=True,
        )

        spikes.to_csv(
            RESULTS_DIR
            /
            "spike_results.csv",
            index=False,
        )

        spike_summary = calculate_spike_metrics(
            spikes
        )

        spike_summary.to_csv(
            RESULTS_DIR
            /
            "spike_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks evaluated: "
        f"{len(evaluated)} / {len(TICKERS)}"
    )

    print(
        "Stocks:",
        ", ".join(evaluated),
    )

    print()
    print("Overall results:")

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    if all_coverage:

        print()
        print(
            "Chronos LoRA coverage:"
        )

        print(
            coverage_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

    if all_spikes:

        print()
        print("Spike results:")

        print(
            spike_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

    print()
    print(
        "Results saved to:",
        RESULTS_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
