"""
CHRONOS-2 vs STATISTICAL ML POC
================================

Project layout expected:

CHRONOS-WRONGS/
├── chronos/
├── data/
│   ├── apple_historical.csv
│   ├── S&P 500 Historical Data.csv
│   └── VIXCLS.csv
└── poc.py

Three experiments:

1) UNIVARIATE
   AAPL Close -> Chronos-2 vs XGBoost

2) AAPL OHLCV
   AAPL Close target + historical OHLCV context -> Chronos-2
   vs XGBoost using OHLCV-derived features

3) CROSS-MARKET
   AAPL Close target + SPX Close + VIX history -> Chronos-2
   vs XGBoost using cross-market lag/return features

The evaluation is walk-forward. At cutoff T, neither model sees data
from T+1 onward. Each model predicts a future endpoint (1/5/10/20
trading days), which is compared to the real future price.

The script also runs a 1-day spike experiment using Chronos quantiles.
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
from xgboost import XGBRegressor

from chronos import BaseChronosPipeline


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

AAPL_FILE = DATA_DIR / "apple_historical.csv"
SPX_FILE = DATA_DIR / "S&P 500 Historical Data.csv"
VIX_FILE = DATA_DIR / "VIXCLS.csv"

CHRONOS_MODEL = "amazon/chronos-2"
DEVICE = "cuda"   # change to "cpu" if you do not have CUDA

# Keep this small for the first run.
N_WINDOWS = 25
HORIZONS = [1, 5, 10, 20]
LOOKBACK = 252
CHRONOS_CONTEXT = 512

# Spike threshold = 2 x recent 20-day volatility.
SPIKE_SIGMA = 2.0

QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


# ============================================================
# CSV HELPERS
# ============================================================

def clean_number(value) -> float:
    """Convert common market-data strings to floats."""
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    # Handle e.g. "$229.98", "2,345,678", "229.98"
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("%", "")

    # Investing/Yahoo sometimes uses '—' for missing.
    if s in {"", "-", "—", "nan", "NaN", "None"}:
        return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column case-insensitively from candidate names."""
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    # More flexible contains matching.
    for candidate in candidates:
        key = candidate.strip().lower()
        for norm, original in normalized.items():
            if key in norm:
                return original

    return None


def parse_market_file(path: Path, prefix: str) -> pd.DataFrame:
    """Load a market CSV and standardize Date + OHLCV where available."""

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            f"Expected file: {path.resolve()}"
        )

    df = pd.read_csv(path)

    date_col = find_column(
        df,
        ["Date", "date", "timestamp", "observation_date"]
    )

    if date_col is None:
        raise ValueError(
            f"Could not find a date column in {path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    result = pd.DataFrame({"Date": df["Date"]})

    close_col = find_column(
        df,
        ["Close/Last", "Close", "Price", "Adj Close", "Last", "VIXCLS"]
    )

    if close_col is not None:
        result[f"{prefix}_Close"] = df[close_col].map(clean_number)

    for field, candidates in {
        "Open": ["Open"],
        "High": ["High"],
        "Low": ["Low"],
        "Volume": ["Volume"],
    }.items():
        col = find_column(df, candidates)
        if col is not None:
            result[f"{prefix}_{field}"] = df[col].map(clean_number)

    result = result.dropna(subset=["Date"])
    result = (
        result
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return result


def load_and_align_data() -> pd.DataFrame:
    """Load AAPL, SPX and VIX and align them on AAPL trading dates."""

    aapl = parse_market_file(
        AAPL_FILE,
        "AAPL"
    )

    spx = parse_market_file(
        SPX_FILE,
        "SPX"
    )

    vix = parse_market_file(
        VIX_FILE,
        "VIX"
    )

    if "AAPL_Close" not in aapl.columns:
        raise ValueError(
            "Could not identify the Apple close column."
        )

    if "SPX_Close" not in spx.columns:
        raise ValueError(
            "Could not identify the S&P 500 close column."
        )

    # VIXCLS.csv usually has just Date + VIXCLS/Close.
    if "VIX_Close" not in vix.columns:
        raise ValueError(
            "Could not identify the VIX close/value column."
        )

    # Start from AAPL trading dates and bring in SPX + VIX.
    df = aapl.merge(
        spx[["Date", "SPX_Close"]],
        on="Date",
        how="left"
    )

    df = df.merge(
        vix[["Date", "VIX_Close"]],
        on="Date",
        how="left"
    )

    # VIX may have occasional missing values on AAPL dates.
    # Forward fill only from information already available at the time.
    df["SPX_Close"] = df["SPX_Close"].ffill()
    df["VIX_Close"] = df["VIX_Close"].ffill()

    df = df.dropna(subset=["AAPL_Close"])
    df = df.reset_index(drop=True)

    return df


# ============================================================
# WALK-FORWARD CUTPOINTS
# ============================================================

def get_cutoffs(n_rows: int) -> list[int]:
    max_horizon = max(HORIZONS)

    start = LOOKBACK + 30
    end = n_rows - max_horizon - 1

    if end <= start:
        raise ValueError(
            f"Not enough data. Need more than {LOOKBACK + max_horizon + 30} rows."
        )

    candidates = np.arange(start, end + 1)

    if len(candidates) <= N_WINDOWS:
        return candidates.tolist()

    indices = np.linspace(
        0,
        len(candidates) - 1,
        N_WINDOWS,
        dtype=int
    )

    return candidates[indices].tolist()


# ============================================================
# STATISTICAL ML FEATURES
# ============================================================

def make_features(
    df: pd.DataFrame,
    cross_market: bool = False
) -> pd.DataFrame:
    """Features available at the cutoff date only."""

    x = df.copy()

    close = x["AAPL_Close"]

    # Target returns.
    x["ret_1"] = close.pct_change(1)
    x["ret_2"] = close.pct_change(2)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)

    # OHLCV-derived information.
    if all(c in x.columns for c in [
        "AAPL_Open",
        "AAPL_High",
        "AAPL_Low"
    ]):
        x["intraday_range"] = (
            x["AAPL_High"] - x["AAPL_Low"]
        ) / close

        x["open_close_return"] = (
            x["AAPL_Close"] - x["AAPL_Open"]
        ) / x["AAPL_Open"]

    if "AAPL_Volume" in x.columns:
        x["volume_change"] = x["AAPL_Volume"].pct_change()
        x["volume_z20"] = (
            x["AAPL_Volume"]
            - x["AAPL_Volume"].rolling(20).mean()
        ) / (
            x["AAPL_Volume"].rolling(20).std()
            + 1e-8
        )

    # Historical volatility.
    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()

    # Momentum.
    x["mom_5"] = close / close.shift(5) - 1
    x["mom_20"] = close / close.shift(20) - 1
    x["mom_60"] = close / close.shift(60) - 1

    if cross_market:
        for col, name in [
            ("SPX_Close", "spx"),
            ("VIX_Close", "vix"),
        ]:
            if col in x.columns:
                x[f"{name}_ret_1"] = x[col].pct_change(1)
                x[f"{name}_ret_5"] = x[col].pct_change(5)
                x[f"{name}_ret_20"] = x[col].pct_change(20)

                x[f"{name}_level_z20"] = (
                    x[col]
                    - x[col].rolling(20).mean()
                ) / (
                    x[col].rolling(20).std()
                    + 1e-8
                )

    return x


def get_feature_columns(
    data: pd.DataFrame,
    cross_market: bool = False
) -> list[str]:

    preferred = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "intraday_range",
        "open_close_return",
        "volume_change",
        "volume_z20",
        "vol_5",
        "vol_20",
        "vol_60",
        "mom_5",
        "mom_20",
        "mom_60",
    ]

    if cross_market:
        preferred += [
            "spx_ret_1",
            "spx_ret_5",
            "spx_ret_20",
            "spx_level_z20",
            "vix_ret_1",
            "vix_ret_5",
            "vix_ret_20",
            "vix_level_z20",
        ]

    return [c for c in preferred if c in data.columns]


# ============================================================
# XGBOOST DIRECT-HORIZON MODEL
# ============================================================

def xgb_predict_endpoint(
    history: pd.DataFrame,
    horizon: int,
    cross_market: bool = False,
) -> float:
    """
    Direct multi-horizon regression:
    predict the return from T -> T+h.

    This avoids using unknown future OHLCV/covariates recursively.
    """

    work = make_features(
        history,
        cross_market=cross_market
    )

    # Target = h-day forward return.
    work["target"] = (
        work["AAPL_Close"].shift(-horizon)
        / work["AAPL_Close"]
        - 1
    )

    features = get_feature_columns(
        work,
        cross_market=cross_market
    )

    train = work.dropna(
        subset=features + ["target"]
    ).tail(LOOKBACK)

    if len(train) < 100:
        raise ValueError(
            "Not enough valid training rows for XGBoost."
        )

    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["target"]
    )

    latest = work.iloc[-1]
    X_latest = latest[features].values.reshape(1, -1)

    predicted_return = float(
        model.predict(X_latest)[0]
    )

    last_price = float(
        history["AAPL_Close"].iloc[-1]
    )

    return last_price * (1 + predicted_return)


# ============================================================
# CHRONOS CONTEXT
# ============================================================

def build_chronos_context(
    history: pd.DataFrame,
    experiment: int,
) -> pd.DataFrame:
    """
    Build the pandas input expected by Chronos-2.

    Exp 1: target = AAPL_Close only
    Exp 2: target = AAPL_Close + AAPL OHLCV as past covariates
    Exp 3: target = AAPL_Close + SPX/VIX as past covariates

    Chronos-2 supports univariate, multivariate and covariate-informed
    forecasting through the same pandas API.
    """

    context = pd.DataFrame({
        "id": "AAPL",
        "timestamp": history["Date"].values,
        "target": history["AAPL_Close"].values,
    })

    if experiment == 2:
        for source, name in [
            ("AAPL_Open", "open"),
            ("AAPL_High", "high"),
            ("AAPL_Low", "low"),
            ("AAPL_Volume", "volume"),
        ]:
            if source in history.columns:
                context[name] = history[source].values

    elif experiment == 3:
        for source, name in [
            ("SPX_Close", "spx"),
            ("VIX_Close", "vix"),
        ]:
            if source in history.columns:
                context[name] = history[source].values

    # Chronos has a maximum context length documented as 8192;
    # using a shorter context keeps the POC practical on a laptop.
    return context.tail(CHRONOS_CONTEXT).copy()


def chronos_predict_endpoint(
    pipeline: BaseChronosPipeline,
    history: pd.DataFrame,
    horizon: int,
    experiment: int,
) -> dict[str, float]:

    context = build_chronos_context(
        history,
        experiment=experiment
    )

    pred = pipeline.predict_df(
        context,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Official Chronos-2 output includes quantile columns such as
    # 0.1 / 0.5 / 0.9.
    return {
        "p05": float(pred["0.05"].iloc[-1]),
        "p10": float(pred["0.1"].iloc[-1]),
        "p50": float(pred["0.5"].iloc[-1]),
        "p90": float(pred["0.9"].iloc[-1]),
        "p95": float(pred["0.95"].iloc[-1]),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    df: pd.DataFrame,
    pipeline: BaseChronosPipeline,
    experiment: int,
) -> pd.DataFrame:

    names = {
        1: "UNIVARIATE",
        2: "OHLCV",
        3: "CROSS_MARKET",
    }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {experiment} — {names[experiment]}")
    print("=" * 80)

    cross_market = experiment == 3
    cutoffs = get_cutoffs(len(df))
    rows = []

    for window_i, cutoff in enumerate(cutoffs, start=1):
        print(
            f"  window {window_i}/{len(cutoffs)} "
            f"(cutoff={df['Date'].iloc[cutoff - 1].date()})"
        )

        history = df.iloc[:cutoff].copy()

        for horizon in HORIZONS:

            future = df.iloc[
                cutoff: cutoff + horizon
            ].copy()

            actual_final = float(
                future["AAPL_Close"].iloc[-1]
            )

            last_price = float(
                history["AAPL_Close"].iloc[-1]
            )

            # -----------------------------
            # XGBoost
            # -----------------------------
            ml_price = xgb_predict_endpoint(
                history,
                horizon=horizon,
                cross_market=cross_market,
            )

            # -----------------------------
            # Chronos-2
            # -----------------------------
            chronos = chronos_predict_endpoint(
                pipeline,
                history,
                horizon=horizon,
                experiment=experiment,
            )

            rows.append({
                "experiment": names[experiment],
                "cutoff_date": history["Date"].iloc[-1],
                "horizon": horizon,
                "last_price": last_price,
                "actual_price": actual_final,

                "xgb_price": ml_price,
                "xgb_abs_error": abs(
                    ml_price - actual_final
                ),

                "chronos_p05": chronos["p05"],
                "chronos_p10": chronos["p10"],
                "chronos_p50": chronos["p50"],
                "chronos_p90": chronos["p90"],
                "chronos_p95": chronos["p95"],

                "chronos_abs_error": abs(
                    chronos["p50"] - actual_final
                ),

                "actual_return": (
                    actual_final / last_price - 1
                ),
                "xgb_return": (
                    ml_price / last_price - 1
                ),
                "chronos_p10_return": (
                    chronos["p10"] / last_price - 1
                ),
                "chronos_p50_return": (
                    chronos["p50"] / last_price - 1
                ),
                "chronos_p90_return": (
                    chronos["p90"] / last_price - 1
                ),
            })

    result = pd.DataFrame(rows)

    filename = {
        1: "experiment_1_univariate.csv",
        2: "experiment_2_ohlcv.csv",
        3: "experiment_3_cross_market.csv",
    }[experiment]

    result.to_csv(
        RESULTS_DIR / filename,
        index=False
    )

    return result


# ============================================================
# SPIKE TEST
# ============================================================

def run_spike_experiment(
    df: pd.DataFrame,
    pipeline: BaseChronosPipeline,
) -> pd.DataFrame:
    """
    One-day-ahead tail-event test.

    Chronos warning:
        P10 or P90 crosses a 2-sigma threshold.

    XGBoost warning:
        point forecast crosses that threshold.

    This is deliberately a simple first benchmark. Later we can
    replace the XGBoost point-forecast warning with a probabilistic
    XGBoost model or quantile regression.
    """

    print("\n" + "=" * 80)
    print("SPIKE EXPERIMENT — 1 DAY AHEAD")
    print("=" * 80)

    rows = []

    cutoffs = get_cutoffs(len(df))

    for i, cutoff in enumerate(cutoffs, start=1):

        history = df.iloc[:cutoff].copy()
        future = df.iloc[cutoff]

        last_price = float(
            history["AAPL_Close"].iloc[-1]
        )

        actual_return = (
            float(future["AAPL_Close"])
            / last_price
            - 1
        )

        returns = history["AAPL_Close"].pct_change().dropna()
        recent_vol = float(returns.tail(20).std())

        threshold = SPIKE_SIGMA * recent_vol

        actual_spike = (
            abs(actual_return) > threshold
        )

        # -----------------------------
        # Chronos
        # -----------------------------
        chronos = chronos_predict_endpoint(
            pipeline,
            history,
            horizon=1,
            experiment=3,
        )

        chronos_p10_return = (
            chronos["p10"] / last_price - 1
        )

        chronos_p90_return = (
            chronos["p90"] / last_price - 1
        )

        chronos_warning = (
            chronos_p10_return < -threshold
            or
            chronos_p90_return > threshold
        )

        # -----------------------------
        # XGBoost
        # -----------------------------
        xgb_price = xgb_predict_endpoint(
            history,
            horizon=1,
            cross_market=True,
        )

        xgb_return = (
            xgb_price / last_price - 1
        )

        xgb_warning = (
            abs(xgb_return) > threshold
        )

        rows.append({
            "cutoff_date": history["Date"].iloc[-1],
            "actual_return": actual_return,
            "recent_vol": recent_vol,
            "spike_threshold": threshold,
            "actual_spike": actual_spike,

            "chronos_p10_return": chronos_p10_return,
            "chronos_p50_return": (
                chronos["p50"] / last_price - 1
            ),
            "chronos_p90_return": chronos_p90_return,
            "chronos_warning": chronos_warning,

            "xgb_return": xgb_return,
            "xgb_warning": xgb_warning,
        })

        if i % 5 == 0 or i == len(cutoffs):
            print(
                f"  processed {i}/{len(cutoffs)}"
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        RESULTS_DIR / "spike_results.csv",
        index=False
    )

    return result


# ============================================================
# METRICS
# ============================================================

def summarize_forecasts(result: pd.DataFrame) -> pd.DataFrame:

    summary = (
        result
        .groupby("horizon")
        .agg(
            xgb_mae=("xgb_abs_error", "mean"),
            chronos_mae=("chronos_abs_error", "mean"),
        )
        .reset_index()
    )

    summary["chronos_mae_improvement_pct"] = (
        (
            summary["xgb_mae"]
            - summary["chronos_mae"]
        )
        / summary["xgb_mae"]
        * 100
    )

    return summary


def binary_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, float]:

    actual = actual.astype(bool).values
    predicted = predicted.astype(bool).values

    tp = np.sum(actual & predicted)
    fp = np.sum(~actual & predicted)
    fn = np.sum(actual & ~predicted)

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

    return precision, recall


def summarize_spikes(spike_df: pd.DataFrame) -> pd.DataFrame:

    chronos_precision, chronos_recall = binary_metrics(
        spike_df["actual_spike"],
        spike_df["chronos_warning"],
    )

    xgb_precision, xgb_recall = binary_metrics(
        spike_df["actual_spike"],
        spike_df["xgb_warning"],
    )

    summary = pd.DataFrame([
        {
            "model": "Chronos-2",
            "precision": chronos_precision,
            "recall": chronos_recall,
        },
        {
            "model": "XGBoost",
            "precision": xgb_precision,
            "recall": xgb_recall,
        },
    ])

    return summary


def summarize_calibration(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether Chronos P10/P90 behave like their nominal tails."""

    rows = []

    for horizon, g in result.groupby("horizon"):

        last = g["last_price"]
        actual_ret = g["actual_return"]

        p10_ret = g["chronos_p10"] / last - 1
        p90_ret = g["chronos_p90"] / last - 1

        rows.append({
            "horizon": horizon,
            "p10_breach_rate": float(
                np.mean(actual_ret < p10_ret)
            ),
            "p90_breach_rate": float(
                np.mean(actual_ret > p90_ret)
            ),
            "expected_p10_breach_rate": 0.10,
            "expected_p90_breach_rate": 0.10,
        })

    return pd.DataFrame(rows)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_last_forecast(
    df: pd.DataFrame,
    pipeline: BaseChronosPipeline,
    experiment: int = 3,
    horizon: int = 20,
) -> None:

    cutoff = len(df) - horizon
    history = df.iloc[:cutoff].copy()
    future = df.iloc[cutoff:cutoff + horizon].copy()

    # XGB endpoint is only for the final horizon value, so don't
    # draw a fake path. Instead, visualize Chronos's full path and
    # show the XGBoost endpoint as a marker.
    chronos_context = build_chronos_context(
        history,
        experiment=experiment
    )

    pred = pipeline.predict_df(
        chronos_context,
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    xgb_endpoint = xgb_predict_endpoint(
        history,
        horizon=horizon,
        cross_market=(experiment == 3),
    )

    p10 = pred["0.1"].values
    p50 = pred["0.5"].values
    p90 = pred["0.9"].values

    plt.figure(figsize=(14, 7))

    plt.plot(
        history["Date"].tail(100),
        history["AAPL_Close"].tail(100),
        label="Known AAPL history",
    )

    plt.plot(
        future["Date"],
        future["AAPL_Close"],
        linewidth=3,
        label="Actual future",
    )

    plt.plot(
        future["Date"],
        p50,
        linestyle="--",
        label="Chronos-2 P50",
    )

    plt.fill_between(
        future["Date"],
        p10,
        p90,
        alpha=0.2,
        label="Chronos-2 P10-P90",
    )

    plt.scatter(
        [future["Date"].iloc[-1]],
        [xgb_endpoint],
        marker="x",
        s=90,
        label="XGBoost endpoint",
    )

    plt.axvline(
        history["Date"].iloc[-1],
        linestyle=":",
        label="Forecast start",
    )

    plt.title(
        "AAPL — Chronos-2 vs Statistical ML"
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "last_forecast.png",
        dpi=150,
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("CHRONOS-2 PROXY / FORECASTING POC")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = load_and_align_data()

    print(
        f"AAPL rows after alignment: {len(df)}"
    )

    print(
        f"Date range: {df['Date'].min().date()} "
        f"-> {df['Date'].max().date()}"
    )

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Load Chronos-2
    # --------------------------------------------------------

    print("\nLoading Chronos-2...")

    pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map=DEVICE,
    )

    print("Chronos-2 loaded.")

    # --------------------------------------------------------
    # Experiments 1, 2, 3
    # --------------------------------------------------------

    all_summaries = []

    for experiment in [1, 2, 3]:

        result = run_experiment(
            df,
            pipeline,
            experiment=experiment,
        )

        summary = summarize_forecasts(
            result
        )

        summary["experiment"] = experiment

        all_summaries.append(summary)

        print("\nSummary:")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}"
            )
        )

        calibration = summarize_calibration(
            result
        )

        calibration.to_csv(
            RESULTS_DIR
            / f"experiment_{experiment}_calibration.csv",
            index=False,
        )

    combined = pd.concat(
        all_summaries,
        ignore_index=True
    )

    combined.to_csv(
        RESULTS_DIR / "all_experiment_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Spike experiment
    # --------------------------------------------------------

    spikes = run_spike_experiment(
        df,
        pipeline,
    )

    spike_summary = summarize_spikes(
        spikes
    )

    spike_summary.to_csv(
        RESULTS_DIR / "spike_summary.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("SPIKE SUMMARY")
    print("=" * 80)
    print(
        spike_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Last-window visualization
    # --------------------------------------------------------

    print("\nCreating final visualization...")

    plot_last_forecast(
        df,
        pipeline,
        experiment=3,
        horizon=20,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Results are in: {RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
