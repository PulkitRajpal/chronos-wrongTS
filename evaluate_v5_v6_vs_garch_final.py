from __future__ import annotations

"""
V5 vs V6 vs GARCH
=================

Purpose:
    Clean final comparison of:
        1. Chronos-LoRA-V5
        2. Chronos-LoRA-V6
        3. GARCH

Same:
    - 10 held-out stocks
    - walk-forward evaluation
    - horizons 1/5/10/20

Main metrics:
    - endpoint price MAE
    - return absolute error
    - directional accuracy
    - stock-level win rate vs GARCH
    - V6 vs V5 win rate
    - exact forecasts
    - Chronos 80% marginal interval coverage

V5:
    target = daily log return
    covariates =
        SPY_RET, QQQ_RET, VIX_RET, US5Y_CHG, US10Y_CHG,
        realized_vol_20, volume_z

V6:
    target = daily log return
    covariates =
        SPY_RET, QQQ_RET, VIX_RET, US5Y_CHG, US10Y_CHG,
        ABS_RETURN, REALIZED_VOL_20, VOLUME_Z

IMPORTANT:
    - No future covariates are supplied.
    - V5/V6 use synthetic regular timestamps to avoid trading-calendar
      frequency inference problems.
    - Chronos adapter loading uses import_allowlist; base model does not.
"""

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from chronos import Chronos2Pipeline

try:
    from arch import arch_model
except ImportError as exc:
    raise RuntimeError(
        "Install arch first: pip install arch"
    ) from exc

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR = (
    BASE_DIR /
    "results_v5_vs_v6_vs_garch"
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

HORIZONS = [1, 5, 10, 20]

N_WINDOWS = 50
MIN_HISTORY = 300

V5_CONTEXT = 512
V6_CONTEXT = 512

Q_LEVELS = [0.10, 0.50, 0.90]

V5_COVARIATES = [
    "realized_vol_20",
    "volume_z",
]

V6_COVARIATES = [
    "SPY_RET",
    "QQQ_RET",
    "VIX_RET",
    "US5Y_CHG",
    "US10Y_CHG",
    "ABS_RETURN",
    "REALIZED_VOL_20",
    "VOLUME_Z",
]

V5_CHECKPOINT = (
    BASE_DIR /
    "models" /
    "chronos2-finance-lora-v5-multivariate" /
    "finetuned-ckpt"
)

V6_CHECKPOINT = (
    BASE_DIR /
    "models" /
    "chronos2-finance-lora-v6-diverse-512" /
    "finetuned-ckpt"
)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:
    if not torch.cuda.is_available():
        print("CUDA unavailable -> CPU")
        return "cpu"

    try:
        torch.randn(
            1,
            device="cuda",
        )
        print(
            "CUDA:",
            torch.cuda.get_device_name(0),
        )
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024 ** 3),
                2,
            ),
            "GB",
        )
        return "cuda"
    except Exception as exc:
        print(
            "CUDA test failed -> CPU:",
            exc,
        )
        return "cpu"


DEVICE = get_device()


# ============================================================
# DATA HELPERS
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}; "
        f"available={list(df.columns)}"
    )


def clean_number(value) -> float:
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s in {
        "",
        "-",
        "—",
        "NA",
        "N/A",
        "null",
        "None",
    }:
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
    except Exception:
        return np.nan


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
        ["close/last", "close", "last", "price"],
    )

    open_col = next(
        (
            c
            for c in [
                "open",
                "Open",
            ]
            if c in raw.columns
        ),
        None,
    )

    high_col = next(
        (
            c
            for c in [
                "high",
                "High",
            ]
            if c in raw.columns
        ),
        None,
    )

    low_col = next(
        (
            c
            for c in [
                "low",
                "Low",
            ]
            if c in raw.columns
        ),
        None,
    )

    volume_col = next(
        (
            c
            for c in [
                "volume",
                "Volume",
            ]
            if c in raw.columns
        ),
        None,
    )

    out = pd.DataFrame({
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
        out
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
# REFERENCE SERIES
# ============================================================

def flatten_yf(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    if isinstance(
        out.columns,
        pd.MultiIndex,
    ):

        lvl0 = [
            str(x)
            for x in
            out.columns.get_level_values(0)
        ]

        expected = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        }

        if expected.intersection(
            set(lvl0)
        ):

            out.columns = (
                out.columns
                .get_level_values(0)
            )

        else:

            out.columns = (
                out.columns
                .get_level_values(-1)
            )

    return out


def download_reference(
    ticker: str,
) -> pd.DataFrame:

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
            f"No reference data for {ticker}"
        )

    raw = flatten_yf(
        raw
    ).copy()

    # Prevent Date index-level/column ambiguity.
    raw.index.name = None

    if "Close" not in raw.columns:
        raise RuntimeError(
            f"Close missing for {ticker}"
        )

    dates = pd.to_datetime(
        raw.index,
        errors="coerce",
    )

    try:
        if getattr(
            dates,
            "tz",
            None,
        ) is not None:
            dates = dates.tz_localize(
                None
            )
    except Exception:
        pass

    out = pd.DataFrame({
        "Date":
            dates,
        "Value":
            pd.to_numeric(
                raw["Close"],
                errors="coerce",
            ).to_numpy(),
    })

    return (
        out
        .reset_index(drop=True)
        .dropna()
        .sort_values("Date")
        .drop_duplicates(
            "Date",
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

    parts = []

    for name, ticker in specs.items():

        print(
            f"Loading {name} ({ticker})"
        )

        x = download_reference(
            ticker
        )

        x = x.rename(
            columns={
                "Value":
                    name
            }
        )

        parts.append(x)

    ref = parts[0]

    for x in parts[1:]:

        ref = ref.merge(
            x,
            on="Date",
            how="outer",
        )

    ref = (
        ref
        .sort_values("Date")
        .drop_duplicates(
            "Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    levels = [
        "SPY",
        "QQQ",
        "VIX",
        "US5Y",
        "US10Y",
    ]

    ref[levels] = (
        ref[levels]
        .ffill(limit=3)
    )

    ref["SPY_RET"] = (
        np.log(ref["SPY"])
        .diff()
    )

    ref["QQQ_RET"] = (
        np.log(ref["QQQ"])
        .diff()
    )

    ref["VIX_RET"] = (
        np.log(ref["VIX"])
        .diff()
    )

    ref["US5Y_CHG"] = (
        ref["US5Y"]
        .diff()
    )

    ref["US10Y_CHG"] = (
        ref["US10Y"]
        .diff()
    )

    return (
        ref[
            [
                "Date",
                "SPY_RET",
                "QQQ_RET",
                "VIX_RET",
                "US5Y_CHG",
                "US10Y_CHG",
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )


# ============================================================
# WALK-FORWARD
# ============================================================

def get_cutoffs(
    n: int,
) -> list[int]:

    required = (
        MIN_HISTORY +
        max(HORIZONS)
    )

    if n <= required:
        raise ValueError(
            f"Only {n} rows; need > {required}"
        )

    candidates = np.arange(
        MIN_HISTORY,
        n - max(HORIZONS),
    )

    if len(candidates) <= N_WINDOWS:
        return [
            int(x)
            for x in candidates
        ]

    idx = np.linspace(
        0,
        len(candidates) - 1,
        N_WINDOWS,
        dtype=int,
    )

    return [
        int(candidates[i])
        for i in idx
    ]


# ============================================================
# CHRONOS CONTEXT
# ============================================================

def make_regular_context(
    df: pd.DataFrame,
    ticker: str,
    context_length: int,
) -> pd.DataFrame:

    d = df.tail(
        context_length
    ).copy()

    d = d.reset_index(
        drop=True
    )

    d["id"] = ticker

    # Regular timestamps eliminate market-calendar frequency inference.
    d["timestamp"] = pd.date_range(
        start="2000-01-01",
        periods=len(d),
        freq="D",
    )

    return d


def build_v5_context(
    history: pd.DataFrame,
    references: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:

    d = history.copy()

    d["target"] = (
        np.log(d["Close"])
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

    d = d.merge(
        references,
        on="Date",
        how="left",
    )

    ref_cols = [
        "SPY_RET",
        "QQQ_RET",
        "VIX_RET",
        "US5Y_CHG",
        "US10Y_CHG",
    ]

    d[ref_cols] = (
        d[ref_cols]
        .ffill(limit=3)
    )

    required = [
        "target",
        *ref_cols,
        "realized_vol_20",
        "volume_z",
    ]

    d = (
        d
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
        )
        .reset_index(drop=True)
    )

    return make_regular_context(
        d,
        ticker,
        V5_CONTEXT,
    )


def build_v6_context(
    history: pd.DataFrame,
    references: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:

    d = history.copy()

    d["target"] = (
        np.log(d["Close"])
        .diff()
    )

    d["ABS_RETURN"] = (
        d["target"].abs()
    )

    d["REALIZED_VOL_20"] = (
        d["target"]
        .rolling(20)
        .std()
    )

    log_volume = np.log1p(
        d["Volume"].clip(
            lower=0
        )
    )

    d["VOLUME_Z"] = (
        (
            log_volume
            -
            log_volume.rolling(20).mean()
        )
        /
        log_volume.rolling(20).std()
    )

    d = d.merge(
        references,
        on="Date",
        how="left",
    )

    ref_cols = [
        "SPY_RET",
        "QQQ_RET",
        "VIX_RET",
        "US5Y_CHG",
        "US10Y_CHG",
    ]

    d[ref_cols] = (
        d[ref_cols]
        .ffill(limit=3)
    )

    required = [
        "target",
        *ref_cols,
        "ABS_RETURN",
        "REALIZED_VOL_20",
        "VOLUME_Z",
    ]

    d = (
        d
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
        )
        .reset_index(drop=True)
    )

    return make_regular_context(
        d,
        ticker,
        V6_CONTEXT,
    )


# ============================================================
# CHRONOS PREDICTION
# ============================================================

def prediction_from_dataframe(
    pipeline,
    context: pd.DataFrame,
    horizon: int,
    context_length: int,
):

    # This is the most stable path in chronos 2.3.1 for the user's
    # installed environment because the timestamp sequence is regular.
    pred = pipeline.predict_df(
        context,
        future_df=None,
        prediction_length=horizon,
        quantile_levels=Q_LEVELS,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        context_length=min(
            context_length,
            len(context),
        ),
        validate_inputs=True,
    )

    return (
        pred[
            str(Q_LEVELS[0])
        ].to_numpy(float),

        pred[
            str(Q_LEVELS[1])
        ].to_numpy(float),

        pred[
            str(Q_LEVELS[2])
        ].to_numpy(float),
    )


def chronos_predict(
    model_name: str,
    pipeline,
    history: pd.DataFrame,
    references: pd.DataFrame,
    ticker: str,
    horizon: int,
):

    if model_name == "Chronos-LoRA-V5":

        context = build_v5_context(
            history,
            references,
            ticker,
        )

        return prediction_from_dataframe(
            pipeline,
            context,
            horizon,
            V5_CONTEXT,
        )

    if model_name == "Chronos-LoRA-V6":

        context = build_v6_context(
            history,
            references,
            ticker,
        )

        return prediction_from_dataframe(
            pipeline,
            context,
            horizon,
            V6_CONTEXT,
        )

    raise ValueError(
        model_name
    )


# ============================================================
# GARCH
# ============================================================

def fit_garch(
    history: pd.DataFrame,
):

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

            fitted = model.fit(
                disp="off",
                update_freq=0,
                options={
                    "maxiter": 2000
                },
            )

            if getattr(
                fitted,
                "convergence_flag",
                0,
            ) == 0:

                return (
                    fitted,
                    "success",
                )

            last_error = (
                "nonconverged"
            )

        except Exception as exc:

            last_error = str(
                exc
            )

    return (
        None,
        f"failed:{last_error}",
    )


def garch_predict(
    fitted,
    last_price: float,
    horizon: int,
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

    return float(
        last_price
        *
        np.exp(
            mean_path.sum()
            /
            100.0
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def score(
    actual: float,
    last_price: float,
    prediction: float,
):

    if not np.isfinite(
        prediction
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    actual_return = (
        actual
        /
        last_price
        -
        1
    )

    predicted_return = (
        prediction
        /
        last_price
        -
        1
    )

    return (
        abs(
            actual
            -
            prediction
        ),

        abs(
            actual_return
            -
            predicted_return
        ),

        float(
            np.sign(
                actual_return
            )
            ==
            np.sign(
                predicted_return
            )
        ),
    )


def evaluate_chronos(
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
        f"\n{ticker} / {model_name} "
        f"({len(cutoffs)} windows)"
    )

    for wi, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = stock.iloc[
            :cutoff
        ].copy()

        if (
            wi == 1
            or
            wi == len(cutoffs)
            or
            wi % 10 == 0
        ):

            print(
                f"  {wi}/{len(cutoffs)}"
            )

        last_price = float(
            history[
                "Close"
            ].iloc[-1]
        )

        for horizon in HORIZONS:

            future = stock.iloc[
                cutoff:
                cutoff + horizon
            ]

            if len(future) < horizon:
                continue

            actual = float(
                future[
                    "Close"
                ].iloc[-1]
            )

            try:

                q10, q50, q90 = (
                    chronos_predict(
                        model_name,
                        pipeline,
                        history,
                        references,
                        ticker,
                        horizon,
                    )
                )

                prediction = float(
                    last_price
                    *
                    np.exp(
                        q50.sum()
                    )
                )

                mae, return_error, direction = (
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
                    "last_price":
                        last_price,
                    "prediction":
                        prediction,
                    "actual":
                        actual,
                    "price_mae":
                        mae,
                    "return_abs_error":
                        return_error,
                    "direction_hit":
                        direction,
                })

                combined = np.concatenate([
                    np.array([
                        last_price
                    ]),
                    future[
                        "Close"
                    ].astype(float).to_numpy(),
                ])

                realized = np.diff(
                    np.log(combined)
                )

                for lead in range(
                    1,
                    horizon + 1,
                ):

                    j = lead - 1

                    if j >= len(q10):
                        break

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
                            float(
                                realized[j]
                            ),
                        "q10":
                            float(
                                q10[j]
                            ),
                        "q50":
                            float(
                                q50[j]
                            ),
                        "q90":
                            float(
                                q90[j]
                            ),
                        "inside_80":
                            float(
                                q10[j]
                                <=
                                realized[j]
                                <=
                                q90[j]
                            ),
                    })

            except Exception as exc:

                print(
                    f"    WARNING "
                    f"{model_name} "
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
                    "last_price":
                        last_price,
                    "prediction":
                        np.nan,
                    "actual":
                        actual,
                    "price_mae":
                        np.nan,
                    "return_abs_error":
                        np.nan,
                    "direction_hit":
                        np.nan,
                })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(interval_rows),
    )


def evaluate_garch(
    stock: pd.DataFrame,
    ticker: str,
):

    rows = []
    diagnostics = []

    cutoffs = get_cutoffs(
        len(stock)
    )

    print(
        f"\n{ticker} / GARCH "
        f"({len(cutoffs)} windows)"
    )

    for wi, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        history = stock.iloc[
            :cutoff
        ].copy()

        if (
            wi == 1
            or
            wi == len(cutoffs)
            or
            wi % 10 == 0
        ):

            print(
                f"  {wi}/{len(cutoffs)}"
            )

        fitted, status = fit_garch(
            history
        )

        diagnostics.append({
            "ticker":
                ticker,
            "cutoff_date":
                history[
                    "Date"
                ].iloc[-1],
            "model":
                "GARCH",
            "status":
                status,
        })

        last_price = float(
            history[
                "Close"
            ].iloc[-1]
        )

        for horizon in HORIZONS:

            future = stock.iloc[
                cutoff:
                cutoff + horizon
            ]

            if len(future) < horizon:
                continue

            actual = float(
                future[
                    "Close"
                ].iloc[-1]
            )

            if fitted is None:

                prediction = np.nan

            else:

                try:

                    prediction = (
                        garch_predict(
                            fitted,
                            last_price,
                            horizon,
                        )
                    )

                except Exception:

                    prediction = np.nan

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
                    "GARCH",
                "last_price":
                    last_price,
                "prediction":
                    prediction,
                "actual":
                    actual,
                "price_mae":
                    mae,
                "return_abs_error":
                    ret_err,
                "direction_hit":
                    direction,
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(diagnostics),
    )


# ============================================================
# SUMMARIES
# ============================================================

def overall_summary(
    points: pd.DataFrame,
) -> pd.DataFrame:

    return (
        points
        .groupby([
            "horizon",
            "model",
        ])
        .agg(
            observations=(
                "price_mae",
                "count",
            ),
            stocks=(
                "ticker",
                "nunique",
            ),
            avg_mae=(
                "price_mae",
                "mean",
            ),
            median_mae=(
                "price_mae",
                "median",
            ),
            avg_return_abs_error=(
                "return_abs_error",
                "mean",
            ),
            directional_accuracy=(
                "direction_hit",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "horizon",
                "avg_mae",
            ]
        )
    )


def pairwise_wins(
    points: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:

    a = points[
        points["model"] == model_a
    ][
        [
            "ticker",
            "cutoff_date",
            "horizon",
            "price_mae",
        ]
    ].rename(
        columns={
            "price_mae":
                "mae_a"
        }
    )

    b = points[
        points["model"] == model_b
    ][
        [
            "ticker",
            "cutoff_date",
            "horizon",
            "price_mae",
        ]
    ].rename(
        columns={
            "price_mae":
                "mae_b"
        }
    )

    merged = a.merge(
        b,
        on=[
            "ticker",
            "cutoff_date",
            "horizon",
        ],
        how="inner",
    ).dropna()

    rows = []

    for horizon, g in merged.groupby(
        "horizon"
    ):

        wins = (
            g["mae_a"]
            <
            g["mae_b"]
        ).sum()

        losses = (
            g["mae_a"]
            >
            g["mae_b"]
        ).sum()

        ties = (
            len(g)
            -
            wins
            -
            losses
        )

        rows.append({
            "model_a":
                model_a,
            "model_b":
                model_b,
            "horizon":
                horizon,
            "comparisons":
                len(g),
            "model_a_wins":
                int(wins),
            "model_b_wins":
                int(losses),
            "ties":
                int(ties),
            "win_rate_excluding_ties":
                (
                    wins /
                    (wins + losses)
                    if wins + losses
                    else np.nan
                ),
        })

    return pd.DataFrame(rows)


def stock_win_table(
    points: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:

    a = points[
        points["model"] == model_a
    ][
        [
            "ticker",
            "horizon",
            "price_mae",
        ]
    ].rename(
        columns={
            "price_mae":
                "mae_a"
        }
    )

    b = points[
        points["model"] == model_b
    ][
        [
            "ticker",
            "horizon",
            "price_mae",
        ]
    ].rename(
        columns={
            "price_mae":
                "mae_b"
        }
    )

    a = (
        a.groupby(
            [
                "ticker",
                "horizon",
            ]
        )["mae_a"]
        .mean()
        .reset_index()
    )

    b = (
        b.groupby(
            [
                "ticker",
                "horizon",
            ]
        )["mae_b"]
        .mean()
        .reset_index()
    )

    m = (
        a.merge(
            b,
            on=[
                "ticker",
                "horizon",
            ],
            how="inner",
        )
        .dropna()
    )

    m["winner"] = np.select(
        [
            m["mae_a"] <
            m["mae_b"],
            m["mae_b"] <
            m["mae_a"],
        ],
        [
            model_a,
            model_b,
        ],
        default="Tie",
    )

    return m


def calibration_summary(
    intervals: pd.DataFrame,
) -> pd.DataFrame:

    if intervals.empty:
        return pd.DataFrame()

    return (
        intervals
        .groupby([
            "model",
            "horizon",
            "lead",
        ])
        .agg(
            observations=(
                "inside_80",
                "size",
            ),
            coverage=(
                "inside_80",
                "mean",
            ),
            avg_interval_width=(
                lambda x: np.nan
            ),
        )
        .reset_index()
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
        "V5 vs V6 vs GARCH"
    )
    print("=" * 90)

    print(
        f"Stocks: {len(TICKERS)}"
    )

    print(
        f"Horizons: {HORIZONS}"
    )

    print(
        f"Walk-forward windows: {N_WINDOWS}"
    )

    print()

    if not V5_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V5 checkpoint missing:\n"
            f"{V5_CHECKPOINT.resolve()}"
        )

    if not V6_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V6 checkpoint missing:\n"
            f"{V6_CHECKPOINT.resolve()}"
        )

    references = load_references()

    all_points = []
    all_intervals = []
    all_diagnostics = []

    # --------------------------------------------------------
    # V5 + V6
    # --------------------------------------------------------

    specs = [
        (
            "Chronos-LoRA-V5",
            V5_CHECKPOINT,
        ),
        (
            "Chronos-LoRA-V6",
            V6_CHECKPOINT,
        ),
    ]

    for model_name, checkpoint in specs:

        print()
        print("=" * 90)
        print(
            f"LOADING {model_name}"
        )
        print("=" * 90)

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

                p, i = (
                    evaluate_chronos(
                        model_name,
                        pipeline,
                        stock,
                        ticker,
                        references,
                    )
                )

                if not p.empty:
                    all_points.append(p)

                if not i.empty:
                    all_intervals.append(i)

            except Exception as exc:

                print(
                    f"[ERROR] "
                    f"{ticker}/{model_name}: "
                    f"{exc}"
                )

        del pipeline
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # GARCH
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "EVALUATING GARCH"
    )
    print("=" * 90)

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

            p, d = evaluate_garch(
                stock,
                ticker,
            )

            if not p.empty:
                all_points.append(p)

            if not d.empty:
                all_diagnostics.append(d)

        except Exception as exc:

            print(
                f"[ERROR] "
                f"{ticker}/GARCH: "
                f"{exc}"
            )

    if not all_points:

        raise RuntimeError(
            "No forecast results generated."
        )

    points = pd.concat(
        all_points,
        ignore_index=True,
    )

    points.to_csv(
        RESULTS_DIR /
        "all_exact_forecasts.csv",
        index=False,
    )

    summary = overall_summary(
        points
    )

    summary.to_csv(
        RESULTS_DIR /
        "v5_v6_garch_overall.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise comparisons.
    # --------------------------------------------------------

    pairs = []

    for a, b in [
        (
            "Chronos-LoRA-V5",
            "GARCH",
        ),
        (
            "Chronos-LoRA-V6",
            "GARCH",
        ),
        (
            "Chronos-LoRA-V6",
            "Chronos-LoRA-V5",
        ),
    ]:

        p = pairwise_wins(
            points,
            a,
            b,
        )

        if not p.empty:
            pairs.append(p)

    pairwise = (
        pd.concat(
            pairs,
            ignore_index=True,
        )
        if pairs
        else pd.DataFrame()
    )

    pairwise.to_csv(
        RESULTS_DIR /
        "pairwise_win_rates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Stock-level comparisons.
    # --------------------------------------------------------

    stock_pairs = []

    for a, b in [
        (
            "Chronos-LoRA-V5",
            "GARCH",
        ),
        (
            "Chronos-LoRA-V6",
            "GARCH",
        ),
        (
            "Chronos-LoRA-V6",
            "Chronos-LoRA-V5",
        ),
    ]:

        p = stock_win_table(
            points,
            a,
            b,
        )

        if not p.empty:

            p["comparison"] = (
                f"{a} vs {b}"
            )

            stock_pairs.append(p)

    if stock_pairs:

        pd.concat(
            stock_pairs,
            ignore_index=True,
        ).to_csv(
            RESULTS_DIR /
            "stock_level_win_comparisons.csv",
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
            RESULTS_DIR /
            "chronos_intervals.csv",
            index=False,
        )

    # --------------------------------------------------------
    # GARCH diagnostics.
    # --------------------------------------------------------

    if all_diagnostics:

        diagnostics = pd.concat(
            all_diagnostics,
            ignore_index=True,
        )

        diagnostics.to_csv(
            RESULTS_DIR /
            "garch_diagnostics.csv",
            index=False,
        )

        fit_summary = (
            diagnostics
            .assign(
                success=(
                    diagnostics[
                        "status"
                    ]
                    ==
                    "success"
                )
            )
            .groupby("model")
            .agg(
                windows=(
                    "success",
                    "size",
                ),
                successful=(
                    "success",
                    "sum",
                ),
            )
            .reset_index()
        )

        fit_summary[
            "success_rate"
        ] = (
            fit_summary[
                "successful"
            ]
            /
            fit_summary[
                "windows"
            ]
        )

        fit_summary.to_csv(
            RESULTS_DIR /
            "garch_fit_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "V5 / V6 / GARCH OVERALL"
    )
    print("=" * 90)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print()
    print("=" * 90)
    print(
        "PAIRWISE WIN RATES"
    )
    print("=" * 90)

    if not pairwise.empty:

        print(
            pairwise.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
        )

    print()
    print(
        "Results:",
        RESULTS_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
