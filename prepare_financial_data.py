from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/chronos_finance")

# These are your FINAL evaluation stocks.
# Do NOT use them for LoRA training if you want a clean
# out-of-sample test.
EVAL_TICKERS = {
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
}

# Broad public financial training universe.
#
# The goal is not to predict these assets specifically.
# The goal is to expose Chronos to financial time-series
# patterns across different types of instruments.
TRAIN_TICKERS = [
    # Large-cap equities
    "GOOG", "META", "AMZN", "TSLA", "NFLX", "ORCL", "ADBE",
    "CRM", "INTC", "AMD", "AVGO", "QCOM", "TXN", "MU",
    "CSCO", "IBM", "JPM", "BAC", "GS", "MS", "C", "WFC",
    "V", "MA", "AXP", "KO", "PEP", "WMT", "COST", "MCD",
    "NKE", "DIS", "SBUX", "BA", "CAT", "GE", "HON",
    "XOM", "CVX", "COP", "SLB", "PFE", "MRK", "JNJ",
    "ABBV", "LLY", "UNH", "TMO",

    # More volatile / growth names
    "COIN", "MSTR", "PLTR", "SNOW", "SHOP", "SQ", "ROKU",
    "SOFI", "DKNG", "RIVN", "LCID", "NIO", "MARA", "RIOT",

    # ETFs / indices
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EEM",
    "TLT", "IEF", "HYG", "LQD",

    # Commodities / commodity ETFs
    "GLD", "SLV", "USO", "UNG", "DBC",

    # Volatility
    "^VIX",

    # Rates / indices where supported
    "^TNX",
    "^TYX",
]

START_DATE = "2015-01-01"

# IMPORTANT:
# Training cutoff should be BEFORE your evaluation period.
#
# Your current benchmark uses 2026 data, so we deliberately
# stop financial fine-tuning data before 2026.
TRAIN_END_DATE = "2025-12-31"

# Validation is carved out from the last part of the
# training period.
CONTEXT_LENGTH = 256
VALIDATION_DAYS = 252
PREDICTION_LENGTH = 20
NUM_STEPS = 5
BATCH_SIZE = 1

# Remove very short series.
MIN_OBSERVATIONS = 500

# Numerical stability.
MAX_ABS_RETURN = 0.50


# ============================================================
# HELPERS
# ============================================================

def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance's output into simple OHLCV columns.

    Handles both:
      - ordinary columns
      - MultiIndex columns when multiple tickers are downloaded
    """
    if isinstance(df.columns, pd.MultiIndex):
        # This function is primarily intended for per-ticker
        # downloads, but handling this makes it safer.
        if len(df.columns.levels) > 1:
            df.columns = [
                "_".join(str(x) for x in col if str(x) != "")
                for col in df.columns
            ]

    # Normalize capitalization
    df = df.rename(columns={c: str(c).lower() for c in df.columns})

    return df


def download_one_ticker(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Download one ticker and convert close prices into log returns.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"[WARN] No data returned for {ticker}")
            return None

        # yfinance can occasionally return MultiIndex even for
        # a single ticker depending on version/configuration.
        if isinstance(df.columns, pd.MultiIndex):
            # For a single ticker, try to extract the ticker level.
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)

        df = flatten_yfinance_columns(df)

        if "close" not in df.columns:
            print(f"[WARN] No close column for {ticker}")
            return None

        df = df[["close"]].copy()

        # Make date explicit.
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Remove duplicate timestamps.
        df = df[~df.index.duplicated(keep="last")]

        # Remove missing values.
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        if len(df) < MIN_OBSERVATIONS:
            print(
                f"[WARN] {ticker}: only {len(df)} observations "
                f"(minimum {MIN_OBSERVATIONS})"
            )
            return None

        # -----------------------------------------
        # LOG RETURNS
        # -----------------------------------------
        #
        # r_t = log(P_t / P_{t-1})
        #
        # This is the target we want Chronos to learn.
        df["target"] = np.log(df["close"] / df["close"].shift(1))

        df = df.dropna(subset=["target"])

        # Remove extreme bad data points.
        df = df[np.isfinite(df["target"])]
        df = df[df["target"].abs() <= MAX_ABS_RETURN]

        # Chronos-2 dataframe format:
        # id | timestamp | target
        out = pd.DataFrame({
            "id": ticker,
            "timestamp": df.index,
            "target": df["target"].astype(np.float32).values,
        })

        return out.reset_index(drop=True)

    except Exception as exc:
        print(f"[ERROR] Failed {ticker}: {exc}")
        return None
def split_by_time(
    df: pd.DataFrame,
    validation_days: int,
    context_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_parts = []
    val_parts = []

    for ticker, group in df.groupby("id", sort=False):

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(group) <= validation_days + context_length:
            print(
                f"[WARN] {ticker}: too short "
                f"({len(group)} rows)"
            )
            continue

        # Last `validation_days` observations are the
        # actual validation targets.
        split_idx = len(group) - validation_days

        # Training ends BEFORE the validation target period.
        train_group = group.iloc[:split_idx].copy()

        # Validation needs historical context immediately
        # before the validation period.
        #
        # Example:
        #
        #       training            validation targets
        # ────────────────────|──────────────────────────
        #                     ↑
        #                 split_idx
        #
        # Validation gets:
        #
        #       context       | validation targets
        #       256 rows      | 252 rows
        # ────────────────────|──────────────────────────
        #
        val_start = max(0, split_idx - context_length)

        val_group = group.iloc[val_start:].copy()

        train_parts.append(train_group)
        val_parts.append(val_group)

    if not train_parts or not val_parts:
        raise RuntimeError(
            "No valid series remained after splitting."
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_df = pd.concat(
        val_parts,
        ignore_index=True,
    )

    return train_df, validation_df


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """
    Sanity checks before saving.
    """
    required = {"id", "timestamp", "target"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{name}: dataset is empty")

    if df["target"].isna().any():
        raise ValueError(f"{name}: NaN targets found")

    if not np.isfinite(df["target"].to_numpy()).all():
        raise ValueError(f"{name}: non-finite targets found")

    # Check chronological ordering per series.
    for ticker, group in df.groupby("id"):
        timestamps = pd.to_datetime(group["timestamp"])
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"{name}: timestamps are not sorted for {ticker}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove accidental overlap with evaluation set.
    training_tickers = [
        t for t in TRAIN_TICKERS
        if t not in EVAL_TICKERS
    ]

    print("=" * 70)
    print("CHRONOS-2 FINANCIAL FINE-TUNING DATA PREPARATION")
    print("=" * 70)

    print(f"Training tickers: {len(training_tickers)}")
    print(f"Evaluation tickers held out: {sorted(EVAL_TICKERS)}")
    print(f"Historical range: {START_DATE} -> {TRAIN_END_DATE}")
    print()

    all_series: list[pd.DataFrame] = []

    for ticker in tqdm(
        training_tickers,
        desc="Downloading financial series",
    ):
        data = download_one_ticker(
            ticker=ticker,
            start=START_DATE,
            end=TRAIN_END_DATE,
        )

        if data is not None:
            all_series.append(data)

    if not all_series:
        raise RuntimeError(
            "No financial data was successfully downloaded."
        )

    full_df = pd.concat(
        all_series,
        ignore_index=True,
    )

    # Sort properly.
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

    full_df = full_df.sort_values(
        ["id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # TIME SPLIT
    # --------------------------------------------------------

    train_df, validation_df = split_by_time(
    full_df,
    validation_days=VALIDATION_DAYS,
    context_length=CONTEXT_LENGTH
)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_dataset(full_df, "full")
    validate_dataset(train_df, "train")
    validate_dataset(validation_df, "validation")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.parquet"
    val_path = OUTPUT_DIR / "validation.parquet"
    full_path = OUTPUT_DIR / "all_financial_series.parquet"

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        val_path,
        index=False,
    )

    full_df.to_parquet(
        full_path,
        index=False,
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    summary = {
        "num_training_tickers": int(train_df["id"].nunique()),
        "num_validation_tickers": int(validation_df["id"].nunique()),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "date_min": str(full_df["timestamp"].min()),
        "date_max": str(full_df["timestamp"].max()),
        "target": "log_return",
        "max_abs_return_filter": MAX_ABS_RETURN,
        "evaluation_tickers": sorted(EVAL_TICKERS),
        "training_tickers": sorted(train_df["id"].unique().tolist()),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Total series:      {full_df['id'].nunique()}"
    )
    print(
        f"Total observations:{len(full_df):,}"
    )
    print(
        f"Train observations:{len(train_df):,}"
    )
    print(
        f"Val observations:  {len(validation_df):,}"
    )

    print()
    print("Observations per ticker:")
    print(
        train_df.groupby("id")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print()
    print(f"Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {full_path}")
    print(f"  {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()