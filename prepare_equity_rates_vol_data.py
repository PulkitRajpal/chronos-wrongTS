# prepare_broad_financial_data.py

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/chronos_finance_equity_rates_vol")

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# FINAL EVALUATION STOCKS
# These must NEVER enter the LoRA training dataset.
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


# ============================================================
# FINANCIAL UNIVERSE
# Crypto intentionally excluded.
# ============================================================

ASSET_UNIVERSE = {

    # --------------------------------------------------------
    # US listed equities — large cap
    # --------------------------------------------------------
    "equity_large": [
        "GOOG", "META", "TSLA", "NFLX", "ORCL", "ADBE",
        "CRM", "INTC", "AMD", "AVGO", "QCOM", "TXN",
        "CSCO", "IBM", "JPM", "BAC", "GS", "MS", "C",
        "WFC", "V", "MA", "AXP", "COF",
        "KO", "PEP", "WMT", "COST", "MCD",
        "NKE", "DIS", "SBUX", "BA", "CAT", "GE", "HON",
        "XOM", "CVX", "COP", "SLB",
        "PFE", "MRK", "JNJ", "ABBV", "LLY", "UNH",
        "TMO", "DHR", "ABT", "BMY",
    ],

    # --------------------------------------------------------
    # US listed equities — growth / volatile
    # --------------------------------------------------------
    "equity_growth": [
        "COIN", "MSTR", "PLTR", "SNOW", "SHOP", "SQ",
        "ROKU", "SOFI", "DKNG", "RIVN", "LCID", "NIO",
        "MARA", "RIOT", "HOOD", "AFRM", "UPST",
        "CRWD", "NET", "DDOG", "PATH", "U",
        "GME", "AMC", "TQQQ", "SQQQ",
    ],

    # --------------------------------------------------------
    # International listed equities / ADRs
    # --------------------------------------------------------
    "international_equity": [
        "BABA", "JD", "PDD", "NIO",
        "SONY", "TM", "HMC", "TCEHY",
        "SAP", "ASML", "NVS", "AZN",
        "BP", "SHEL", "RIO", "BHP",
        "VALE", "UL", "DEO",
    ],

    # --------------------------------------------------------
    # Equity ETFs / indices
    # --------------------------------------------------------
    "equity_etf": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
        "EEM", "EFA", "VEA", "VWO",
        "IVV", "RSP", "MDY",
        "XLF", "XLK", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLB", "XLU",
    ],

    # --------------------------------------------------------
    # Rates / bonds
    # --------------------------------------------------------
    "rates_bonds": [
        "TLT", "IEF", "SHY", "TIP",
        "LQD", "HYG", "AGG", "BND",
        "GOVT", "EDV", "TMF", "TMV",
        "^TNX", "^TYX",
    ],

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------
    "volatility": [
        "^VIX",
        "^VIX3M",
        "^VVIX",
    ],
}


# ============================================================
# PARAMETERS
# ============================================================

CONTEXT_LENGTH = 256
PREDICTION_LENGTH = 20

# Last 252 observations are validation targets.
VALIDATION_DAYS = 252

# Remove obviously bad/extreme data points.
MAX_ABS_RETURN = 0.50

# Minimum observations after cleaning.
MIN_OBSERVATIONS = (
    CONTEXT_LENGTH
    + PREDICTION_LENGTH
    + VALIDATION_DAYS
    + 100
)


# ============================================================
# HELPERS
# ============================================================

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns, including MultiIndex output."""

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [str(c[0]) for c in df.columns]

    df.columns = [str(c).lower() for c in df.columns]

    return df


def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download a ticker and return Chronos-ready log returns."""

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"[WARN] No data: {ticker}")
            return None

        df = flatten_columns(df)

        if "close" not in df.columns:
            print(f"[WARN] No Close column: {ticker}")
            return None

        df = df[["close"]].copy()

        # ----------------------------------------------------
        # Cleaning
        # ----------------------------------------------------

        df.index = pd.to_datetime(df.index)

        df = (
            df
            .sort_index()
            .loc[~df.index.duplicated(keep="last")]
        )

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        df = df.dropna(subset=["close"])

        # ----------------------------------------------------
        # LOG RETURNS
        # ----------------------------------------------------

        df["target"] = np.log(
            df["close"] / df["close"].shift(1)
        )

        df = df.dropna(subset=["target"])

        df = df[np.isfinite(df["target"])]

        # Remove obviously bad/extreme observations.
        df = df[df["target"].abs() <= MAX_ABS_RETURN]

        if len(df) < MIN_OBSERVATIONS:
            print(
                f"[SKIP] {ticker}: "
                f"only {len(df)} observations"
            )
            return None

        return pd.DataFrame({
            "id": ticker,
            "timestamp": df.index,
            "target": df["target"].astype(np.float32),
        }).reset_index(drop=True)

    except Exception as exc:
        print(f"[ERROR] {ticker}: {exc}")
        return None


# ============================================================
# TIME SPLIT
# ============================================================

def split_series(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically.

    Validation contains:
      256 context observations
      +
      252 validation target observations
    """

    train_parts = []
    validation_parts = []

    for ticker, group in df.groupby("id", sort=False):

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        required = CONTEXT_LENGTH + VALIDATION_DAYS

        if len(group) <= required:
            print(
                f"[SKIP SPLIT] {ticker}: "
                f"{len(group)} rows"
            )
            continue

        split_idx = len(group) - VALIDATION_DAYS

        # All observations before the validation target period.
        train_group = group.iloc[:split_idx].copy()

        # Validation gets the historical context immediately
        # preceding the validation target period.
        val_start = max(
            0,
            split_idx - CONTEXT_LENGTH,
        )

        validation_group = group.iloc[val_start:].copy()

        train_parts.append(train_group)
        validation_parts.append(validation_group)

    if not train_parts:
        raise RuntimeError(
            "No training series survived split."
        )

    if not validation_parts:
        raise RuntimeError(
            "No validation series survived split."
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    return train_df, validation_df


# ============================================================
# VALIDATION
# ============================================================

def validate(
    df: pd.DataFrame,
    name: str,
) -> None:

    required = {
        "id",
        "timestamp",
        "target",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{name}: empty dataset"
        )

    if df["target"].isna().any():
        raise ValueError(
            f"{name}: NaN target"
        )

    if not np.isfinite(
        df["target"].to_numpy()
    ).all():
        raise ValueError(
            f"{name}: non-finite targets"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Flatten ticker universe and remove evaluation assets.
    # --------------------------------------------------------

    all_tickers = []
    ticker_to_asset_class = {}

    for asset_class, tickers in ASSET_UNIVERSE.items():

        for ticker in tickers:

            # Evaluation leakage prevention.
            if ticker in EVAL_TICKERS:
                continue

            if ticker not in all_tickers:
                all_tickers.append(ticker)
                ticker_to_asset_class[ticker] = asset_class

    print("=" * 70)
    print("EQUITY + RATES + VOLATILITY DATASET")
    print("=" * 70)

    print(
        f"Candidate tickers: {len(all_tickers)}"
    )

    print(
        f"Held-out evaluation tickers: "
        f"{len(EVAL_TICKERS)}"
    )

    print()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    series = []

    for ticker in tqdm(
        all_tickers,
        desc="Downloading market data",
    ):

        data = download_ticker(ticker)

        if data is not None:

            data["asset_class"] = (
                ticker_to_asset_class[ticker]
            )

            series.append(data)

        # Small delay to reduce request pressure.
        time.sleep(0.05)

    if not series:
        raise RuntimeError(
            "No financial series downloaded."
        )

    full_df = pd.concat(
        series,
        ignore_index=True,
    )

    full_df["timestamp"] = pd.to_datetime(
        full_df["timestamp"]
    )

    full_df = (
        full_df
        .sort_values(["id", "timestamp"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, validation_df = split_series(
        full_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate(full_df, "full")
    validate(train_df, "train")
    validate(validation_df, "validation")

    # --------------------------------------------------------
    # Ensure evaluation tickers are absent.
    # --------------------------------------------------------

    train_ids = set(
        train_df["id"].unique()
    )

    validation_ids = set(
        validation_df["id"].unique()
    )

    leakage = (
        train_ids | validation_ids
    ) & EVAL_TICKERS

    if leakage:
        raise RuntimeError(
            "EVALUATION LEAKAGE DETECTED: "
            f"{sorted(leakage)}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"
    full_path = OUTPUT_DIR / "all_financial_series.parquet"

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        validation_path,
        index=False,
    )

    full_df.to_parquet(
        full_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "validation_days": VALIDATION_DAYS,
        "target": "log_return",
        "max_abs_return": MAX_ABS_RETURN,
        "candidate_tickers": len(all_tickers),
        "successful_tickers": int(full_df["id"].nunique()),
        "train_tickers": int(train_df["id"].nunique()),
        "validation_tickers": int(
            validation_df["id"].nunique()
        ),
        "held_out_tickers": sorted(EVAL_TICKERS),
        "asset_class_counts": {
            str(k): int(v)
            for k, v in (
                full_df.groupby("asset_class")["id"]
                .nunique()
                .items()
            )
        },
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Successful series: "
        f"{full_df['id'].nunique()}"
    )

    print(
        f"Training series: "
        f"{train_df['id'].nunique()}"
    )

    print(
        f"Validation series: "
        f"{validation_df['id'].nunique()}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print()
    print("Asset-class distribution:")

    print(
        full_df
        .groupby("asset_class")["id"]
        .nunique()
        .sort_values(ascending=False)
        .to_string()
    )

    print()
    print("Saved:")
    print(train_path)
    print(validation_path)
    print(full_path)
    print(OUTPUT_DIR / "metadata.json")

    print()
    print("Evaluation stocks excluded:")
    print(sorted(EVAL_TICKERS))


if __name__ == "__main__":
    main()
