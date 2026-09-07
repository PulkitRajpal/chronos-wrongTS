from __future__ import annotations

"""
CHRONOS-2 V6 ROBUST TRAINING SCRIPT
===================================

This version fixes the two most likely causes of:
    Usable series: 0
    Train inputs: 0
    Validation inputs: 0

1. It accepts the V6 parquet layout produced by the V6 data-preparation
   script and checks the files before preprocessing.
2. It converts actual trading dates to a REGULAR synthetic daily index for
   Chronos-2's dataframe preprocessing, avoiding frequency-inference
   failures caused by weekends/holidays.
3. It prints the exact reason each series is skipped.
4. It never silently treats an empty dataset as valid.

Expected layout:
    data/
        chronos_finance_v6/
            AAPL/              <-- should NOT exist for held-out stocks
                train.parquet
                validation.parquet
            ABBV/
                train.parquet
                validation.parquet
            ...

Run:
    python train_chronos_lora_v6_robust.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame
from peft import LoraConfig


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "amazon/chronos-2"

DATA_DIR = Path(
    "data/chronos_finance_v6"
)

OUTPUT_DIR = Path(
    "models/chronos2-finance-lora-v6-diverse-512"
)

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 20

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

LEARNING_RATE = 2e-5
NUM_STEPS = 2000

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LR_SCHEDULER_TYPE = (
    "constant_with_warmup"
)

WARMUP_STEPS = 100

SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3

TARGET_COLUMN = "target"

COVARIATE_COLUMNS = [
    "SPY_RET",
    "QQQ_RET",
    "VIX_RET",
    "US5Y_CHG",
    "US10Y_CHG",
    "ABS_RETURN",
    "REALIZED_VOL_20",
    "VOLUME_Z",
]

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
# DATAFRAME NORMALIZATION
# ============================================================

def normalize_series(
    raw: pd.DataFrame,
    ticker: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Normalize one parquet file.

    We deliberately DO NOT feed the original market-calendar dates to
    Chronos preprocessing. Trading dates have weekend/holiday gaps, which
    may make frequency inference fail.

    Numerical observations are untouched.

    Only the timestamp labels are replaced later with a regular synthetic
    sequence.
    """

    df = raw.copy()

    required = {
        TARGET_COLUMN,
        *COVARIATE_COLUMNS,
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"missing columns "
            f"{sorted(missing)}; "
            f"available={list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find date column.
    # --------------------------------------------------------

    candidates = [
        "timestamp",
        "Timestamp",
        "Date",
        "date",
        "datetime",
        "Datetime",
        "index",
    ]

    date_col = next(
        (
            c
            for c in candidates
            if c in df.columns
        ),
        None,
    )

    if date_col is None:

        # Date may be in the index.
        idx = pd.to_datetime(
            df.index,
            errors="coerce",
        )

        if (
            len(idx) == 0
            or
            idx.notna().mean()
            < 0.99
        ):
            raise ValueError(
                "could not find a datetime "
                "column or datetime index"
            )

        df = df.reset_index(
            drop=False
        )

        date_col = None

        for c in df.columns:

            parsed = pd.to_datetime(
                df[c],
                errors="coerce",
            )

            if (
                parsed.notna().mean()
                >= 0.99
            ):

                date_col = c
                break

        if date_col is None:
            raise ValueError(
                "datetime index was reset but "
                "no datetime column was found"
            )

    # --------------------------------------------------------
    # Copy actual dates to a temporary column.
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df["_original_date"] = dates

    # --------------------------------------------------------
    # Convert numerical columns.
    # --------------------------------------------------------

    numeric_cols = [
        TARGET_COLUMN,
        *COVARIATE_COLUMNS,
    ]

    for c in numeric_cols:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows.
    # --------------------------------------------------------

    before = len(df)

    df = (
        df
        .dropna(
            subset=[
                "_original_date",
                *numeric_cols,
            ]
        )
        .sort_values(
            "_original_date"
        )
        .drop_duplicates(
            "_original_date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    if removed:
        print(
            f"      removed {removed} invalid rows"
        )

    if len(df) < (
        PREDICTION_LENGTH + 20
    ):
        raise ValueError(
            f"only {len(df)} usable rows "
            f"after cleaning"
        )

    values = df[
        numeric_cols
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "non-finite numerical values remain"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Synthetic REGULAR timestamps.
    # --------------------------------------------------------

    df["timestamp"] = pd.date_range(
        start="2000-01-01",
        periods=len(df),
        freq="D",
    )

    df["id"] = ticker

    out = df[
        [
            "id",
            "timestamp",
            TARGET_COLUMN,
            *COVARIATE_COLUMNS,
        ]
    ].copy()

    return out


# ============================================================
# CHRONOS PREPROCESSING
# ============================================================

def prepare_input(
    df: pd.DataFrame,
    ticker: str,
):
    """
    Convert normalized long-format V6 series into Chronos PreparedInput.

    `future_df=None` means all non-target columns are past-only covariates.
    """

    # Print schema before entering Chronos.
    expected = [
        "id",
        "timestamp",
        TARGET_COLUMN,
        *COVARIATE_COLUMNS,
    ]

    missing = (
        set(expected)
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"preprocessing input missing "
            f"{sorted(missing)}"
        )

    prepared = from_data_frame(
        df=df,
        target_columns=[
            TARGET_COLUMN
        ],
        prediction_length=
            PREDICTION_LENGTH,
        future_df=None,
        id_column="id",
        timestamp_column="timestamp",
        validate_inputs=True,
    )

    return prepared


# ============================================================
# LOAD SERIES
# ============================================================

def load_one_series(
    ticker_dir: Path,
):
    ticker = ticker_dir.name

    train_path = (
        ticker_dir
        /
        "train.parquet"
    )

    val_path = (
        ticker_dir
        /
        "validation.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"missing {train_path.name}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"missing {val_path.name}"
        )

    train_raw = pd.read_parquet(
        train_path
    )

    val_raw = pd.read_parquet(
        val_path
    )

    print(
        f"  raw train={len(train_raw)} "
        f"raw validation={len(val_raw)}"
    )

    train = normalize_series(
        train_raw,
        ticker,
        "train",
    )

    validation = normalize_series(
        val_raw,
        ticker,
        "validation",
    )

    train_prepared = prepare_input(
        train,
        ticker,
    )

    validation_prepared = (
        prepare_input(
            validation,
            ticker,
        )
    )

    return (
        train,
        validation,
        train_prepared,
        validation_prepared,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CHRONOS-2 V6 ROBUST LoRA TRAINING"
    )
    print(
        "DIVERSE EQUITIES + MARKET CONTEXT + 512 CONTEXT"
    )
    print("=" * 80)

    print()
    print(
        "Python working directory:",
        Path.cwd(),
    )

    print(
        "Data directory:",
        DATA_DIR.resolve(),
    )

    print(
        "Output directory:",
        OUTPUT_DIR.resolve(),
    )

    print()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print(
            "CUDA:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )

        device = "cuda"

    else:

        print(
            "WARNING: CUDA unavailable."
        )

        device = "cpu"

    # --------------------------------------------------------
    # Verify data directory.
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            "\nV6 DATA DIRECTORY DOES NOT EXIST:\n"
            f"{DATA_DIR.resolve()}\n\n"
            "Run the V6 data-preparation script first:\n"
            "python prepare_chronos_v6_diverse_equity_data.py"
        )

    # --------------------------------------------------------
    # Discover ticker directories.
    # --------------------------------------------------------

    ticker_dirs = sorted(
        [
            p
            for p in DATA_DIR.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )

    print(
        "Ticker directories found:",
        len(ticker_dirs),
    )

    if not ticker_dirs:

        raise RuntimeError(
            "\nThe V6 directory exists but "
            "contains NO ticker folders.\n"
            f"Expected something like:\n"
            f"{DATA_DIR.resolve()}\\ABBV\\train.parquet"
        )

    print()

    # --------------------------------------------------------
    # Safety: remove evaluation stocks.
    # --------------------------------------------------------

    ticker_dirs = [
        p
        for p in ticker_dirs
        if p.name
        not in EVAL_TICKERS
    ]

    print(
        "After excluding held-out stocks:",
        len(ticker_dirs),
    )

    if not ticker_dirs:

        raise RuntimeError(
            "All discovered directories were held-out "
            "evaluation stocks."
        )

    # --------------------------------------------------------
    # Prepare all series.
    # --------------------------------------------------------

    train_inputs = []
    validation_inputs = []

    used_tickers = []
    failures = []

    series_info = []

    print()
    print("=" * 80)
    print(
        "PREPARING V6 SERIES"
    )
    print("=" * 80)

    for ticker_dir in ticker_dirs:

        ticker = ticker_dir.name

        print()
        print(
            f"[CHECK] {ticker}"
        )

        try:

            (
                train_df,
                val_df,
                train_prepared,
                val_prepared,
            ) = load_one_series(
                ticker_dir
            )

            train_inputs.extend(
                train_prepared
            )

            validation_inputs.extend(
                val_prepared
            )

            used_tickers.append(
                ticker
            )

            series_info.append({
                "ticker":
                    ticker,
                "train_rows":
                    len(train_df),
                "validation_rows":
                    len(val_df),
            })

            print(
                f"  [OK] {ticker}"
            )

        except Exception as exc:

            failures.append({
                "ticker":
                    ticker,
                "error":
                    repr(exc),
            })

            print(
                f"  [SKIP] {ticker}"
            )

            print(
                f"         {type(exc).__name__}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Final checks.
    # --------------------------------------------------------

    leakage = (
        set(used_tickers)
        &
        EVAL_TICKERS
    )

    if leakage:

        raise RuntimeError(
            "EVALUATION LEAKAGE: "
            f"{sorted(leakage)}"
        )

    print()
    print("=" * 80)
    print(
        "DATA PREPARATION SUMMARY"
    )
    print("=" * 80)

    print(
        "Candidate directories:",
        len(ticker_dirs),
    )

    print(
        "Usable series:",
        len(used_tickers),
    )

    print(
        "Train inputs:",
        len(train_inputs),
    )

    print(
        "Validation inputs:",
        len(validation_inputs),
    )

    print(
        "Failures:",
        len(failures),
    )

    if failures:

        print()
        print(
            "FIRST FAILURES:"
        )

        for item in failures[:20]:

            print(
                f"  {item['ticker']}: "
                f"{item['error']}"
            )

    if not train_inputs:

        raise RuntimeError(
            "\nZERO TRAINING INPUTS.\n"
            "The exact [SKIP] reason(s) above identify the problem. "
            "No model was loaded and no training was attempted."
        )

    if not validation_inputs:

        raise RuntimeError(
            "\nZERO VALIDATION INPUTS.\n"
            "The exact [SKIP] reason(s) above identify the problem. "
            "No model was loaded and no training was attempted."
        )

    # --------------------------------------------------------
    # Save preflight metadata before model loading.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    preflight = {
        "version":
            "V6",

        "context_length":
            CONTEXT_LENGTH,

        "prediction_length":
            PREDICTION_LENGTH,

        "usable_series":
            len(used_tickers),

        "training_inputs":
            len(train_inputs),

        "validation_inputs":
            len(validation_inputs),

        "used_tickers":
            used_tickers,

        "excluded_evaluation_tickers":
            sorted(EVAL_TICKERS),

        "failures":
            failures,

        "series_info":
            series_info,
    }

    with open(
        OUTPUT_DIR
        /
        "preflight.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            preflight,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Load Chronos.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "LOADING CHRONOS-2"
    )
    print("=" * 80)

    pipeline = (
        Chronos2Pipeline
        .from_pretrained(
            MODEL_ID,
            device_map=device,
        )
    )

    # --------------------------------------------------------
    # LoRA.
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )

    print()
    print(
        "LoRA configuration:"
    )

    print(
        lora_config
    )

    # --------------------------------------------------------
    # Trainer kwargs.
    # --------------------------------------------------------

    trainer_kwargs = {

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "lr_scheduler_type":
            LR_SCHEDULER_TYPE,

        "warmup_steps":
            WARMUP_STEPS,

        "logging_steps":
            10,

        "save_steps":
            SAVE_STEPS,

        "eval_steps":
            EVAL_STEPS,

        "save_total_limit":
            SAVE_TOTAL_LIMIT,

        "report_to":
            "none",

        "remove_unused_columns":
            False,

        "dataloader_num_workers":
            0,

        "tf32":
            False,

        "optim":
            "adamw_torch",
    }

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STARTING V6 LoRA FINE-TUNING"
    )
    print("=" * 80)

    print(
        "Training series:",
        len(used_tickers),
    )

    print(
        "Context:",
        CONTEXT_LENGTH,
    )

    print(
        "Prediction:",
        PREDICTION_LENGTH,
    )

    print(
        "Steps:",
        NUM_STEPS,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    try:

        pipeline.fit(
            inputs=train_inputs,
            prediction_length=
                PREDICTION_LENGTH,
            validation_inputs=
                validation_inputs,
            finetune_mode="lora",
            lora_config=lora_config,
            context_length=
                CONTEXT_LENGTH,
            learning_rate=
                LEARNING_RATE,
            num_steps=
                NUM_STEPS,
            batch_size=
                BATCH_SIZE,
            output_dir=
                OUTPUT_DIR,
            disable_data_parallel=True,
            **trainer_kwargs,
        )

    except Exception:

        print()
        print("=" * 80)
        print(
            "V6 TRAINING FAILED"
        )
        print("=" * 80)

        raise

    # --------------------------------------------------------
    # Save training config.
    # --------------------------------------------------------

    config = {
        "version":
            "V6",

        "model_id":
            MODEL_ID,

        "data_dir":
            str(
                DATA_DIR
            ),

        "context_length":
            CONTEXT_LENGTH,

        "prediction_length":
            PREDICTION_LENGTH,

        "target":
            TARGET_COLUMN,

        "past_only_covariates":
            COVARIATE_COLUMNS,

        "lora": {
            "r":
                LORA_R,
            "alpha":
                LORA_ALPHA,
            "dropout":
                LORA_DROPOUT,
        },

        "learning_rate":
            LEARNING_RATE,

        "lr_scheduler_type":
            LR_SCHEDULER_TYPE,

        "warmup_steps":
            WARMUP_STEPS,

        "num_steps":
            NUM_STEPS,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "training_series":
            len(used_tickers),

        "excluded_evaluation_tickers":
            sorted(EVAL_TICKERS),

        "series_info":
            series_info,

        "failures":
            failures,
    }

    with open(
        OUTPUT_DIR
        /
        "training_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "V6 TRAINING COMPLETE"
    )
    print("=" * 80)

    print(
        "Checkpoint:",
        OUTPUT_DIR
        /
        "finetuned-ckpt",
    )

    print(
        "Config:",
        OUTPUT_DIR
        /
        "training_config.json",
    )


if __name__ == "__main__":
    main()
