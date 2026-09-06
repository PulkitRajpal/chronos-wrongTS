from __future__ import annotations

"""
Chronos-2 Financial LoRA V5
===========================

Multivariate / past-covariate fine-tuning.

TARGET
------
Daily log return of each stock.

PAST-ONLY COVARIATES
--------------------
SPY_RET
QQQ_RET
VIX_RET
US5Y_CHG
US10Y_CHG
realized_vol_20
volume_z

The important difference from V1-V4 is that Chronos sees the stock target
together with related market/rates/volatility history.

The 10 held-out evaluation stocks are explicitly excluded.

This script expects the output of:

    prepare_v5_multivariate_data.py

under:

    data/chronos_v5_multivariate/<TICKER>/
        train.parquet
        validation.parquet

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
    "data/chronos_v5_multivariate"
)

OUTPUT_DIR = Path(
    "models/chronos2-finance-lora-v5-multivariate"
)


# ------------------------------------------------------------
# Forecast setup
# ------------------------------------------------------------

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 20


# ------------------------------------------------------------
# LoRA
# Keep same basic LoRA configuration as V2-V4.
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


# ------------------------------------------------------------
# Optimizer
# Keep same optimizer setup as V2-V4.
# ------------------------------------------------------------

LEARNING_RATE = 2e-5
NUM_STEPS = 2000

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LR_SCHEDULER_TYPE = (
    "constant_with_warmup"
)

WARMUP_STEPS = 100


# ------------------------------------------------------------
# Checkpoints / evaluation
# ------------------------------------------------------------

SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3


# ------------------------------------------------------------
# V5 schema
# ------------------------------------------------------------

TARGET_COLUMN = "target"

COVARIATE_COLUMNS = [
    "SPY_RET",
    "QQQ_RET",
    "VIX_RET",
    "US5Y_CHG",
    "US10Y_CHG",
    "realized_vol_20",
    "volume_z",
]


# ------------------------------------------------------------
# Final held-out evaluation universe.
# Never allow these into fine-tuning.
# ------------------------------------------------------------

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

        props = torch.cuda.get_device_properties(
            0
        )

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
            "CUDA test failed -> CPU"
        )

        print(exc)

        return "cpu"


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
    ticker: str,
    name: str,
) -> pd.DataFrame:
    """
    Normalize a V5 dataframe before Chronos preprocessing.

    Handles the exact issue from the previous run:
    parquet files may have the datetime stored as the DataFrame index,
    and reset_index() can produce a column called Date/index rather than
    timestamp.
    """

    work = df.copy()

    # --------------------------------------------------------
    # Required numerical fields.
    # --------------------------------------------------------

    required = {
        TARGET_COLUMN,
        *COVARIATE_COLUMNS,
    }

    missing = (
        required
        -
        set(work.columns)
    )

    if missing:

        raise ValueError(
            f"{ticker}/{name}: "
            f"missing columns "
            f"{sorted(missing)}; "
            f"available={list(work.columns)}"
        )

    # --------------------------------------------------------
    # Extract timestamp.
    #
    # Case 1:
    # timestamp already exists as a column.
    # Case 2:
    # datetime is in the index.
    # Case 3:
    # reset_index produces Date/date/index.
    # --------------------------------------------------------

    timestamp_candidates = [
        "timestamp",
        "Timestamp",
        "Date",
        "date",
        "datetime",
        "Datetime",
        "index",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in work.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        # Try the DataFrame index.
        if isinstance(
            work.index,
            pd.DatetimeIndex,
        ):

            index_values = work.index

        else:

            try:

                index_values = pd.to_datetime(
                    work.index,
                    errors="coerce",
                )

            except Exception as exc:

                raise ValueError(
                    f"{ticker}/{name}: "
                    "could not identify timestamp."
                ) from exc

        if pd.isna(index_values).all():

            raise ValueError(
                f"{ticker}/{name}: "
                "could not identify timestamp "
                f"from columns={list(work.columns)} "
                f"or index."
            )

        work = work.reset_index(
            drop=False
        )

        # Find the new datetime-like column.
        for column in work.columns:

            parsed = pd.to_datetime(
                work[column],
                errors="coerce",
            )

            if (
                parsed.notna().mean()
                >= 0.99
            ):

                timestamp_column = column
                break

    if timestamp_column is None:

        raise ValueError(
            f"{ticker}/{name}: "
            "no usable timestamp column after "
            f"normalization. Columns={list(work.columns)}"
        )

    # Rename exactly what Chronos expects.
    if timestamp_column != "timestamp":

        work = work.rename(
            columns={
                timestamp_column:
                    "timestamp"
            }
        )

    # --------------------------------------------------------
    # Normalize timestamp.
    # --------------------------------------------------------

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    work = work.dropna(
        subset=["timestamp"]
    )

    # Remove duplicate timestamps.
    work = (
        work
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Numeric validation.
    # --------------------------------------------------------

    numeric_columns = [
        TARGET_COLUMN,
        *COVARIATE_COLUMNS,
    ]

    for column in numeric_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.dropna(
        subset=numeric_columns
    )

    values = work[
        numeric_columns
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():

        raise ValueError(
            f"{ticker}/{name}: "
            "NaN/inf remains after cleaning."
        )

    # --------------------------------------------------------
    # Add series ID.
    # --------------------------------------------------------

    work["id"] = ticker

    # --------------------------------------------------------
    # Chronos long-format layout.
    # --------------------------------------------------------

    work = work[
        [
            "id",
            "timestamp",
            TARGET_COLUMN,
            *COVARIATE_COLUMNS,
        ]
    ]

    return work


# ============================================================
# LOAD ONE SERIES
# ============================================================

def load_train_validation(
    ticker_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    ticker = ticker_dir.name

    train_path = (
        ticker_dir
        /
        "train.parquet"
    )

    validation_path = (
        ticker_dir
        /
        "validation.parquet"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            train_path
        )

    if not validation_path.exists():

        raise FileNotFoundError(
            validation_path
        )

    train_raw = pd.read_parquet(
        train_path
    )

    validation_raw = pd.read_parquet(
        validation_path
    )

    train = validate_dataframe(
        train_raw,
        ticker,
        "train",
    )

    validation = validate_dataframe(
        validation_raw,
        ticker,
        "validation",
    )

    required_length = (
        CONTEXT_LENGTH
        +
        PREDICTION_LENGTH
    )

    if len(train) < required_length:

        raise ValueError(
            f"{ticker}/train: "
            f"only {len(train)} rows; "
            f"need at least {required_length}"
        )

    if len(validation) < required_length:

        raise ValueError(
            f"{ticker}/validation: "
            f"only {len(validation)} rows; "
            f"need at least {required_length}"
        )

    return (
        train,
        validation,
    )


# ============================================================
# CHRONOS INPUT PREPARATION
# ============================================================

def dataframe_to_prepared_input(
    df: pd.DataFrame,
    ticker: str,
):
    """
    Convert long-format DataFrame into Chronos PreparedInput.

    All columns other than target are interpreted as PAST-ONLY covariates
    because future_df=None.

    This is the intended Chronos-2 preprocessing path for covariates.
    """

    work = df.copy()

    # Ensure correct chronological order.
    work = (
        work
        .sort_values(
            [
                "id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # Extra safety: exactly one ID.
    unique_ids = (
        work["id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if unique_ids != [ticker]:

        raise ValueError(
            f"{ticker}: unexpected IDs "
            f"in prepared dataframe: "
            f"{unique_ids}"
        )

    # No future_df:
    # all remaining columns become past-only covariates.
    prepared = from_data_frame(
        df=work,
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
# LEAKAGE CHECK
# ============================================================

def validate_no_evaluation_leakage(
    tickers: list[str],
) -> None:

    leakage = (
        set(tickers)
        &
        EVAL_TICKERS
    )

    if leakage:

        raise RuntimeError(
            "EVALUATION LEAKAGE DETECTED: "
            f"{sorted(leakage)}"
        )

    print()
    print(
        "Evaluation leakage check: PASS"
    )

    print(
        "Held-out stocks are absent from "
        "V5 fine-tuning data."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CHRONOS-2 FINANCIAL LoRA V5"
    )
    print(
        "MULTIVARIATE / PAST-COVARIATE FINE-TUNING"
    )
    print("=" * 80)

    print()
    print(
        "Model:",
        MODEL_ID,
    )

    print(
        "Context length:",
        CONTEXT_LENGTH,
    )

    print(
        "Prediction length:",
        PREDICTION_LENGTH,
    )

    print(
        "LoRA rank:",
        LORA_R,
    )

    print(
        "LoRA alpha:",
        LORA_ALPHA,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Scheduler:",
        LR_SCHEDULER_TYPE,
    )

    print(
        "Warmup:",
        WARMUP_STEPS,
    )

    print(
        "Training steps:",
        NUM_STEPS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print()
    print(
        "Target:",
        TARGET_COLUMN,
    )

    print(
        "Past-only covariates:"
    )

    for column in COVARIATE_COLUMNS:
        print(
            f"  - {column}"
        )

    print()
    print(
        "Data directory:",
        DATA_DIR,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = resolve_device()

    # ========================================================
    # DISCOVER SERIES
    # ========================================================

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"V5 data directory does not exist:\n"
            f"{DATA_DIR.resolve()}\n\n"
            f"Run prepare_v5_multivariate_data.py first."
        )

    ticker_dirs = sorted(
        [
            p
            for p in DATA_DIR.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )

    # Explicitly exclude evaluation tickers.
    ticker_dirs = [
        p
        for p in ticker_dirs
        if p.name
        not in EVAL_TICKERS
    ]

    if not ticker_dirs:

        raise RuntimeError(
            "No candidate V5 training series found."
        )

    train_inputs = []
    validation_inputs = []

    used_tickers = []

    # Keep a compact metadata report.
    series_metadata = []

    print()
    print("=" * 80)
    print(
        "PREPARING CHRONOS INPUTS"
    )
    print("=" * 80)

    for ticker_dir in ticker_dirs:

        ticker = ticker_dir.name

        try:

            train_df, validation_df = (
                load_train_validation(
                    ticker_dir
                )
            )

            train_prepared = (
                dataframe_to_prepared_input(
                    train_df,
                    ticker,
                )
            )

            validation_prepared = (
                dataframe_to_prepared_input(
                    validation_df,
                    ticker,
                )
            )

            train_inputs.extend(
                train_prepared
            )

            validation_inputs.extend(
                validation_prepared
            )

            used_tickers.append(
                ticker
            )

            series_metadata.append({
                "ticker":
                    ticker,
                "train_rows":
                    len(train_df),
                "validation_rows":
                    len(validation_df),
            })

            print(
                f"[OK] {ticker}: "
                f"train={len(train_df)}, "
                f"validation={len(validation_df)}"
            )

        except Exception as exc:

            print(
                f"[SKIP] {ticker}: {exc}"
            )

    validate_no_evaluation_leakage(
        used_tickers
    )

    print()
    print(
        "Usable series:",
        len(used_tickers),
    )

    print(
        "Train prepared inputs:",
        len(train_inputs),
    )

    print(
        "Validation prepared inputs:",
        len(validation_inputs),
    )

    if not train_inputs:

        raise RuntimeError(
            "No usable training inputs."
        )

    if not validation_inputs:

        raise RuntimeError(
            "No usable validation inputs."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOADING CHRONOS-2"
    )
    print("=" * 80)

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
    )

    # ========================================================
    # LoRA
    # ========================================================

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

    # ========================================================
    # TRAINER KWARGS
    # ========================================================

    training_kwargs = {

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

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STARTING LoRA V5 TRAINING"
    )
    print("=" * 80)

    print()
    print(
        "This is the first Chronos experiment "
        "in the project where the target stock "
        "return is trained jointly with "
        "past-only market/rates/volatility covariates."
    )

    print()

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
            **training_kwargs,
        )

    except Exception:

        print()
        print("=" * 80)
        print(
            "V5 TRAINING FAILED"
        )
        print("=" * 80)

        raise

    # ========================================================
    # SAVE METADATA
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {

        "version":
            "V5",

        "model_id":
            MODEL_ID,

        "dataset":
            "multivariate_equity_market_context",

        "context_length":
            CONTEXT_LENGTH,

        "prediction_length":
            PREDICTION_LENGTH,

        "target":
            TARGET_COLUMN,

        "past_only_covariates":
            COVARIATE_COLUMNS,

        "future_covariates":
            [],

        "lora": {

            "r":
                LORA_R,

            "alpha":
                LORA_ALPHA,

            "dropout":
                LORA_DROPOUT,

            "target_modules": [

                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
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

        "series_metadata":
            series_metadata,
    }

    config_path = (
        OUTPUT_DIR
        /
        "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "V5 TRAINING COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "Checkpoint:"
    )

    print(
        OUTPUT_DIR
        /
        "finetuned-ckpt"
    )

    print()
    print(
        "Training configuration:"
    )

    print(
        config_path
    )


if __name__ == "__main__":
    main()
