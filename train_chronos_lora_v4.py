from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline
from peft import LoraConfig


# ============================================================
# CHRONOS-2 FINANCIAL LoRA V4
#
# Training corpus:
#   - US listed equities
#   - International equities / ADRs
#   - Equity ETFs / indices
#   - Rates / bonds
#   - Volatility
#
# Excluded:
#   - Crypto
#   - Commodities
#   - FX
#   - 10 final evaluation stocks
# ============================================================


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "amazon/chronos-2"

TRAIN_FILE = Path(
    "data/chronos_finance_equity_rates_vol/train.parquet"
)

VALIDATION_FILE = Path(
    "data/chronos_finance_equity_rates_vol/validation.parquet"
)

OUTPUT_DIR = Path(
    "models/chronos2-finance-lora-v4-equity-rates-vol"
)


# ------------------------------------------------------------
# Forecast objective
# ------------------------------------------------------------

CONTEXT_LENGTH = 256
PREDICTION_LENGTH = 20


# ------------------------------------------------------------
# LoRA
#
# Keep identical to V2/V3 so this experiment primarily tests
# training-data composition.
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


# ------------------------------------------------------------
# Training
#
# Keep identical to V2/V3.
# ------------------------------------------------------------

LEARNING_RATE = 2e-5

NUM_STEPS = 2000

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 4

LR_SCHEDULER_TYPE = "constant_with_warmup"

WARMUP_STEPS = 100


# ------------------------------------------------------------
# Validation / checkpoints
# ------------------------------------------------------------

SAVE_STEPS = 100
EVAL_STEPS = 100

SAVE_TOTAL_LIMIT = 3


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

DISABLE_DATA_PARALLEL = True


# ============================================================
# DATA LOADING
# ============================================================

def load_dataframe(
    path: Path,
) -> pd.DataFrame:

    print(f"Loading {path}")

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{path.resolve()}"
        )

    df = pd.read_parquet(path)

    required = {
        "id",
        "timestamp",
        "target",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "id",
            "timestamp",
            "target",
        ]
    )

    df = (
        df
        .sort_values(
            [
                "id",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# CONVERT TO CHRONOS INPUT FORMAT
# ============================================================

def convert_to_chronos_inputs(
    df: pd.DataFrame,
) -> list[dict]:

    inputs = []

    required_length = (
        CONTEXT_LENGTH
        +
        PREDICTION_LENGTH
    )

    for series_id, group in df.groupby(
        "id",
        sort=False,
    ):

        group = (
            group
            .sort_values("timestamp")
        )

        values = (
            group["target"]
            .astype(np.float32)
            .to_numpy()
        )

        if len(values) < required_length:

            print(
                f"[SKIP] {series_id}: "
                f"{len(values)} observations"
            )

            continue

        if not np.isfinite(values).all():

            print(
                f"[SKIP] {series_id}: "
                "non-finite values"
            )

            continue

        inputs.append(
            {
                "target": values
            }
        )

    return inputs


# ============================================================
# SAFETY CHECKS
# ============================================================

def validate_no_eval_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:

    eval_tickers = {
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

    train_ids = set(
        train_df["id"].unique()
    )

    val_ids = set(
        val_df["id"].unique()
    )

    leakage = (
        train_ids
        |
        val_ids
    ) & eval_tickers

    if leakage:

        raise RuntimeError(
            "EVALUATION LEAKAGE DETECTED: "
            f"{sorted(leakage)}"
        )

    print(
        "Evaluation leakage check: PASS"
    )

    print(
        "Held-out stocks absent from "
        "training/validation datasets."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CHRONOS-2 FINANCIAL LoRA V4"
    )
    print(
        "EQUITIES + RATES + VOLATILITY"
    )
    print("=" * 70)

    print(
        f"Model:              {MODEL_ID}"
    )

    print(
        f"Context length:     "
        f"{CONTEXT_LENGTH}"
    )

    print(
        f"Prediction length:  "
        f"{PREDICTION_LENGTH}"
    )

    print(
        f"LoRA rank:          {LORA_R}"
    )

    print(
        f"LoRA alpha:         "
        f"{LORA_ALPHA}"
    )

    print(
        f"Learning rate:      "
        f"{LEARNING_RATE}"
    )

    print(
        f"Scheduler:          "
        f"{LR_SCHEDULER_TYPE}"
    )

    print(
        f"Warmup steps:       "
        f"{WARMUP_STEPS}"
    )

    print(
        f"Training steps:     "
        f"{NUM_STEPS}"
    )

    print(
        f"Batch size:         "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accum.:    "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Output:             "
        f"{OUTPUT_DIR}"
    )

    print()

    # ========================================================
    # GPU
    # ========================================================

    if torch.cuda.is_available():

        device = "cuda"

        print(
            "CUDA available:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            round(
                torch.cuda
                .get_device_properties(0)
                .total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )

    else:

        device = "cpu"

        print(
            "WARNING: CUDA unavailable."
        )

        print(
            "Training will be very slow."
        )

    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = load_dataframe(
        TRAIN_FILE
    )

    validation_df = load_dataframe(
        VALIDATION_FILE
    )

    print(
        f"Train rows:       "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows:  "
        f"{len(validation_df):,}"
    )

    print(
        f"Train series:      "
        f"{train_df['id'].nunique()}"
    )

    print(
        f"Validation series: "
        f"{validation_df['id'].nunique()}"
    )

    validate_no_eval_leakage(
        train_df,
        validation_df,
    )

    # ========================================================
    # CHRONOS INPUTS
    # ========================================================

    train_inputs = convert_to_chronos_inputs(
        train_df
    )

    validation_inputs = convert_to_chronos_inputs(
        validation_df
    )

    print()

    print(
        f"Usable training series:   "
        f"{len(train_inputs)}"
    )

    print(
        f"Usable validation series: "
        f"{len(validation_inputs)}"
    )

    if not train_inputs:

        raise RuntimeError(
            "No usable training series."
        )

    if not validation_inputs:

        raise RuntimeError(
            "No usable validation series."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD CHRONOS-2
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Loading Chronos-2"
    )
    print("=" * 70)

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
    )

    # ========================================================
    # LoRA CONFIGURATION
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
    print("LoRA configuration:")
    print(lora_config)

    # ========================================================
    # TRAINER ARGUMENTS
    # ========================================================

    training_kwargs = {

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        # Keep the LR around 2e-5 after warmup.
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

        # Prevent W&B / other integrations
        # from starting automatically.
        "report_to":
            "none",

        "remove_unused_columns":
            False,

        # Windows-safe.
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
    print("=" * 70)
    print(
        "Starting LoRA V4 fine-tuning"
    )
    print("=" * 70)
    print()

    try:

        pipeline.fit(

            inputs=train_inputs,

            prediction_length=
                PREDICTION_LENGTH,

            validation_inputs=
                validation_inputs,

            finetune_mode=
                "lora",

            lora_config=
                lora_config,

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

            disable_data_parallel=
                DISABLE_DATA_PARALLEL,

            **training_kwargs,
        )

    except Exception:

        print()
        print("=" * 70)
        print(
            "TRAINING FAILED"
        )
        print("=" * 70)

        raise

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {

        "version":
            "V4",

        "dataset":
            "equities_rates_volatility",

        "model_id":
            MODEL_ID,

        "train_file":
            str(TRAIN_FILE),

        "validation_file":
            str(VALIDATION_FILE),

        "context_length":
            CONTEXT_LENGTH,

        "prediction_length":
            PREDICTION_LENGTH,

        "finetune_mode":
            "lora",

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

        "train_series":
            len(train_inputs),

        "validation_series":
            len(validation_inputs),
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
    print("=" * 70)
    print(
        "V4 TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        "Fine-tuned checkpoint:"
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
