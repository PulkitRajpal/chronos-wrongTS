from __future__ import annotations

"""
Chronos-2 LoRA V5:
Target + past-only market/rates/volatility covariates.

This experiment is intentionally different from V1-V4:
instead of giving Chronos only a stock return series, it gives the
target stock return plus several historical covariates.

Covariates:
    SPY_RET
    QQQ_RET
    VIX_RET
    US5Y_CHG
    US10Y_CHG
    realized_vol_20
    volume_z

All are PAST-ONLY.
No future covariate values are supplied during forecasting.

The Chronos-2 documentation recommends using the public
`from_data_frame` / `from_list_of_dicts` preprocessing helpers when
fine-tuning with covariates.
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

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 20

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

LEARNING_RATE = 2e-5
NUM_STEPS = 2000

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LR_SCHEDULER_TYPE = "constant_with_warmup"
WARMUP_STEPS = 100

SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# Training series only.
EVAL_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "ENPH",
    "SMCI", "CVNA", "PLUG", "RKLB", "IONQ",
}

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


# ============================================================
# LOAD FILES
# ============================================================

def load_series(
    ticker_dir: Path,
) -> pd.DataFrame:

    train = pd.read_parquet(
        ticker_dir / "train.parquet"
    )

    validation = pd.read_parquet(
        ticker_dir / "validation.parquet"
    )

    for name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        required = {
            TARGET_COLUMN,
            *COVARIATE_COLUMNS,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{ticker_dir.name}/{name}: "
                f"missing {sorted(missing)}"
            )

        if not np.isfinite(
            df[
                [
                    TARGET_COLUMN,
                    *COVARIATE_COLUMNS,
                ]
            ].to_numpy()
        ).all():
            raise ValueError(
                f"{ticker_dir.name}/{name}: "
                "non-finite values"
            )

        if len(df) < (
            CONTEXT_LENGTH
            +
            PREDICTION_LENGTH
        ):
            raise ValueError(
                f"{ticker_dir.name}/{name}: "
                f"too short ({len(df)})"
            )

    return train, validation


def dataframe_to_input(
    df: pd.DataFrame,
    ticker: str,
):
    """
    Convert one stock dataframe to a Chronos-2 PreparedInput.

    Important:
        Remaining columns in a long-format dataframe are treated as
        past-only covariates unless future_df is supplied.
    """

    work = df.copy()

    # Chronos expects a regular timestamp column.
    work = work.reset_index()

    if "index" in work.columns:
        work = work.rename(
            columns={"index": "timestamp"}
        )

    work["id"] = ticker

    return from_data_frame(
        df=work,
        target_columns=[
            TARGET_COLUMN
        ],
        prediction_length=PREDICTION_LENGTH,
        future_df=None,
        id_column="id",
        timestamp_column="timestamp",
        validate_inputs=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Missing V5 data directory:\n"
            f"{DATA_DIR.resolve()}\n"
            f"Run prepare_v5_multivariate_data.py first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if torch.cuda.is_available():
        device = "cuda"
        print(
            "CUDA available:",
            torch.cuda.get_device_name(0),
        )
    else:
        device = "cpu"
        print(
            "WARNING: CUDA unavailable."
        )

    # --------------------------------------------------------
    # Discover training series.
    # --------------------------------------------------------

    ticker_dirs = [
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
        and p.name not in EVAL_TICKERS
    ]

    if not ticker_dirs:
        raise RuntimeError(
            "No V5 training series found."
        )

    train_inputs = []
    validation_inputs = []

    used_tickers = []

    print()
    print("=" * 80)
    print("LOADING V5 MULTIVARIATE TRAINING DATA")
    print("=" * 80)

    for ticker_dir in sorted(
        ticker_dirs
    ):

        ticker = ticker_dir.name

        try:

            train_df, validation_df = (
                load_series(
                    ticker_dir
                )
            )

            train_input = (
                dataframe_to_input(
                    train_df,
                    ticker,
                )
            )

            validation_input = (
                dataframe_to_input(
                    validation_df,
                    ticker,
                )
            )

            train_inputs.extend(
                train_input
            )

            validation_inputs.extend(
                validation_input
            )

            used_tickers.append(
                ticker
            )

        except Exception as exc:

            print(
                f"[SKIP] {ticker}: {exc}"
            )

    leakage = (
        set(used_tickers)
        &
        EVAL_TICKERS
    )

    if leakage:
        raise RuntimeError(
            "Evaluation leakage detected: "
            f"{sorted(leakage)}"
        )

    print()
    print(
        f"Usable series: {len(used_tickers)}"
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

    # --------------------------------------------------------
    # Load Chronos-2.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING CHRONOS-2")
    print("=" * 80)

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
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
    print("LoRA configuration:")
    print(lora_config)

    # --------------------------------------------------------
    # Trainer kwargs.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fit.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING LoRA V5 FINE-TUNING")
    print("=" * 80)

    print(
        "Target:",
        TARGET_COLUMN,
    )

    print(
        "Past-only covariates:",
        ", ".join(COVARIATE_COLUMNS),
    )

    print(
        "Context:",
        CONTEXT_LENGTH,
    )

    print(
        "Prediction length:",
        PREDICTION_LENGTH,
    )

    print(
        "Steps:",
        NUM_STEPS,
    )

    print()

    pipeline.fit(
        inputs=train_inputs,
        prediction_length=PREDICTION_LENGTH,
        validation_inputs=validation_inputs,
        finetune_mode="lora",
        lora_config=lora_config,
        context_length=CONTEXT_LENGTH,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
        disable_data_parallel=True,
        **training_kwargs,
    )

    # --------------------------------------------------------
    # Save config.
    # --------------------------------------------------------

    config = {
        "version": "V5",
        "model_id": MODEL_ID,
        "dataset": "multivariate_equity_market_context",
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "target": TARGET_COLUMN,
        "past_covariates": COVARIATE_COLUMNS,
        "future_covariates": [],
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": [
                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
        },
        "learning_rate": LEARNING_RATE,
        "scheduler": LR_SCHEDULER_TYPE,
        "warmup_steps": WARMUP_STEPS,
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "training_series": len(used_tickers),
        "evaluation_series_excluded":
            sorted(EVAL_TICKERS),
    }

    with open(
        OUTPUT_DIR / "training_config.json",
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
    print("V5 TRAINING COMPLETE")
    print("=" * 80)

    print(
        "Checkpoint:",
        OUTPUT_DIR / "finetuned-ckpt",
    )

    print(
        "Config:",
        OUTPUT_DIR / "training_config.json",
    )


if __name__ == "__main__":
    main()
