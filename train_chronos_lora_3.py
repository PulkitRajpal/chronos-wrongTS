from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline
from peft import LoraConfig


# ============================================================
# CONFIG — LoRA V2
# ============================================================

MODEL_ID = "amazon/chronos-2"

TRAIN_FILE = Path("data/chronos_finance/train.parquet")
VALIDATION_FILE = Path("data/chronos_finance/validation.parquet")

OUTPUT_DIR = Path("models/chronos2-finance-lora-v2")

# Forecast setup
CONTEXT_LENGTH = 256
PREDICTION_LENGTH = 20

# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

LEARNING_RATE = 2e-5
NUM_STEPS = 2000

# Small batch because you have ~8 GB VRAM
BATCH_SIZE = 1

# Effective batch size = 1 * 4 = 4
GRADIENT_ACCUMULATION_STEPS = 4

# IMPORTANT:
# Instead of the default linear decay to ~0,
# keep the LR constant after a short warmup.
LR_SCHEDULER_TYPE = "constant_with_warmup"
WARMUP_STEPS = 100

# Validation/checkpointing
SAVE_STEPS = 100
EVAL_STEPS = 100

DISABLE_DATA_PARALLEL = True


# ============================================================
# DATA
# ============================================================

def load_dataframe(path: Path) -> pd.DataFrame:

    print(f"Loading {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    df = pd.read_parquet(path)

    required = {"id", "timestamp", "target"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["id", "timestamp", "target"]
    )

    df = df.sort_values(
        ["id", "timestamp"]
    ).reset_index(drop=True)

    return df


def convert_to_chronos_inputs(
    df: pd.DataFrame,
) -> list[dict]:

    inputs = []

    for series_id, group in df.groupby(
        "id",
        sort=False,
    ):

        group = group.sort_values("timestamp")

        values = (
            group["target"]
            .astype(np.float32)
            .to_numpy()
        )

        required_length = (
            CONTEXT_LENGTH +
            PREDICTION_LENGTH
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CHRONOS-2 FINANCIAL LoRA V2")
    print("=" * 70)

    print(f"Model:              {MODEL_ID}")
    print(f"Context length:     {CONTEXT_LENGTH}")
    print(f"Prediction length:  {PREDICTION_LENGTH}")

    print(f"LoRA rank:          {LORA_R}")
    print(f"LoRA alpha:         {LORA_ALPHA}")

    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Scheduler:          {LR_SCHEDULER_TYPE}")
    print(f"Warmup steps:       {WARMUP_STEPS}")

    print(f"Training steps:     {NUM_STEPS}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(
        f"Gradient accum.:    "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(f"Output:             {OUTPUT_DIR}")
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
                torch.cuda.get_device_properties(0)
                .total_memory / (1024 ** 3),
                2,
            ),
            "GB",
        )

    else:

        device = "cpu"

        print(
            "WARNING: CUDA unavailable."
        )

    print()

    # ========================================================
    # DATA
    # ========================================================

    train_df = load_dataframe(
        TRAIN_FILE
    )

    val_df = load_dataframe(
        VALIDATION_FILE
    )

    print(
        f"Train rows:       {len(train_df):,}"
    )

    print(
        f"Validation rows:  {len(val_df):,}"
    )

    print(
        f"Train series:      "
        f"{train_df['id'].nunique()}"
    )

    print(
        f"Validation series: "
        f"{val_df['id'].nunique()}"
    )

    # ========================================================
    # CONVERT
    # ========================================================

    train_inputs = convert_to_chronos_inputs(
        train_df
    )

    val_inputs = convert_to_chronos_inputs(
        val_df
    )

    print()

    print(
        f"Usable training series:   "
        f"{len(train_inputs)}"
    )

    print(
        f"Usable validation series: "
        f"{len(val_inputs)}"
    )

    if not train_inputs:

        raise RuntimeError(
            "No usable training series."
        )

    if not val_inputs:

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
    # LOAD CHRONOS
    # ========================================================

    print()
    print("=" * 70)
    print("Loading Chronos-2")
    print("=" * 70)

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
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
    print("LoRA configuration:")
    print(lora_config)

    # ========================================================
    # TRAINER KWARGS
    # ========================================================

    training_kwargs = {

        # Effective batch = 4
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        # ----------------------------------------------------
        # LR scheduling
        # ----------------------------------------------------

        "lr_scheduler_type":
            LR_SCHEDULER_TYPE,

        "warmup_steps":
            WARMUP_STEPS,

        # ----------------------------------------------------
        # Logging / evaluation
        # ----------------------------------------------------

        "logging_steps": 10,

        "save_steps":
            SAVE_STEPS,

        "eval_steps":
            EVAL_STEPS,

        # Current Transformers uses eval_strategy.
        # Chronos-2 supplies its own defaults when validation
        # inputs are provided, so we do not override it here.
        #
        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        "save_total_limit": 3,

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        "report_to": "none",

        "remove_unused_columns": False,

        "dataloader_num_workers": 0,

        "tf32": False,

        "optim": "adamw_torch",
    }

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("Starting LoRA V2 fine-tuning")
    print("=" * 70)
    print()

    try:

        finetuned_pipeline = pipeline.fit(

            inputs=train_inputs,

            prediction_length=
                PREDICTION_LENGTH,

            validation_inputs=
                val_inputs,

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

            disable_data_parallel=
                DISABLE_DATA_PARALLEL,

            **training_kwargs,
        )

    except Exception:

        print()
        print("=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)

        raise

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {

        "version": "v2",

        "model_id":
            MODEL_ID,

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
            len(val_inputs),
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

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Fine-tuned pipeline saved to:"
    )

    print(
        OUTPUT_DIR / "finetuned-ckpt"
    )

    print()
    print(
        "Training configuration saved to:"
    )

    print(
        OUTPUT_DIR / "training_config.json"
    )


if __name__ == "__main__":
    main()