from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline
from peft import LoraConfig


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "amazon/chronos-2"

TRAIN_FILE = Path("data/chronos_finance/train.parquet")
VALIDATION_FILE = Path("data/chronos_finance/validation.parquet")

OUTPUT_DIR = Path("models/chronos2-finance-lora")

# Your benchmark evaluates these horizons.
PREDICTION_LENGTH = 20

# Start conservatively.
CONTEXT_LENGTH = 256

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Training
LEARNING_RATE = 1e-5
NUM_STEPS = 1000

# IMPORTANT for a small GPU
BATCH_SIZE = 1

# Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 4

# Save/evaluate periodically.
SAVE_STEPS = 100
EVAL_STEPS = 100

# Windows + Chronos recommendation
DISABLE_DATA_PARALLEL = True


# ============================================================
# DATA LOADING
# ============================================================

def load_dataframe(path: Path) -> pd.DataFrame:
    print(f"Loading {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    df = pd.read_parquet(path)

    required_columns = {
        "id",
        "timestamp",
        "target",
    }

    missing = required_columns - set(df.columns)

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
    """
    Convert long-format dataframe to the input structure
    accepted by Chronos2Pipeline.fit().

    Each item becomes:

        {
            "target": numpy_array
        }
    """

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

        if len(values) < CONTEXT_LENGTH + PREDICTION_LENGTH:
            print(
                f"[SKIP] {series_id}: "
                f"{len(values)} observations"
            )
            continue

        if not np.isfinite(values).all():
            print(
                f"[SKIP] {series_id}: non-finite values"
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
    print("CHRONOS-2 FINANCIAL LoRA FINE-TUNING")
    print("=" * 70)

    print(f"Model:              {MODEL_ID}")
    print(f"Context length:     {CONTEXT_LENGTH}")
    print(f"Prediction length:  {PREDICTION_LENGTH}")
    print(f"LoRA rank:          {LORA_R}")
    print(f"LoRA alpha:         {LORA_ALPHA}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Training steps:     {NUM_STEPS}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Gradient accum.:    {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Output:             {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "cuda"

        print(
            "CUDA available:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
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
            "Training on CPU will be extremely slow."
        )

    print()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_df = load_dataframe(TRAIN_FILE)
    val_df = load_dataframe(VALIDATION_FILE)

    print(
        f"Train rows:       {len(train_df):,}"
    )

    print(
        f"Validation rows:  {len(val_df):,}"
    )

    print(
        f"Train series:      {train_df['id'].nunique()}"
    )

    print(
        f"Validation series: {val_df['id'].nunique()}"
    )

    # --------------------------------------------------------
    # CONVERT TO CHRONOS FORMAT
    # --------------------------------------------------------

    train_inputs = convert_to_chronos_inputs(
        train_df
    )

    val_inputs = convert_to_chronos_inputs(
        val_df
    )

    print()
    print(
        f"Usable training series:   {len(train_inputs)}"
    )

    print(
        f"Usable validation series: {len(val_inputs)}"
    )

    if len(train_inputs) == 0:
        raise RuntimeError(
            "No usable training series."
        )

    if len(val_inputs) == 0:
        raise RuntimeError(
            "No usable validation series."
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD CHRONOS-2
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading Chronos-2")
    print("=" * 70)

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # --------------------------------------------------------
    # LoRA CONFIG
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
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting LoRA fine-tuning")
    print("=" * 70)

    # HF Trainer arguments are passed via **extra_trainer_kwargs
    # in Chronos2Pipeline.fit().
    #
    # fp16 is deliberately OFF for maximum compatibility.
    #
    # report_to="none" prevents unwanted integrations.
    #
    # gradient_accumulation_steps increases effective batch
    # size without requiring a large GPU batch.
    #

    training_kwargs = {
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,

        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "logging_steps": 10,

        "save_total_limit": 2,

        "report_to": "none",

        "remove_unused_columns": False,

        "dataloader_num_workers": 0,

        "tf32": False,

        "optim": "adamw_torch",
    }

    try:

        finetuned_pipeline = pipeline.fit(

            inputs=train_inputs,

            prediction_length=PREDICTION_LENGTH,

            validation_inputs=val_inputs,

            finetune_mode="lora",

            lora_config=lora_config,

            context_length=CONTEXT_LENGTH,

            learning_rate=LEARNING_RATE,

            num_steps=NUM_STEPS,

            batch_size=BATCH_SIZE,

            output_dir=OUTPUT_DIR,

            disable_data_parallel=DISABLE_DATA_PARALLEL,

            **training_kwargs,
        )

    except Exception:

        print()
        print("=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)

        raise

    # --------------------------------------------------------
    # SAVE CONFIG
    # --------------------------------------------------------

    config = {
        "model_id": MODEL_ID,
        "prediction_length": PREDICTION_LENGTH,
        "context_length": CONTEXT_LENGTH,

        "finetune_mode": "lora",

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
        "num_steps": NUM_STEPS,

        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "train_series": len(train_inputs),
        "validation_series": len(val_inputs),
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

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

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