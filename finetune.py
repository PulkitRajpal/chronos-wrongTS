import pandas as pd
import torch

from chronos import Chronos2Pipeline

MODEL_ID = "amazon/chronos-2"

pipeline = Chronos2Pipeline.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    torch_dtype=torch.float32,
)

train_df = pd.read_parquet("financial_train.parquet")

# Example:
# columns = id, timestamp, target

train_inputs = []

for series_id, group in train_df.groupby("id"):
    group = group.sort_values("timestamp")

    train_inputs.append({
        "id": series_id,
        "target": group["target"].values,
    })

# Chronos-2 fine-tuning configuration
finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=20,
    num_steps=1000,
    learning_rate=1e-5,
    batch_size=32,
    finetune_mode="lora",
)

finetuned_pipeline.save_pretrained(
    "./chronos2-finance-lora"
)