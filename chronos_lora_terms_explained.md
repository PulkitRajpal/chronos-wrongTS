# Chronos-2 LoRA Fine-Tuning — Terms Explained

## 1. Fine-Tuning

**Fine-tuning** means taking a model that has already been pretrained and training it further on a specific type of data or task.

In this project:

```text
Pretrained Chronos-2
        ↓
Financial time series
        ↓
Financially adapted Chronos-2
```

We are **not training Chronos from scratch**.

The purpose is to teach a general time-series foundation model to become better adapted to financial behavior.

---

## 2. LoRA

**LoRA = Low-Rank Adaptation.**

Instead of changing all of Chronos-2's original model weights, LoRA keeps the original weights frozen and adds small trainable adapter parameters.

Conceptually:

```text
                    Chronos-2
                       │
             ┌─────────┴─────────┐
             │                   │
       Original weights      LoRA weights
          FROZEN               TRAINED
```

This reduces the amount of parameters that need to be trained and lowers the memory requirement compared with full fine-tuning.

In our code:

```python
finetune_mode="lora"
```

means that only the LoRA adapters are trained.

---

## 3. LoRA Rank (`r`)

The **LoRA rank** controls the capacity of the LoRA adapter.

Our current setting is:

```python
r = 8
```

Conceptually:

```text
rank 4
   ↓
smaller adapter

rank 8
   ↓
larger adapter

rank 16
   ↓
even larger adapter
```

A higher rank gives the adapter more capacity to learn complex changes, but also increases the number of trainable parameters and memory usage.

Our current experiment uses rank 8. The actual V2 and V3 runs show:

```text
LoRA rank: 8
```

---

## 4. LoRA Alpha

We use:

```python
lora_alpha = 16
```

`lora_alpha` is the scaling factor that controls the strength of the LoRA update.

A useful simplified way to think about it is:

```text
LoRA learns an adjustment
        ↓
alpha controls how strongly that adjustment
is applied
```

Its effect is considered together with the rank.

For our setup:

```text
alpha = 16
rank  = 8
```

so the standard LoRA scaling is based on:

```text
alpha / rank = 2
```

This does **not** mean the adapter is simply "2x stronger"; it is the scaling factor used by the LoRA formulation.

---

## 5. LoRA Dropout

We use:

```python
lora_dropout = 0.0
```

Dropout is a regularization technique.

It can randomly disable part of the adapter during training so the model is less likely to overfit.

Conceptually:

```text
dropout = 0.0
    ↓
no LoRA dropout

dropout = 0.1
    ↓
some LoRA activations randomly disabled
```

We currently use 0.0 for a simple baseline.

---

## 6. Target Modules

Our LoRA configuration applies adapters to:

```text
self_attention.q
self_attention.k
self_attention.v
self_attention.o
output_patch_embedding.output_layer
```

The first four belong to the attention mechanism:

```text
Q = Query
K = Key
V = Value
O = Output
```

Conceptually:

```text
Input time series
       ↓
Attention mechanism
       ↓
Q / K / V / O
       ↓
LoRA adaptation
```

This allows the adapter to modify how Chronos processes relationships in the sequence.

The `output_patch_embedding.output_layer` is another part of the Chronos-2 architecture that the LoRA configuration adapts.

---

## 7. Context Length

We use:

```python
CONTEXT_LENGTH = 256
```

The **context length** is how much historical data Chronos can look at when making a forecast.

Conceptually:

```text
<----------- 256 observations ----------->
                                            ↓
                                         forecast
```

For our return-based dataset, this is approximately the most recent 256 trading observations.

---

## 8. Prediction Length

We use:

```python
PREDICTION_LENGTH = 20
```

The **prediction length** is how many future observations the model has to forecast.

So our training problem is approximately:

```text
256 historical observations
            ↓
       Chronos-2
            ↓
20 future observations
```

Twenty trading observations is roughly one trading month.

This matches the longest forecast horizon used in our benchmarking.

---

## 9. Training Sample / Window

The combination of context length and prediction length creates a training sample.

For example:

```text
Historical:
[1 ... 256]
     ↓
Predict:
[257 ... 276]
```

Then the window moves:

```text
[2 ... 257]
      ↓
[258 ... 277]
```

and so on.

This means one long financial time series can produce many training examples.

---

## 10. Learning Rate

We use:

```python
LEARNING_RATE = 2e-5
```

The **learning rate** controls how large each parameter update is.

Conceptually:

```text
learning rate too small
        ↓
very slow adaptation

reasonable learning rate
        ↓
steady learning

learning rate too large
        ↓
unstable / overshooting
```

Because we are using LoRA, this controls the size of updates to the trainable adapter parameters.

Our V1 used `1e-5`, while V2 and V3 used `2e-5`.

---

## 11. Learning-Rate Scheduler

A **learning-rate scheduler** determines how the learning rate changes during training.

Our original V1 effectively used a decaying schedule:

```text
1e-5
 ↓
 ↓
 ↓
 ↓
~0
```

This meant the model was making increasingly tiny updates toward the end.

For V2 and V3, we changed to:

```python
lr_scheduler_type = "constant_with_warmup"
```

Conceptually:

```text
small LR
   ↓
warmup
   ↓
2e-5
   ↓
2e-5
   ↓
2e-5
```

This keeps the learning rate useful for a much larger portion of training.

---

## 12. Warmup Steps

We use:

```python
WARMUP_STEPS = 100
```

Warmup means gradually increasing the learning rate at the beginning instead of immediately using the full learning rate.

Our V2 log demonstrates this:

```text
early:
1.8e-6
3.8e-6
5.8e-6
...
1.98e-5

then:
2e-5
```

So:

```text
Step 1       → small learning rate
Step 50      → larger learning rate
Step 100     → approximately full learning rate
After 100    → approximately 2e-5
```

Warmup is commonly used to make early optimization more stable.

---

## 13. Training Step

A **training step** is one optimizer update.

Very simplified:

```text
Training data
     ↓
Model
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Update LoRA parameters
```

That is one training step.

Our V2/V3 experiments use:

```python
NUM_STEPS = 2000
```

so the model performs approximately 2,000 optimizer updates.

---

## 14. Batch Size

We use:

```python
BATCH_SIZE = 1
```

A **batch** is the number of training samples processed together before the gradient is calculated.

For example:

```text
batch size = 1

sample 1
   ↓
gradient
```

Compared with:

```text
batch size = 16

sample 1 ─┐
sample 2  │
...       ├──→ combined gradient
sample 16 ┘
```

A larger batch generally requires more GPU memory.

Because the RTX 5060 Laptop GPU has approximately 8 GB of VRAM in our environment, we use batch size 1 for safety.

---

## 15. Gradient Accumulation

We use:

```python
GRADIENT_ACCUMULATION_STEPS = 4
```

Normally:

```text
batch 1 → update
batch 2 → update
batch 3 → update
```

With gradient accumulation:

```text
batch 1 → accumulate gradient
batch 2 → accumulate gradient
batch 3 → accumulate gradient
batch 4 → accumulate gradient
                    ↓
                update
```

So instead of updating after every small batch, gradients from several batches are collected first.

This allows us to simulate a larger effective batch without requiring all samples to fit in GPU memory simultaneously.

---

## 16. Effective Batch Size

With:

```text
batch size = 1
gradient accumulation = 4
```

and one GPU:

```text
effective batch size ≈ 1 × 1 × 4
                     = 4
```

So the optimizer behaves approximately as though it is seeing a batch of 4 samples per update.

---

## 17. Loss

During training we see values such as:

```text
loss: 5.598
```

The **loss** is the model's training objective: it represents how poorly the model is performing according to the objective used during fine-tuning.

The goal is:

```text
loss ↓
```

However, the loss value is **not the same as price MAE**.

For example:

```text
loss = 5
```

does not mean:

```text
5 dollars of error
```

This is why the final financial evaluation must still use metrics such as MAE, return error, directional accuracy, spike detection, and prediction-interval coverage.

---

## 18. Validation Loss

We also see:

```text
eval_loss: 4.388
```

This is the loss calculated on the validation data.

The idea is:

```text
Training data
      ↓
learn

Validation data
      ↓
check generalization
```

Validation loss is therefore useful for deciding whether additional training is actually helping.

For example, our V2 experiment reached a substantially lower validation loss than its starting point, while the V3 broader-data experiment reached an even lower value.

---

## 19. Gradient

Training output also contains:

```text
grad_norm: 1.227
```

A **gradient** tells the optimizer which direction the trainable parameters should move to reduce the loss.

Conceptually:

```text
Loss
 ↓
Gradient
 ↓
Direction of improvement
 ↓
Update LoRA parameters
```

`grad_norm` describes the overall magnitude of the gradient.

We mainly monitor it for abnormal behavior such as exploding gradients.

---

## 20. Epoch

An **epoch** is approximately one complete pass through the training dataset.

For example:

```text
epoch = 1
```

means the training process has gone through roughly one full pass of the training dataset.

Our experiments are controlled primarily by **training steps**, so steps are more important than epochs for comparing V1, V2, and V3.

---

## 21. Checkpoint

A **checkpoint** is a saved version of the model during training.

For example:

```text
checkpoint-100
checkpoint-200
checkpoint-300
...
```

This is useful because the best model does not necessarily occur at the final training step.

Example:

```text
step 1000 → validation loss 4.30
step 1500 → validation loss 4.25  ← best
step 2000 → validation loss 4.29
```

In that case, checkpoint 1500 would be preferable to checkpoint 2000.

---

## 22. Log Returns

Our target is based on **daily log returns** instead of raw stock price.

The transformation is:

```text
r_t = log(P_t / P_(t-1))
```

Instead of:

```text
AAPL:
150
151
149
152
...
```

the model sees:

```text
+0.006
-0.013
+0.020
...
```

This makes the representation less dependent on the absolute price level.

For example:

```text
$100 → $105 = +5%
$500 → $525 = +5%
```

Both are represented as approximately the same percentage movement.

This is also useful because our benchmark includes volatility models such as GARCH, which are naturally concerned with return dynamics and volatility.

---

## 23. Domain Adaptation

**Domain adaptation** is the broader idea behind this experiment.

Chronos-2 starts as:

```text
General time-series foundation model
```

We expose it to:

```text
Financial time series
```

through LoRA.

The result is:

```text
General Chronos-2
       ↓
Financial data
       ↓
Financially adapted Chronos-2
```

This is what we mean when we say that we are adapting Chronos to the financial domain.

---

## 24. Why We Increased the Number of Series

Our first fine-tuning corpus contained:

```text
79 financial series
```

The broader experiment contained:

```text
152 financial series
```

while keeping the main LoRA configuration unchanged.

The purpose was to test whether:

> More diverse financial experience improves domain adaptation.

This is different from simply changing the learning rate, LoRA rank, or number of steps.

V3 used 152 training series and 517,030 training observations.

---

## 25. The Three LoRA Experiments

### LoRA V1

```text
79 series
rank = 8
learning rate = 1e-5
steps = 1000
```

This established that LoRA fine-tuning works.

---

### LoRA V2

```text
79 series
rank = 8
learning rate = 2e-5
steps = 2000
constant_with_warmup
warmup = 100
```

This tested whether a better optimization configuration could improve adaptation.

---

### LoRA V3

```text
152 series
rank = 8
learning rate = 2e-5
steps = 2000
constant_with_warmup
warmup = 100
```

This kept the main training configuration from V2 and increased the financial training corpus.

The purpose was to isolate the effect of broader financial data.

---

# Putting Everything Together

Your current setup can be summarized as:

```text
                    Chronos-2
                       │
                pretrained model
                       │
                base weights frozen
                       │
                  add LoRA
                  rank = 8
                       │
              financial log returns
                       │
                256 observations
                  of context
                       │
                       ▼
                predict next 20
                       │
              2000 training steps
                       │
                       ▼
              Financial Chronos-2
```

The most important distinction is:

### Training controls

These determine **how we train**:

```text
LoRA rank
LoRA alpha
learning rate
scheduler
warmup
batch size
gradient accumulation
training steps
```

### Forecasting problem definition

These determine **what we train the model to do**:

```text
log returns
context length = 256
prediction length = 20
```

### Domain adaptation

This determines **what type of data the model learns from**:

```text
79 financial series
        ↓
152 broader financial series
```

---

# Current POC Direction

The goal is **not necessarily to prove that Chronos replaces GARCH**.

The more interesting hypothesis is:

> Can a financially adapted Chronos-2 provide useful information that complements specialized statistical models?

The final comparison should therefore evaluate:

```text
Chronos-2 zero-shot
Chronos-2 LoRA V1
Chronos-2 LoRA V2
Chronos-2 LoRA V3
        VS
GARCH
GJR-GARCH
EGARCH
ARIMA
ETS
XGBoost
LightGBM
```

on the same held-out 10-stock test set.

The final metrics should include:

```text
Point forecasting
    → MAE
    → return absolute error

Direction
    → directional accuracy

Extreme movements
    → spike precision
    → spike recall
    → F1

Probabilistic forecasting
    → P10-P90 coverage
    → tail breach rates
    → interval width
```

The final question is therefore not simply:

> "Which model has the lowest MAE?"

It is:

> **"Where does a financially adapted Chronos-2 add information that the existing statistical models do not already provide?"**
