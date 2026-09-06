# Chronos-2 Financial LoRA Fine-Tuning POC

## 1. Objective

The objective of this phase of the POC is to determine whether **domain adaptation of Chronos-2 using LoRA** can improve forecasting behavior on financial time series.

The key research question is:

> Does exposing Chronos-2 to a broader and more diverse set of financial time series during LoRA fine-tuning improve its ability to forecast unseen listed stocks?

The final evaluation set is kept separate from the fine-tuning data.

---

## 2. Final Evaluation Universe

The following 10 stocks are treated as the held-out evaluation set:

- AAPL
- MSFT
- NVDA
- AMZN
- ENPH
- SMCI
- CVNA
- PLUG
- RKLB
- IONQ

These stocks are excluded from the financial fine-tuning corpus to reduce the risk of asset-level leakage.

The goal is therefore to test transfer:

```text
Broad financial training data
            ↓
       LoRA adaptation
            ↓
      Chronos-2
            ↓
    unseen evaluation stocks
```

---

## 3. Baseline Chronos-2 Results

Before fine-tuning, Chronos-2 was compared against several statistical and machine-learning models across 10 stocks and horizons of 1, 5, 10, and 20 trading sessions.

The main overall result was:

| Horizon | Best model | Best MAE | Chronos-2 MAE |
|---:|---|---:|---:|
| 1 | GARCH | 1.216 | 1.302 |
| 5 | GARCH | 2.678 | 2.757 |
| 10 | GARCH | 3.803 | 3.912 |
| 20 | GARCH | 5.359 | 5.690 |

Thus, in the current benchmark, **GARCH was the strongest point-forecasting model**.

Chronos-2 was generally competitive with ARIMA and ETS, and substantially better than XGBoost and LightGBM at longer horizons.

The current evidence therefore does **not** support replacing specialized statistical models such as GARCH with zero-shot Chronos-2 for ordinary point forecasting.

---

## 4. Why LoRA Was Tested

The motivation for LoRA fine-tuning was not simply to force Chronos-2 to beat GARCH.

Instead, the hypothesis was:

> A pretrained, general-purpose time-series foundation model may benefit from financial-domain adaptation.

Potentially useful financial patterns include:

- volatility clustering
- large return shocks
- regime changes
- reversals
- long-horizon uncertainty
- extreme movements

LoRA was selected because it allows a small set of adapter parameters to be trained while keeping the pretrained Chronos-2 weights frozen.

Conceptually:

```text
                Chronos-2 pretrained model
                         │
              ┌──────────┴──────────┐
              │                     │
        Frozen base model      Trainable LoRA
                                    │
                                    ▼
                         Financial adaptation
```

---

## 5. Training Data Representation

The initial fine-tuning experiments used **daily log returns** rather than raw prices.

The return transformation is:

```text
r_t = log(P_t / P_(t-1))
```

This reduces the direct dependence on the absolute price level.

For example, a 5% move in a $100 stock and a 5% move in a $500 stock are represented similarly as returns.

The training data uses Chronos-style long-format series:

| id | timestamp | target |
|---|---|---:|
| GOOG | date | log return |
| GOOG | date | log return |
| META | date | log return |
| ... | ... | ... |

---

## 6. Chronos-2 Training Configuration

The initial experiments used:

```text
Model:              amazon/chronos-2
Context length:     256
Prediction length:  20
LoRA rank:          8
LoRA alpha:         16
LoRA dropout:       0.0
```

The LoRA target modules were:

```text
self_attention.q
self_attention.v
self_attention.k
self_attention.o
output_patch_embedding.output_layer
```

These correspond to the LoRA target modules used by the Chronos-2 fine-tuning implementation.

---

## 7. LoRA V1 — Initial Experiment

### Configuration

```text
Training series:              79
Training observations:        186,126
Validation series:            79

Context:                      256
Prediction horizon:           20

LoRA rank:                    8
LoRA alpha:                   16

Learning rate:                1e-5
Training steps:               1,000
Batch size:                   1
Gradient accumulation:       8
```

### Result

Validation loss improved only slightly, from approximately:

```text
4.388 → 4.377
```

This indicated that the pipeline was working but that the degree of domain adaptation was limited.

A contributing factor was the learning-rate schedule: the learning rate decayed toward approximately zero by the end of training.

### Interpretation

V1 successfully demonstrated:

- Chronos-2 can be fine-tuned with LoRA in the environment.
- The financial return dataset is accepted by the training pipeline.
- Training and validation both execute successfully.
- The adapter can be saved.

However, the validation improvement was small.

---

## 8. LoRA V2 — Training Schedule Experiment

V2 kept the same 79-series financial corpus but changed the optimization setup.

### Configuration

```text
Training series:              79
Training observations:        186,126

Context:                      256
Prediction horizon:           20

LoRA rank:                    8
LoRA alpha:                   16

Learning rate:                2e-5
Training steps:               2,000
Batch size:                   1
Gradient accumulation:       4

Learning-rate schedule:
    constant_with_warmup

Warmup steps:
    100
```

### Result

The validation loss improved more than V1:

```text
Best validation loss ≈ 4.355
```

The best region occurred roughly around the middle of training rather than at the final step.

The final result was still only a modest improvement, but it showed that the learning-rate schedule affected adaptation.

### Interpretation

V2 suggested that:

> The optimization configuration matters, but changing training hyperparameters alone was not producing a dramatic domain adaptation.

This led to the next hypothesis:

> A broader and more diverse financial training corpus may be more important than further hyperparameter tuning.

---

## 9. Why More Financial Series Were Added

The next experiment increased the diversity of the fine-tuning corpus.

The reasoning was that a TSFM may benefit from seeing more examples of financial behavior rather than repeatedly seeing a small number of related equity series.

The broader corpus included:

- US equities
- international equities
- equity ETFs and indices
- bonds and rates
- commodities
- FX
- volatility series

**Crypto was intentionally removed** because the primary application focus is listed-stock risk and forecasting.

The objective was not simply to maximize the number of rows.

The objective was to increase the diversity of financial regimes and behaviors.

---

## 10. LoRA V3 — Broader Financial Corpus

### Configuration

The LoRA and optimization parameters were kept the same as V2 so that the effect of the larger training corpus could be isolated.

```text
Training series:              152
Training observations:        517,030
Validation series:            152
Validation observations:      77,216

Context:                      256
Prediction horizon:           20

LoRA rank:                    8
LoRA alpha:                   16

Learning rate:                2e-5
Training steps:               2,000
Batch size:                   1
Gradient accumulation:       4

Scheduler:
    constant_with_warmup

Warmup steps:
    100
```

The V3 run completed successfully on the RTX 5060 Laptop GPU with approximately 8 GB VRAM.

---

## 11. V3 Results

The broad-corpus experiment produced a noticeably lower validation loss.

The run reached:

```text
Best observed validation loss ≈ 4.291
```

and finished at approximately:

```text
Final validation loss ≈ 4.297
```

The best region occurred relatively early/midway through training.

The V3 training log shows:

```text
step 100   → 4.303
step 200   → 4.291
step 300   → 4.366
step 400   → 4.374
step 500   → 4.374
step 600   → 4.294
step 700   → 4.301
step 800   → 4.300
step 900   → 4.324
step 1000  → 4.306
step 2000  → 4.297
```

The exact validation trajectory is noisy, but the important comparison is:

```text
V1: ~4.377
V2: ~4.355
V3: ~4.291
```

This is considerably more encouraging than the V1 → V2 improvement.

---

## 12. Current Interpretation

The current experiments support the following preliminary hypothesis:

> **Increasing the amount and diversity of financial time-series data appears more promising for Chronos-2 LoRA adaptation than simply increasing training steps or changing basic optimization parameters.**

However, this conclusion is based on **fine-tuning validation loss**.

It has **not yet been established that the lower validation loss translates into better forecasting of the held-out 10 stocks**.

That distinction is critical.

---

## 13. What Has Been Proven

### Engineering feasibility

The following have now been demonstrated:

```text
Chronos-2 2.3.1
        ↓
PEFT / LoRA
        ↓
RTX 5060 Laptop GPU
        ↓
CUDA 12.8
        ↓
financial time series
        ↓
training
        ↓
validation
        ↓
saved LoRA checkpoint
```

All of these steps are working.

### Data scaling signal

Moving from:

```text
79 series
```

to:

```text
152 series
```

produced a substantially lower validation loss.

This is evidence that broader financial data can improve the domain-adaptation objective.

It is **not yet proof of improved out-of-sample stock forecasting**.

---

## 14. What Has Not Been Proven

The following questions remain open:

1. Does LoRA improve Chronos-2 point-forecast MAE on unseen stocks?
2. Does financial LoRA improve directional accuracy?
3. Does it improve detection of extreme moves?
4. Does it improve prediction-interval calibration?
5. Does the broader 152-series corpus outperform the original 79-series corpus on the actual 10-stock test?
6. Does the best validation-loss checkpoint also produce the best financial forecasting performance?

These should be answered before further increasing LoRA rank or performing extensive hyperparameter sweeps.

---

## 15. Next Experiment

The next experiment should be **evaluation, not more training**.

Compare:

```text
1. Chronos-2 zero-shot
2. Chronos-2 LoRA V1
3. Chronos-2 LoRA V2
4. Chronos-2 LoRA V3
5. GARCH
6. GJR-GARCH
7. EGARCH
8. ARIMA
9. ETS
10. XGBoost
11. LightGBM
```

using exactly the same held-out 10-stock benchmark.

The evaluation should include:

### Point forecasting

```text
MAE
median MAE
return absolute error
```

### Direction

```text
directional accuracy
```

### Extreme movements

```text
spike precision
spike recall
F1
```

### Probabilistic forecasting

```text
P10-P90 coverage
lower-tail breach rate
upper-tail breach rate
interval width
```

---

## 16. Important Model-Selection Principle

The final V3 checkpoint should not automatically be treated as the best model just because it is the last checkpoint.

The validation results indicate the lowest loss occurred before the end of training.

Therefore, the final evaluation should preferably compare the best validation checkpoint(s), rather than assuming:

```text
latest checkpoint = best checkpoint
```

This is particularly important because the V3 curve is noisy and shows degradation and recovery across training.

---

## 17. Final POC Hypothesis

The broader project should now test a more interesting proposition than:

> "Can Chronos replace GARCH?"

The stronger hypothesis is:

> **Can financially adapted Chronos-2 provide information that is complementary to specialized statistical risk models?**

A potential future architecture is:

```text
                    Market history
                          │
             ┌────────────┴────────────┐
             │                         │
           GARCH                  Chronos-2 LoRA
             │                         │
      volatility/path          probabilistic forecast
             │                         │
             └────────────┬────────────┘
                          │
                 model disagreement
                          │
                          ▼
                  risk signal
```

This would position Chronos as a **complementary risk signal**, particularly if it can improve uncertainty estimation or extreme-move detection while GARCH remains stronger for ordinary point forecasting.

---

## 18. Current Status

```text
Baseline statistical comparison       ✅
Zero-shot Chronos benchmark           ✅
LoRA pipeline                          ✅
LoRA V1                                ✅
LoRA V2                                ✅
Broader financial corpus               ✅
LoRA V3                                ✅
Held-out 10-stock LoRA evaluation     NEXT
Rank-16 experiment                     LATER
Extreme-event enriched training        LATER
Multivariate/covariate training        LATER
```

### Current conclusion

**Do not claim yet that LoRA improves stock forecasting.**

The current evidence only supports:

> Chronos-2 can be financially adapted with LoRA, and increasing the diversity of the financial fine-tuning corpus produced a materially lower validation loss than the earlier 79-series runs.

The decisive next step is to test whether that improvement transfers to the unseen 10-stock benchmark.
