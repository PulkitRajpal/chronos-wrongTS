# Chronos-2 V1–V6: What Changed and How They Compare With GARCH

## 1. Purpose

This document summarizes the six Chronos financial fine-tuning variants developed during the POC and compares their held-out forecasting results against **GARCH**, the main specialized statistical benchmark.

The evaluation uses the same held-out equity set:

```text
AAPL, MSFT, NVDA, AMZN, ENPH, SMCI, CVNA, PLUG, RKLB, IONQ
```

with:

```text
50 walk-forward windows per stock
10 stocks
4 horizons: 1, 5, 10, 20 trading days
500 forecast observations per model per horizon
```

The primary metrics are:

- endpoint-price MAE
- directional accuracy
- relative MAE versus GARCH

The important interpretation is not simply "which Chronos version wins," but **what additional information each version introduces and whether that translates into out-of-sample gains over GARCH**.

---

## 2. V1 → V6 at a glance

| Version | What it is | Training / context | Main change |
|---|---|---|---|
| **V1** | Initial financial LoRA | 79 financial series, context 256 | First financial-domain adaptation |
| **V2** | Better-optimized V1 | 79 series, context 256 | Higher LR, 2,000 steps, warmup |
| **V3** | Broader financial corpus | 152 financial series, context 256 | More/diverse financial series |
| **V4** | Equities + rates + volatility | 132 series, context 256 | Explicitly broadens the training corpus toward equities, rates and volatility |
| **V5** | Multivariate/covariate Chronos | 132-series financial corpus, context 512 | Adds explicit market/rates/volatility/volume context |
| **V6** | Broader V5-style model | Diverse equity universe, context 512 | Adds more industry/style diversity plus absolute-return/volatility information |

### The conceptual progression

```text
V1
  Financial LoRA
      ↓
V2
  Better optimization
      ↓
V3
  More financial series
      ↓
V4
  Equities + rates + volatility in training data
      ↓
V5
  Explicit market/rates/volatility covariates
      ↓
V6
  Broader/diverse equity corpus
  + explicit volatility features
```

---

## 3. Detailed explanation of each version

### V1 — Initial financial adaptation

**Goal:** establish whether LoRA fine-tuning on financial returns can adapt Chronos-2 to the domain.

Configuration:

```text
79 financial series
context = 256
prediction = 20
LoRA rank = 8
LoRA alpha = 16
learning rate = 1e-5
steps = 1,000
```

V1 is the baseline against which the later adaptations should be understood.

Its main contribution is demonstrating that the pretrained foundation model can be adapted to financial return behavior without changing the underlying Chronos architecture.

---

### V2 — Optimization improvement

**Goal:** determine whether the initial V1 result was limited by the optimization setup.

Configuration:

```text
79 financial series
context = 256
prediction = 20
LoRA rank = 8
LoRA alpha = 16
learning rate = 2e-5
steps = 2,000
constant-with-warmup scheduler
warmup = 100
```

The training corpus stayed the same as V1.

So V1 → V2 primarily asks:

> Does better optimization produce a better financial adapter?

---

### V3 — Broader financial training corpus

**Goal:** determine whether broader financial exposure helps Chronos learn patterns that transfer to unseen equities.

Configuration:

```text
152 financial series
context = 256
prediction = 20
same main optimization setup as V2
```

The documented V3 run used 152 series and achieved a materially lower validation loss than the earlier 79-series versions.

The interesting hypothesis here is **cross-series transfer**:

```text
many different financial series
          ↓
shared temporal patterns
          ↓
financial representation
          ↓
better generalization
```

V3 did **not** explicitly give the model SPY/VIX/rates covariates during training. Its advantage, where it exists, should therefore be interpreted as coming from broader financial exposure rather than explicit macro features.

---

### V4 — Equities + rates + volatility

**Goal:** make the training corpus more explicitly representative of the risk factors relevant to listed equities.

The V4 run used:

```text
132 training/validation series
context = 256
prediction = 20
LoRA rank = 8
LR = 2e-5
2,000 steps
constant-with-warmup
```

The V4 experiment was explicitly described as **EQUITIES + RATES + VOLATILITY**.

This is different from V3: V3 primarily broadened the number/diversity of financial series; V4 deliberately emphasized equity/rate/volatility-related instruments.

---

### V5 — Explicit cross-market covariates

V5 is the most important architectural change before V6.

Instead of only training on a collection of financial target series, V5 gives the model explicit historical contextual variables.

The target is the stock's daily log return.

The past-only context includes:

```text
SPY return
QQQ return
VIX return
5Y yield change
10Y yield change
20-day realized volatility
volume z-score
```

V5 therefore tests:

> Does Chronos become better when it can explicitly see the broader market regime?

The context length was increased to 512.

---

### V6 — Broader equity diversity + richer volatility representation

V6 keeps the V5-style contextual approach but changes the training universe and feature set.

The V6 design uses:

```text
512 context
diverse equity universe
multiple industries / styles
```

with:

```text
SPY return
QQQ return
VIX return
5Y change
10Y change
absolute return
20-day realized volatility
volume z-score
```

The idea was to test:

> Does a broader and more diverse equity training population improve the V5 approach?

The held-out evaluation remained unchanged.

---

# 4. Actual results versus GARCH

## 1-day horizon

| Model | MAE | Direction |
|---|---:|---:|
| **V1** | 1.2224 | 51.8% |
| **V2** | 1.2140 | 53.4% |
| **V3** | 1.2146 | 52.4% |
| **V4** | 1.2195 | 49.0% |
| **V5** | **1.2104** | 54.8% |
| **V6** | 1.2112 | 54.0% |
| **GARCH** | 1.2164 | **55.0%** |

### Interpretation

**V5 is the strongest Chronos version on 1-day MAE and beats GARCH.**

Relative to GARCH:

```text
V5 MAE improvement ≈ 0.49%
```

V6 is almost as good, but V5 is marginally better.

---

## 5-day horizon

| Model | MAE | Direction |
|---|---:|---:|
| V1 | 2.7332 | 55.4% |
| V2 | 2.7423 | 54.6% |
| V3 | 2.7258 | 54.8% |
| V4 | 2.7467 | 54.2% |
| **V5** | **2.7412** | **55.0%** |
| V6 | 2.7409 | 54.6% |
| **GARCH** | **2.6783** | **55.8%** |

GARCH clearly remains better on point accuracy.

V6 is essentially tied with V5 but does not close the gap to GARCH enough to claim superiority.

---

## 10-day horizon

| Model | MAE | Direction |
|---|---:|---:|
| V1 | **3.8379** | 56.4% |
| V2 | 3.8568 | 57.2% |
| V3 | 3.8460 | 56.8% |
| V4 | 3.8495 | 56.6% |
| V5 | 3.8668 | **57.6%** |
| V6 | 3.8839 | **57.6%** |
| **GARCH** | **3.8027** | 55.8% |

The point-forecast advantage still belongs to GARCH.

But both V5 and V6 are **1.8 percentage points better than GARCH in directional accuracy**.

---

## 20-day horizon

| Model | MAE | Direction |
|---|---:|---:|
| V1 | 5.5155 | 58.4% |
| V2 | 5.5427 | 58.4% |
| V3 | 5.5317 | 58.0% |
| V4 | 5.5333 | 57.6% |
| V5 | 5.5631 | 58.2% |
| V6 | 5.5930 | **58.6%** |
| **GARCH** | **5.3588** | 56.4% |

This is the clearest example of the different strengths:

```text
GARCH
    → lower point-forecast MAE

V6
    → higher directional accuracy
```

V6 is **2.2 percentage points better than GARCH on direction** at 20 days, despite being worse on endpoint-price MAE.

---

# 5. Summary of V1–V6 relative to GARCH

| Horizon | Best Chronos MAE | Best Chronos | GARCH MAE | Chronos vs GARCH |
|---:|---:|---:|---:|---|
| 1D | **1.2104** | **V5** | 1.2164 | **Chronos wins** |
| 5D | 2.7409 | **V6** | **2.6783** | GARCH wins |
| 10D | 3.8379 | **V1** | **3.8027** | GARCH wins |
| 20D | 5.5155 | **V1** | **5.3588** | GARCH wins |

There is therefore **no evidence that the latest V6 simply dominates all earlier models**.

The results show different optima for different objectives.

---

# 6. V5 vs V6: what did the extra training data actually do?

This is probably the most important finding from the final experiments.

### V5

```text
MAE:
1D  = 1.2104
5D  = 2.7412
10D = 3.8668
20D = 5.5631
```

### V6

```text
MAE:
1D  = 1.2112
5D  = 2.7409
10D = 3.8839
20D = 5.5930
```

V6 did **not** improve aggregate MAE.

The pairwise comparison shows:

```text
1D   V6 wins 44.6% of windows
5D   V6 wins 50.0%
10D  V6 wins 45.2%
20D  V6 wins 49.4%
```

So the experiment does not support:

> "More diverse equities automatically make V5 better."

However, V6 did slightly improve long-horizon directional accuracy:

```text
10D: 57.6% vs V5 57.6%
20D: 58.6% vs V5 58.2%
```

This suggests the broader corpus may have changed the model's **directional behavior** more than its point-forecast accuracy.

---

# 7. What appears to work best

Based on the complete sequence, there are three distinct lessons.

### Lesson 1 — Financial adaptation helps

Moving from the pretrained model to financial LoRA produces competitive results, particularly at shorter horizons and on directional metrics.

### Lesson 2 — More financial data can help, but not monotonically

V3's broader corpus was useful during fine-tuning, but the downstream test does not show a simple:

```text
more training data
    =
better forecast
```

relationship.

### Lesson 3 — Explicit context is more promising than simply adding stocks

The largest qualitative jump came with V5:

```text
stock history
+
market
+
rates
+
volatility
+
volume
```

V5 became the strongest Chronos configuration in aggregate MAE.

That is more compelling than V6's simple expansion of the equity universe.

---

# 8. How to position Chronos vs GARCH

The evidence does **not** support:

> "Chronos replaces GARCH."

The evidence supports:

> **"Chronos and GARCH exhibit different strengths."**

GARCH is stronger for:

```text
multi-day endpoint-price accuracy
conditional volatility structure
```

Chronos V5/V6 is interesting for:

```text
cross-market context
longer-horizon directional information
generalized learned representations
potential regime-sensitive signals
```

That suggests an architecture like:

```text
              Market / portfolio history
                         │
             ┌───────────┴───────────┐
             │                       │
           GARCH                  Chronos V5/V6
             │                       │
       volatility path        cross-market forecast
             │                       │
             └───────────┬───────────┘
                         ↓
                  risk signal layer
                         ↓
                 existing risk engine
```

---

# 9. Final recommendation

For this POC, I would treat:

**V5 as the current best Chronos configuration for forecasting performance.**

Not V6.

The V6 experiment is still useful because it tells us that simply increasing equity diversity did **not** provide a clear incremental improvement.

The strongest next research direction is therefore **better contextual information**, not simply more stocks.

The most promising future experiment would be a V7 that adds:

```text
sector ETF return
stock - sector return
stock - SPY relative return
VIX level / term structure
multi-horizon realized volatility
yield-curve slope
credit spread proxy
DXY / FX
momentum / drawdown
```

while keeping the V5/V6 512 context.

The central research question then becomes:

> **Can Chronos learn cross-asset and regime relationships that GARCH, which is primarily a single-series conditional-volatility model, does not capture?**

That is a much stronger POC question than simply trying to force Chronos to produce a lower MAE than GARCH.
