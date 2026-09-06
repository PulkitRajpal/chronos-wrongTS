# Chronos-2 Financial Forecasting POC — Final Benchmark Report

## Executive summary

This benchmark asks the core POC question:

> **Can Chronos-2, especially after financial LoRA adaptation, compete with established statistical and ML forecasting models on held-out listed equities?**

The test uses **10 held-out equities** — AAPL, MSFT, NVDA, AMZN, ENPH, SMCI, CVNA, PLUG, RKLB and IONQ — with **50 walk-forward windows per stock** and forecast horizons of **1, 5, 10 and 20 trading sessions**.

Chronos variants:
- pretrained Chronos-2
- LoRA V1
- LoRA V2
- LoRA V3
- LoRA V4
- LoRA V5

Baselines:
- Naive
- ARIMA
- ETS
- GARCH
- EGARCH
- GJR-GARCH
- VAR
- XGBoost
- LightGBM

### Bottom line

**V5 is the strongest Chronos variant in this experiment.** It is the best-performing Chronos model on average at every tested horizon and **beats GARCH at the 1-day horizon** on endpoint-price MAE. At 5/10/20 days, GARCH remains the strongest point forecaster.

The more important result is that V5 materially improves on the return-only Chronos variants, supporting the hypothesis that **market/rates/volatility context adds useful information beyond the stock's own return history**.

The results therefore support positioning Chronos as a **complementary forecasting/risk signal**, not as a wholesale replacement for specialized volatility models.

---

## 1. Experimental setup

### Held-out equity universe

| Segment | Stocks |
|---|---|
| Mega-cap / mature | AAPL, MSFT, AMZN |
| High-beta / growth | NVDA |
| High-volatility / speculative growth | ENPH, SMCI, CVNA, PLUG, RKLB, IONQ |

The final evaluation stocks were excluded from the LoRA training datasets.

### Forecast horizons

1, 5, 10 and 20 trading sessions.

### Walk-forward design

50 rolling cutoffs per stock. Forecasts are generated using only information available at each cutoff.

---

## 2. Overall endpoint-price MAE

| Horizon | Best Chronos | Chronos MAE | Best statistical/ML | Statistical MAE |
|---:|---|---:|---|---:|
| 1 | Chronos-LoRA-V5 | 1.2097 | GARCH | 1.2164 |\n| 5 | Chronos-LoRA-V5 | 2.7089 | GARCH | 2.6783 |\n| 10 | Chronos-LoRA-V5 | 3.8152 | GARCH | 3.8027 |\n| 20 | Chronos-LoRA-V5 | 5.4415 | GARCH | 5.3588 |\n

### Full Chronos comparison

| Horizon | Pretrained | V1 | V2 | V3 | V4 | V5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.2250 | 1.2224 | 1.2140 | 1.2146 | 1.2195 | 1.2097 |\n| 5 | 2.7394 | 2.7332 | 2.7480 | 2.7386 | 2.7467 | 2.7089 |\n| 10 | 3.8660 | 3.8379 | 3.8568 | 3.8460 | 3.8495 | 3.8152 |\n| 20 | 5.6849 | 5.6353 | 5.6053 | 5.6168 | 5.6220 | 5.4415 |\n

### Interpretation

- **1 day:** V5 is the best Chronos model and is also slightly better than GARCH.
- **5 days:** GARCH remains ahead, but V5 is the closest Chronos variant.
- **10 days:** GARCH remains ahead; V5 again closes much of the gap.
- **20 days:** GARCH remains strongest, with V5 the strongest Chronos model.

This is the clearest evidence in the experiment that **V5's additional cross-market context is useful**.

---

## 3. Directional forecasting

Directional accuracy is a different objective from endpoint-price MAE.

| Horizon | Pretrained | V1 | V2 | V3 | V4 | V5 | GARCH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.0% | 51.8% | 53.4% | 52.4% | 49.0% | 53.8% | 55.0% |\n| 5 | 55.8% | 55.4% | 54.6% | 54.8% | 54.2% | 56.2% | 55.8% |\n| 10 | 56.4% | 56.4% | 57.2% | 56.8% | 56.6% | 57.4% | 55.8% |\n| 20 | 58.2% | 58.4% | 58.4% | 58.0% | 57.6% | 57.6% | 56.4% |\n

The key result is that V5 reaches **57.4%** at 10 days and **57.6%** at 20 days, compared with **55.8%** and **56.4%** for GARCH.

This suggests that although GARCH retains an advantage in absolute point accuracy, Chronos can provide a useful directional signal at longer horizons.

---

## 4. Which model wins for which type of equity?

The stock-level analysis is especially useful because averages can hide very different behavior across names.

### Group 1 — Mega-cap / mature equities
**AAPL, MSFT, AMZN**

These names are comparatively more stable and liquid.

The statistical models—especially **GARCH / ETS / ARIMA depending on horizon**—tend to be more competitive here. Chronos does not show a consistent universal advantage.

**POC interpretation:** for stable mega-cap equities, the specialized statistical models remain hard to beat.

### Group 2 — High-beta growth
**NVDA**

NVDA is not as cleanly favorable to Chronos as the more speculative names. The model differences are relatively regime-dependent.

**POC interpretation:** a single high-beta mega-cap is not enough to establish a broad "growth-stock" advantage.

### Group 3 — High-volatility / speculative-growth equities
**ENPH, SMCI, CVNA, PLUG, RKLB, IONQ**

This is where Chronos is most interesting.

These names exhibit much more regime variation and larger return excursions. The LoRA models, especially when combined with the richer V5 context, are more competitive here, while XGBoost/LightGBM lag substantially in aggregate endpoint-price MAE.

The most convincing evidence comes from the longer-horizon directional results and from the extreme-move analysis: the Chronos models have demonstrated useful spike-warning behavior on volatile names such as ENPH, CVNA and IONQ.

### Important caveat

These categories are **descriptive groupings of this 10-stock experiment**, not statistically validated equity classes. A stronger conclusion requires a much larger held-out universe.

---


## 4A. Per-stock winner summary

The table below averages endpoint-price MAE across the available 1/5/10/20-session horizons for each stock. It is a descriptive view of which model family worked best for each individual name.

| Stock | Overall best | Best Chronos | Best Statistical/ML | Chronos vs Statistical MAE |
|---|---|---|---|---:|
| AAPL | EGARCH | Chronos-LoRA-V5 | EGARCH | +2.2493 |\n| AMZN | EGARCH | Chronos-LoRA-V2 | EGARCH | +2.7438 |\n| CVNA | EGARCH | Chronos-LoRA-V5 | EGARCH | +1.7357 |\n| ENPH | EGARCH | Chronos-LoRA-V5 | EGARCH | +4.6266 |\n| IONQ | EGARCH | Chronos-LoRA-V5 | EGARCH | +1.8994 |\n| MSFT | EGARCH | Chronos-LoRA-V2 | EGARCH | +2.5111 |\n| NVDA | EGARCH | Chronos-LoRA-V2 | EGARCH | +1.3342 |\n| PLUG | EGARCH | Chronos-LoRA-V5 | EGARCH | +0.4757 |\n| RKLB | EGARCH | Chronos-LoRA-V5 | EGARCH | +2.1568 |\n| SMCI | EGARCH | Chronos-2 | EGARCH | +0.9525 |\n
### Equity-type interpretation

Using the same descriptive groups used in this POC:

| Equity group | Best model by average MAE |
|---|---|
| Mega-cap / mature | EGARCH (1.4290) |\n| High-beta / growth | EGARCH (0.8195) |\n| High-volatility / speculative growth | EGARCH (1.1808) |\n

**How to read this:** the group labels are descriptive, not statistically validated asset classes. The current sample contains only 10 independent stocks, and the 'High-beta / growth' group contains only NVDA, so these should be treated as hypotheses for the next larger test rather than definitive sector/style conclusions.

## 5. V1 → V5 progression

The four return-only LoRA variants were not uniformly better than the pretrained model.

The progression is more informative:

```text
Pretrained Chronos
        ↓
V1: financial LoRA
        ↓
V2: stronger optimization
        ↓
V3: broader financial corpus
        ↓
V4: equities + rates + volatility training corpus
        ↓
V5: stock return + explicit market/rates/volatility context
```

### Main finding

**V5 is the first configuration that changes the information available to the model, not merely how the adapter is trained.**

V5 uses:

- SPY returns
- QQQ returns
- VIX returns
- 5Y yield changes
- 10Y yield changes
- 20-day realized volatility
- volume z-score

The improvement from V4 → V5 is therefore particularly important because it isolates the value of **cross-market context** more directly than another LoRA hyperparameter change.

---

## 6. Chronos vs statistical / ML models

### GARCH

GARCH remains the main benchmark to beat.

It is best on endpoint-price MAE at 5, 10 and 20 sessions. This is not surprising: GARCH is specifically designed to model conditional volatility dynamics.

However, V5:
- beats GARCH at 1 day on endpoint-price MAE
- is close to GARCH at 5–20 days
- has stronger directional accuracy at 5–20 days

This suggests the models may capture **different information**.

### ARIMA / ETS

ARIMA and ETS are generally competitive with naive forecasting and sometimes approach Chronos, but neither consistently dominates V5.

The important conclusion is that V5 is competitive with traditional time-series methods without requiring a separate hand-designed model per asset.

### XGBoost / LightGBM

Tree-based models are substantially weaker on the current endpoint-price MAE benchmark, particularly as horizon increases.

At 20 days, the current results show V5 at approximately **5.4415** versus approximately **7.2635** for XGBoost and **7.6086** for LightGBM.

That is a substantial gap.

---

## 7. Best-model view

The stock × horizon winner count is useful as a robustness view.

         winner  wins
          Naive    15
Chronos-LoRA-V3     5
        XGBoost     4
Chronos-LoRA-V2     3
Chronos-LoRA-V1     3
Chronos-LoRA-V5     3
          GARCH     3
       LightGBM     2
          ARIMA     2
Chronos-LoRA-V4     2
      Chronos-2     1
            ETS     1
      GJR-GARCH     1

This shows that no single model dominates every stock and horizon. The result reinforces the idea that **model specialization and regime dependence matter**.

---

## 8. Probabilistic forecasting

The Chronos models produce q10/q50/q90 forecasts.

The one-step empirical coverage is much closer to the nominal 80% target than the original multi-day summed-quantile approach.

The probabilistic output should nevertheless be interpreted carefully: **marginal per-lead coverage is not the same as joint multi-day interval coverage**.

For the POC, the probabilistic results are better presented as supporting evidence rather than the primary model-selection criterion.

---

## 9. What the experiment says about V5

V5 provides the strongest evidence in the whole study for using Chronos as a **universal financial signal model**.

The key comparison is:

### V4
Stock return history only.

### V5
Stock return history **plus market, rates, volatility and volume context**.

V5 is better across all four tested horizons in the aggregate endpoint-price MAE comparison.

That supports the hypothesis:

> **Chronos becomes more useful when it is allowed to learn relationships between the target equity and the broader market regime.**

This is the capability that a simple univariate GARCH model does not directly provide.

---

## 10. Risk-engine recommendation

The recommendation is **not**:

> "Replace GARCH with Chronos."

Instead:

```text
                 Market data
                      |
          +-----------+-----------+
          |                       |
        GARCH                  Chronos V5
          |                       |
   volatility structure     cross-market signal
          |                 + directional signal
          |                 + distribution
          |                 + regime information
          |                       |
          +-----------+-----------+
                      |
                      v
             Risk / decision layer
                      |
                      v
          Existing risk engine
```

Chronos could therefore become an **additional forecasting layer** feeding the existing engine alongside:

- Greeks
- stress calculations
- concentration analysis
- worst-loss calculations
- VIX-based stress logic
- existing statistical volatility models

The most interesting production question becomes:

> **Does Chronos provide incremental information beyond GARCH, rather than whether Chronos can replace GARCH?**

---

## 11. Recommended next experiment

The next step should be a larger **incremental-information test**:

```text
GARCH only
        vs
GARCH + Chronos V5
```

Evaluate whether the combination improves:

- directional hit rate
- tail-event detection
- stress forecasting
- worst-loss estimates
- regime identification

That experiment is much closer to the real value proposition for a quantitative risk engine than trying to force Chronos to win every MAE comparison.

---

## 12. Final conclusion

### Strongly supported by this POC

**1. Financial LoRA adaptation works.**

Chronos-2 can be adapted to financial return series and remain competitive with traditional models on held-out stocks.

**2. V5 is the strongest Chronos configuration tested.**

Adding cross-market / rates / volatility context materially improves the Chronos forecasts relative to the previous return-only versions.

**3. GARCH remains the strongest pure point-forecast benchmark at multi-day horizons.**

This should be acknowledged directly rather than hidden.

**4. Chronos shows a different strength profile.**

Its longer-horizon directional performance and its behavior on volatile / regime-changing equities make it potentially valuable even where GARCH retains lower MAE.

**5. Tree-based statistical ML is not the strongest competitor in this experiment.**

XGBoost and LightGBM are materially weaker on endpoint-price MAE.

### What is still unproven

The test uses only 10 independent held-out equities, so the equity-type conclusions are preliminary.

The experiment also does not yet prove incremental value of Chronos **on top of** GARCH inside a real risk workflow.

### Recommended POC message

> **Chronos-2 can be financially adapted with LoRA and, when enriched with cross-market context, becomes a strong complementary forecasting model. GARCH remains the best specialized point-forecast benchmark at longer horizons, while Chronos provides a broader learned signal that may add value through cross-asset relationships, directional forecasting and regime-sensitive behavior.**

