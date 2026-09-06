# Chronos-2 Data Ingestion & Forecasting Pipeline Evaluation

## 1. Objective

This POC evaluates whether **Chronos-2**, a Time-Series Foundation Model (TSFM), can provide useful forward forecasts from public market data and whether it can outperform an initial **statistical ML baseline (XGBoost)**.

The work is deliberately structured as a proxy for the broader **new-product / limited-history problem**:

> Given information available up to a historical cutoff, can a model reconstruct or forecast the subsequent behavior of a financial time series well enough to be useful for downstream risk calculations?

The current experiment uses Apple (AAPL) as the primary series, with S&P 500 and VIX available for a cross-market experiment.

---

## 2. Current Data Environment

### Project structure

```text
CHRONOS-WRONGTS/
├── chronos/
├── data/
│   ├── apple_historical.csv
│   ├── S&P 500 Historical Data.csv
│   └── VIXCLS.csv
└── poc.py
```

### AAPL source data

The AAPL file contains:

```text
Date
Close/Last
Volume
Open
High
Low
```

After ingestion and cleaning:

- **AAPL rows:** 2,514
- **Date range:** 2016-09-06 → 2026-09-04
- `AAPL_Close`, `AAPL_Open`, `AAPL_High`, `AAPL_Low`, and `AAPL_Volume` were all parsed as numeric `float64` values.
- No missing values were detected in the cleaned AAPL OHLCV fields.

### Cross-market alignment

The current common-date dataset contains:

```text
AAPL_Close
AAPL_Open
AAPL_High
AAPL_Low
AAPL_Volume
SPX_Close
VIX_Close
```

Current aligned observations:

- **805 rows**
- **2016-09-06 → 2019-11-14**

The shorter cross-market range is caused by the currently available VIX data and is an important limitation of the current Experiment 3.

---

# 3. Data Ingestion Pipeline

The ingestion pipeline performs the following steps:

```text
Raw CSV
   ↓
Column identification
   ↓
Date parsing
   ↓
Numeric cleaning
   ↓
Remove invalid rows
   ↓
Sort chronologically
   ↓
Remove duplicate dates
   ↓
Cross-market date alignment
   ↓
Model-ready time series
```

### Numeric cleaning

The loader handles common market-data formats such as:

```text
$229.98
1,234,567
229.98
```

and converts them to numeric values.

### Date handling

Market data is stored as actual exchange dates.

However, stock-market observations are irregular in calendar time because of:

- weekends
- exchange holidays

Chronos-2's dataframe interface attempts to infer a regular frequency. This initially caused:

```text
ValueError: Could not infer frequency for series AAPL
```

To resolve this, the pipeline creates an internal **synthetic regular Chronos timestamp**:

```text
Observed market session 1 → 2000-01-01
Observed market session 2 → 2000-01-02
Observed market session 3 → 2000-01-03
...
```

Each synthetic step represents one observed trading session.

The original market date is retained separately and is used for:

- evaluation
- plotting
- reporting

This avoids inventing weekend/holiday prices.

---

# 4. Chronos-2 Evaluation Design

The evaluation is **walk-forward**.

At every historical cutoff `T`:

```text
Information available at T
          ↓
    Chronos-2 forecast
          ↓
    Future horizon
          ↓
   Compare with actual
```

Future data is not provided to the model during forecast generation.

The current horizons are:

```text
1 trading session
5 trading sessions
10 trading sessions
20 trading sessions
```

The current run used **25 walk-forward windows**.

This is sufficient for a first POC iteration but is not yet large enough for a statistically strong conclusion.

---

# 5. Models Being Compared

## Model A — Statistical ML

The current baseline is XGBoost.

It uses past information and engineered features such as:

- lagged returns
- OHLC range
- open-to-close return
- volume change
- rolling volatility
- momentum
- moving-average distance

For the cross-market experiment, historical SPX and VIX return features are also included.

## Model B — Chronos-2

Chronos-2 is used for:

### Experiment 1
AAPL Close only.

### Experiment 2
AAPL Close with historical OHLCV as past covariates.

### Experiment 3
AAPL Close with historical S&P 500 and VIX as past covariates.

Chronos produces probabilistic forecasts including:

```text
P05
P10
P50
P90
P95
```

The current analysis focuses primarily on:

```text
P10
P50
P90
```

---

# 6. Experiment 1 — Univariate AAPL Close

## Setup

Input:

```text
AAPL Close
```

Models:

```text
AAPL Close → XGBoost
AAPL Close → Chronos-2
```

The models forecast the future AAPL price over 1, 5, 10, and 20 trading-session horizons.

## Results

| Horizon | XGBoost MAE | Chronos MAE | Chronos MAE Improvement |
|---:|---:|---:|---:|
| 1 | 3.60179 | 3.42843 | +4.81% |
| 5 | 6.66751 | 5.49069 | +17.65% |
| 10 | 8.12313 | 7.11048 | +12.47% |
| 20 | 14.67226 | 10.83392 | +26.16% |

### Directional accuracy

| Horizon | XGBoost | Chronos-2 |
|---:|---:|---:|
| 1 | 52% | 56% |
| 5 | 60% | 52% |
| 10 | 56% | 44% |
| 20 | 48% | 52% |

### Initial interpretation

The current experiment shows a strong preliminary pattern:

> Chronos-2 has lower endpoint MAE than the current XGBoost baseline at all four horizons, with the largest observed advantage at the 20-session horizon.

However, Chronos does **not** consistently outperform XGBoost on direction.

This suggests that the apparent advantage may be more related to **forecasting the level / trajectory / magnitude** than simply predicting the direction of the next move.

---

# 7. Experiment 1 — Probabilistic Forecast Calibration

Chronos-2 also produces prediction intervals.

The current evaluation checks how often the actual value falls outside the P10/P90 boundaries.

A well-calibrated P10/P90 pair would ideally have approximately:

```text
P10 breach rate ≈ 10%
P90 breach rate ≈ 10%
```

## Observed results

| Horizon | P10 Breach | P90 Breach |
|---:|---:|---:|
| 1 | 4% | 20% |
| 5 | 4% | 20% |
| 10 | 12% | 16% |
| 20 | 20% | 12% |

### Interpretation

The upper tail is reasonably close to the desired 10% level at longer horizons, while the lower tail is more problematic at the 20-session horizon.

Specifically:

```text
20-session P10 breach rate = 20%
```

instead of approximately 10%.

This suggests that the current Chronos lower-tail forecast is **too optimistic / insufficiently wide** at that horizon.

This is important for the eventual risk use case because downside-tail calibration matters more than median price accuracy.

---

# 8. Experiment 2 — AAPL OHLCV

## Setup

Chronos-2 receives:

```text
AAPL Close
AAPL Open
AAPL High
AAPL Low
AAPL Volume
```

as the target plus historical/past covariates.

XGBoost receives engineered historical OHLCV features.

## Results

| Horizon | XGBoost MAE | Chronos MAE | Chronos MAE Improvement |
|---:|---:|---:|---:|
| 1 | 3.60179 | 3.27368 | +9.11% |
| 5 | 6.66751 | 5.22506 | +21.63% |
| 10 | 8.12313 | 6.64295 | +18.22% |
| 20 | 14.67226 | 10.42934 | +28.92% |

### Directional accuracy

| Horizon | XGBoost | Chronos-2 |
|---:|---:|---:|
| 1 | 52% | 48% |
| 5 | 60% | 52% |
| 10 | 56% | 48% |
| 20 | 48% | 60% |

### Initial interpretation

Adding OHLCV information improves Chronos's endpoint MAE relative to Experiment 1.

Again, its largest relative advantage occurs at longer horizons.

At 20 sessions:

```text
XGBoost MAE   = 14.67
Chronos MAE   = 10.43
```

This is a promising result, although the XGBoost and Chronos probabilistic capabilities are not yet perfectly symmetric.

---

# 9. Experiment 3 — Cross-Market AAPL + SPX + VIX

## Setup

Chronos-2 receives:

```text
Target:
    AAPL Close

Past covariates:
    S&P 500 Close
    VIX Close
```

The statistical ML model receives historical AAPL, SPX, and VIX-derived features.

## Results

| Horizon | XGBoost MAE | Chronos MAE | Chronos MAE Improvement |
|---:|---:|---:|---:|
| 1 | 0.65965 | 0.70062 | -6.21% |
| 5 | 1.29977 | 1.26004 | +3.06% |
| 10 | 2.79419 | 2.43373 | +12.90% |
| 20 | 5.27411 | 3.74874 | +28.92% |

### Directional accuracy

| Horizon | XGBoost | Chronos-2 |
|---:|---:|---:|
| 1 | 56% | 44% |
| 5 | 52% | 52% |
| 10 | 60% | 44% |
| 20 | 40% | 56% |

### Important limitation

Experiment 3 only covers:

```text
2016-09-06 → 2019-11-14
```

because the currently aligned VIX data stops in 2019.

Therefore this experiment is **not directly comparable** with Experiments 1 and 2, which cover the full AAPL history.

A longer VIX series should be added before drawing conclusions.

---

# 10. Price Spike Experiment

A spike is currently defined as:

```text
|1-day return|
>
2 × recent 20-session volatility
```

The objective is not simply to predict the exact price.

Instead, the experiment asks:

> Does the model recognize when an unusually large move is possible?

## Current result

Only **5 actual spike events** occurred in the current sampled test windows.

| Model | Precision | Recall | True Positives | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.00 | 0.00 | 0 | 0 | 5 |
| Chronos-2 | 0.20 | 0.40 | 2 | 8 | 3 |

### Initial interpretation

Chronos-2 detected:

```text
2 / 5 = 40%
```

of the observed spike events.

The current XGBoost point-forecast baseline detected none.

This is directionally encouraging, but the sample is **far too small** to make a reliable claim.

The spike experiment needs to be expanded substantially.

---

# 11. What the First Run Suggests

The strongest preliminary observation is:

> **Chronos-2 appears to become relatively more competitive as the forecast horizon increases.**

The current 20-session results are especially notable:

```text
Univariate:
Chronos MAE improvement ≈ 26%

OHLCV:
Chronos MAE improvement ≈ 29%

Cross-market:
Chronos MAE improvement ≈ 29%
```

The result is less compelling at the 1-session horizon.

This points to a hypothesis worth testing:

> **Chronos-2 may provide more incremental value for medium-horizon forecasting than for very short-horizon point prediction.**

This aligns better with the broader risk scenarios being investigated, where multi-day and multi-week horizons matter.

---

# 12. Current Limitations

The current POC should **not** yet be presented as proof that Chronos-2 is superior to statistical ML.

### Limited number of windows

Only 25 walk-forward windows are currently used.

The next version should use many more eligible cutoffs and report uncertainty/confidence intervals.

### Incomplete cross-market period

VIX currently limits the common AAPL/SPX/VIX dataset to 2019.

A longer VIX series is required.

### Statistical baseline is not yet probabilistic

Chronos produces:

```text
P10 / P50 / P90
```

while the current XGBoost benchmark is primarily a point predictor.

A stronger and fairer comparison should add **quantile XGBoost** or another probabilistic statistical baseline.

### Spike sample is too small

Only 5 spike events appear in the current sampled evaluation.

More windows and a larger historical dataset/event definition are required.

### Evaluation is currently endpoint-focused

The current MAE comparison emphasizes the final forecast point for each horizon.

The next iteration should also evaluate the complete forecast path.

---

# 13. Next Evaluation Iteration

The next version of the POC should focus on making the comparison statistically stronger rather than immediately adding more model types.

## A. Increase walk-forward windows

Move from:

```text
25 windows
```

to:

```text
all eligible historical cutoffs
```

or several hundred representative windows.

## B. Add a stronger statistical probabilistic baseline

Compare:

```text
XGBoost point forecast
XGBoost quantile forecast
Chronos P10/P50/P90
```

This enables fair comparison of uncertainty estimates.

## C. Add risk-oriented metrics

In addition to MAE/RMSE:

- Quantile / pinball loss
- CRPS
- P10/P90 coverage
- Calibration error
- Tail-event recall
- Tail-event precision
- Maximum drawdown forecast error
- Expected Shortfall error

## D. Separate market regimes

Evaluate:

```text
Calm
Normal
High volatility
Crisis / spike
```

This will test whether Chronos's advantage is concentrated around regime changes.

## E. Increase the spike sample

Evaluate many more historical extreme-move events and separate:

```text
Upside spikes
Downside spikes
Multi-day shocks
```

## F. Extend the cross-market dataset

Use a complete VIX series covering the whole AAPL evaluation period.

---

# 14. Relevance to the New-Product Proxy Problem

The current AAPL experiment is a controlled forecasting benchmark.

The eventual business-oriented experiment should simulate a **new product**.

For a historical product:

```text
                    Product exists historically
                              │
                     Pretend launch at T
                              │
                  Product has limited history
                              │
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
          MSCI Barra      Statistical ML   Chronos-2
              Proxy            Proxy          Forecast
                │             │             │
                └─────────────┼─────────────┘
                              ↓
                       Actual future data
                              │
                              ↓
                         Compare error
```

This is the experiment that most directly answers the team's real question:

> **Can a TSFM provide a better proxy/forecast for a new product when product-specific historical data is limited?**

The current AAPL results are therefore best treated as **Phase 1 feasibility evidence**, not the final proxy benchmark.

---

# 15. Current Conclusion

The first Chronos-2 evaluation is **promising but preliminary**.

The strongest evidence so far is:

1. Chronos-2 has lower endpoint MAE than the initial XGBoost baseline at 5-, 10-, and 20-session horizons.
2. The relative advantage is largest around the 20-session horizon.
3. Chronos produces useful probabilistic forecasts, but downside-tail calibration still needs improvement.
4. The initial spike experiment suggests Chronos can identify some extreme moves that the current point-forecast baseline misses, but the sample is too small to draw a conclusion.
5. Cross-market forecasting shows a similar longer-horizon advantage, but the current VIX history limits the test period.

Therefore the next step should be:

> **Strengthen the benchmark and specifically test whether the observed Chronos advantage survives a large walk-forward evaluation, a probabilistic statistical baseline, regime-separated testing, and finally a simulated new-product/Barra proxy experiment.**

---

## Current POC Position

```text
                Phase 1 — Completed
                        │
                        ▼
              Chronos-2 ingestion
                        │
                        ▼
              AAPL forecasting
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
     Statistical ML               Chronos-2
          │                           │
          └─────────────┬─────────────┘
                        ↓
                  Initial signal
                        │
                        ▼
              Strengthen benchmark
                        │
                        ▼
              New-product simulation
                        │
             ┌──────────┼──────────┐
             ↓          ↓          ↓
          Barra       Stat ML    Chronos
             │          │          │
             └──────────┼──────────┘
                        ↓
                 Actual product
                    behavior
                        │
                        ▼
                Final evaluation
```

**Status: Promising early signal; not yet a production or model-selection conclusion.**
