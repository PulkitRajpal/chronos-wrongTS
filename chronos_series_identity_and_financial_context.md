# Chronos-2 Series Identity and Financial Context

## Does Chronos-2 understand the name of the time series?

### Short answer

**Not automatically in the financial-semantic sense.**

When Chronos-2 receives a time series, the `id` identifies which observations belong to the same series. For example:

```text
id = "NVDA"
```

mainly tells the preprocessing / pipeline:

> These observations belong to the NVDA series.

That should **not** be interpreted as:

> The model knows that NVDA is NVIDIA, a semiconductor company, a high-beta growth stock, etc.

The ticker string is not the same thing as an explicit semantic description of the asset.

---

## 1. What the series ID does

A typical input can look conceptually like:

```text
id        timestamp       target
NVDA      ...             0.012
NVDA      ...            -0.008
NVDA      ...             0.004
```

The `id` is primarily a **series identifier**.

Its practical purpose is to distinguish:

```text
NVDA series
AAPL series
JPM series
XOM series
```

so that observations from different time series are not mixed together incorrectly.

---

## 2. What the model should not be assumed to know

We should not assume that supplying:

```text
id = "NVDA"
```

automatically gives the model explicit knowledge such as:

```text
Company: NVIDIA
Sector: Technology
Industry: Semiconductors
Style: Growth
Beta: High
Market cap: Large
```

The model instead sees the numerical patterns in the series and whatever additional contextual variables we explicitly supply.

Therefore:

```text
NVDA
```

is not equivalent to providing:

```text
sector = technology
industry = semiconductors
beta = high
style = growth
```

---

## 3. Why this matters for the financial POC

Your held-out universe contains very different equity behaviors:

```text
Semiconductors
Banks
Energy
Healthcare
Utilities
Consumer stocks
High-beta / speculative growth
```

These assets can react very differently to the same market regime.

For example:

```text
NVDA → technology / semiconductor regime
JPM  → financial regime
XOM  → energy regime
LLY  → healthcare regime
```

Giving Chronos only the stock's return history does not explicitly tell it which economic environment the stock belongs to.

That is why broader training data can potentially help: the model may learn recurring temporal patterns across many financial series.

However, that is different from explicitly providing the identity or economic characteristics of the asset.

---

# 4. Why V3 is particularly interesting

V3 increased the financial training universe from roughly:

```text
79 financial series
```

to:

```text
152 broader financial series
```

while keeping the main training configuration broadly aligned with V2.

The purpose of V3 was to test whether broader financial data could improve domain adaptation.

V3 therefore provides a plausible mechanism for **implicit transfer across different financial series**:

```text
many different financial series
            ↓
shared temporal patterns
            ↓
common financial dynamics
            ↓
better representation for unseen series
```

But we should be careful with the interpretation.

We cannot conclude that V3 learned:

```text
"NVDA is a semiconductor"
```

merely because NVDA or similar names appeared in the training corpus.

A more defensible interpretation is:

> The broader training corpus may expose Chronos to recurring temporal structures that transfer across financial series.

---

# 5. Explicit financial context is different

V5 and V6 move toward explicit contextual information.

For example, the model can receive numerical covariates such as:

```text
SPY return
QQQ return
VIX return
5Y yield change
10Y yield change
realized volatility
volume
absolute return
```

This tells the model something about the **environment in which the target stock is moving**.

Conceptually:

```text
                    Target stock
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
          returns     market       volatility
                         │
                  SPY / QQQ / VIX
                         │
                     rates
                    5Y / 10Y
                         │
                         ↓
                     Chronos
```

This is different from simply giving the ticker name.

---

# 6. Static asset metadata could be another experiment

A potentially interesting extension is to provide explicit information about the asset itself.

For example:

```text
sector = technology
industry = semiconductors
market_cap_bucket = large
beta_bucket = high
style = growth
```

Conceptually:

```text
                       Stock
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       history        market          metadata
          │          context             │
          │        SPY/VIX/rates         │
          │                              │
          └──────────────┬───────────────┘
                         ↓
                      Chronos
```

This would test a different question:

> **Does explicitly telling Chronos what type of asset it is improve forecasting?**

---

# 7. Sector information may be more useful than the ticker itself

Instead of relying on:

```text
id = "NVDA"
```

a stronger numerical representation could be:

```text
sector_return
stock_return - sector_return
stock_return - SPY_return
sector_volatility
```

For example:

```text
SPY        +1.0%
Sector     +2.0%
NVDA       +5.0%
```

can be transformed into:

```text
NVDA relative to SPY     = +4.0%
NVDA relative to sector  = +3.0%
```

This gives Chronos actual numerical information about the relationship between the stock and its economic peers.

---

# 8. Recommended next experiment

The most useful next step would be to build a model based on V6 and add **sector / relative-strength context**.

For example:

```text
V6 baseline

+

sector ETF return
stock - sector return
stock - SPY return
sector volatility
```

Then compare:

```text
V5
V6
V7 = V6 + sector context
GARCH
```

on the same held-out stocks.

This would let us separate several possible effects:

```text
More training data?
        ↓
V5 → V6

More context?
        ↓
V6 → V7

Specialized statistical model?
        ↓
V7 → GARCH
```

That is a much cleaner experiment than simply adding more ticker names.

---

# 9. Key takeaway

The distinction to keep in mind is:

```text
Series ID
    ≠
financial identity
```

A ticker such as:

```text
"NVDA"
```

primarily identifies the series.

Financial meaning has to come from:

```text
1. patterns learned from training data
2. explicit numerical covariates
3. potentially explicit static metadata
```

For your POC, this distinction is important because it gives us a concrete experimental question:

> **Does Chronos benefit from being explicitly told about the economic context and type of the asset it is forecasting?**

That is a more interesting next step than assuming that the ticker string itself provides that information.
