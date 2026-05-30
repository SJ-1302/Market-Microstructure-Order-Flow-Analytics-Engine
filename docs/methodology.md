# Methodology: Market Microstructure Metrics

> **Document Version**: 1.0
> **Last Updated**: 2024
> **Authors**: Market Microstructure & Order Flow Analytics Engine Team

This document provides detailed mathematical definitions, formulas, interpretations, and academic references for every microstructure metric computed by the engine.

---

## Table of Contents

1. [Bid-Ask Spread Analysis](#1-bid-ask-spread-analysis)
2. [Order Book Imbalance](#2-order-book-imbalance)
3. [Trade-to-Quote Ratio](#3-trade-to-quote-ratio)
4. [Adverse Selection Models](#4-adverse-selection-models)
5. [Price Impact Analysis](#5-price-impact-analysis)
6. [Queue Position Decay](#6-queue-position-decay)
7. [Hawkes Process Intensity](#7-hawkes-process-intensity)
8. [Signal Generation](#8-signal-generation)
9. [Adaptive Calibration](#9-adaptive-calibration)

---

## 1. Bid-Ask Spread Analysis

The bid-ask spread is the most fundamental measure of market liquidity and transaction costs. It represents the cost of immediacy — the premium paid for immediate execution versus patient limit-order placement.

### 1.1 Quoted Spread

**Definition**: The difference between the best ask and best bid prices at a given point in time.

**Formula**:

```
S_quoted(t) = P_ask,1(t) - P_bid,1(t)
```

**Relative Quoted Spread** (normalized by midpoint):

```
s_quoted(t) = S_quoted(t) / M(t)

where M(t) = [P_ask,1(t) + P_bid,1(t)] / 2
```

**Interpretation**: The quoted spread reflects the minimum round-trip transaction cost for a market order of size ≤ min(Q_ask,1, Q_bid,1). Narrower spreads indicate higher liquidity and lower trading costs. The quoted spread is directly observable and serves as a lower bound on effective transaction costs.

**Reference**: Demsetz, H. (1968). *The Cost of Transacting*. Quarterly Journal of Economics.

### 1.2 Effective Spread

**Definition**: The actual transaction cost incurred based on the trade execution price relative to the prevailing midpoint.

**Formula**:

```
S_effective(t) = 2 × d(t) × [P_trade(t) - M(t)]

where:
  d(t) = +1 for buyer-initiated trades
  d(t) = -1 for seller-initiated trades
  M(t) = midpoint at time of trade
```

**Interpretation**: The effective spread captures the true cost of trading, accounting for trades that may execute at prices different from the quoted prices (e.g., trades that walk the book). It is always ≥ 0 for correctly classified trades and is typically ≤ quoted spread for trades within the best quotes, but can exceed it for large orders.

**Reference**: Bessembinder, H. (2003). *Trade Execution Costs and Market Quality after Decimalization*. Journal of Financial and Quantitative Analysis.

### 1.3 Realized Spread

**Definition**: The market maker's profit after accounting for subsequent adverse price movement.

**Formula**:

```
S_realized(t) = 2 × d(t) × [P_trade(t) - M(t + Δ)]

where Δ is the evaluation horizon (typically 5 seconds or 30 seconds)
```

**Interpretation**: The realized spread decomposes the effective spread into a component earned by the market maker (realized spread) and a component lost to informed traders (price impact). A low or negative realized spread indicates that trades are predominantly information-driven.

**Decomposition**:
```
S_effective = S_realized + Price Impact
```

**Reference**: Huang, R.D. & Stoll, H.R. (1996). *Dealer versus Auction Markets*. Journal of Financial Economics.

### 1.4 Time-Weighted Average Spread (TWAS)

**Definition**: The duration-weighted mean spread over a specified time interval.

**Formula**:

```
TWAS(T₁, T₂) = [1 / (T₂ - T₁)] × ∫_{T₁}^{T₂} S_quoted(t) dt

Discrete approximation:
TWAS ≈ Σᵢ S_quoted(tᵢ) × Δtᵢ / Σᵢ Δtᵢ
```

**Interpretation**: TWAS provides a fair representation of average spreads, giving more weight to periods where the spread persists for longer durations. This is preferred over simple averages when quote updates are unevenly spaced.

---

## 2. Order Book Imbalance

Order book imbalance quantifies the asymmetry between buy-side and sell-side liquidity, serving as a leading indicator of short-term price direction.

### 2.1 Volume Order Imbalance (OI)

**Definition**: The normalized difference between bid and ask volumes at the best price level.

**Formula**:

```
OI(t) = [V_bid,1(t) - V_ask,1(t)] / [V_bid,1(t) + V_ask,1(t)]
```

**Range**: [-1, +1], where +1 indicates all liquidity on the bid side (bullish) and -1 indicates all liquidity on the ask side (bearish).

### 2.2 Multi-Level Book Imbalance (OBI)

**Definition**: Depth-weighted imbalance across multiple price levels.

**Formula**:

```
OBI(t) = [Σₖ wₖ × V_bid,k(t) - Σₖ wₖ × V_ask,k(t)] / [Σₖ wₖ × V_bid,k(t) + Σₖ wₖ × V_ask,k(t)]

where k = 1, ..., K (price levels)
```

### 2.3 Weighted OBI (Exponential Decay)

**Definition**: OBI with exponentially decaying weights that prioritize near-touch levels.

**Weights**:

```
wₖ = exp(-β × (k - 1))

where β > 0 is the decay parameter (default: 0.5)
```

**Interpretation**: Deeper levels contribute less to the imbalance signal, reflecting the empirical observation that book depth far from the touch has lower predictive power for short-term price movements.

**Reference**: Cao, C., Hansch, O. & Wang, X. (2009). *The Information Content of an Open Limit-Order Book*. Journal of Futures Markets.

---

## 3. Trade-to-Quote Ratio (TQR)

### Definition

The ratio of trade messages to quote update messages within a given time window.

**Formula**:

```
TQR(T₁, T₂) = N_trades(T₁, T₂) / N_quotes(T₁, T₂)
```

### Interpretation

- **Low TQR** (< 0.1): High quote-to-trade ratio, indicative of HFT market-making activity (frequent quote updates with few executions).
- **High TQR** (> 0.5): More trades relative to quote updates, suggestive of directional flow or reduced market-making.
- **Spikes in TQR**: Can signal shifts in market regime, such as transitions from calm to volatile periods.

**Reference**: Hendershott, T., Jones, C.M. & Menkveld, A.J. (2011). *Does Algorithmic Trading Improve Liquidity?* Journal of Finance.

---

## 4. Adverse Selection Models

### 4.1 Kyle's Lambda (λ)

**Definition**: The permanent price impact coefficient from Kyle's (1985) model, measuring the informativeness of order flow.

**Model**:

```
ΔPₜ = λ × OFₜ + εₜ

where:
  ΔPₜ = price change (midpoint return)
  OFₜ = signed order flow (net buy volume - net sell volume)
  εₜ  = noise term
  λ   = Kyle's Lambda (price impact per unit of order flow)
```

**Estimation**: OLS regression of price changes on signed order flow over rolling windows.

**Interpretation**: Higher λ implies greater adverse selection risk — each unit of order flow moves prices more, indicating a higher proportion of informed traders. Market makers widen spreads when λ is high to compensate for adverse selection losses.

**Reference**: Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315-1335.

### 4.2 PIN (Probability of Informed Trading)

**Definition**: The unconditional probability that a randomly selected trade is initiated by an informed trader.

**Model (Easley & O'Hara, 1992, 1996)**:

The sequential trade model assumes:
- With probability α, an information event occurs at the start of each trading period
- Given an information event, with probability δ it is bad news (1-δ good news)
- Informed traders arrive at rate μ
- Uninformed buyers arrive at rate εᵦ, uninformed sellers at rate εₛ

**Likelihood for a day with B buys and S sells**:

```
L(B, S | θ) = (1 - α) × f(B|εᵦ) × f(S|εₛ)
            + α × δ × f(B|εᵦ) × f(S|εₛ + μ)
            + α × (1 - δ) × f(B|εᵦ + μ) × f(S|εₛ)

where f(n|λ) = e^(-λ) × λⁿ / n!  (Poisson PMF)
      θ = (α, δ, μ, εᵦ, εₛ)
```

**PIN Formula**:

```
PIN = (α × μ) / (α × μ + εᵦ + εₛ)
```

**Interpretation**: PIN ranges from 0 to 1. Higher PIN indicates a greater fraction of trading is information-driven. Typical values for liquid large-cap stocks are 0.10–0.25.

**Reference**:
- Easley, D. & O'Hara, M. (1992). *Time and the Process of Security Price Adjustment*. Journal of Finance, 47(2), 577-605.
- Easley, D., Kiefer, N.M., O'Hara, M. & Paperman, J.B. (1996). *Liquidity, Information, and Infrequently Traded Stocks*. Journal of Finance, 51(4), 1405-1436.

### 4.3 VPIN (Volume-Synchronized PIN)

**Definition**: A real-time estimator of order flow toxicity that does not require MLE estimation.

**Algorithm**:

1. Partition trades into equal-volume buckets of size V
2. For each bucket τ, classify buy volume (V^B_τ) and sell volume (V^S_τ) using bulk volume classification
3. Compute VPIN over n buckets:

```
VPIN = (1/n) × Σᵢ |V^B_τᵢ - V^S_τᵢ| / V
```

**Interpretation**: VPIN measures order flow imbalance on a volume clock. High VPIN indicates toxic flow (informed trading dominates), which can predict periods of market stress and flash crashes.

**Reference**: Easley, D., López de Prado, M.M. & O'Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World*. Review of Financial Studies, 25(5), 1457-1493.

---

## 5. Price Impact Analysis

### 5.1 Amihud Illiquidity Ratio

**Definition**: The average ratio of absolute daily returns to daily trading volume, measuring price sensitivity to trading activity.

**Formula**:

```
ILLIQ = (1/N) × Σₜ |rₜ| / Vₜ

where:
  rₜ = return in period t
  Vₜ = trading volume (in currency) in period t
  N  = number of periods
```

**Interpretation**: Higher ILLIQ indicates lower liquidity — prices move more per unit of volume. This measure is particularly useful for cross-sectional comparison of liquidity across instruments.

**Reference**: Amihud, Y. (2002). *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects*. Journal of Financial Markets, 5(1), 31-56.

### 5.2 Temporary vs. Permanent Price Impact

**Definition**: Decomposition of total price impact into a temporary (transient) component and a permanent (information) component.

**Model**:

```
Total Impact = P_trade - M_pre

Temporary Impact = P_trade - M_post(Δ)
Permanent Impact = M_post(Δ) - M_pre

where:
  M_pre     = midpoint before the trade
  M_post(Δ) = midpoint Δ seconds after the trade
```

**Interpretation**:
- **Permanent impact** reflects the information content of the trade (price discovery)
- **Temporary impact** reflects the liquidity premium (inventory/order-processing costs)
- The ratio of permanent to total impact indicates the fraction of informed trading

**Reference**: Hasbrouck, J. (1991). *Measuring the Information Content of Stock Trades*. Journal of Finance, 46(1), 179-207.

---

## 6. Queue Position Decay

### Definition

Queue position decay models the time-to-cancellation or time-to-fill of limit orders at each price level, using survival analysis techniques.

### Survival Function

```
S(t) = P(T > t) = 1 - F(t)

where T is the time until the order is cancelled or filled
```

### Exponential Decay Model

**Assumption**: Constant hazard rate (memoryless property).

```
S(t) = exp(-λ × t)

Hazard rate: h(t) = λ (constant)
Mean survival time: E[T] = 1/λ
```

**Estimation**: Maximum likelihood estimation from observed order lifetimes.

### Interpretation

- **High decay rate (λ)**: Orders are short-lived, indicating active management or fleeting liquidity
- **Low decay rate**: Orders persist, suggesting genuine liquidity provision
- **Level-dependent decay**: Orders further from the touch typically have longer lifetimes
- **Asymmetric decay**: Differences between bid and ask decay rates can signal directional intent

### Queue Priority Value

The economic value of queue position can be estimated as:

```
QPV(k, t) = P(fill before cancel | position k) × E[profit | fill at level k]
```

**Reference**: Cont, R., Stoikov, S. & Talreja, R. (2010). *A Stochastic Model for Order Book Dynamics*. Operations Research, 58(3), 549-563.

---

## 7. Hawkes Process Intensity

### Definition

A Hawkes process is a self-exciting point process where past events increase the probability of future events, capturing the empirical clustering of market events.

### Univariate Model

**Conditional intensity**:

```
λ(t) = μ + Σ_{tᵢ < t} g(t - tᵢ)

where:
  μ        = baseline intensity (events/second)
  g(s)     = kernel function (triggering function)
  {tᵢ}     = historical event times
```

**Exponential kernel**:

```
g(s) = α × exp(-ω × s), s ≥ 0

Parameters:
  α = excitation magnitude (jump in intensity per event)
  ω = decay rate (how quickly excitation fades)
```

### Key Quantities

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Branching ratio | n = α/ω | Fraction of events triggered by past events. Must be < 1 for stationarity. |
| Unconditional mean | E[λ] = μ/(1 - n) | Long-run average intensity |
| Half-life | t_½ = ln(2)/ω | Time for excitation to decay by half |
| Endogenous fraction | n/(1 - n + n) = n | Fraction of activity that is self-generated |

### Estimation

**Maximum Likelihood Estimation (MLE)**:

```
log L = -∫₀ᵀ λ(t)dt + Σᵢ log λ(tᵢ)
```

The log-likelihood is maximized numerically using L-BFGS-B with multiple restarts from random initial conditions.

### Interpretation

- **High branching ratio** (n → 1): Market activity is largely self-exciting; events trigger cascades of follow-on events. Common during volatile periods.
- **Low branching ratio** (n → 0): Events are predominantly driven by exogenous information, not self-excitation.
- **Short half-life**: Excitation decays quickly, suggesting transient market impact.
- **Long half-life**: Prolonged impact, possibly indicating persistent information or momentum.

**Reference**: Hawkes, A.G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes*. Biometrika, 58(1), 83-90.

---

## 8. Signal Generation

### Composite Signal Construction

The composite microstructure signal is a weighted combination of individual normalized metric signals:

```
S_composite(t) = Σₖ wₖ × z_k(t)

where:
  wₖ = weight for metric k (Σwₖ = 1)
  z_k(t) = z-score normalized value of metric k at time t
```

**Default weights** (from `config/settings.yaml`):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Spread signal | 0.15 | Liquidity cost proxy |
| OBI signal | 0.25 | Strongest short-term directional predictor |
| Kyle's Lambda | 0.20 | Adverse selection / information asymmetry |
| PIN signal | 0.15 | Informed trading probability |
| Hawkes intensity | 0.15 | Event clustering / volatility regime |
| Queue decay | 0.10 | Liquidity provision dynamics |

### Z-Score Normalization

Each raw metric is normalized using a rolling z-score:

```
z_k(t) = [x_k(t) - μ_k(t, L)] / σ_k(t, L)

where L = lookback period (default: 200 observations)
```

### Entry/Exit Rules

| Signal | Condition | Action |
|--------|-----------|--------|
| Entry Long | S_composite > θ_entry_long | Open long position |
| Entry Short | S_composite < θ_entry_short | Open short position |
| Exit Long | S_composite < θ_exit_long | Close long position |
| Exit Short | S_composite > θ_exit_short | Close short position |

---

## 9. Adaptive Calibration

### Walk-Forward Optimization

The threshold calibration uses a rolling walk-forward framework to prevent overfitting:

```
For each forward step i:
  1. Train window: [t_i, t_i + W_train]
  2. Optimize: θ* = argmax_{θ} Sharpe(θ; train_data)
  3. Test window: [t_i + W_train, t_i + W_train + W_test]
  4. Record: out-of-sample performance with θ*
  5. Roll forward: t_{i+1} = t_i + W_test
```

### Sharpe Ratio Maximization

The objective function for threshold optimization:

```
Sharpe(θ) = [E(R_portfolio) - R_f] / σ(R_portfolio)

where:
  R_portfolio = portfolio returns under thresholds θ
  R_f = risk-free rate
  θ = {θ_entry_long, θ_entry_short, θ_exit_long, θ_exit_short}
```

**Optimization**: Constrained Nelder-Mead or L-BFGS-B with bounds on threshold values and minimum trade frequency constraints.

### Anti-Overfitting Measures

1. **Walk-forward validation**: Out-of-sample testing prevents in-sample overfitting
2. **Minimum trade constraint**: Require ≥ 30 trades per calibration window
3. **Weight bounds**: Each signal weight constrained to [0.05, 0.40]
4. **Regularization**: Penalty for extreme threshold values

---

## References Summary

| # | Citation | Key Contribution |
|---|----------|-----------------|
| 1 | Kyle (1985) | Price impact coefficient (λ) |
| 2 | Easley & O'Hara (1992) | PIN model of informed trading |
| 3 | Easley et al. (1996) | Extended PIN estimation |
| 4 | Easley, López de Prado & O'Hara (2012) | VPIN real-time toxicity metric |
| 5 | Amihud (2002) | Illiquidity ratio |
| 6 | Hawkes (1971) | Self-exciting point processes |
| 7 | Lee & Ready (1991) | Trade classification algorithm |
| 8 | Hasbrouck (1991) | Trade informativeness / VAR decomposition |
| 9 | Huang & Stoll (1996) | Spread decomposition |
| 10 | Cont, Stoikov & Talreja (2010) | Order book dynamics model |
| 11 | Hendershott, Jones & Menkveld (2011) | Algorithmic trading and liquidity |
| 12 | Bessembinder (2003) | Effective spread measurement |
