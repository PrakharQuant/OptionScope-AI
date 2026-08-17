# OptionScope AI 📊

### Quantitative Intelligence for the Indian Index Options Market

OptionScope AI is an interactive quantitative analytics platform designed to explore long-run patterns in the Indian index options market.

The platform combines market activity analytics, liquidity indicators, option sentiment measures, unsupervised market-regime detection, anomaly detection and indicative forecasting in a single interactive dashboard.

---

## 🚀 Features

### 1. Market Overview

Track:

- Index option contracts
- Index option turnover
- Put/Call ratio
- OptionScope Market Intelligence Score
- Long-run market activity

### 2. Liquidity Intelligence

Measures include:

- Average premium per contract
- Premium growth
- 12-month rolling premium
- Turnover per contract

Average premium is calculated as:

Average Premium = Index Options Turnover / Index Options Contracts

### 3. Market Sentiment

The platform calculates:

- Put/Call Contract Ratio
- Put/Call Turnover Ratio
- Put participation share
- Call participation share

These indicators provide a descriptive view of changes in option-market positioning.

### 4. Market Regime Detection

K-Means clustering is used to identify periods with similar combinations of:

- Contract growth
- Turnover growth
- Premium growth
- Put/Call ratio

The resulting clusters are interpreted as:

- Expansion
- Balanced
- Defensive

The labels are assigned after examining cluster characteristics rather than being supplied to the model beforehand.

### 5. Anomaly Detection

Isolation Forest identifies observations whose combination of:

- Market growth
- Turnover growth
- Premium dynamics
- Put/Call ratio

differs substantially from the rest of the sample.

### 6. Forecasting

A 12-month exponential-trend model provides indicative projections for:

- Index option contracts
- Index option turnover

Approximate uncertainty intervals are also displayed.

Forecasts are analytical estimates and are not trading recommendations.

### 7. OptionScope Market Intelligence Score

OMIS is a project-defined composite score ranging from 0 to 100.

It combines:

- Market participation
- Turnover momentum
- Premium dynamics
- Sentiment balance

OMIS is an original analytical feature of this project and is not a standard market index.

### 8. Data Integrity

The platform automatically verifies:

Index Contracts ≈ Call Contracts + Put Contracts

and

Index Turnover ≈ Call Turnover + Put Turnover

---

## 📊 Dataset

The project uses monthly Indian index-options market data containing:

- Date
- Index Options Contracts
- Index Options Turnover
- Call Contracts
- Call Turnover
- Put Contracts
- Put Turnover

The dataset contains 301 monthly observations.

---

## 🧠 Quantitative Methodology

### Liquidity

Average premium:

Premium_t = Turnover_t / Contracts_t

### Sentiment

Put/Call ratio:

PCR_t = Put Contracts_t / Call Contracts_t

### Market Regimes

K-Means clustering is performed after standardizing selected market features.

### Anomaly Detection

Isolation Forest is used to detect observations with unusual multivariate characteristics.

### Forecasting

Exponential smoothing with an additive trend is used for indicative 12-month projections.

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Statsmodels
- Streamlit
- Render
- GitHub

---

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/PrakharQuant/OptionScope-AI.git
cd OptionScope-AI