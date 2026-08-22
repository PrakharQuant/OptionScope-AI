import numpy as np
import pandas as pd


# ============================================================
# FORMATTERS
# ============================================================

def format_number(value):
    """Format large numbers for human-readable output."""

    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.0f}"


def format_percent(value):
    """Format percentage values."""

    if pd.isna(value):
        return "N/A"

    return f"{value:+.1f}%"


# ============================================================
# MARKET ACTIVITY INSIGHT
# ============================================================

def activity_insight(df):
    """
    Explain whether current market activity is unusually
    high or low relative to its recent baseline.
    """

    latest = df.iloc[-1]

    contracts_vs_baseline = latest["contracts_vs_12m"]
    turnover_vs_baseline = latest["turnover_vs_12m"]

    if (
        contracts_vs_baseline > 25
        and turnover_vs_baseline > 25
    ):
        return (
            f"Option activity is elevated: contracts are "
            f"{contracts_vs_baseline:+.1f}% above the 12-month "
            f"baseline and turnover is "
            f"{turnover_vs_baseline:+.1f}% above baseline."
        )

    if (
        contracts_vs_baseline < -25
        and turnover_vs_baseline < -25
    ):
        return (
            f"Option activity is subdued: contracts are "
            f"{contracts_vs_baseline:+.1f}% below the 12-month "
            f"baseline and turnover is "
            f"{turnover_vs_baseline:+.1f}% below baseline."
        )

    return (
        f"Market activity is relatively close to its recent "
        f"baseline, with contracts at "
        f"{contracts_vs_baseline:+.1f}% and turnover at "
        f"{turnover_vs_baseline:+.1f}% versus the 12-month average."
    )


# ============================================================
# LIQUIDITY INSIGHT
# ============================================================

def liquidity_insight(df):
    """
    Explain the behaviour of average premium per contract.
    """

    latest = df.iloc[-1]

    premium_vs_baseline = latest["premium_vs_12m"]

    if premium_vs_baseline > 20:
        return (
            f"Average premium per contract is elevated by "
            f"{premium_vs_baseline:+.1f}% versus its 12-month "
            f"baseline, indicating higher value traded per contract."
        )

    if premium_vs_baseline < -20:
        return (
            f"Average premium per contract is "
            f"{premium_vs_baseline:+.1f}% below its 12-month "
            f"baseline."
        )

    return (
        f"Average premium per contract is broadly stable, "
        f"currently {premium_vs_baseline:+.1f}% versus its "
        f"12-month baseline."
    )


# ============================================================
# SENTIMENT INSIGHT
# ============================================================

def sentiment_insight(df):
    """
    Explain put/call positioning using the contract PCR.

    This is a descriptive positioning signal, not a prediction
    of market direction.
    """

    latest = df.iloc[-1]

    pcr = latest["pcr_contracts"]
    put_share = latest["put_share"]

    if pcr > 1.2:
        interpretation = "put participation dominates call participation"

    elif pcr < 0.8:
        interpretation = "call participation dominates put participation"

    else:
        interpretation = "put and call participation is relatively balanced"

    return (
        f"Contract PCR is {pcr:.2f}, with puts representing "
        f"{put_share:.1f}% of contracts; {interpretation}."
    )


# ============================================================
# OMIS INSIGHT
# ============================================================

def omis_insight(df):
    """
    Interpret the project-defined OptionScope Market
    Intelligence Score.
    """

    latest = df.iloc[-1]
    score = latest["omis"]

    if score >= 70:
        level = "high"
    elif score >= 55:
        level = "elevated"
    elif score <= 30:
        level = "low"
    elif score <= 45:
        level = "below-average"
    else:
        level = "moderate"

    return (
        f"OptionScope Market Intelligence Score (OMIS) is "
        f"{score:.1f}, indicating {level} overall market intensity."
    )


# ============================================================
# ANOMALY INSIGHT
# ============================================================

def anomaly_insight(df):
    """
    Explain whether the latest observation is classified
    as unusual by the anomaly detector.
    """

    if "anomaly_flag" not in df.columns:
        return (
            "Anomaly detection has not yet been run."
        )

    latest = df.iloc[-1]

    if latest["anomaly_flag"]:
        return (
            "The latest observation is classified as "
            "statistically unusual relative to the patterns "
            "learned by the anomaly detector."
        )

    return (
        "The latest observation is not classified as an "
        "anomaly by the current detection model."
    )


# ============================================================
# REGIME INSIGHT
# ============================================================

def regime_insight(df):
    """
    Explain the latest detected market regime.
    """

    if "regime_name" not in df.columns:
        return (
            "Market-regime detection has not yet been run."
        )

    latest = df.iloc[-1]

    regime = latest["regime_name"]

    if pd.isna(regime):
        return (
            "A market regime could not be assigned to "
            "the latest observation."
        )

    return (
        f"The latest observation belongs to the "
        f"'{regime}' market regime based on the clustering "
        f"model's feature profile."
    )


# ============================================================
# FORECAST INSIGHT
# ============================================================

def forecast_insight(forecast_df):
    """
    Summarize the direction of the 12-month forecast.
    """

    if forecast_df is None or forecast_df.empty:
        return "No forecast is currently available."

    first_contracts = forecast_df.iloc[0]["contracts_forecast"]
    last_contracts = forecast_df.iloc[-1]["contracts_forecast"]

    first_turnover = forecast_df.iloc[0]["turnover_forecast"]
    last_turnover = forecast_df.iloc[-1]["turnover_forecast"]

    if first_contracts != 0:
        contracts_change = (
            (last_contracts / first_contracts) - 1
        ) * 100
    else:
        contracts_change = np.nan

    if first_turnover != 0:
        turnover_change = (
            (last_turnover / first_turnover) - 1
        ) * 100
    else:
        turnover_change = np.nan

    return (
        f"The 12-month model forecast implies a "
        f"{contracts_change:+.1f}% change in contracts and "
        f"a {turnover_change:+.1f}% change in turnover "
        f"from the beginning to the end of the forecast horizon."
    )


# ============================================================
# MASTER INSIGHT GENERATOR
# ============================================================

def generate_insights(
    df,
    forecast_df=None
):
    """
    Generate the complete set of automated OptionScope
    intelligence statements.
    """

    return {
        "activity": activity_insight(df),
        "liquidity": liquidity_insight(df),
        "sentiment": sentiment_insight(df),
        "omis": omis_insight(df),
        "anomaly": anomaly_insight(df),
        "regime": regime_insight(df),
        "forecast": forecast_insight(forecast_df),
    }