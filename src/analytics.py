import numpy as np
import pandas as pd


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_MAP = {
    "Index Options Contracts": "index_contracts",
    "Index Options Turnover": "index_turnover",
    "Call Contracts": "call_contracts",
    "Call Turnover": "call_turnover",
    "Put Contract": "put_contracts",
    "Put Turnover": "put_turnover",
}

# ============================================================
# DATE PARSING
# ============================================================

def parse_dates(date_series):
    """
    Convert source dates such as Jun-01 and Jun-26 into
    explicit four-digit years: Jun-2001 and Jun-2026.

    The dataset uses two-digit years and represents the
    period 2001-2026.
    """

    date_parts = (
        date_series
        .astype(str)
        .str.strip()
        .str.split("-", expand=True)
    )

    return pd.to_datetime(
        date_parts[0] + "-20" + date_parts[1],
        format="%b-%Y",
        errors="coerce"
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Clean the raw options dataset and generate the complete
    feature set used by OptionScope AI.

    Returns
    -------
    pandas.DataFrame
        Monthly observations with engineered market features.
    """

    df = df.copy()

    # --------------------------------------------------------
    # 1. Parse dates
    # --------------------------------------------------------

    df["Date"] = parse_dates(df["Date"])

    # --------------------------------------------------------
    # 2. Standardize column names
    # --------------------------------------------------------

    df = df.rename(columns=COLUMN_MAP)

    # --------------------------------------------------------
    # 3. Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 4. Basic data hygiene
    # --------------------------------------------------------

    numeric_columns = [
        "index_contracts",
        "index_turnover",
        "call_contracts",
        "call_turnover",
        "put_contracts",
        "put_turnover",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 5. Market activity
    # --------------------------------------------------------

    df["contracts_growth"] = (
        df["index_contracts"]
        .pct_change()
        * 100
    )

    df["turnover_growth"] = (
        df["index_turnover"]
        .pct_change()
        * 100
    )

    # --------------------------------------------------------
    # 6. Year-over-year growth
    # --------------------------------------------------------

    df["contracts_yoy"] = (
        df["index_contracts"]
        .pct_change(12)
        * 100
    )

    df["turnover_yoy"] = (
        df["index_turnover"]
        .pct_change(12)
        * 100
    )

    # --------------------------------------------------------
    # 7. Average premium / value per contract
    # --------------------------------------------------------

    df["avg_premium"] = (
        df["index_turnover"]
        / df["index_contracts"].replace(0, np.nan)
    )

    df["premium_growth"] = (
        df["avg_premium"]
        .pct_change()
        * 100
    )

    df["premium_yoy"] = (
        df["avg_premium"]
        .pct_change(12)
        * 100
    )

    # --------------------------------------------------------
    # 8. Rolling market baselines
    # --------------------------------------------------------

    df["contracts_12m"] = (
        df["index_contracts"]
        .rolling(12)
        .mean()
    )

    df["turnover_12m"] = (
        df["index_turnover"]
        .rolling(12)
        .mean()
    )

    df["premium_12m"] = (
        df["avg_premium"]
        .rolling(12)
        .mean()
    )

    # --------------------------------------------------------
    # 9. Distance from historical baseline
    # --------------------------------------------------------

    df["contracts_vs_12m"] = (
        (
            df["index_contracts"]
            / df["contracts_12m"]
        ) - 1
    ) * 100

    df["turnover_vs_12m"] = (
        (
            df["index_turnover"]
            / df["turnover_12m"]
        ) - 1
    ) * 100

    df["premium_vs_12m"] = (
        (
            df["avg_premium"]
            / df["premium_12m"]
        ) - 1
    ) * 100

    # --------------------------------------------------------
    # 10. Put/Call ratios
    # --------------------------------------------------------

    df["pcr_contracts"] = (
        df["put_contracts"]
        / df["call_contracts"].replace(0, np.nan)
    )

    df["pcr_turnover"] = (
        df["put_turnover"]
        / df["call_turnover"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 11. Put / Call market shares
    # --------------------------------------------------------

    df["put_share"] = (
        df["put_contracts"]
        / df["index_contracts"].replace(0, np.nan)
    ) * 100

    df["call_share"] = (
        df["call_contracts"]
        / df["index_contracts"].replace(0, np.nan)
    ) * 100

    # --------------------------------------------------------
    # 12. Put/Call changes
    # --------------------------------------------------------

    df["pcr_change"] = (
        df["pcr_contracts"].pct_change()
        * 100
    )

    df["pcr_yoy"] = (
        df["pcr_contracts"].pct_change(12)
        * 100
    )

    # --------------------------------------------------------
    # 13. Participation growth
    # --------------------------------------------------------

    df["call_growth"] = (
        df["call_contracts"]
        .pct_change()
        * 100
    )

    df["put_growth"] = (
        df["put_contracts"]
        .pct_change()
        * 100
    )

    # --------------------------------------------------------
    # 14. Market composition
    # --------------------------------------------------------

    df["call_turnover_share"] = (
        df["call_turnover"]
        / df["index_turnover"].replace(0, np.nan)
    ) * 100

    df["put_turnover_share"] = (
        df["put_turnover"]
        / df["index_turnover"].replace(0, np.nan)
    ) * 100

    # --------------------------------------------------------
    # 15. Rolling volatility of activity
    # --------------------------------------------------------

    df["contracts_growth_vol"] = (
        df["contracts_growth"]
        .rolling(12)
        .std()
    )

    df["turnover_growth_vol"] = (
        df["turnover_growth"]
        .rolling(12)
        .std()
    )

    df["premium_growth_vol"] = (
        df["premium_growth"]
        .rolling(12)
        .std()
    )

    # --------------------------------------------------------
    # 16. Standardized activity indicators
    # --------------------------------------------------------

    df["contracts_zscore"] = rolling_zscore(
        df["index_contracts"],
        window=24
    )

    df["turnover_zscore"] = rolling_zscore(
        df["index_turnover"],
        window=24
    )

    df["premium_zscore"] = rolling_zscore(
        df["avg_premium"],
        window=24
    )

    df["pcr_zscore"] = rolling_zscore(
        df["pcr_contracts"],
        window=24
    )

    # --------------------------------------------------------
    # 17. Composite activity measures
    # --------------------------------------------------------

    # Activity intensity:
    # combines contract and turnover deviations from
    # their rolling historical baselines.

    df["activity_intensity"] = (
        0.5 * safe_zscore(df["contracts_vs_12m"])
        + 0.5 * safe_zscore(df["turnover_vs_12m"])
    )

    # Liquidity intensity:
    # captures unusual changes in value per contract.

    df["liquidity_intensity"] = (
        safe_zscore(df["premium_vs_12m"])
    )

    # Sentiment intensity:
    # captures unusual PCR behaviour.

    df["sentiment_intensity"] = (
        safe_zscore(df["pcr_contracts"])
    )

    # --------------------------------------------------------
    # 18. OptionScope Market Intelligence Score
    # --------------------------------------------------------

    df["omis"] = calculate_omis(df)

    # --------------------------------------------------------
    # 19. Event indicator
    # --------------------------------------------------------

    # BSE derivatives relaunch date:
    # 15 May 2023.

    df["post_bse_relaunch"] = (
        df["Date"]
        >= pd.Timestamp("2023-05-15")
    )

    df["event_period"] = np.where(
        df["post_bse_relaunch"],
        "Post-relaunch",
        "Pre-relaunch"
    )

    # --------------------------------------------------------
    # 20. Data reconciliation
    # --------------------------------------------------------

    df["contract_reconciliation"] = (
        df["index_contracts"]
        - df["call_contracts"]
        - df["put_contracts"]
    )

    df["turnover_reconciliation"] = (
        df["index_turnover"]
        - df["call_turnover"]
        - df["put_turnover"]
    )

    return df


# ============================================================
# ROLLING Z-SCORE
# ============================================================

def rolling_zscore(series, window=24):
    """
    Calculate a rolling z-score.

    This measures how unusual the current observation is
    relative to its recent historical distribution.
    """

    rolling_mean = (
        series
        .rolling(window)
        .mean()
    )

    rolling_std = (
        series
        .rolling(window)
        .std()
    )

    return (
        (series - rolling_mean)
        / rolling_std.replace(0, np.nan)
    )


# ============================================================
# STANDARD Z-SCORE
# ============================================================

def safe_zscore(series):
    """
    Standardize a series while safely handling zero variance.
    """

    mean = series.mean()
    std = series.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (series - mean) / std


# ============================================================
# OPTION SCOPE MARKET INTELLIGENCE SCORE
# ============================================================

def calculate_omis(df):
    """
    Calculate the OptionScope Market Intelligence Score (OMIS).

    OMIS is a project-defined composite indicator from 0-100
    designed to summarize unusual market activity, liquidity
    behaviour and sentiment conditions.

    It is NOT an established market index.
    """

    activity = safe_zscore(
        df["activity_intensity"]
    )

    liquidity = safe_zscore(
        df["liquidity_intensity"]
    )

    sentiment = safe_zscore(
        df["sentiment_intensity"]
    )

    # Weighted composite.
    raw_score = (
        0.50 * activity
        + 0.30 * liquidity
        + 0.20 * sentiment
    )

    # Convert standardized score into approximately 0-100.
    score = (
        50
        + 10 * raw_score
    )

    return score.clip(0, 100)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df):
    """
    Validate the relationship between aggregate Index
    Options data and its Call + Put components.
    """

    contract_difference = (
        df["index_contracts"]
        - df["call_contracts"]
        - df["put_contracts"]
    )

    turnover_difference = (
        df["index_turnover"]
        - df["call_turnover"]
        - df["put_turnover"]
    )

    contract_ok = np.allclose(
        contract_difference,
        0,
        atol=0.01
    )

    turnover_ok = np.allclose(
        turnover_difference,
        0,
        atol=1.01
    )

    return {
        "contract_ok": contract_ok,
        "turnover_ok": turnover_ok,
        "max_contract_difference": float(
            contract_difference.abs().max()
        ),
        "max_turnover_difference": float(
            turnover_difference.abs().max()
        ),
        "rows_checked": len(df),
    }


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def get_market_summary(df):
    """
    Return a compact summary of the market dataset.
    """

    if df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        "observations": len(df),

        "start_date": df["Date"].min(),

        "end_date": df["Date"].max(),

        "total_contracts": df["index_contracts"].sum(),

        "total_turnover": df["index_turnover"].sum(),

        "latest_contracts": latest["index_contracts"],

        "latest_turnover": latest["index_turnover"],

        "latest_pcr": latest["pcr_contracts"],

        "latest_premium": latest["avg_premium"],

        "latest_omis": latest["omis"],
    }


# ============================================================
# EVENT ANALYSIS
# ============================================================

def compare_periods(
    df,
    event_date="2023-05-15"
):
    """
    Compare market characteristics before and after
    the BSE derivatives relaunch period.

    This is descriptive/event analysis and does not claim
    causal identification.
    """

    event_date = pd.Timestamp(event_date)

    before = df[
        df["Date"] < event_date
    ].copy()

    after = df[
        df["Date"] >= event_date
    ].copy()

    metrics = [
        "index_contracts",
        "index_turnover",
        "avg_premium",
        "pcr_contracts",
        "pcr_turnover",
        "put_share",
        "omis",
    ]

    results = []

    for metric in metrics:

        before_value = before[metric].mean()
        after_value = after[metric].mean()

        if before_value != 0:
            percentage_change = (
                (after_value / before_value) - 1
            ) * 100
        else:
            percentage_change = np.nan

        results.append({
            "metric": metric,
            "pre_event_mean": before_value,
            "post_event_mean": after_value,
            "percentage_change": percentage_change,
        })

    return pd.DataFrame(results)


# ============================================================
# LATEST MARKET STATE
# ============================================================

def get_latest_state(df):
    """
    Produce a compact representation of the latest
    observed market conditions.
    """

    if df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        "date": latest["Date"],
        "contracts": latest["index_contracts"],
        "turnover": latest["index_turnover"],
        "avg_premium": latest["avg_premium"],
        "pcr": latest["pcr_contracts"],
        "put_share": latest["put_share"],
        "call_share": latest["call_share"],
        "omis": latest["omis"],
        "contracts_zscore": latest["contracts_zscore"],
        "turnover_zscore": latest["turnover_zscore"],
        "premium_zscore": latest["premium_zscore"],
        "pcr_zscore": latest["pcr_zscore"],
    }