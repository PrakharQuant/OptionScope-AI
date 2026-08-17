import numpy as np
import pandas as pd


def prepare_data(df):
    """Clean the raw dataset and create quantitative indicators."""

    df = df.copy()

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%b-%y"
    )

    # ---------------------------------------------------------
    # Standardized column names
    # ---------------------------------------------------------
    df = df.rename(columns={
        "Index Options Contracts": "index_contracts",
        "Index Options Turnover": "index_turnover",
        "Call Contracts": "call_contracts",
        "Call Turnover": "call_turnover",
        "Put Contract": "put_contracts",
        "Put Turnover": "put_turnover",
    })

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Growth
    # ---------------------------------------------------------
    df["contracts_growth"] = (
        df["index_contracts"].pct_change() * 100
    )

    df["turnover_growth"] = (
        df["index_turnover"].pct_change() * 100
    )

    # ---------------------------------------------------------
    # Liquidity
    # ---------------------------------------------------------
    df["avg_premium"] = (
        df["index_turnover"]
        / df["index_contracts"].replace(0, np.nan)
    )

    df["premium_growth"] = (
        df["avg_premium"].pct_change() * 100
    )

    df["premium_12m"] = (
        df["avg_premium"]
        .rolling(12)
        .mean()
    )

    # ---------------------------------------------------------
    # Sentiment
    # ---------------------------------------------------------
    df["pcr_contracts"] = (
        df["put_contracts"]
        / df["call_contracts"].replace(0, np.nan)
    )

    df["pcr_turnover"] = (
        df["put_turnover"]
        / df["call_turnover"].replace(0, np.nan)
    )

    df["put_share"] = (
        df["put_contracts"]
        / df["index_contracts"].replace(0, np.nan)
        * 100
    )

    df["call_share"] = (
        df["call_contracts"]
        / df["index_contracts"].replace(0, np.nan)
        * 100
    )

    # ---------------------------------------------------------
    # Rolling market activity
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Data integrity
    # ---------------------------------------------------------
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


def validate_data(df):
    """Validate index totals against call + put components."""

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

    # Turnover may contain rounding differences.
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
        )
    }


def calculate_omis(df):
    """
    OptionScope Market Intelligence Score.

    This is a project-defined composite indicator,
    not a standard financial-market index.
    """

    data = df.copy()

    # Participation
    participation = (
        data["contracts_growth"]
        .rolling(6)
        .mean()
    )

    # Turnover momentum
    turnover = (
        data["turnover_growth"]
        .rolling(6)
        .mean()
    )

    # Premium relative to its 12-month average
    premium_signal = (
        data["avg_premium"]
        / data["premium_12m"]
        - 1
    ) * 100

    # Sentiment balance:
    # PCR near 1 = more balanced
    sentiment_balance = (
        1
        - (data["pcr_contracts"] - 1).abs()
    )

    def percentile_score(series):
        return series.rank(pct=True) * 100

    score = (
        0.30 * percentile_score(participation)
        + 0.30 * percentile_score(turnover)
        + 0.20 * percentile_score(premium_signal)
        + 0.20 * percentile_score(sentiment_balance)
    )

    return score.clip(0, 100)