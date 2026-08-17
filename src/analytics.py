import numpy as np
import pandas as pd


def prepare_data(df):
    """Clean the raw options dataset and create derived market indicators."""

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Parse monthly dates
    # ---------------------------------------------------------
    # Raw data uses two-digit years such as Jun-01 and Jun-26.
    # Explicitly prepend "20" so the years are interpreted as
    # 2001 ... 2026 rather than relying on Python's YY convention.
    date_parts = df["Date"].astype(str).str.split("-", expand=True)

    df["Date"] = pd.to_datetime(
        date_parts[0] + "-20" + date_parts[1],
        format="%b-%Y"
    )

    # ---------------------------------------------------------
    # 2. Standardize column names
    # ---------------------------------------------------------
    df = df.rename(columns={
        "Index Options Contracts": "index_contracts",
        "Index Options Turnover": "index_turnover",
        "Call Contracts": "call_contracts",
        "Call Turnover": "call_turnover",
        "Put Contract": "put_contracts",
        "Put Turnover": "put_turnover",
    })

    # ---------------------------------------------------------
    # 3. Sort chronologically
    # ---------------------------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Market activity
    # ---------------------------------------------------------
    df["contracts_growth"] = (
        df["index_contracts"].pct_change() * 100
    )

    df["turnover_growth"] = (
        df["index_turnover"].pct_change() * 100
    )

    # ---------------------------------------------------------
    # 5. Liquidity / average premium
    # ---------------------------------------------------------
    # Approximate premium/value per contract:
    # Turnover / Contracts
    df["avg_premium"] = (
        df["index_turnover"]
        / df["index_contracts"].replace(0, np.nan)
    )

    df["premium_growth"] = (
        df["avg_premium"].pct_change() * 100
    )

    # 12-month rolling average premium
    df["premium_12m"] = (
        df["avg_premium"]
        .rolling(12)
        .mean()
    )

    # ---------------------------------------------------------
    # 6. Sentiment indicators
    # ---------------------------------------------------------

    # Put/Call Contract Ratio
    df["pcr_contracts"] = (
        df["put_contracts"]
        / df["call_contracts"].replace(0, np.nan)
    )

    # Put/Call Turnover Ratio
    df["pcr_turnover"] = (
        df["put_turnover"]
        / df["call_turnover"].replace(0, np.nan)
    )

    # Put and Call participation shares
    df["put_share"] = (
        df["put_contracts"]
        / df["index_contracts"].replace(0, np.nan)
    ) * 100

    df["call_share"] = (
        df["call_contracts"]
        / df["index_contracts"].replace(0, np.nan)
    ) * 100

    # ---------------------------------------------------------
    # 7. Participation / activity trends
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
    # 8. Reconciliation checks
    # ---------------------------------------------------------
    # These allow us to verify that the index totals are
    # consistent with the Call + Put components.

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
    """
    Check whether index totals reconcile with
    Call + Put components.
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

    # Contracts should reconcile essentially exactly.
    contract_ok = np.allclose(
        contract_difference,
        0,
        atol=0.01
    )

    # Turnover may contain small rounding differences.
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
    }