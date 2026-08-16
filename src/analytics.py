import numpy as np
import pandas as pd


def prepare_data(df):
    """Clean and create derived market indicators."""

    df = df.copy()

    # Parse monthly dates
    df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")

    # Standardize column names
    df = df.rename(columns={
        "Index Options Contracts": "index_contracts",
        "Index Options Turnover": "index_turnover",
        "Call Contracts": "call_contracts",
        "Call Turnover": "call_turnover",
        "Put Contract": "put_contracts",
        "Put Turnover": "put_turnover",
    })

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------
    # Market activity
    # --------------------------------------------------

    df["contracts_growth"] = df["index_contracts"].pct_change() * 100
    df["turnover_growth"] = df["index_turnover"].pct_change() * 100

    # --------------------------------------------------
    # Liquidity / average premium
    # --------------------------------------------------

    df["avg_premium"] = (
        df["index_turnover"] / df["index_contracts"]
    )

    df["premium_growth"] = df["avg_premium"].pct_change() * 100

    # 12-month rolling premium
    df["premium_12m"] = (
        df["avg_premium"].rolling(12).mean()
    )

    # --------------------------------------------------
    # Sentiment
    # --------------------------------------------------

    # Put/Call Contract Ratio
    df["pcr_contracts"] = (
        df["put_contracts"] /
        df["call_contracts"].replace(0, np.nan)
    )

    # Put/Call Turnover Ratio
    df["pcr_turnover"] = (
        df["put_turnover"] /
        df["call_turnover"].replace(0, np.nan)
    )

    # Share of total contracts
    df["put_share"] = (
        df["put_contracts"] /
        df["index_contracts"]
    ) * 100

    df["call_share"] = (
        df["call_contracts"] /
        df["index_contracts"]
    ) * 100

    # --------------------------------------------------
    # Participation
    # --------------------------------------------------

    df["contracts_12m"] = (
        df["index_contracts"].rolling(12).mean()
    )

    df["turnover_12m"] = (
        df["index_turnover"].rolling(12).mean()
    )

    return df


def validate_data(df):
    """Check whether index totals reconcile with call + put."""

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
    }