import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_data(df):
    """
    Clean the raw options dataset and create derived
    market activity, liquidity and sentiment indicators.
    """

    df = df.copy()

    # -----------------------------------------------------
    # 1. Parse dates
    # -----------------------------------------------------
    # Raw dataset contains values such as:
    # Jun-01, Jul-01 ... Jun-26
    #
    # Explicitly interpret them as 2001 ... 2026.
    date_parts = (
        df["Date"]
        .astype(str)
        .str.split("-", expand=True)
    )

    df["Date"] = pd.to_datetime(
        date_parts[0] + "-20" + date_parts[1],
        format="%b-%Y"
    )

    # -----------------------------------------------------
    # 2. Standardize column names
    # -----------------------------------------------------

    df = df.rename(
        columns={
            "Index Options Contracts": "index_contracts",
            "Index Options Turnover": "index_turnover",
            "Call Contracts": "call_contracts",
            "Call Turnover": "call_turnover",
            "Put Contract": "put_contracts",
            "Put Turnover": "put_turnover",
        }
    )

    # -----------------------------------------------------
    # 3. Sort chronologically
    # -----------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # 4. Growth indicators
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 5. Liquidity
    # -----------------------------------------------------

    df["avg_premium"] = (
        df["index_turnover"]
        / df["index_contracts"].replace(0, np.nan)
    )

    df["premium_growth"] = (
        df["avg_premium"]
        .pct_change()
        * 100
    )

    df["premium_12m"] = (
        df["avg_premium"]
        .rolling(12)
        .mean()
    )

    # -----------------------------------------------------
    # 6. Sentiment
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 7. Rolling participation indicators
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 8. Reconciliation checks
    # -----------------------------------------------------

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


# =========================================================
# DATA VALIDATION
# =========================================================

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


# =========================================================
# MARKET REGIME DETECTION
# =========================================================

def detect_regimes(df):
    """
    Detect market regimes using K-Means clustering.

    Features:
        - Contracts growth
        - Turnover growth
        - Premium growth
        - Put/Call ratio

    Returns:
        result       : dataframe with regime labels
        profiles     : cluster-level feature profiles
        model        : fitted KMeans model
    """

    result = df.copy()

    features = [
        "contracts_growth",
        "turnover_growth",
        "premium_growth",
        "pcr_contracts",
    ]

    # -----------------------------------------------------
    # Prepare ML sample
    # -----------------------------------------------------

    ml_data = result[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = ml_data.notna().all(axis=1)

    valid_data = ml_data.loc[valid_mask]

    # Not enough observations for clustering
    if len(valid_data) < 12:

        result["regime"] = "Insufficient Data"

        profiles = pd.DataFrame()

        return result, profiles, None

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(valid_data)

    # -----------------------------------------------------
    # K-Means
    # -----------------------------------------------------

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=20
    )

    clusters = model.fit_predict(X)

    result.loc[
        valid_mask,
        "cluster"
    ] = clusters

    # -----------------------------------------------------
    # Build cluster profiles
    # -----------------------------------------------------

    profiles = (
        result.loc[valid_mask]
        .groupby("cluster")[
            features
        ]
        .mean()
    )

    # -----------------------------------------------------
    # Give clusters interpretable names
    # -----------------------------------------------------
    #
    # We do NOT assume cluster 0 = Expansion etc.
    # Instead, examine the profile of each cluster.
    #
    # Expansion:
    # relatively high activity growth.
    #
    # Defensive:
    # relatively weak activity combined with
    # higher Put/Call ratio.
    #
    # Balanced:
    # middle profile.

    activity_score = (
        profiles["contracts_growth"].rank()
        + profiles["turnover_growth"].rank()
        + profiles["premium_growth"].rank()
    )

    defensive_score = (
        profiles["pcr_contracts"].rank()
    )

    expansion_cluster = (
        activity_score.idxmax()
    )

    remaining = [
        c for c in profiles.index
        if c != expansion_cluster
    ]

    if len(remaining) == 2:

        defensive_cluster = (
            profiles.loc[
                remaining,
                "pcr_contracts"
            ].idxmax()
        )

        balanced_cluster = [
            c for c in remaining
            if c != defensive_cluster
        ][0]

    elif len(remaining) == 1:

        defensive_cluster = remaining[0]
        balanced_cluster = remaining[0]

    else:

        defensive_cluster = expansion_cluster
        balanced_cluster = expansion_cluster

    cluster_names = {
        expansion_cluster: "Expansion",
        balanced_cluster: "Balanced",
        defensive_cluster: "Defensive",
    }

    result["regime"] = (
        result["cluster"]
        .map(cluster_names)
        .fillna("Insufficient Data")
    )

    # -----------------------------------------------------
    # Clean helper column
    # -----------------------------------------------------

    result = result.drop(
        columns=["cluster"],
        errors="ignore"
    )

    profiles = profiles.copy()

    profiles["Regime"] = [
        cluster_names.get(
            cluster,
            "Unknown"
        )
        for cluster in profiles.index
    ]

    profiles = profiles.set_index("Regime")

    return result, profiles, model


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomalies(df):
    """
    Detect unusual market observations using Isolation Forest.

    Features:
        - Contracts growth
        - Turnover growth
        - Premium growth
        - Put/Call ratio
        - Average premium
    """

    result = df.copy()

    features = [
        "contracts_growth",
        "turnover_growth",
        "premium_growth",
        "pcr_contracts",
        "avg_premium",
    ]

    ml_data = (
        result[features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    valid_mask = ml_data.notna().all(axis=1)

    valid_data = ml_data.loc[valid_mask]

    # Default values
    result["anomaly"] = False
    result["anomaly_score"] = np.nan

    if len(valid_data) < 20:

        return result, None

    # -----------------------------------------------------
    # Standardize
    # -----------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(valid_data)

    # -----------------------------------------------------
    # Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=200
    )

    predictions = model.fit_predict(X)

    scores = model.decision_function(X)

    # Isolation Forest:
    #
    # +1 = normal
    # -1 = anomaly

    result.loc[
        valid_mask,
        "anomaly"
    ] = predictions == -1

    result.loc[
        valid_mask,
        "anomaly_score"
    ] = scores

    return result, model


# =========================================================
# OMIS
# =========================================================

def calculate_omis(df):
    """
    Calculate the OptionScope Market Intelligence Score (OMIS).

    OMIS is a project-defined composite indicator from 0 to 100.

    Components:
        1. Participation
        2. Liquidity
        3. Market activity

    Higher OMIS means stronger overall market activity
    relative to the historical sample.

    This is NOT a standard market index.
    """

    data = df.copy()

    # -----------------------------------------------------
    # Helper: percentile score
    # -----------------------------------------------------

    def percentile_score(series):

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return (
            series
            .rank(pct=True)
            * 100
        )

    # -----------------------------------------------------
    # 1. Participation score
    # -----------------------------------------------------

    participation = percentile_score(
        np.log1p(
            data["index_contracts"]
        )
    )

    # -----------------------------------------------------
    # 2. Turnover / liquidity score
    # -----------------------------------------------------

    turnover = percentile_score(
        np.log1p(
            data["index_turnover"]
        )
    )

    # -----------------------------------------------------
    # 3. Premium score
    # -----------------------------------------------------

    premium = percentile_score(
        data["avg_premium"]
    )

    # -----------------------------------------------------
    # 4. Growth score
    # -----------------------------------------------------

    growth = percentile_score(
        data["turnover_growth"]
    )

    # -----------------------------------------------------
    # Composite score
    # -----------------------------------------------------
    #
    # Weights:
    # Participation : 30%
    # Turnover      : 30%
    # Premium       : 20%
    # Growth        : 20%

    omis = (
        0.30 * participation
        + 0.30 * turnover
        + 0.20 * premium
        + 0.20 * growth
    )

    return omis.clip(
        lower=0,
        upper=100
    )