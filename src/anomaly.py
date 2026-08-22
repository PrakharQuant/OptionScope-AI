import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURES USED FOR ANOMALY DETECTION
# ============================================================

ANOMALY_FEATURES = [
    "contracts_growth",
    "turnover_growth",
    "premium_growth",
    "pcr_contracts",
    "contracts_zscore",
    "turnover_zscore",
    "premium_zscore",
]


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_anomaly_features(df):
    """
    Prepare the quantitative features used by the
    Isolation Forest anomaly detector.
    """

    feature_data = df[
        ["Date"] + ANOMALY_FEATURES
    ].copy()

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_data = feature_data.dropna().reset_index(drop=True)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        feature_data[ANOMALY_FEATURES]
    )

    return feature_data, X_scaled, scaler


# ============================================================
# RUN ANOMALY DETECTION
# ============================================================

def detect_anomalies(
    df,
    contamination=0.05,
    random_state=42
):
    """
    Detect unusual market observations using Isolation Forest.

    Parameters
    ----------
    df : DataFrame
        Output of analytics.prepare_data()

    contamination : float
        Expected proportion of anomalous observations.

    random_state : int
        Ensures reproducible results.

    Returns
    -------
    result : DataFrame
        Dates with anomaly labels and scores.

    model : IsolationForest
        Fitted anomaly detection model.

    scaler : StandardScaler
        Feature scaler.
    """

    feature_data, X_scaled, scaler = (
        prepare_anomaly_features(df)
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(X_scaled)

    # Isolation Forest:
    # +1 = normal
    # -1 = anomaly

    feature_data["anomaly_flag"] = (
        predictions == -1
    )

    # Higher values are more normal.
    # We invert the score so that higher values
    # represent greater unusualness.
    feature_data["anomaly_score"] = (
        -model.decision_function(X_scaled)
    )

    return feature_data, model, scaler


# ============================================================
# ADD ANOMALIES TO COMPLETE DATASET
# ============================================================

def add_anomalies_to_data(
    df,
    contamination=0.05,
    random_state=42
):
    """
    Detect anomalies and attach the results to the
    complete monthly dataset.
    """

    anomaly_data, model, scaler = detect_anomalies(
        df,
        contamination=contamination,
        random_state=random_state
    )

    result = df.merge(
        anomaly_data[
            [
                "Date",
                "anomaly_flag",
                "anomaly_score",
            ]
        ],
        on="Date",
        how="left"
    )

    result["anomaly_flag"] = (
        result["anomaly_flag"]
        .fillna(False)
        .astype(bool)
    )

    result["anomaly_score"] = (
        result["anomaly_score"]
        .fillna(0)
    )

    return result, model, scaler


# ============================================================
# RANK MOST UNUSUAL OBSERVATIONS
# ============================================================

def get_top_anomalies(
    df,
    n=10
):
    """
    Return the most unusual observations ranked by
    anomaly score.
    """

    if "anomaly_score" not in df.columns:
        raise ValueError(
            "Run add_anomalies_to_data() first."
        )

    return (
        df[
            df["anomaly_flag"]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(n)
        [
            [
                "Date",
                "index_contracts",
                "index_turnover",
                "avg_premium",
                "pcr_contracts",
                "anomaly_score",
            ]
        ]
    )


# ============================================================
# ANOMALY SUMMARY
# ============================================================

def get_anomaly_summary(df):
    """
    Return a compact summary of anomaly detection results.
    """

    if "anomaly_flag" not in df.columns:
        raise ValueError(
            "Run add_anomalies_to_data() first."
        )

    anomalies = df[
        df["anomaly_flag"]
    ].copy()

    return {
        "total_observations": len(df),
        "anomaly_count": len(anomalies),
        "anomaly_percentage": (
            len(anomalies) / len(df) * 100
            if len(df) > 0
            else 0
        ),
        "most_unusual_date": (
            anomalies
            .sort_values(
                "anomaly_score",
                ascending=False
            )
            .iloc[0]["Date"]
            if not anomalies.empty
            else None
        ),
    }