import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "contracts_growth",
    "turnover_growth",
    "premium_growth",
    "pcr_contracts",
]


def detect_regimes(df, n_clusters=3):
    """
    Detect market regimes using K-Means clustering.

    Cluster labels are interpreted based on their
    feature profiles rather than hard-coded dates.
    """

    result = df.copy()

    features = result[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    raw_labels = model.fit_predict(X)

    features = features.copy()
    features["cluster"] = raw_labels

    # ---------------------------------------------------------
    # Interpret clusters
    # ---------------------------------------------------------

    cluster_profiles = (
        features
        .groupby("cluster")[FEATURES]
        .mean()
    )

    # Higher PCR + premium growth and weaker participation
    # indicates a more defensive/stress-like profile.
    cluster_profiles["stress_score"] = (
        cluster_profiles["pcr_contracts"]
        + cluster_profiles["premium_growth"]
        - cluster_profiles["contracts_growth"]
        - cluster_profiles["turnover_growth"]
    )

    ordered_clusters = (
        cluster_profiles["stress_score"]
        .sort_values()
        .index
        .tolist()
    )

    names = [
        "Expansion",
        "Balanced",
        "Defensive"
    ]

    mapping = {
        cluster: names[min(i, len(names) - 1)]
        for i, cluster in enumerate(ordered_clusters)
    }

    result["regime"] = "Insufficient Data"

    result.loc[
        features.index,
        "regime"
    ] = [
        mapping[label]
        for label in raw_labels
    ]

    # Add numerical cluster ID
    result["regime_cluster"] = np.nan

    result.loc[
        features.index,
        "regime_cluster"
    ] = raw_labels

    return result, cluster_profiles, model