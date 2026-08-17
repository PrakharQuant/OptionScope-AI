import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ANOMALY_FEATURES = [
    "contracts_growth",
    "turnover_growth",
    "premium_growth",
    "pcr_contracts",
]


def detect_anomalies(df, contamination=0.05):

    result = df.copy()

    features = (
        result[ANOMALY_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=300
    )

    predictions = model.fit_predict(X)

    scores = model.decision_function(X)

    result["anomaly"] = False
    result["anomaly_score"] = np.nan

    result.loc[
        features.index,
        "anomaly"
    ] = predictions == -1

    result.loc[
        features.index,
        "anomaly_score"
    ] = scores

    return result, model