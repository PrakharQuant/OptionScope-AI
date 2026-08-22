import numpy as np
import pandas as pd


# ============================================================
# MAJOR MARKET EVENTS
# ============================================================

EVENTS = {
    "BSE Derivatives Relaunch": {
        "date": pd.Timestamp("2023-05-15"),
        "description": (
            "BSE relaunched Sensex and Bankex derivatives."
        ),
    },

    "Russia-Ukraine War": {
        "date": pd.Timestamp("2022-02-24"),
        "description": (
            "Russia launched its full-scale invasion of Ukraine."
        ),
    },

    "Trump India Tariffs": {
        "date": pd.Timestamp("2025-08-27"),
        "description": (
            "Additional US tariff measures on Indian imports "
            "took effect."
        ),
    },
}


# ============================================================
# GET EVENT
# ============================================================

def get_event(event_name):
    """Return configuration for a named event."""

    if event_name not in EVENTS:
        raise ValueError(
            f"Unknown event: {event_name}"
        )

    return EVENTS[event_name]


# ============================================================
# ADD EVENT WINDOW
# ============================================================

def add_event_window(
    df,
    event_date,
    window_months=12,
):
    """
    Add months-from-event information to the dataset.
    """

    result = df.copy()

    event_date = pd.Timestamp(event_date)

    result["months_from_event"] = (
        (
            result["Date"].dt.year
            - event_date.year
        ) * 12
        +
        (
            result["Date"].dt.month
            - event_date.month
        )
    )

    result["event_window"] = np.select(
        [
            result["months_from_event"] < -window_months,

            result["months_from_event"].between(
                -window_months,
                -1,
            ),

            result["months_from_event"].between(
                0,
                window_months,
            ),

            result["months_from_event"] > window_months,
        ],
        [
            "Earlier History",
            "12M Pre-Event",
            "12M Post-Event",
            "Later History",
        ],
        default="Other",
    )

    result["event_period"] = np.where(
        result["months_from_event"] < 0,
        "Pre-Event",
        "Post-Event",
    )

    return result


# ============================================================
# GENERIC EVENT COMPARISON
# ============================================================

def compare_event(
    df,
    event_name,
    window_months=12,
):
    """
    Compare market characteristics during the 12 months
    before and after a selected event.

    This is descriptive event analysis, not causal inference.
    """

    event = get_event(event_name)

    result = add_event_window(
        df,
        event["date"],
        window_months=window_months,
    )

    pre = result[
        result["months_from_event"].between(
            -window_months,
            -1,
        )
    ]

    post = result[
        result["months_from_event"].between(
            0,
            window_months,
        )
    ]

    metrics = [
        "index_contracts",
        "index_turnover",
        "avg_premium",
        "pcr_contracts",
        "pcr_turnover",
        "put_share",
        "call_share",
        "omis",
    ]

    rows = []

    for metric in metrics:

        pre_mean = pre[metric].mean()
        post_mean = post[metric].mean()

        if (
            pre_mean != 0
            and not pd.isna(pre_mean)
        ):
            percentage_change = (
                (post_mean / pre_mean) - 1
            ) * 100
        else:
            percentage_change = np.nan

        rows.append(
            {
                "metric": metric,
                "pre_12m_mean": pre_mean,
                "post_12m_mean": post_mean,
                "percentage_change": percentage_change,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EVENT TRAJECTORY
# ============================================================

def get_event_trajectory(
    df,
    event_name,
    window_months=12,
):
    """
    Return monthly observations surrounding a selected event.
    """

    event = get_event(event_name)

    result = add_event_window(
        df,
        event["date"],
        window_months=window_months,
    )

    result = result[
        result["months_from_event"].between(
            -window_months,
            window_months,
        )
    ]

    return (
        result[
            [
                "Date",
                "months_from_event",
                "index_contracts",
                "index_turnover",
                "avg_premium",
                "pcr_contracts",
                "omis",
            ]
        ]
        .sort_values("months_from_event")
        .reset_index(drop=True)
    )


# ============================================================
# EVENT SUMMARY
# ============================================================

def get_event_summary(
    df,
    event_name,
):
    """
    Return a compact summary for a selected event.
    """

    event = get_event(event_name)

    comparison = compare_event(
        df,
        event_name,
        window_months=12,
    )

    return {
        "event_name": event_name,
        "event_date": event["date"],
        "description": event["description"],
        "comparison": comparison,
    }


# ============================================================
# BACKWARD-COMPATIBILITY FUNCTION
# ============================================================

def compare_12m_event_window(
    df,
    event_date=pd.Timestamp("2023-05-15"),
):
    """
    Backward-compatible BSE event comparison.

    Kept so older code can still call this function.
    """

    event_date = pd.Timestamp(event_date)

    result = add_event_window(
        df,
        event_date,
        window_months=12,
    )

    pre = result[
        result["months_from_event"].between(
            -12,
            -1,
        )
    ]

    post = result[
        result["months_from_event"].between(
            0,
            12,
        )
    ]

    metrics = [
        "index_contracts",
        "index_turnover",
        "avg_premium",
        "pcr_contracts",
        "pcr_turnover",
        "put_share",
        "call_share",
        "omis",
    ]

    rows = []

    for metric in metrics:

        pre_mean = pre[metric].mean()
        post_mean = post[metric].mean()

        if (
            pre_mean != 0
            and not pd.isna(pre_mean)
        ):
            change = (
                (post_mean / pre_mean) - 1
            ) * 100
        else:
            change = np.nan

        rows.append(
            {
                "metric": metric,
                "pre_12m_mean": pre_mean,
                "post_12m_mean": post_mean,
                "percentage_change": change,
            }
        )

    return pd.DataFrame(rows)