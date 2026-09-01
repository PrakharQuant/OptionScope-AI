import pandas as pd


# ============================================================
# MAJOR EVENTS
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
            "Additional US tariffs on Indian imports took effect."
        ),
    },
}


# ============================================================
# EVENT HELPERS
# ============================================================

def get_event(event_name):
    """
    Return event metadata for a named event.
    """

    if event_name not in EVENTS:
        raise ValueError(
            f"Unknown event: {event_name}"
        )

    return EVENTS[event_name]


def get_event_date(event_name):
    """
    Return the actual calendar date of the event.
    """

    return get_event(event_name)["date"]


def get_event_marker_date(event_name):
    """
    Return the month-start date used for plotting.

    The underlying dataset is monthly, so an event occurring
    during a month is visually aligned with that month's
    observation.
    """

    event_date = get_event_date(event_name)

    return pd.Timestamp(
        year=event_date.year,
        month=event_date.month,
        day=1,
    )


# ============================================================
# EVENT WINDOW
# ============================================================

def get_event_trajectory(
    df,
    event_name,
    window_months=12,
):
    """
    Return observations around an event.

    The event window is expressed in months before and after
    the event.

    Because the dataset is monthly, the event itself is
    aligned to the first day of its month.
    """

    event_date = get_event_date(event_name)

    # Align event to monthly observation
    event_month = pd.Timestamp(
        year=event_date.year,
        month=event_date.month,
        day=1,
    )

    start_date = (
        event_month
        - pd.DateOffset(months=window_months)
    )

    end_date = (
        event_month
        + pd.DateOffset(months=window_months)
    )

    trajectory = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

    return trajectory.sort_values(
        "Date"
    ).reset_index(drop=True)


# ============================================================
# PRE / POST EVENT COMPARISON
# ============================================================

def compare_event(
    df,
    event_name,
    window_months=12,
):
    """
    Compare average market characteristics during the
    pre-event and post-event windows.
    """

    event_date = get_event_date(event_name)

    event_month = pd.Timestamp(
        year=event_date.year,
        month=event_date.month,
        day=1,
    )

    pre_start = (
        event_month
        - pd.DateOffset(months=window_months)
    )

    post_end = (
        event_month
        + pd.DateOffset(months=window_months)
    )

    pre = df[
        (df["Date"] >= pre_start)
        & (df["Date"] < event_month)
    ].copy()

    post = df[
        (df["Date"] > event_month)
        & (df["Date"] <= post_end)
    ].copy()

    metrics = [
        "index_contracts",
        "index_turnover",
        "avg_premium",
        "pcr_contracts",
        "pcr_turnover",
    ]

    rows = []

    for metric in metrics:

        if metric not in df.columns:
            continue

        pre_value = pre[metric].mean()
        post_value = post[metric].mean()

        if pd.isna(pre_value):
            change_pct = None
        elif pre_value == 0:
            change_pct = None
        else:
            change_pct = (
                (post_value - pre_value)
                / abs(pre_value)
                * 100
            )

        rows.append(
            {
                "Metric": metric,
                "Pre-Event Average": pre_value,
                "Post-Event Average": post_value,
                "Change %": change_pct,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 12-MONTH EVENT COMPARISON
# ============================================================

def compare_12m_event_window(
    df,
    event_name=None,
):
    """
    Backward-compatible helper.

    If event_name is supplied, compare that event.
    If omitted, compare the first configured event.
    """

    if event_name is None:
        event_name = list(EVENTS.keys())[0]

    return compare_event(
        df,
        event_name,
        window_months=12,
    )