import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    prepare_data,
    validate_data,
)

from src.regime import (
    add_regimes_to_data,
)

from src.anomaly import (
    add_anomalies_to_data,
    get_top_anomalies,
)

from src.event_analysis import (
    EVENTS,
    compare_event,
    get_event_trajectory,
)

from src.forecast import (
    generate_forecasts,
)

from src.insights import (
    generate_insights,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OptionScope AI",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.7rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    .insight-box {
        background: #111820;
        border: 1px solid #26313d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .insight-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    .insight-text {
        color: #c7d0da;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .footer {
        text-align: center;
        color: #7f8a97;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #202833;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    raw = pd.read_csv(
        "data/bse_index_options_market_data.csv"
    )

    return prepare_data(raw)


# ============================================================
# RUN MODELS
# ============================================================

@st.cache_data
def run_models(df):

    # Market regimes
    result, regime_summary, _, _ = (
        add_regimes_to_data(df)
    )

    # Anomalies
    result, _, _ = (
        add_anomalies_to_data(result)
    )

    # Forecast
    forecast, _, _ = (
        generate_forecasts(result)
    )

    return (
        result,
        regime_summary,
        forecast,
    )


# ============================================================
# INITIALIZE DATA
# ============================================================

df = load_data()

df, regime_summary, forecast = run_models(df)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("OptionScope AI")

st.sidebar.caption(
    "Quantitative Intelligence for the Indian "
    "Index Options Market"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Explore",
    [
        "📊 Market Overview",
        "💧 Liquidity Intelligence",
        "🎯 Market Sentiment",
        "🧠 Market Regimes",
        "🚨 Anomaly Detection",
        "📅 Event Analysis",
        "🔍 Data Quality",
    ],
)

st.sidebar.divider()

st.sidebar.caption("Dataset")


# ============================================================
# DATE RANGE SLIDER
# ============================================================

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY",
)

start_date, end_date = date_range

filtered = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
].copy()


# ============================================================
# HEADER
# ============================================================

st.title("📊 OptionScope AI")

st.markdown(
    """
    <div class="subtitle">
    Quantitative intelligence for the Indian Index Options Market
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MARKET OVERVIEW
# ============================================================

if page == "📊 Market Overview":

    latest = filtered.iloc[-1]

    st.header("Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Value per Contract",
        f"{latest['avg_premium']:.2f}",
    )

    col2.metric(
        "Value per Contract vs 12M",
        f"{latest['premium_vs_12m']:+.1f}%",
    )

    col3.metric(
        "Value per Contract Growth",
        f"{latest['premium_growth']:+.1f}%",
    )

    st.divider()
    st.subheader("Market Activity")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["index_contracts"],
            mode="lines",
            name="Contracts",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["contracts_12m"],
            mode="lines",
            name="12M Average",
            line=dict(
                dash="dash"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        yaxis_title="Contracts",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Automated Intelligence")

    insights = generate_insights(
        filtered,
        forecast,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="insight-box">
            <div class="insight-title">
            📈 Activity
            </div>
            <div class="insight-text">
            {insights["activity"]}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="insight-box">
            <div class="insight-title">
            💧 Liquidity
            </div>
            <div class="insight-text">
            {insights["liquidity"]}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="insight-box">
            <div class="insight-title">
            🎯 Sentiment
            </div>
            <div class="insight-text">
            {insights["sentiment"]}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="insight-box">
            <div class="insight-title">
            🧠 Intelligence Score
            </div>
            <div class="insight-text">
            {insights["omis"]}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LIQUIDITY
# ============================================================

elif page == "💧 Liquidity Intelligence":

    st.header("💧 Liquidity Intelligence")

    latest = filtered.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Premium",
        f"{latest['avg_premium']:.2f}",
    )

    col2.metric(
        "Premium vs 12M",
        f"{latest['premium_vs_12m']:+.1f}%",
    )

    col3.metric(
        "Premium Growth",
        f"{latest['premium_growth']:+.1f}%",
    )

    st.divider()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["avg_premium"],
            mode="lines",
            name="Average Value per Contract",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["premium_12m"],
            mode="lines",
            name="12M Average",
            line=dict(
                dash="dash"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        yaxis_title="Turnover / Contract",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Liquidity Interpretation")

    insights = generate_insights(
        filtered,
        forecast,
    )

    st.info(insights["liquidity"])


# ============================================================
# SENTIMENT
# ============================================================

elif page == "🎯 Market Sentiment":

    st.header("🎯 Market Sentiment")

    latest = filtered.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Contract PCR",
        f"{latest['pcr_contracts']:.2f}",
    )

    col2.metric(
        "Turnover PCR",
        f"{latest['pcr_turnover']:.2f}",
    )

    col3.metric(
        "Put Share",
        f"{latest['put_share']:.1f}%",
    )

    st.divider()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["pcr_contracts"],
            mode="lines",
            name="Put / Call Ratio",
        )
    )

    fig.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="PCR = 1",
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        yaxis_title="Put / Call",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Positioning Interpretation")

    insights = generate_insights(
        filtered,
        forecast,
    )

    st.info(insights["sentiment"])


# ============================================================
# MARKET REGIMES
# ============================================================

elif page == "🧠 Market Regimes":

    st.header("🧠 Market Regimes")

    st.markdown(
        """
        K-Means clustering identifies recurring market
        environments from activity, contract-value and
        positioning characteristics.
        """
    )

    latest = filtered.iloc[-1]

    if pd.isna(latest["regime_name"]):

        st.warning(
            "The selected period does not contain enough "
            "observations for regime classification."
        )

    else:

        st.metric(
            "Current Regime",
            latest["regime_name"],
        )

    st.divider()

    st.subheader("Regime Distribution")

    regime_counts = (
        filtered["regime_name"]
        .value_counts()
        .reset_index()
    )

    regime_counts.columns = [
        "Regime",
        "Observations",
    ]

    fig = go.Figure(
        go.Bar(
            x=regime_counts["Regime"],
            y=regime_counts["Observations"],
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_title="",
        yaxis_title="Months",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Regime Through Time")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["regime_cluster"],
            mode="markers",
            name="Regime",
        )
    )

    fig2.update_layout(
        template="plotly_dark",
        height=350,
        yaxis_title="Cluster",
        xaxis_title="",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    st.subheader("Regime Profiles")

    st.dataframe(
        regime_summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    st.markdown(
        """
        Isolation Forest identifies observations whose
        combination of activity, contract value and positioning
        characteristics differs substantially from historical
        patterns.
        """
    )

    anomaly_count = int(
        filtered["anomaly_flag"].sum()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Detected Anomalies",
        anomaly_count,
    )

    col2.metric(
        "Anomaly Rate",
        f"{anomaly_count / len(filtered) * 100:.1f}%",
    )

    st.divider()

    st.subheader("Most Unusual Observations")

    anomalies = get_top_anomalies(
        filtered,
        n=10,
    )

    st.dataframe(
        anomalies,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Anomaly Timeline")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["index_contracts"],
            mode="lines",
            name="Contracts",
        )
    )

    unusual = filtered[
        filtered["anomaly_flag"]
    ]

    fig.add_trace(
        go.Scatter(
            x=unusual["Date"],
            y=unusual["index_contracts"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=10,
                symbol="x",
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        yaxis_title="Contracts",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# EVENT ANALYSIS
# ============================================================

elif page == "📅 Event Analysis":

    st.title("📅 Event Analysis")

    st.markdown(
        """
        Examine how index-options market activity behaved around
        major market and geopolitical events.

        The analysis is **descriptive rather than causal**: it
        identifies changes in market behavior around major event
        dates but does not claim that an event caused those changes.
        """
    )

    # --------------------------------------------------------
    # EVENT DEFINITIONS
    # --------------------------------------------------------

    event_definitions = [
        {
            "name": "Russia-Ukraine War",
            "date": pd.Timestamp("2022-02-24"),
            "description": (
                "Russia launched its full-scale invasion of Ukraine."
            ),
            "line_color": "#ff6b6b",
        },
        {
            "name": "BSE Derivatives Relaunch",
            "date": pd.Timestamp("2023-05-15"),
            "description": (
                "BSE relaunched Sensex and Bankex derivatives."
            ),
            "line_color": "#4dabf7",
        },
        {
            "name": "Trump India Tariffs",
            "date": pd.Timestamp("2025-08-27"),
            "description": (
                "Additional US tariffs on Indian imports took effect."
            ),
            "line_color": "#ffd43b",
        },
    ]

    event_definitions = sorted(
        event_definitions,
        key=lambda x: x["date"],
    )

    # --------------------------------------------------------
    # EVENT TIMELINE
    # --------------------------------------------------------

    st.subheader("Major Event Timeline")

    st.caption(
        "Dashed vertical lines mark major events. "
        "The chart follows the date range selected in the sidebar."
    )

    # --------------------------------------------------------
    # MARKET ACTIVITY CHART
    # --------------------------------------------------------

    fig_events = go.Figure()

    # Use FILTERED data so the chart responds to the
    # global date slider.
    if not filtered.empty:

        # -----------------------------------------------
        # Contracts
        # -----------------------------------------------

        if "index_contracts" in filtered.columns:

            fig_events.add_trace(
                go.Scatter(
                    x=filtered["Date"],
                    y=filtered["index_contracts"],
                    mode="lines",
                    name="Index Options Contracts",
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Contracts: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

        # -----------------------------------------------
        # Turnover
        # -----------------------------------------------

        if "index_turnover" in filtered.columns:

            fig_events.add_trace(
                go.Scatter(
                    x=filtered["Date"],
                    y=filtered["index_turnover"],
                    mode="lines",
                    name="Index Options Turnover",
                    yaxis="y2",
                    hovertemplate=(
                        "%{x|%b %Y}<br>"
                        "Turnover: %{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    # --------------------------------------------------------
    # DATE RANGE OF FILTERED DATA
    # --------------------------------------------------------

    if not filtered.empty:

        chart_min_date = filtered["Date"].min()
        chart_max_date = filtered["Date"].max()

        # ----------------------------------------------------
        # ADD EVENT MARKERS ONLY IF INSIDE SELECTED RANGE
        # ----------------------------------------------------

        for event_item in event_definitions:

            actual_date = event_item["date"]

            # Dataset is monthly, so align the event marker
            # to the first day of the event month.
            marker_date = pd.Timestamp(
                year=actual_date.year,
                month=actual_date.month,
                day=1,
            )

            # Only show the event if its month falls inside
            # the currently selected date range.
            if (
                marker_date >= chart_min_date
                and marker_date <= chart_max_date
            ):

                fig_events.add_vline(
                    x=marker_date,
                    line_dash="dash",
                    line_width=2.5,
                    line_color=event_item["line_color"],
                )

                fig_events.add_annotation(
                    x=marker_date,
                    y=1,
                    yref="paper",
                    text=(
                        f"<b>{event_item['name']}</b><br>"
                        f"{actual_date:%d %b %Y}"
                    ),
                    showarrow=False,
                    yshift=10,
                    xanchor="left",
                    textangle=-90,
                    font=dict(
                        size=11,
                    ),
                )

    # --------------------------------------------------------
    # CHART LAYOUT
    # --------------------------------------------------------

    fig_events.update_layout(
        template="plotly_dark",
        title="Index Options Activity Around Major Events",
        xaxis_title="Date",
        yaxis_title="Contracts",
        yaxis2=dict(
            title="Turnover",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            l=60,
            r=80,
            t=130,
            b=60,
        ),
    )

    if filtered.empty:

        st.warning(
            "No data is available for the selected date range."
        )

    else:

        st.plotly_chart(
            fig_events,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # EVENT REFERENCE
    # --------------------------------------------------------

    st.subheader("Events")

    event_rows = []

    for event_item in event_definitions:

        event_rows.append(
            {
                "Event": event_item["name"],
                "Date": event_item["date"].strftime(
                    "%d %B %Y"
                ),
                "Description": event_item["description"],
            }
        )

    st.dataframe(
        pd.DataFrame(event_rows),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 12-MONTH EVENT COMPARISON
    # --------------------------------------------------------

    st.subheader("12-Month Pre / Post Event Comparison")

    st.caption(
        "The comparison uses the available historical data around "
        "each event. The event month itself is excluded."
    )

    comparison_rows = []

    for event_item in event_definitions:

        event_date = event_item["date"]

        event_month = pd.Timestamp(
            year=event_date.year,
            month=event_date.month,
            day=1,
        )

        pre_start = (
            event_month
            - pd.DateOffset(months=12)
        )

        post_end = (
            event_month
            + pd.DateOffset(months=12)
        )

        # Use the FULL dataset for the 12-month event study.
        # This is intentional: otherwise moving the dashboard
        # slider would incorrectly truncate the event windows.
        pre = df[
            (df["Date"] >= pre_start)
            & (df["Date"] < event_month)
        ].copy()

        post = df[
            (df["Date"] > event_month)
            & (df["Date"] <= post_end)
        ].copy()

        # -----------------------------------------------
        # Contracts
        # -----------------------------------------------

        pre_contracts = (
            pre["index_contracts"].mean()
            if "index_contracts" in pre.columns
            else None
        )

        post_contracts = (
            post["index_contracts"].mean()
            if "index_contracts" in post.columns
            else None
        )

        if (
            pre_contracts is not None
            and post_contracts is not None
            and pre_contracts != 0
        ):

            contracts_change = (
                (post_contracts - pre_contracts)
                / abs(pre_contracts)
                * 100
            )

        else:

            contracts_change = None

        # -----------------------------------------------
        # Turnover
        # -----------------------------------------------

        pre_turnover = (
            pre["index_turnover"].mean()
            if "index_turnover" in pre.columns
            else None
        )

        post_turnover = (
            post["index_turnover"].mean()
            if "index_turnover" in post.columns
            else None
        )

        if (
            pre_turnover is not None
            and post_turnover is not None
            and pre_turnover != 0
        ):

            turnover_change = (
                (post_turnover - pre_turnover)
                / abs(pre_turnover)
                * 100
            )

        else:

            turnover_change = None

        comparison_rows.append(
            {
                "Event": event_item["name"],
                "Date": event_date.strftime(
                    "%d %b %Y"
                ),
                "Pre-Event Contracts": pre_contracts,
                "Post-Event Contracts": post_contracts,
                "Contracts Change %": contracts_change,
                "Pre-Event Turnover": pre_turnover,
                "Post-Event Turnover": post_turnover,
                "Turnover Change %": turnover_change,
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # FORMAT DISPLAY
    # --------------------------------------------------------

    display_df = comparison_df.copy()

    for column in [
        "Pre-Event Contracts",
        "Post-Event Contracts",
        "Pre-Event Turnover",
        "Post-Event Turnover",
    ]:

        if column in display_df.columns:

            display_df[column] = display_df[column].map(
                lambda x: (
                    f"{x:,.0f}"
                    if pd.notna(x)
                    else "—"
                )
            )

    for column in [
        "Contracts Change %",
        "Turnover Change %",
    ]:

        if column in display_df.columns:

            display_df[column] = display_df[column].map(
                lambda x: (
                    f"{x:+.1f}%"
                    if pd.notna(x)
                    else "—"
                )
            )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader("How to Read This")

    st.markdown(
        """
        **Event markers** show when major events occurred relative
        to monthly options-market activity.

        **The chart follows the global date slider.** Events outside
        the selected period are automatically hidden.

        **12-month pre/post comparison** evaluates the average activity
        in the 12 months before and after each event using the available
        historical dataset. This comparison is kept independent of the
        dashboard slider so that the event-study window remains intact.

        These comparisons are descriptive and should not be interpreted
        as evidence of causality.
        """
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "🔍 Data Quality":

    st.header("🔍 Data Quality")

    validation = validate_data(
        df
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Observations",
        validation["rows_checked"],
    )

    col2.metric(
        "Max Contract Difference",
        f"{validation['max_contract_difference']:.2f}",
    )

    col3.metric(
        "Max Turnover Difference",
        f"{validation['max_turnover_difference']:.2f}",
    )

    st.divider()

    if (
        validation["contract_ok"]
        and validation["turnover_ok"]
    ):

        st.success(
            "✓ Dataset integrity verified. "
            "Index totals reconcile with Call + Put components."
        )

    else:

        st.warning(
            "Minor reconciliation differences detected."
        )

    st.subheader("Dataset Coverage")

    st.write(
        f"""
        **Start:** {df["Date"].min():%B %Y}

        **End:** {df["Date"].max():%B %Y}

        **Monthly observations:** {len(df):,}

        **Variables:** {len(df.columns)}
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>OptionScope AI</strong><br><br>

        Built by <strong>Prakhar Gupta</strong>
        &nbsp;&nbsp;
        <a href="mailto:bestofprakhar@gmail.com"
           title="Email Prakhar Gupta">
            ✉
        </a>

        &nbsp;&nbsp;|&nbsp;&nbsp;

        <a href="https://www.linkedin.com/in/prakhar-gupta-5b7250372"
           target="_blank"
           title="LinkedIn">
            in
        </a>

        &nbsp;&nbsp;|&nbsp;&nbsp;

        <a href="https://x.com/PrakharQuant"
           target="_blank"
           title="X">
            𝕏
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)