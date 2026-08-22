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
        "Latest Contracts",
        f"{latest['index_contracts']:,.0f}",
    )

    col2.metric(
        "Latest Turnover",
        f"₹{latest['index_turnover'] / 1e9:.2f}B",
    )

    col3.metric(
        "Put / Call Ratio",
        f"{latest['pcr_contracts']:.2f}",
    )

    col4.metric(
        "OMIS",
        f"{latest['omis']:.1f}",
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
            name="Average Premium",
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
        environments from activity, premium and positioning
        characteristics.
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
        combination of activity, premium and positioning
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

    st.header("📅 Major Market Events")

    st.markdown(
        """
        Examine how Indian index-options activity behaved
        around major geopolitical, policy and market-structure
        events.

        **Important:** these are descriptive event comparisons,
        not causal estimates.
        """
    )

    # --------------------------------------------------------
    # EVENT SELECTOR
    # --------------------------------------------------------

    event_name = st.selectbox(
        "Select Event",
        list(EVENTS.keys()),
    )

    event = EVENTS[event_name]

    st.info(
        f"**{event_name}** — "
        f"{event['date']:%d %B %Y}\n\n"
        f"{event['description']}"
    )

    # --------------------------------------------------------
    # PRE / POST COMPARISON
    # --------------------------------------------------------

    st.subheader("12-Month Pre / Post Comparison")

    comparison = compare_event(
        df,
        event_name,
        window_months=12,
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # EVENT TRAJECTORY
    # --------------------------------------------------------

    st.subheader(
        "Market Activity Around the Event"
    )

    trajectory = get_event_trajectory(
        df,
        event_name,
        window_months=12,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trajectory["Date"],
            y=trajectory["index_contracts"],
            mode="lines+markers",
            name="Contracts",
        )
    )

    fig.add_vline(
        x=event["date"],
        line_dash="dash",
        annotation_text=event_name,
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        yaxis_title="Contracts",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    st.subheader("Turnover Response")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=trajectory["Date"],
            y=trajectory["index_turnover"],
            mode="lines+markers",
            name="Turnover",
        )
    )

    fig2.add_vline(
        x=event["date"],
        line_dash="dash",
        annotation_text=event_name,
    )

    fig2.update_layout(
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        yaxis_title="Turnover",
        xaxis_title="",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
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

    Built by <strong>Prakhar Gupta</strong><br><br>

    <a href="mailto:bestofprakhar@gmail.com">
    ✉ Email
    </a>

    &nbsp;&nbsp;|&nbsp;&nbsp;

    <a href="https://www.linkedin.com/in/prakhar-gupta-5b7250372"
       target="_blank">
    LinkedIn
    </a>

    &nbsp;&nbsp;|&nbsp;&nbsp;

    <a href="https://x.com/PrakharQuant"
       target="_blank">
    X
    </a>

    </div>
    """,
    unsafe_allow_html=True,
)