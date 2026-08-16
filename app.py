import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import prepare_data, validate_data


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="OptionScope AI",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 700;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-label {
        color: #9aa4b2;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/bse_index_options_market_data.csv"
    )

    return prepare_data(df)


df = load_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 OptionScope AI")

st.markdown(
    """
    <div class="subtitle">
    Quantitative intelligence for the Indian Index Options Market
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("OptionScope")

st.sidebar.markdown(
    """
    **Market Intelligence Platform**

    Explore:

    • Market Activity  
    • Liquidity  
    • Sentiment  
    • Market Regimes  
    • Anomalies  
    """
)

st.sidebar.divider()

start_date = st.sidebar.date_input(
    "Start Date",
    value=df["Date"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df["Date"].max().date()
)

filtered = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
].copy()


# --------------------------------------------------
# KPI calculations
# --------------------------------------------------

total_contracts = filtered["index_contracts"].sum()
total_turnover = filtered["index_turnover"].sum()

latest = filtered.iloc[-1]

latest_pcr = latest["pcr_contracts"]
latest_premium = latest["avg_premium"]


# --------------------------------------------------
# KPI cards
# --------------------------------------------------

st.subheader("Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Contracts",
    f"{total_contracts:,.0f}"
)

col2.metric(
    "Total Turnover",
    f"₹{total_turnover / 1e9:,.2f} B"
)

col3.metric(
    "Latest Put/Call Ratio",
    f"{latest_pcr:.2f}"
)

col4.metric(
    "Latest Avg. Premium",
    f"{latest_premium:,.2f}"
)


# --------------------------------------------------
# Market Activity
# --------------------------------------------------

st.divider()

st.subheader("Market Activity")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["Date"],
        y=filtered["index_contracts"],
        mode="lines",
        name="Contracts",
        line=dict(width=2)
    )
)

fig.update_layout(
    template="plotly_dark",
    height=430,
    hovermode="x unified",
    xaxis_title="",
    yaxis_title="Contracts",
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# Turnover
# --------------------------------------------------

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=filtered["Date"],
        y=filtered["index_turnover"],
        mode="lines",
        name="Turnover",
        line=dict(width=2)
    )
)

fig2.update_layout(
    template="plotly_dark",
    height=430,
    hovermode="x unified",
    xaxis_title="",
    yaxis_title="Turnover",
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# --------------------------------------------------
# Liquidity + Sentiment
# --------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Liquidity Intelligence")

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["avg_premium"],
            mode="lines",
            name="Average Premium"
        )
    )

    fig3.update_layout(
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Turnover / Contract"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


with right:

    st.subheader("Market Sentiment")

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["pcr_contracts"],
            mode="lines",
            name="Put/Call Ratio"
        )
    )

    fig4.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="PCR = 1"
    )

    fig4.update_layout(
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Put / Call"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# --------------------------------------------------
# Data Quality
# --------------------------------------------------

st.divider()

st.subheader("Data Integrity")

validation = validate_data(df)

if validation["contract_ok"] and validation["turnover_ok"]:

    st.success(
        "✓ Dataset integrity verified — "
        "Index totals reconcile with Call + Put components."
    )

else:

    st.warning(
        "⚠ Minor reconciliation differences detected."
    )


st.caption(
    f"Dataset: {len(df):,} monthly observations | "
    f"{df['Date'].min():%b %Y} – {df['Date'].max():%b %Y}"
)