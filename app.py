import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    prepare_data,
    validate_data,
    calculate_omis,
)

from src.regime import detect_regimes

from src.anomaly import detect_anomalies


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OptionScope AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0f14;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }

    h2 {
        margin-top: 1rem !important;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 1.05rem;
        margin-top: -15px;
        margin-bottom: 25px;
    }

    .section-description {
        color: #8f9aaa;
        font-size: 0.92rem;
        margin-bottom: 12px;
    }

    .regime-card {
        background: #111820;
        border: 1px solid #27313d;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
    }

    .regime-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .small-muted {
        color: #8f9aaa;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    raw = pd.read_csv(
        "data/bse_index_options_market_data.csv"
    )

    return prepare_data(raw)


df = load_data()


# =========================================================
# MACHINE LEARNING
# =========================================================

@st.cache_data
def run_regime_detection(data):

    result, profiles, _ = detect_regimes(data)

    return result, profiles


@st.cache_data
def run_anomaly_detection(data):

    result, _ = detect_anomalies(data)

    return result


regime_df, regime_profiles = run_regime_detection(df)

analysis_df = run_anomaly_detection(regime_df)

analysis_df["omis"] = calculate_omis(analysis_df)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("OptionScope AI")

st.sidebar.caption(
    "Indian Index Options Market Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "EXPLORE",
    [
        "Market Overview",
        "Liquidity Intelligence",
        "Market Sentiment",
        "Market Regimes",
        "Anomaly Detection",
        "Forecast",
        "Data Quality",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset: 301 monthly observations"
)


# =========================================================
# GLOBAL DATE FILTER
# =========================================================

st.sidebar.subheader("Date Range")

start_date = st.sidebar.date_input(
    "From",
    value=df["Date"].min().date(),
)

end_date = st.sidebar.date_input(
    "To",
    value=df["Date"].max().date(),
)

filtered = analysis_df[
    (analysis_df["Date"].dt.date >= start_date)
    & (analysis_df["Date"].dt.date <= end_date)
].copy()


if filtered.empty:

    st.error(
        "No observations exist for the selected date range."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("📊 OptionScope AI")

st.markdown(
    """
    <div class="subtitle">
    Quantitative intelligence for the Indian Index Options Market
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MARKET OVERVIEW
# =========================================================

if page == "Market Overview":

    st.header("Market Overview")

    st.markdown(
        """
        <div class="section-description">
        Long-run evolution of market participation, turnover,
        liquidity and option sentiment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = filtered.iloc[-1]

    total_contracts = filtered["index_contracts"].sum()

    total_turnover = filtered["index_turnover"].sum()

    avg_pcr = filtered["pcr_contracts"].mean()

    current_omis = latest["omis"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Contracts",
        f"{total_contracts:,.0f}",
    )

    c2.metric(
        "Total Turnover",
        f"₹{total_turnover / 1e9:,.2f} B",
    )

    c3.metric(
        "Average PCR",
        f"{avg_pcr:.2f}",
    )

    c4.metric(
        "OMIS Score",
        f"{current_omis:.0f}/100"
        if pd.notna(current_omis)
        else "N/A",
    )

    st.divider()

    # -----------------------------------------------------
    # Contracts
    # -----------------------------------------------------

    st.subheader("Index Options Contracts")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["index_contracts"],
            mode="lines",
            name="Contracts",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        hovermode="x unified",
        yaxis_title="Contracts",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Turnover
    # -----------------------------------------------------

    st.subheader("Index Options Turnover")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["Date"],
            y=filtered["index_turnover"],
            mode="lines",
            name="Turnover",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        hovermode="x unified",
        yaxis_title="Turnover",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# LIQUIDITY
# =========================================================

elif page == "Liquidity Intelligence":

    st.header("Liquidity Intelligence")

    st.markdown(
        """
        <div class="section-description">
        Average premium per contract provides a simple proxy
        for the value traded per option contract.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = filtered.iloc[-1]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest Avg. Premium",
        f"{latest['avg_premium']:,.2f}",
    )

    c2.metric(
        "12M Avg. Premium",
        f"{latest['premium_12m']:,.2f}"
        if pd.notna(latest["premium_12m"])
        else "N/A",
    )

    c3.metric(
        "Premium Growth",
        f"{latest['premium_growth']:.2f}%"
        if pd.notna(latest["premium_growth"])
        else "N/A",
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
            name="12M Rolling Average",
            line=dict(dash="dash"),
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

    st.subheader("Liquidity Growth")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            x=filtered["Date"],
            y=filtered["premium_growth"],
            name="Premium Growth",
        )
    )

    fig2.update_layout(
        template="plotly_dark",
        height=380,
        yaxis_title="Growth (%)",
        xaxis_title="",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )


# =========================================================
# SENTIMENT
# =========================================================

elif page == "Market Sentiment":

    st.header("Market Sentiment")

    st.markdown(
        """
        <div class="section-description">
        Put/Call ratios and option participation shares provide
        a descriptive view of changes in market positioning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = filtered.iloc[-1]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Put/Call Contracts",
        f"{latest['pcr_contracts']:.2f}",
    )

    c2.metric(
        "Put/Call Turnover",
        f"{latest['pcr_turnover']:.2f}",
    )

    c3.metric(
        "Put Share",
        f"{latest['put_share']:.1f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Put/Call Contract Ratio")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["pcr_contracts"],
                mode="lines",
                name="PCR",
            )
        )

        fig.add_hline(
            y=1,
            line_dash="dash",
            annotation_text="PCR = 1",
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            hovermode="x unified",
            yaxis_title="Put / Call",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        st.subheader("Put vs Call Participation")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["put_share"],
                mode="lines",
                name="Put Share",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=filtered["Date"],
                y=filtered["call_share"],
                mode="lines",
                name="Call Share",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            hovermode="x unified",
            yaxis_title="Share (%)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =========================================================
# REGIMES
# =========================================================

elif page == "Market Regimes":

    st.header("Market Regime Detection")

    st.markdown(
        """
        <div class="section-description">
        K-Means clustering identifies periods with similar combinations
        of market participation, turnover growth, premium dynamics and
        Put/Call ratios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = filtered.iloc[-1]

    regime = latest["regime"]

    st.markdown(
        f"""
        <div class="regime-card">
            <div class="small-muted">CURRENT MARKET REGIME</div>
            <div class="regime-name">{regime}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # -----------------------------------------------------
    # Regime timeline
    # -----------------------------------------------------

    st.subheader("Regime History")

    regime_numbers = {
        "Expansion": 1,
        "Balanced": 2,
        "Defensive": 3,
        "Insufficient Data": 0,
    }

    plot_df = filtered.copy()

    plot_df["regime_number"] = (
        plot_df["regime"]
        .map(regime_numbers)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["regime_number"],
            mode="markers+lines",
            name="Regime",
            text=plot_df["regime"],
            hovertemplate=(
                "%{x|%b %Y}"
                "<br>Regime: %{text}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[1, 2, 3],
        ticktext=[
            "Expansion",
            "Balanced",
            "Defensive",
        ],
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        yaxis_title="",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Regime statistics
    # -----------------------------------------------------

    st.subheader("Regime Characteristics")

    regime_summary = (
        filtered
        .groupby("regime")
        .agg(
            Months=("Date", "count"),
            Avg_Contracts_Growth=(
                "contracts_growth",
                "mean",
            ),
            Avg_Turnover_Growth=(
                "turnover_growth",
                "mean",
            ),
            Avg_PCR=(
                "pcr_contracts",
                "mean",
            ),
            Avg_Premium=(
                "avg_premium",
                "mean",
            ),
        )
        .round(2)
    )

    st.dataframe(
        regime_summary,
        use_container_width=True,
    )

    st.caption(
        "Regime names are interpretations of the clustering "
        "profiles, not pre-labelled historical events."
    )


# =========================================================
# ANOMALIES
# =========================================================

elif page == "Anomaly Detection":

    st.header("Anomaly Detection")

    st.markdown(
        """
        <div class="section-description">
        Isolation Forest identifies observations whose combination
        of growth, premium and sentiment characteristics differs
        substantially from the rest of the sample.
        </div>
        """,
        unsafe_allow_html=True,
    )

    anomalies = filtered[
        filtered["anomaly"]
    ].copy()

    c1, c2 = st.columns(2)

    c1.metric(
        "Detected Anomalies",
        f"{len(anomalies)}",
    )

    c2.metric(
        "Anomaly Rate",
        f"{len(anomalies) / len(filtered) * 100:.1f}%",
    )

    st.divider()

    fig = go.Figure()

    normal = filtered[
        ~filtered["anomaly"]
    ]

    fig.add_trace(
        go.Scatter(
            x=normal["Date"],
            y=normal["index_turnover"],
            mode="lines",
            name="Normal",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["Date"],
            y=anomalies["index_turnover"],
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
        height=450,
        hovermode="x unified",
        yaxis_title="Turnover",
        xaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Detected Unusual Periods")

    if anomalies.empty:

        st.info(
            "No anomalies detected in the selected period."
        )

    else:

        display_columns = [
            "Date",
            "index_contracts",
            "index_turnover",
            "contracts_growth",
            "turnover_growth",
            "pcr_contracts",
            "avg_premium",
            "regime",
        ]

        st.dataframe(
            anomalies[
                display_columns
            ]
            .sort_values("Date", ascending=False)
            .style.format({
                "index_contracts": "{:,.0f}",
                "index_turnover": "{:,.2f}",
                "contracts_growth": "{:.2f}%",
                "turnover_growth": "{:.2f}%",
                "pcr_contracts": "{:.2f}",
                "avg_premium": "{:,.2f}",
            }),
            use_container_width=True,
        )


# =========================================================
# FORECAST
# =========================================================

elif page == "Forecast":

    st.header("12-Month Market Forecast")

    st.markdown(
        """
        <div class="section-description">
        An exponential-trend model provides an indicative 12-month
        projection of contracts and turnover. Forecasts are analytical
        estimates, not trading recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    forecast_months = 12

    def create_forecast(series):

        series = series.dropna()

        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )

        fitted = model.fit(
            optimized=True
        )

        forecast = fitted.forecast(
            forecast_months
        )

        residuals = (
            series
            - fitted.fittedvalues
        )

        residual_std = residuals.std()

        lower = (
            forecast
            - 1.96 * residual_std
        )

        upper = (
            forecast
            + 1.96 * residual_std
        )

        return forecast, lower, upper

    contracts_forecast, contracts_lower, contracts_upper = (
        create_forecast(
            df.set_index("Date")["index_contracts"]
        )
    )

    turnover_forecast, turnover_lower, turnover_upper = (
        create_forecast(
            df.set_index("Date")["index_turnover"]
        )
    )

    forecast_dates = pd.date_range(
        start=df["Date"].max()
        + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )

    # -----------------------------------------------------
    # Contracts forecast
    # -----------------------------------------------------

    st.subheader("Index Options Contracts")

    fig = go.Figure()

    history = df[
        df["Date"] >=
        df["Date"].max() - pd.DateOffset(years=5)
    ]

    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["index_contracts"],
            mode="lines",
            name="Historical",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=contracts_forecast,
            mode="lines",
            name="Forecast",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=contracts_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=contracts_lower,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Approx. 95% interval",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        yaxis_title="Contracts",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Turnover forecast
    # -----------------------------------------------------

    st.subheader("Index Options Turnover")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["index_turnover"],
            mode="lines",
            name="Historical",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=turnover_forecast,
            mode="lines",
            name="Forecast",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=turnover_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=turnover_lower,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Approx. 95% interval",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified",
        yaxis_title="Turnover",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.warning(
        "Forecast intervals are approximate and based on historical "
        "model residuals. They should not be interpreted as guaranteed "
        "prediction intervals."
    )


# =========================================================
# DATA QUALITY
# =========================================================

elif page == "Data Quality":

    st.header("Data Quality & Integrity")

    validation = validate_data(df)

    c1, c2 = st.columns(2)

    with c1:

        if validation["contract_ok"]:

            st.success(
                "✓ Contract totals reconcile."
            )

        else:

            st.warning(
                "Contract reconciliation differences detected."
            )

        st.metric(
            "Maximum Contract Difference",
            f"{validation['max_contract_difference']:,.4f}",
        )

    with c2:

        if validation["turnover_ok"]:

            st.success(
                "✓ Turnover totals reconcile within tolerance."
            )

        else:

            st.warning(
                "Turnover reconciliation differences detected."
            )

        st.metric(
            "Maximum Turnover Difference",
            f"{validation['max_turnover_difference']:,.4f}",
        )

    st.divider()

    st.subheader("Dataset Information")

    info = {
        "Observations": len(df),
        "Variables": 7,
        "Start": df["Date"].min().strftime("%b %Y"),
        "End": df["Date"].max().strftime("%b %Y"),
        "Frequency": "Monthly",
        "Missing Values": int(df.isna().sum().sum()),
    }

    info_df = pd.DataFrame(
        info.items(),
        columns=["Metric", "Value"],
    )

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Raw Data")

    st.dataframe(
        df[
            [
                "Date",
                "index_contracts",
                "index_turnover",
                "call_contracts",
                "call_turnover",
                "put_contracts",
                "put_turnover",
            ]
        ],
        use_container_width=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "OptionScope AI • Quantitative Market Intelligence "
    "for Indian Index Options"
)