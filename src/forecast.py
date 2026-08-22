import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# FORECAST CONFIGURATION
# ============================================================

FORECAST_HORIZON = 12


# ============================================================
# PREPARE TIME SERIES
# ============================================================

def prepare_series(df, column):
    """
    Prepare a monthly time series for forecasting.
    """

    series = (
        df[["Date", column]]
        .dropna()
        .sort_values("Date")
        .set_index("Date")[column]
        .astype(float)
    )

    # Ensure a regular monthly frequency.
    series = series.asfreq("MS")

    # Fill any missing observations by interpolation.
    series = series.interpolate()

    return series


# ============================================================
# HOLT-WINTERS FORECAST
# ============================================================

def forecast_series(
    df,
    column,
    horizon=FORECAST_HORIZON
):
    """
    Forecast a market variable using Holt-Winters
    exponential smoothing.

    The model captures:
    - level
    - trend
    - monthly seasonality
    """

    series = prepare_series(df, column)

    if len(series) < 36:
        raise ValueError(
            "At least 36 monthly observations are required "
            "for forecasting."
        )

    # Multiplicative seasonality can fail when values are zero,
    # so use additive seasonality for robustness.
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    )

    fitted_model = model.fit(
        optimized=True
    )

    forecast = fitted_model.forecast(
        horizon
    )

    # --------------------------------------------------------
    # Approximate prediction intervals
    # --------------------------------------------------------
    residuals = (
        fitted_model.resid
        .dropna()
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

    forecast_df = pd.DataFrame({
        "Date": forecast.index,
        "forecast": forecast.values,
        "lower": lower.values,
        "upper": upper.values,
    })

    return forecast_df, fitted_model


# ============================================================
# FORECAST CONTRACTS
# ============================================================

def forecast_contracts(
    df,
    horizon=FORECAST_HORIZON
):
    """
    Forecast monthly index option contracts.
    """

    return forecast_series(
        df,
        "index_contracts",
        horizon=horizon
    )


# ============================================================
# FORECAST TURNOVER
# ============================================================

def forecast_turnover(
    df,
    horizon=FORECAST_HORIZON
):
    """
    Forecast monthly index option turnover.
    """

    return forecast_series(
        df,
        "index_turnover",
        horizon=horizon
    )


# ============================================================
# COMBINED FORECAST
# ============================================================

def generate_forecasts(
    df,
    horizon=FORECAST_HORIZON
):
    """
    Generate forecasts for both contracts and turnover.
    """

    contracts, contracts_model = forecast_contracts(
        df,
        horizon=horizon
    )

    turnover, turnover_model = forecast_turnover(
        df,
        horizon=horizon
    )

    contracts = contracts.rename(
        columns={
            "forecast": "contracts_forecast",
            "lower": "contracts_lower",
            "upper": "contracts_upper",
        }
    )

    turnover = turnover.rename(
        columns={
            "forecast": "turnover_forecast",
            "lower": "turnover_lower",
            "upper": "turnover_upper",
        }
    )

    result = contracts.merge(
        turnover,
        on="Date",
        how="outer"
    )

    return (
        result,
        contracts_model,
        turnover_model
    )


# ============================================================
# FORECAST SUMMARY
# ============================================================

def get_forecast_summary(
    forecast_df
):
    """
    Return a compact summary of the forecast horizon.
    """

    if forecast_df.empty:
        return {}

    first = forecast_df.iloc[0]
    last = forecast_df.iloc[-1]

    return {
        "forecast_start": first["Date"],
        "forecast_end": last["Date"],
        "starting_contract_forecast": (
            first["contracts_forecast"]
        ),
        "ending_contract_forecast": (
            last["contracts_forecast"]
        ),
        "starting_turnover_forecast": (
            first["turnover_forecast"]
        ),
        "ending_turnover_forecast": (
            last["turnover_forecast"]
        ),
    }