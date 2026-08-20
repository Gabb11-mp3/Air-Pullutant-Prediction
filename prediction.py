"""Air-pollution prediction calculations."""

from typing import Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


class ChartData(TypedDict):
    """Values needed to draw one pollutant chart."""

    pollutant: str
    years: NDArray[np.float64]
    concentrations: NDArray[np.float64]
    curve_years: NDArray[np.float64]
    curve_values: NDArray[np.float64]
    last_observed_year: float
    prediction: float


PredictionResult = Tuple[Dict[str, float], List[ChartData], List[str]]


def calculate_predictions(
    data: pd.DataFrame,
    target_year: int,
) -> PredictionResult:
    """Calculate cubic-spline predictions and their chart values."""
    predictions: Dict[str, float] = {}
    chart_data: List[ChartData] = []
    skipped_pollutants: List[str] = []

    pollutants = sorted(data["pollutant"].unique())

    for pollutant in pollutants:
        pollutant_name = str(pollutant)
        pollutant_data = data[data["pollutant"] == pollutant].copy()

        # A spline requires unique x-values. Average measurements that share
        # the same pollutant and date before performing the calculation.
        pollutant_data = (
            pollutant_data.groupby("date", as_index=False)["concentration"]
            .mean()
            .sort_values("date")
        )

        if len(pollutant_data) < 2:
            skipped_pollutants.append(pollutant_name)
            continue

        years = (
            pollutant_data["date"].dt.year
            + (pollutant_data["date"].dt.dayofyear - 1) / 365.25
        ).to_numpy(dtype=float)

        concentrations = pollutant_data["concentration"].to_numpy(
            dtype=float
        )

        spline = CubicSpline(years, concentrations)
        predicted_value = float(spline(float(target_year)))
        predictions[pollutant_name] = predicted_value

        graph_start = min(float(years.min()), float(target_year))
        graph_end = max(float(years.max()), float(target_year))
        curve_years = np.linspace(graph_start, graph_end, 500)

        chart_data.append(
            {
                "pollutant": pollutant_name,
                "years": years,
                "concentrations": concentrations,
                "curve_years": curve_years,
                "curve_values": spline(curve_years),
                "last_observed_year": float(years.max()),
                "prediction": predicted_value,
            }
        )

    if not chart_data:
        raise ValueError(
            "At least two dated observations are required "
            "for each pollutant."
        )

    return predictions, chart_data, skipped_pollutants
