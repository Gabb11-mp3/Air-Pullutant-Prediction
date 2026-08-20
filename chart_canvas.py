"""Matplotlib chart component used by AirScope."""

from typing import Sequence

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QWidget

from config import CHART_COLORS
from prediction import ChartData


class PollutionChartCanvas(FigureCanvas):
    """Matplotlib figure embedded in the PyQt5 window."""

    def __init__(
        self,
        parent: QWidget = None,
    ) -> None:
        self.figure = Figure(
            figsize=(12, 8),
            dpi=100,
            facecolor="#F8FAFC",
        )

        super().__init__(self.figure)

        self.setParent(parent)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.setMinimumWidth(750)
        self.setMinimumHeight(650)

    def display_charts(
        self,
        chart_data: Sequence[ChartData],
        target_year: int,
    ) -> None:
        """Draw every pollutant graph in one canvas."""
        self.figure.clear()

        columns = 2
        rows = (
            len(chart_data) + columns - 1
        ) // columns

        axes = self.figure.subplots(
            rows,
            columns,
            squeeze=False,
        ).ravel()

        for index, item in enumerate(chart_data):
            axis = axes[index]

            color = CHART_COLORS[
                index % len(CHART_COLORS)
            ]

            years = item["years"]
            concentrations = item["concentrations"]
            prediction = item["prediction"]

            axis.set_facecolor("#FFFFFF")

            # Historical observations.
            axis.scatter(
                years,
                concentrations,
                s=42,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label="Observed data",
                zorder=3,
            )

            # Cubic-spline line.
            axis.plot(
                item["curve_years"],
                item["curve_values"],
                color=color,
                linewidth=2.2,
                label="Cubic spline",
                zorder=2,
            )

            # Highlight the forecast region.
            if target_year > item["last_observed_year"]:
                axis.axvspan(
                    item["last_observed_year"],
                    target_year,
                    color="#EF4444",
                    alpha=0.06,
                )

            # Prediction year indicator.
            axis.axvline(
                target_year,
                color="#EF4444",
                linestyle="--",
                linewidth=1.5,
                label=f"{target_year} prediction",
            )

            # Predicted concentration.
            axis.scatter(
                target_year,
                prediction,
                s=75,
                color="#EF4444",
                edgecolor="white",
                linewidth=1,
                zorder=5,
            )

            axis.annotate(
                f"{prediction:,.4f}",
                (target_year, prediction),
                xytext=(8, 9),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#B91C1C",
            )

            axis.set_title(
                f"{item['pollutant']} concentration",
                fontsize=12,
                fontweight="bold",
                color="#0F172A",
                pad=12,
            )

            axis.set_xlabel(
                "Year",
                color="#475569",
            )

            axis.set_ylabel(
                "Concentration",
                color="#475569",
            )

            axis.grid(
                True,
                color="#CBD5E1",
                alpha=0.55,
                linewidth=0.7,
            )

            axis.tick_params(
                colors="#475569",
                labelsize=9,
            )

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#CBD5E1")
            axis.spines["bottom"].set_color("#CBD5E1")

            axis.legend(
                frameon=False,
                fontsize=8,
                loc="best",
            )

        # Hide unused subplot positions.
        for axis in axes[len(chart_data):]:
            axis.set_visible(False)

        self.figure.suptitle(
            f"Pollutant trends and {target_year} predictions",
            fontsize=16,
            fontweight="bold",
            color="#0F172A",
            y=0.985,
        )

        self.figure.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.065,
            top=0.93,
            hspace=0.42,
            wspace=0.24,
        )

        self.setMinimumHeight(
            max(650, rows * 410)
        )

        self.draw_idle()