"""Matplotlib chart component used by the AirScope window."""

from typing import Sequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QWidget

from config import CHART_COLORS
from prediction import ChartData


class PollutionChartCanvas(FigureCanvas):
    """Matplotlib figure embedded inside the PyQt5 window."""

    def __init__(self, parent: QWidget = None) -> None:
        self.figure = Figure(
            figsize=(12, 8),
            dpi=100,
            facecolor="#F8FAFC",
        )
        super().__init__(self.figure)

        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        rows = (len(chart_data) + columns - 1) // columns
        axes = self.figure.subplots(
            rows,
            columns,
            squeeze=False,
        ).ravel()

        for index, item in enumerate(chart_data):
            axis = axes[index]
            color = CHART_COLORS[index % len(CHART_COLORS)]
            years = item["years"]
            concentrations = item["concentrations"]
            prediction = item["prediction"]

            axis.set_facecolor("#FFFFFF")
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

            # Shade the 95% prediction range calculated from model residuals.
            axis.fill_between(
                item["curve_years"],
                item["curve_lower"],
                item["curve_upper"],
                color=color,
                alpha=0.12,
                label="95% prediction range",
                zorder=1,
            )

            # Draw the line learned by gradient descent.
            axis.plot(
                item["curve_years"],
                item["curve_values"],
                color=color,
                linewidth=2.2,
                label="Linear regression",
                zorder=2,
            )

            if target_year > item["last_observed_year"]:
                axis.axvspan(
                    item["last_observed_year"],
                    target_year,
                    color="#EF4444",
                    alpha=0.06,
                )

            axis.axvline(
                target_year,
                color="#EF4444",
                linestyle="--",
                linewidth=1.5,
                label=f"{target_year} prediction",
            )
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
                (
                    f"{prediction:,.4f}\n"
                    f"95%: {item['prediction_lower']:,.4f} to "
                    f"{item['prediction_upper']:,.4f}"
                ),
                (target_year, prediction),
                xytext=(8, 9),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#B91C1C",
            )

            # Display the cost and accuracy so model error is transparent.
            axis.text(
                0.02,
                0.98,
                (
                    f"Cost: {item['final_cost']:.6g}\n"
                    f"RMSE: {item['rmse']:.4f}\n"
                    f"R²: {item['r_squared']:.4f}\n"
                    f"Trend/year: {item['slope_per_year']:+.4f}\n"
                    f"Iterations: {item['iterations']}"
                ),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#334155",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#CBD5E1",
                    "alpha": 0.88,
                },
                zorder=6,
            )
            axis.set_title(
                f"{item['pollutant']} concentration",
                fontsize=12,
                fontweight="bold",
                color="#0F172A",
                pad=12,
            )
            axis.set_xlabel("Year", color="#475569")
            axis.set_ylabel("Concentration", color="#475569")
            axis.grid(
                True,
                color="#CBD5E1",
                alpha=0.55,
                linewidth=0.7,
            )
            axis.tick_params(colors="#475569", labelsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#CBD5E1")
            axis.spines["bottom"].set_color("#CBD5E1")
            axis.legend(frameon=False, fontsize=8, loc="best")

        for axis in axes[len(chart_data) :]:
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

        # Give every row enough height; the enclosing scroll area handles the
        # overflow when many pollutants are displayed.
        self.setMinimumHeight(max(650, rows * 410))
        self.draw_idle()
