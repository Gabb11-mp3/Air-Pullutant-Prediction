import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import CubicSpline


REQUIRED_COLUMNS = {"date", "pollutant", "concentration"}
DEFAULT_DATASET = Path(__file__).resolve().parent / "air_pollution.csv"
CHART_COLORS = (
    "#2563EB",
    "#7C3AED",
    "#0891B2",
    "#059669",
    "#D97706",
    "#DB2777",
)


def load_and_clean_data(file_path):
    """Read and validate an air-pollution CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{file_path}")

    data = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The selected CSV is missing these columns: {missing}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["pollutant"] = data["pollutant"].astype("string").str.strip()
    data["concentration"] = pd.to_numeric(
        data["concentration"], errors="coerce"
    )

    data = data.dropna(subset=["date", "pollutant"])
    data = data[data["pollutant"] != ""]
    data = data.sort_values(["pollutant", "date"])

    data["concentration"] = data.groupby("pollutant")[
        "concentration"
    ].transform(
        lambda values: values.interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    data = data.dropna(subset=["concentration"])

    if data.empty:
        raise ValueError("The selected CSV does not contain any valid data rows.")

    return data


def calculate_predictions(data, target_year):
    """Create spline models and return predictions plus chart-ready data."""
    predictions = {}
    chart_data = []
    skipped = []

    for pollutant in sorted(data["pollutant"].unique()):
        pollutant_data = data[data["pollutant"] == pollutant]
        pollutant_data = (
            pollutant_data.groupby("date", as_index=False)["concentration"]
            .mean()
            .sort_values("date")
        )

        if len(pollutant_data) < 2:
            skipped.append(str(pollutant))
            continue

        years = (
            pollutant_data["date"].dt.year
            + (pollutant_data["date"].dt.dayofyear - 1) / 365.25
        ).to_numpy(dtype=float)
        concentrations = pollutant_data["concentration"].to_numpy(dtype=float)

        spline = CubicSpline(years, concentrations)
        predicted_value = float(spline(float(target_year)))
        predictions[str(pollutant)] = predicted_value

        graph_start = min(float(years.min()), float(target_year))
        graph_end = max(float(years.max()), float(target_year))
        extended_years = np.linspace(graph_start, graph_end, 500)

        chart_data.append(
            {
                "pollutant": str(pollutant),
                "years": years,
                "concentrations": concentrations,
                "curve_years": extended_years,
                "curve_values": spline(extended_years),
                "last_observed_year": float(years.max()),
                "prediction": predicted_value,
            }
        )

    if not chart_data:
        raise ValueError(
            "At least two dated observations are required for each pollutant."
        )

    return predictions, chart_data, skipped


class PollutionChartCanvas(FigureCanvas):
    """Matplotlib canvas embedded directly inside the Qt application."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8), dpi=100, facecolor="#F8FAFC")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(760)

    def display_charts(self, chart_data, target_year):
        self.figure.clear()
        columns = 2
        rows = (len(chart_data) + columns - 1) // columns
        axes = self.figure.subplots(rows, columns, squeeze=False).ravel()

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
            axis.plot(
                item["curve_years"],
                item["curve_values"],
                color=color,
                linewidth=2.2,
                label="Cubic spline",
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
            axis.set_xlabel("Year", color="#475569")
            axis.set_ylabel("Concentration", color="#475569")
            axis.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.7)
            axis.tick_params(colors="#475569", labelsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#CBD5E1")
            axis.spines["bottom"].set_color("#CBD5E1")
            axis.legend(frameon=False, fontsize=8, loc="best")

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
        self.setMinimumHeight(max(620, rows * 410))
        self.draw_idle()


class AirPollutionWindow(QMainWindow):
    """Main dashboard window for exploring pollutant predictions."""

    def __init__(self):
        super().__init__()
        self.dataset_path = DEFAULT_DATASET
        self.setWindowTitle("AirScope | Pollutant Prediction Dashboard")
        self.resize(1500, 900)
        self.setMinimumSize(1050, 700)
        self.build_interface()
        self.apply_styles()
        QTimer.singleShot(0, self.generate_predictions)

    def build_interface(self):
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(22, 18, 22, 18)
        root_layout.setSpacing(14)
        self.setCentralWidget(central_widget)

        header = QFrame()
        header.setObjectName("HeaderCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 16)
        header_layout.setSpacing(3)

        title = QLabel("AirScope")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "Explore historical air-quality measurements and cubic-spline forecasts"
        )
        subtitle.setObjectName("AppSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        controls = QFrame()
        controls.setObjectName("Card")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(12)

        dataset_caption = QLabel("Dataset")
        dataset_caption.setObjectName("FieldLabel")
        controls_layout.addWidget(dataset_caption)

        self.path_label = QLabel(str(self.dataset_path))
        self.path_label.setObjectName("PathLabel")
        self.path_label.setToolTip(str(self.dataset_path))
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls_layout.addWidget(self.path_label, 1)

        self.browse_button = QPushButton("Choose CSV")
        self.browse_button.setObjectName("SecondaryButton")
        self.browse_button.clicked.connect(self.choose_dataset)
        controls_layout.addWidget(self.browse_button)

        year_caption = QLabel("Prediction year")
        year_caption.setObjectName("FieldLabel")
        controls_layout.addWidget(year_caption)

        self.year_input = QSpinBox()
        self.year_input.setRange(1900, 2200)
        self.year_input.setValue(2023)
        self.year_input.setAlignment(Qt.AlignCenter)
        self.year_input.setFixedWidth(96)
        controls_layout.addWidget(self.year_input)

        self.generate_button = QPushButton("Generate prediction")
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.clicked.connect(self.generate_predictions)
        controls_layout.addWidget(self.generate_button)
        root_layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        results_card = QFrame()
        results_card.setObjectName("Card")
        results_card.setMinimumWidth(300)
        results_card.setMaximumWidth(410)
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(16, 16, 16, 16)
        results_layout.setSpacing(10)

        results_title = QLabel("Prediction summary")
        results_title.setObjectName("SectionTitle")
        self.summary_label = QLabel("Waiting for data…")
        self.summary_label.setObjectName("MutedLabel")
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(results_title)
        results_layout.addWidget(self.summary_label)

        self.prediction_table = QTableWidget(0, 2)
        self.prediction_table.setHorizontalHeaderLabels(
            ["Pollutant", "Concentration"]
        )
        self.prediction_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.prediction_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.prediction_table.setAlternatingRowColors(True)
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.prediction_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        results_layout.addWidget(self.prediction_table, 1)

        note = QLabel(
            "Forecasts are spline extrapolations. Treat results beyond the observed "
            "date range as estimates, not measured values."
        )
        note.setObjectName("InfoNote")
        note.setWordWrap(True)
        results_layout.addWidget(note)

        chart_card = QFrame()
        chart_card.setObjectName("Card")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.setSpacing(6)

        self.canvas = PollutionChartCanvas(chart_card)
        self.toolbar = NavigationToolbar2QT(self.canvas, chart_card)
        self.toolbar.setObjectName("ChartToolbar")
        chart_layout.addWidget(self.toolbar)

        chart_scroll = QScrollArea()
        chart_scroll.setObjectName("ChartScroll")
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setFrameShape(QFrame.NoFrame)
        chart_scroll.setWidget(self.canvas)
        chart_layout.addWidget(chart_scroll, 1)

        splitter.addWidget(results_card)
        splitter.addWidget(chart_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1100])
        root_layout.addWidget(splitter, 1)

        self.statusBar().showMessage("Ready")

    def choose_dataset(self):
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an air-pollution dataset",
            str(self.dataset_path.parent),
            "CSV files (*.csv);;All files (*)",
        )

        if selected_path:
            self.dataset_path = Path(selected_path)
            self.path_label.setText(str(self.dataset_path))
            self.path_label.setToolTip(str(self.dataset_path))
            self.generate_predictions()

    def generate_predictions(self):
        self.generate_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            target_year = self.year_input.value()
            data = load_and_clean_data(self.dataset_path)
            predictions, chart_data, skipped = calculate_predictions(
                data, target_year
            )

            self.update_prediction_table(predictions)
            self.canvas.display_charts(chart_data, target_year)

            first_date = data["date"].min().strftime("%d %b %Y")
            last_date = data["date"].max().strftime("%d %b %Y")
            summary = (
                f"{len(predictions)} pollutants predicted for {target_year}.\n"
                f"{len(data):,} records · {first_date} to {last_date}"
            )
            if skipped:
                summary += f"\nSkipped (insufficient data): {', '.join(skipped)}"

            self.summary_label.setText(summary)
            self.statusBar().showMessage(
                f"Prediction dashboard updated for {target_year}", 6000
            )
        except Exception as error:
            self.show_error("Unable to generate predictions", str(error))
            self.statusBar().showMessage("Prediction failed", 6000)
        finally:
            QApplication.restoreOverrideCursor()
            self.generate_button.setEnabled(True)
            self.browse_button.setEnabled(True)

    def update_prediction_table(self, predictions):
        self.prediction_table.setSortingEnabled(False)
        self.prediction_table.setRowCount(len(predictions))

        for row, (pollutant, concentration) in enumerate(predictions.items()):
            pollutant_item = QTableWidgetItem(pollutant)
            concentration_item = QTableWidgetItem(f"{concentration:,.4f}")
            concentration_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            concentration_item.setForeground(QColor("#1D4ED8"))
            concentration_font = QFont()
            concentration_font.setBold(True)
            concentration_item.setFont(concentration_font)
            self.prediction_table.setItem(row, 0, pollutant_item)
            self.prediction_table.setItem(row, 1, concentration_item)

        self.prediction_table.resizeRowsToContents()

    def show_error(self, title, message):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(title)
        dialog.setInformativeText(message)
        dialog.exec_()

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #F1F5F9;
                color: #0F172A;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QFrame#HeaderCard {
                background-color: #0F172A;
                border-radius: 12px;
            }
            QLabel#AppTitle {
                color: #FFFFFF;
                font-size: 22pt;
                font-weight: 700;
                background: transparent;
            }
            QLabel#AppSubtitle {
                color: #CBD5E1;
                font-size: 10.5pt;
                background: transparent;
            }
            QFrame#Card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QLabel#SectionTitle {
                font-size: 14pt;
                font-weight: 700;
                color: #0F172A;
                background: transparent;
            }
            QLabel#FieldLabel {
                font-weight: 600;
                color: #334155;
                background: transparent;
            }
            QLabel#PathLabel, QLabel#MutedLabel {
                color: #64748B;
                background: transparent;
            }
            QLabel#InfoNote {
                color: #475569;
                background-color: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-radius: 7px;
                padding: 10px;
            }
            QPushButton {
                min-height: 36px;
                border-radius: 7px;
                padding: 0 15px;
                font-weight: 600;
            }
            QPushButton#PrimaryButton {
                color: #FFFFFF;
                background-color: #2563EB;
                border: 1px solid #2563EB;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton#SecondaryButton {
                color: #1E293B;
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #F8FAFC;
                border-color: #94A3B8;
            }
            QPushButton:disabled {
                color: #94A3B8;
                background-color: #E2E8F0;
                border-color: #E2E8F0;
            }
            QSpinBox {
                min-height: 34px;
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 0 7px;
                font-weight: 600;
            }
            QSpinBox:focus {
                border: 2px solid #3B82F6;
            }
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 7px;
                gridline-color: #E2E8F0;
                selection-background-color: #DBEAFE;
                selection-color: #0F172A;
            }
            QHeaderView::section {
                background-color: #F8FAFC;
                color: #475569;
                border: none;
                border-bottom: 1px solid #E2E8F0;
                padding: 8px;
                font-weight: 600;
            }
            QScrollArea#ChartScroll {
                background-color: #F8FAFC;
                border: none;
            }
            QToolBar#ChartToolbar {
                background-color: #FFFFFF;
                border: none;
                spacing: 5px;
                padding: 2px;
            }
            QStatusBar {
                background-color: #FFFFFF;
                color: #475569;
                border-top: 1px solid #E2E8F0;
            }
            QSplitter::handle {
                background-color: #E2E8F0;
                width: 5px;
            }
            QScrollBar:vertical {
                background: #F1F5F9;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                border-radius: 6px;
                min-height: 32px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )


def show_unhandled_exception(exception_type, exception, exception_traceback):
    """Display unexpected errors in the GUI instead of the terminal."""
    dialog = QMessageBox()
    dialog.setIcon(QMessageBox.Critical)
    dialog.setWindowTitle("Unexpected application error")
    dialog.setText("The application encountered an unexpected error.")
    dialog.setInformativeText(str(exception))
    dialog.setDetailedText(
        "".join(
            traceback.format_exception(
                exception_type,
                exception,
                exception_traceback,
            )
        )
    )
    dialog.exec_()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    application = QApplication(sys.argv)
    application.setApplicationName("AirScope")
    application.setStyle("Fusion")
    sys.excepthook = show_unhandled_exception

    window = AirPollutionWindow()
    window.showMaximized()
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
