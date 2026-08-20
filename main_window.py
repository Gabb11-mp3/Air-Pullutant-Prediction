"""Main PyQt5 window for the AirScope dashboard."""

from pathlib import Path
from typing import Dict

from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chart_canvas import PollutionChartCanvas
from config import DEFAULT_CSV_FILE
from file_upload import (
    choose_csv_file,
    load_and_clean_data,
)
from prediction import calculate_predictions
from styles import apply_styles


class AirPollutionWindow(QMainWindow):
    """Main air-pollution prediction dashboard."""

    def __init__(self) -> None:
        super().__init__()

        self.dataset_path = Path(
            DEFAULT_CSV_FILE
        )

        self.setWindowTitle(
            "AirScope | Pollutant Prediction Dashboard"
        )

        self.resize(1500, 900)
        self.setMinimumSize(1050, 700)

        self.build_interface()
        apply_styles(self)

        QTimer.singleShot(
            0,
            self.generate_predictions,
        )

    def build_interface(self) -> None:
        """Create and connect the dashboard widgets."""
        central_widget = QWidget()
        root_layout = QVBoxLayout(
            central_widget
        )

        root_layout.setContentsMargins(
            22,
            18,
            22,
            18,
        )

        root_layout.setSpacing(14)

        self.setCentralWidget(
            central_widget
        )

        # Header
        header = QFrame()
        header.setObjectName(
            "HeaderCard"
        )

        header_layout = QVBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            22,
            16,
            22,
            16,
        )

        title = QLabel("AirScope")
        title.setObjectName("AppTitle")

        subtitle = QLabel(
            "Explore historical air-quality measurements "
            "and cubic-spline forecasts"
        )

        subtitle.setObjectName(
            "AppSubtitle"
        )

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        # Dataset controls
        controls = self._build_controls()
        root_layout.addWidget(controls)

        # Main content
        splitter = QSplitter(
            Qt.Horizontal
        )

        splitter.setChildrenCollapsible(
            False
        )

        splitter.addWidget(
            self._build_results_panel()
        )

        splitter.addWidget(
            self._build_graph_panel()
        )

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1100])

        root_layout.addWidget(
            splitter,
            1,
        )

        self.statusBar().showMessage(
            "Ready"
        )

    def _build_controls(self) -> QFrame:
        """Build dataset and prediction controls."""
        controls = QFrame()
        controls.setObjectName("Card")

        layout = QHBoxLayout(controls)

        layout.setContentsMargins(
            16,
            12,
            16,
            12,
        )

        dataset_label = QLabel("Dataset")

        dataset_label.setObjectName(
            "FieldLabel"
        )

        self.path_label = QLabel(
            str(self.dataset_path)
        )

        self.path_label.setObjectName(
            "PathLabel"
        )

        self.path_label.setToolTip(
            str(self.dataset_path)
        )

        self.path_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.browse_button = QPushButton(
            "Choose CSV"
        )

        self.browse_button.setObjectName(
            "SecondaryButton"
        )

        self.browse_button.clicked.connect(
            self.choose_dataset
        )

        prediction_year_label = QLabel(
            "Prediction year"
        )

        prediction_year_label.setObjectName(
            "FieldLabel"
        )

        self.year_input = QSpinBox()
        self.year_input.setRange(1900, 2200)
        self.year_input.setValue(2023)

        self.year_input.setAlignment(
            Qt.AlignCenter
        )

        self.year_input.setFixedWidth(100)

        self.generate_button = QPushButton(
            "Generate prediction"
        )

        self.generate_button.setObjectName(
            "PrimaryButton"
        )

        self.generate_button.clicked.connect(
            self.generate_predictions
        )

        layout.addWidget(dataset_label)

        layout.addWidget(
            self.path_label,
            1,
        )

        layout.addWidget(
            self.browse_button
        )

        layout.addWidget(
            prediction_year_label
        )

        layout.addWidget(
            self.year_input
        )

        layout.addWidget(
            self.generate_button
        )

        return controls

    def _build_results_panel(self) -> QFrame:
        """Build the summary and prediction table."""
        results_card = QFrame()
        results_card.setObjectName("Card")
        results_card.setMinimumWidth(310)
        results_card.setMaximumWidth(420)

        layout = QVBoxLayout(
            results_card
        )

        layout.setContentsMargins(
            16,
            16,
            16,
            16,
        )

        layout.setSpacing(10)

        results_title = QLabel(
            "Prediction summary"
        )

        results_title.setObjectName(
            "SectionTitle"
        )

        self.summary_label = QLabel(
            "Waiting for data..."
        )

        self.summary_label.setObjectName(
            "MutedLabel"
        )

        self.summary_label.setWordWrap(
            True
        )

        self.prediction_table = QTableWidget(
            0,
            2,
        )

        self.prediction_table.setHorizontalHeaderLabels(
            [
                "Pollutant",
                "Concentration",
            ]
        )

        self.prediction_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.prediction_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        self.prediction_table.setAlternatingRowColors(
            True
        )

        self.prediction_table.verticalHeader().setVisible(
            False
        )

        self.prediction_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )

        self.prediction_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        information_note = QLabel(
            "Forecasts are spline extrapolations. "
            "Treat values beyond the observed date range "
            "as estimates rather than measured values."
        )

        information_note.setObjectName(
            "InfoNote"
        )

        information_note.setWordWrap(True)

        layout.addWidget(results_title)
        layout.addWidget(
            self.summary_label
        )

        layout.addWidget(
            self.prediction_table,
            1,
        )

        layout.addWidget(
            information_note
        )

        return results_card

    def _build_graph_panel(self) -> QFrame:
        """Build the toolbar and graph panel."""
        graph_card = QFrame()
        graph_card.setObjectName("Card")

        layout = QVBoxLayout(
            graph_card
        )

        layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        layout.setSpacing(6)

        self.canvas = PollutionChartCanvas(
            graph_card
        )

        self.toolbar = NavigationToolbar2QT(
            self.canvas,
            graph_card,
        )

        self.toolbar.setObjectName(
            "ChartToolbar"
        )

        graph_scroll = QScrollArea()

        graph_scroll.setObjectName(
            "ChartScroll"
        )

        graph_scroll.setWidgetResizable(
            True
        )

        graph_scroll.setFrameShape(
            QFrame.NoFrame
        )

        graph_scroll.setWidget(
            self.canvas
        )

        layout.addWidget(self.toolbar)

        layout.addWidget(
            graph_scroll,
            1,
        )

        return graph_card

    def choose_dataset(self) -> None:
        """Ask the user for a CSV file."""
        selected_path = choose_csv_file(
            self,
            self.dataset_path,
        )

        if selected_path is None:
            return

        self.dataset_path = selected_path

        self.path_label.setText(
            str(self.dataset_path)
        )

        self.path_label.setToolTip(
            str(self.dataset_path)
        )

        self.generate_predictions()

    def generate_predictions(self) -> None:
        """Load data, predict, and update the interface."""
        self.generate_button.setEnabled(
            False
        )

        self.browse_button.setEnabled(
            False
        )

        QApplication.setOverrideCursor(
            Qt.WaitCursor
        )

        try:
            target_year = (
                self.year_input.value()
            )

            data = load_and_clean_data(
                self.dataset_path
            )

            predictions, chart_data, skipped = (
                calculate_predictions(
                    data,
                    target_year,
                )
            )

            self.update_prediction_table(
                predictions
            )

            self.canvas.display_charts(
                chart_data,
                target_year,
            )

            first_date = (
                data["date"]
                .min()
                .strftime("%d %b %Y")
            )

            last_date = (
                data["date"]
                .max()
                .strftime("%d %b %Y")
            )

            summary = (
                f"{len(predictions)} pollutants "
                f"predicted for {target_year}.\n"
                f"{len(data):,} records · "
                f"{first_date} to {last_date}"
            )

            if skipped:
                summary += (
                    "\nSkipped because of "
                    "insufficient data: "
                    + ", ".join(skipped)
                )

            self.summary_label.setText(
                summary
            )

            self.statusBar().showMessage(
                f"Dashboard updated for "
                f"{target_year}",
                6000,
            )

        except Exception as error:
            self.show_error(
                "Unable to generate predictions",
                str(error),
            )

            self.statusBar().showMessage(
                "Prediction failed",
                6000,
            )

        finally:
            QApplication.restoreOverrideCursor()

            self.generate_button.setEnabled(
                True
            )

            self.browse_button.setEnabled(
                True
            )

    def update_prediction_table(
        self,
        predictions: Dict[str, float],
    ) -> None:
        """Display prediction values in the table."""
        self.prediction_table.setRowCount(
            len(predictions)
        )

        for row, (
            pollutant,
            concentration,
        ) in enumerate(predictions.items()):
            pollutant_item = QTableWidgetItem(
                pollutant
            )

            concentration_item = QTableWidgetItem(
                f"{concentration:,.4f}"
            )

            concentration_item.setTextAlignment(
                Qt.AlignRight
                | Qt.AlignVCenter
            )

            self.prediction_table.setItem(
                row,
                0,
                pollutant_item,
            )

            self.prediction_table.setItem(
                row,
                1,
                concentration_item,
            )

        self.prediction_table.resizeRowsToContents()

    def show_error(
        self,
        title: str,
        message: str,
    ) -> None:
        """Display a data or prediction error."""
        dialog = QMessageBox(self)

        dialog.setIcon(
            QMessageBox.Critical
        )

        dialog.setWindowTitle(title)
        dialog.setText(title)
        dialog.setInformativeText(message)
        dialog.exec_()