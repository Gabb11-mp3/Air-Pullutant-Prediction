"""CSV selection, loading, validation, and cleaning."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QWidget

from config import REQUIRED_COLUMNS


PathLike = Union[str, Path]


def choose_csv_file(
    parent: QWidget,
    current_path: PathLike,
) -> Optional[Path]:
    """Open a file picker and return the selected CSV path, if any."""
    current_path = Path(current_path)

    selected_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select an air-pollution dataset",
        str(current_path.parent),
        "CSV files (*.csv);;All files (*)",
    )

    return Path(selected_path) if selected_path else None


def load_and_clean_data(file_path: PathLike) -> pd.DataFrame:
    """Load, validate, and clean an air-pollution CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"The dataset could not be found:\n\n{file_path}"
        )

    data = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "The CSV file is missing these required columns:\n\n"
            f"{missing}"
        )

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["pollutant"] = data["pollutant"].astype("string").str.strip()
    data["concentration"] = pd.to_numeric(
        data["concentration"],
        errors="coerce",
    )

    # Treat infinite values as missing before interpolation.
    data["concentration"] = data["concentration"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(subset=["date", "pollutant"])
    data = data[data["pollutant"] != ""]
    data = data.sort_values(["pollutant", "date"])

    # Interpolate each pollutant independently.
    data["concentration"] = (
        data.groupby("pollutant")["concentration"]
        .transform(
            lambda values: values.interpolate(
                method="linear",
                limit_direction="both",
            )
        )
    )

    data = data.dropna(subset=["concentration"])

    if data.empty:
        raise ValueError(
            "The CSV file does not contain any valid pollutant data."
        )

    return data