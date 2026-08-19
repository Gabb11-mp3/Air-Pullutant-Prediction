from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# Locate the CSV file beside this Python script
file_path = Path(__file__).resolve().parent / "air_pollution.csv"

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

# Load the dataset
data = pd.read_csv(file_path)

# Confirm that required columns exist
required_columns = {"date", "pollutant", "concentration"}
missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise ValueError(
        f"CSV file is missing these columns: {', '.join(sorted(missing_columns))}"
    )

# Clean the data
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["concentration"] = pd.to_numeric(
    data["concentration"], errors="coerce"
)

data = data.dropna(subset=["date", "pollutant"])
data = data.sort_values(["pollutant", "date"])

# Interpolate missing concentrations separately for each pollutant
data["concentration"] = data.groupby("pollutant")[
    "concentration"
].transform(
    lambda values: values.interpolate(
        method="linear",
        limit_direction="both",
    )
)

data = data.dropna(subset=["concentration"])

# Extract pollutants
pollutants = sorted(data["pollutant"].unique())

if not pollutants:
    raise ValueError("No valid pollutant data was found in the CSV file.")

# Prediction settings
year_to_predict = 2023.0
predictions = {}

# Create one UI window containing every graph
columns = 2
rows = (len(pollutants) + columns - 1) // columns

figure, axes = plt.subplots(
    rows,
    columns,
    figsize=(14, 5 * rows),
    squeeze=False,
)

axes = axes.ravel()

# Create one subplot for each pollutant
for index, pollutant in enumerate(pollutants):
    axis = axes[index]

    pollutant_data = data[data["pollutant"] == pollutant].copy()

    # Average duplicate dates to keep spline x-values unique
    pollutant_data = (
        pollutant_data.groupby("date", as_index=False)["concentration"]
        .mean()
        .sort_values("date")
    )

    if len(pollutant_data) < 2:
        axis.text(
            0.5,
            0.5,
            f"Not enough data for {pollutant}",
            ha="center",
            va="center",
        )
        axis.set_title(str(pollutant))
        continue

    # Convert dates into fractional years
    years = (
        pollutant_data["date"].dt.year
        + (pollutant_data["date"].dt.dayofyear - 1) / 365.25
    ).to_numpy()

    concentrations = pollutant_data["concentration"].to_numpy()

    # Fit the cubic spline
    spline = CubicSpline(years, concentrations)

    # Predict the concentration for 2023
    predicted_concentration = float(spline(year_to_predict))
    predictions[pollutant] = predicted_concentration

    # Generate values for the spline curve
    graph_end_year = max(float(years.max()), year_to_predict)
    extended_years = np.linspace(
        float(years.min()),
        graph_end_year,
        500,
    )

    # Draw the subplot
    axis.scatter(
        years,
        concentrations,
        label=f"{pollutant} data",
        color="blue",
    )

    axis.plot(
        extended_years,
        spline(extended_years),
        label=f"{pollutant} spline",
        color="orange",
    )

    axis.axvline(
        year_to_predict,
        color="red",
        linestyle="--",
        label="2023 prediction",
    )

    axis.scatter(
        year_to_predict,
        predicted_concentration,
        color="red",
        zorder=5,
    )

    axis.set_title(f"{pollutant} Concentration Prediction")
    axis.set_xlabel("Year")
    axis.set_ylabel("Concentration")
    axis.grid(alpha=0.3)
    axis.legend()

# Hide unused subplot positions
for axis in axes[len(pollutants):]:
    axis.set_visible(False)

# Configure and display the single UI window
figure.suptitle(
    "Air Pollutant Concentration Predictions",
    fontsize=16,
)

figure.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# Print predictions
print("\nPredicted Gas Concentrations in 2023:")

for pollutant, concentration in predictions.items():
    print(f"{pollutant}: {concentration:.4f}")