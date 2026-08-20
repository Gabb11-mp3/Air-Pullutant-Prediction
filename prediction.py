"""Linear-regression calculations for air-pollution predictions."""

# Import reusable type descriptions so the function inputs and outputs are clear.
from typing import Dict, List, Tuple, TypedDict

# NumPy performs the vector mathematics used by cost and gradient descent.
import numpy as np

# Pandas supplies the DataFrame that contains the cleaned CSV records.
import pandas as pd

# NDArray lets the type hints describe NumPy arrays more precisely.
from numpy.typing import NDArray

# Student's t distribution is used to calculate a 95% prediction range.
from scipy.stats import t as student_t


# This learning rate controls how large each gradient-descent update is.
DEFAULT_LEARNING_RATE = 0.05

# This limit prevents gradient descent from running forever.
DEFAULT_MAX_ITERATIONS = 10_000

# Training stops when the cost changes by less than this small amount.
DEFAULT_TOLERANCE = 1e-10

# This value requests a 95% prediction range around the regression line.
PREDICTION_CONFIDENCE = 0.95


class ChartData(TypedDict):
    """Values needed to draw and explain one pollutant regression chart."""

    # Store the pollutant name displayed above the chart.
    pollutant: str

    # Store the decimal years of the real observations.
    years: NDArray[np.float64]

    # Store the real concentration measurements.
    concentrations: NDArray[np.float64]

    # Store closely spaced years used to draw a smooth straight line.
    curve_years: NDArray[np.float64]

    # Store the regression prediction at every curve year.
    curve_values: NDArray[np.float64]

    # Store the lower edge of the 95% prediction area.
    curve_lower: NDArray[np.float64]

    # Store the upper edge of the 95% prediction area.
    curve_upper: NDArray[np.float64]

    # Store the last year that has a real observation.
    last_observed_year: float

    # Store the predicted concentration for the user's requested year.
    prediction: float

    # Store the lower prediction-range value for the requested year.
    prediction_lower: float

    # Store the upper prediction-range value for the requested year.
    prediction_upper: float

    # Store the final half-mean-squared-error cost after training.
    final_cost: float

    # Store the root-mean-squared error in concentration units.
    rmse: float

    # Store R-squared, which describes how much variation the line explains.
    r_squared: float

    # Store the learned change in concentration for one calendar year.
    slope_per_year: float

    # Store how many gradient-descent updates were required.
    iterations: int


# Describe the three values returned to the main window.
PredictionResult = Tuple[
    Dict[str, float],
    List[ChartData],
    List[str],
]


def calculate_cost(
    actual_values: NDArray[np.float64],
    predicted_values: NDArray[np.float64],
) -> float:
    """Return the standard linear-regression half-MSE cost."""

    # Confirm that every actual value has one matching predicted value.
    if actual_values.shape != predicted_values.shape:
        raise ValueError(
            "Actual and predicted values must have matching shapes."
        )

    # A cost cannot be calculated when there are no observations.
    if actual_values.size == 0:
        raise ValueError("At least one value is required to calculate cost.")

    # Subtract the actual value from the prediction to get each model error.
    errors = predicted_values - actual_values

    # Square each error so negative and positive errors cannot cancel out.
    squared_errors = errors**2

    # Divide the mean squared error by two to simplify the gradient formula.
    cost = np.mean(squared_errors) / 2.0

    # Convert NumPy's number into an ordinary Python float for the UI.
    return float(cost)


def fit_linear_regression_with_gradient_descent(
    scaled_years: NDArray[np.float64],
    actual_values: NDArray[np.float64],
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[float, float, List[float]]:
    """Learn a slope and intercept by repeatedly reducing prediction cost."""

    # Confirm that each year has exactly one concentration value.
    if scaled_years.shape != actual_values.shape:
        raise ValueError("Years and concentrations must have matching shapes.")

    # Reject an empty dataset because the gradients would be undefined.
    if scaled_years.size == 0:
        raise ValueError("Gradient descent requires at least one observation.")

    # The learning rate must be positive so updates move toward lower error.
    if learning_rate <= 0:
        raise ValueError("The learning rate must be greater than zero.")

    # At least one update must be allowed for the model to train.
    if max_iterations <= 0:
        raise ValueError("The maximum iteration count must be positive.")

    # Start with a flat line because the initial slope is zero.
    slope = 0.0

    # Start the line at the average concentration to reduce initial error.
    intercept = float(np.mean(actual_values))

    # Record every cost so the learning progress can be inspected or plotted.
    cost_history: List[float] = []

    # Infinity guarantees that the first calculated cost is treated as new.
    previous_cost = float("inf")

    # Repeat small updates until the cost stops changing or the limit is met.
    for _ in range(max_iterations):
        # Apply y = intercept + slope*x to every standardized year.
        predicted_values = intercept + slope * scaled_years

        # Measure how far every prediction is from its actual concentration.
        errors = predicted_values - actual_values

        # Count observations because the gradients use their average effect.
        observation_count = float(actual_values.size)

        # Calculate how strongly the slope contributes to the current errors.
        slope_gradient = float(
            np.sum(errors * scaled_years) / observation_count
        )

        # Calculate how strongly the intercept contributes to current errors.
        intercept_gradient = float(np.sum(errors) / observation_count)

        # Move the slope opposite its gradient to reduce the cost.
        slope -= learning_rate * slope_gradient

        # Move the intercept opposite its gradient to reduce the cost.
        intercept -= learning_rate * intercept_gradient

        # Recalculate predictions after changing the model parameters.
        updated_predictions = intercept + slope * scaled_years

        # Measure the new error with the separate cost function above.
        current_cost = calculate_cost(actual_values, updated_predictions)

        # Save the cost so the number and progress of updates remain visible.
        cost_history.append(current_cost)

        # Stop when another update changes the cost by almost nothing.
        if abs(previous_cost - current_cost) < tolerance:
            break

        # Save this cost so it can be compared with the next iteration.
        previous_cost = current_cost

    # Return the trained model and its complete training-cost history.
    return slope, intercept, cost_history


def calculate_prediction_range(
    years: NDArray[np.float64],
    actual_values: NDArray[np.float64],
    fitted_values: NDArray[np.float64],
    requested_years: NDArray[np.float64],
    requested_predictions: NDArray[np.float64],
    confidence: float = PREDICTION_CONFIDENCE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a prediction range from residual error and year distance."""

    # Count the observations used to train this pollutant's regression line.
    observation_count = years.size

    # Two points can form a line but cannot estimate residual uncertainty.
    if observation_count <= 2:
        return requested_predictions.copy(), requested_predictions.copy()

    # Calculate the differences between real values and fitted training values.
    residuals = actual_values - fitted_values

    # Add the squared residuals to measure unexplained model variation.
    sum_squared_errors = float(np.sum(residuals**2))

    # Linear regression estimates two parameters: slope and intercept.
    degrees_of_freedom = observation_count - 2

    # Estimate the typical unexplained error remaining after model training.
    residual_standard_error = np.sqrt(
        sum_squared_errors / degrees_of_freedom
    )

    # Find the center of the observed years.
    mean_year = float(np.mean(years))

    # Measure how widely the real years are spread around their center.
    year_spread = float(np.sum((years - mean_year) ** 2))

    # Distinct dates should give a positive spread; guard against bad input.
    if np.isclose(year_spread, 0.0):
        return requested_predictions.copy(), requested_predictions.copy()

    # Convert 95% confidence into the probability left in the two tails.
    alpha = 1.0 - confidence

    # Get the Student-t multiplier appropriate for this sample size.
    critical_value = float(
        student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    )

    # Predictions farther from the observed mean year are less certain.
    distance_effect = (
        (requested_years - mean_year) ** 2 / year_spread
    )

    # Include uncertainty for both the fitted mean and a future observation.
    prediction_standard_error = residual_standard_error * np.sqrt(
        1.0 + (1.0 / observation_count) + distance_effect
    )

    # Multiply standard error by the t value to obtain the range margin.
    margin = critical_value * prediction_standard_error

    # Subtract the margin to create the bottom edge of the prediction area.
    lower_values = requested_predictions - margin

    # Add the margin to create the top edge of the prediction area.
    upper_values = requested_predictions + margin

    # Return both edges so the chart can shade the entire prediction area.
    return lower_values, upper_values


def calculate_r_squared(
    actual_values: NDArray[np.float64],
    fitted_values: NDArray[np.float64],
) -> float:
    """Calculate the proportion of concentration variation explained."""

    # Calculate the variation that the regression line did not explain.
    residual_variation = float(np.sum((actual_values - fitted_values) ** 2))

    # Calculate the total variation around the average concentration.
    total_variation = float(
        np.sum((actual_values - np.mean(actual_values)) ** 2)
    )

    # A constant series has no variation, so treat a perfect fit as R-squared 1.
    if np.isclose(total_variation, 0.0):
        return 1.0 if np.isclose(residual_variation, 0.0) else 0.0

    # Compare unexplained variation with total variation.
    return float(1.0 - residual_variation / total_variation)


def calculate_predictions(
    data: pd.DataFrame,
    target_year: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PredictionResult:
    """Train one linear-regression model for every pollutant and predict."""

    # This dictionary supplies the prediction values used by the summary table.
    predictions: Dict[str, float] = {}

    # This list supplies observations, lines, ranges, and metrics to the charts.
    chart_data: List[ChartData] = []

    # This list tells the user which pollutants did not have enough data.
    skipped_pollutants: List[str] = []

    # Sorting produces a stable pollutant order in both the table and charts.
    pollutants = sorted(data["pollutant"].unique())

    # Train an independent linear-regression model for each pollutant gas.
    for pollutant in pollutants:
        # Convert Pandas or NumPy text values into a regular Python string.
        pollutant_name = str(pollutant)

        # Select only the observations that belong to the current pollutant.
        pollutant_data = data[data["pollutant"] == pollutant].copy()

        # Average duplicate dates so every regression x-value is unique.
        pollutant_data = (
            pollutant_data.groupby("date", as_index=False)["concentration"]
            .mean()
            .sort_values("date")
        )

        # A straight line requires at least two different dated observations.
        if len(pollutant_data) < 2:
            skipped_pollutants.append(pollutant_name)
            continue

        # Convert each date into a decimal year, preserving its position in-year.
        years = (
            pollutant_data["date"].dt.year
            + (pollutant_data["date"].dt.dayofyear - 1) / 365.25
        ).to_numpy(dtype=float)

        # Convert concentrations into a NumPy array for vector calculations.
        concentrations = pollutant_data["concentration"].to_numpy(dtype=float)

        # Find the year center used to standardize large calendar-year numbers.
        year_mean = float(np.mean(years))

        # Find the year scale so gradient descent receives manageable inputs.
        year_standard_deviation = float(np.std(years))

        # Skip an invalid group whose dates somehow map to the same decimal year.
        if np.isclose(year_standard_deviation, 0.0):
            skipped_pollutants.append(pollutant_name)
            continue

        # Center and scale years; this prevents gradient descent from diverging.
        scaled_years = (years - year_mean) / year_standard_deviation

        # Learn the best slope and intercept by minimizing the cost function.
        slope, intercept, cost_history = (
            fit_linear_regression_with_gradient_descent(
                scaled_years,
                concentrations,
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        )

        # Apply the trained line to observed years to measure training accuracy.
        fitted_values = intercept + slope * scaled_years

        # Convert the requested year onto the same scale used during training.
        scaled_target_year = (
            float(target_year) - year_mean
        ) / year_standard_deviation

        # Apply the learned equation to estimate the unseen year's concentration.
        predicted_value = float(intercept + slope * scaled_target_year)

        # Store the requested prediction for the summary table.
        predictions[pollutant_name] = predicted_value

        # Start the graph at the earliest point among data and requested year.
        graph_start = min(float(years.min()), float(target_year))

        # End the graph at the latest point among data and requested year.
        graph_end = max(float(years.max()), float(target_year))

        # Create many closely spaced years so the regression line draws cleanly.
        curve_years = np.linspace(graph_start, graph_end, 500)

        # Standardize curve years with exactly the same training transformation.
        scaled_curve_years = (
            curve_years - year_mean
        ) / year_standard_deviation

        # Predict the regression value for every position along the chart line.
        curve_values = intercept + slope * scaled_curve_years

        # Calculate the 95% lower and upper edges across the complete chart.
        curve_lower, curve_upper = calculate_prediction_range(
            years,
            concentrations,
            fitted_values,
            curve_years,
            curve_values,
        )

        # Put the target into an array because the range function is vectorized.
        target_year_array = np.array([float(target_year)], dtype=float)

        # Put its prediction into a matching one-value NumPy array.
        target_prediction_array = np.array([predicted_value], dtype=float)

        # Calculate the requested year's specific lower and upper range values.
        target_lower, target_upper = calculate_prediction_range(
            years,
            concentrations,
            fitted_values,
            target_year_array,
            target_prediction_array,
        )

        # Convert half-MSE cost back into RMSE, which uses concentration units.
        rmse = float(np.sqrt(2.0 * cost_history[-1]))

        # Convert the standardized slope into concentration change per year.
        slope_per_year = float(slope / year_standard_deviation)

        # Store every value required to explain and draw this pollutant model.
        chart_data.append(
            {
                "pollutant": pollutant_name,
                "years": years,
                "concentrations": concentrations,
                "curve_years": curve_years,
                "curve_values": curve_values,
                "curve_lower": curve_lower,
                "curve_upper": curve_upper,
                "last_observed_year": float(years.max()),
                "prediction": predicted_value,
                "prediction_lower": float(target_lower[0]),
                "prediction_upper": float(target_upper[0]),
                "final_cost": float(cost_history[-1]),
                "rmse": rmse,
                "r_squared": calculate_r_squared(
                    concentrations,
                    fitted_values,
                ),
                "slope_per_year": slope_per_year,
                "iterations": len(cost_history),
            }
        )

    # Report a useful error when no pollutant can produce a regression model.
    if not chart_data:
        raise ValueError(
            "At least two different dated observations are required "
            "for each pollutant."
        )

    # Return table values, graph values, and any insufficient-data warnings.
    return predictions, chart_data, skipped_pollutants
