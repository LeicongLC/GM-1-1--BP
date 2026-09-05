"""Run a reproducible GM(1,1)-BP forecasting workflow.

'BP' is implemented with scikit-learn's MLPRegressor. This keeps the project
easy to run while preserving the back-propagation neural network idea from the
original notebook.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6"]
REQUIRED_COLUMNS = ["year"] + FEATURE_COLUMNS + ["Y"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class Metrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "explained_variance": self.explained_variance,
        }


class GM11Model:
    """GM(1,1) model for strictly positive time-series values."""

    def __init__(self) -> None:
        self.alpha = None
        self.u = None
        self.x0 = None

    def fit(self, series: np.ndarray) -> "GM11Model":
        x = np.asarray(series, dtype=float).reshape(-1)
        if len(x) < 4:
            raise ValueError("GM(1,1) needs at least 4 observations.")
        if np.any(x <= 0):
            raise ValueError("GM(1,1) requires positive input values.")

        x1 = np.cumsum(x)
        z1 = 0.5 * (x1[1:] + x1[:-1])
        b = np.column_stack((-z1, np.ones(len(z1))))
        y = x[1:]
        self.alpha, self.u = np.linalg.lstsq(b, y, rcond=None)[0]
        self.x0 = x[0]
        return self

    def _cumulative_hat(self, k: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.u is None or self.x0 is None:
            raise RuntimeError("GM11Model must be fitted before predicting.")
        if np.isclose(self.alpha, 0.0):
            raise ValueError("GM(1,1) fit is degenerate because alpha is approximately zero.")
        return (self.x0 - self.u / self.alpha) * np.exp(-self.alpha * k) + self.u / self.alpha

    def predict_next(self, steps: int, history_length: int) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        k = np.arange(history_length, history_length + steps)
        current = self._cumulative_hat(k)
        previous = self._cumulative_hat(k - 1)
        return current - previous


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Input data contains missing values in required columns.")
    return df[REQUIRED_COLUMNS].sort_values("year").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_index = int(round(len(df) * (1 - test_size)))
    split_index = min(max(split_index, 4), len(df) - 1)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def build_bp_model(random_state: int, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.005,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mse = mean_squared_error(y_true, y_pred)
    return Metrics(
        mse=float(mse),
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        explained_variance=float(explained_variance_score(y_true, y_pred)),
    )


def forecast_features_with_gm(df: pd.DataFrame, future_steps: int) -> pd.DataFrame:
    if future_steps <= 0:
        raise ValueError("future_steps must be positive.")
    predictions = {}
    for column in FEATURE_COLUMNS:
        model = GM11Model().fit(df[column].values)
        predictions[column] = model.predict_next(future_steps, history_length=len(df))

    last_year = int(df["year"].max())
    future = pd.DataFrame(predictions)
    future.insert(0, "year", np.arange(last_year + 1, last_year + future_steps + 1))
    return future


def save_plots(
    output_dir: Path,
    train_losses: List[float],
    test_df: pd.DataFrame,
    test_predictions: np.ndarray,
    future_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, color="#2563eb", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("BP Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(test_df["Y"], test_predictions, color="#059669", edgecolors="#064e3b", alpha=0.75)
    lower = min(test_df["Y"].min(), test_predictions.min())
    upper = max(test_df["Y"].max(), test_predictions.max())
    plt.plot([lower, upper], [lower, upper], color="#dc2626", linestyle="--", linewidth=2)
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Actual vs Predicted Values")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(future_df["year"], future_df["Predicted Values"], marker="o", color="#7c3aed", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Predicted Y")
    plt.title("Future Forecast")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "future_forecast.png", dpi=160)
    plt.close()


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    future_steps: int,
    test_size: float,
    random_state: int,
    max_iter: int,
) -> Metrics:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    train_df, test_df = chronological_split(df, test_size)

    model = build_bp_model(random_state=random_state, max_iter=max_iter)
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(train_df[["Y"]]).reshape(-1)
    model.fit(train_df[FEATURE_COLUMNS], y_train)
    test_predictions_scaled = model.predict(test_df[FEATURE_COLUMNS]).reshape(-1, 1)
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled).reshape(-1)
    metrics = evaluate(test_df["Y"].values, test_predictions)

    result_df = test_df[["year", "Y"]].copy()
    result_df["Predicted Values"] = test_predictions
    result_df["Absolute Error"] = (result_df["Y"] - result_df["Predicted Values"]).abs()
    result_df.to_csv(output_dir / "test_predictions.csv", index=False)

    future_features = forecast_features_with_gm(df, future_steps)
    future_predictions_scaled = model.predict(future_features[FEATURE_COLUMNS]).reshape(-1, 1)
    future_predictions = target_scaler.inverse_transform(future_predictions_scaled).reshape(-1)
    future_df = future_features[["year"]].copy()
    future_df["Predicted Values"] = future_predictions
    future_features.to_csv(output_dir / "gm_feature_forecast.csv", index=False)
    future_df.to_csv(output_dir / "predicted_values.csv", index=False)

    loss_curve = list(getattr(model.named_steps["model"], "loss_curve_", []))
    pd.DataFrame([metrics.as_dict()]).to_csv(output_dir / "metrics.csv", index=False)
    save_plots(output_dir, loss_curve, test_df, test_predictions, future_df)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GM(1,1)-BP forecasting workflow.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"), help="Path to the input CSV file.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for generated files.")
    parser.add_argument("--future-steps", type=int, default=10, help="Number of future years to forecast.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of latest rows used as test data.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the BP model.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum BP training iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        future_steps=args.future_steps,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    print("Evaluation metrics:")
    for key, value in metrics.as_dict().items():
        print("  {}: {:.6f}".format(key, value))
    print("Generated outputs in: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
