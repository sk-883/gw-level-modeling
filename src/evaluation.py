import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def compute_metrics(true, pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAE": mean_absolute_error(true, pred),
        "R2": r2_score(true, pred)
    }


def load_test_df():
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    _, test = train_test_split_time_series(df)
    return test


def evaluate_model(name: str, pred_file: str):
    test = load_test_df()
    true = test["level"].values
    pred = np.load(PROCESSED_DATA_DIR / pred_file)
    metrics = compute_metrics(true, pred)
    metrics["model"] = name
    return metrics


def main():
    results = [
        evaluate_model("Bayesian", "bayesian_pred.npy"),
        evaluate_model("ARIMA", "arima_forecast.npy"),
        evaluate_model("RandomForest", "rf_preds.npy"),
        evaluate_model("GaussianProcess", "gp_preds.npy"),
    ]
    df = pd.DataFrame(results)
    print(df[['model', 'RMSE', 'MAE', 'R2']].to_markdown(index=False))
    df.to_csv(
        PROCESSED_DATA_DIR / "evaluation_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()