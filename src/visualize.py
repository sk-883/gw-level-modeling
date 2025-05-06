import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def plot_time_series():
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_processed.csv",
        parse_dates=["date"]
    )
    plt.figure()
    for well_id, grp in df.groupby("well_id"):
        plt.plot(grp["date"], grp["level"], label=str(well_id))
    plt.xlabel("Date")
    plt.ylabel("Water Level")
    plt.title("Groundwater Levels Over Time")
    plt.legend()
    plt.show()


def plot_prediction_intervals():
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    _, test = train_test_split_time_series(df)
    dates = test["date"]
    preds = np.load(
        PROCESSED_DATA_DIR / "bayesian_pred.npy"
    )
    plt.figure()
    plt.plot(dates, test["level"], label="Actual")
    plt.plot(dates, preds, label="Bayesian Pred")
    plt.xlabel("Date")
    plt.ylabel("Water Level")
    plt.title("Bayesian Predictions vs Actual")
    plt.legend()
    plt.show()


def map_uncertainty():
    gdf = gpd.read_file(
        PROCESSED_DATA_DIR / "wells_features.geojson"
    )
    std = np.load(
        PROCESSED_DATA_DIR / "gp_std.npy"
    )
    # align last N rows
    gdf = gdf.iloc[-len(std):].copy()
    gdf["std"] = std
    ax = gdf.plot(
        column="std",
        legend=True,
        figsize=(8, 6)
    )
    ax.set_title("Prediction Uncertainty (Std Dev)")
    plt.show()

if __name__ == "__main__":
    plot_time_series()
    plot_prediction_intervals()
    map_uncertainty()