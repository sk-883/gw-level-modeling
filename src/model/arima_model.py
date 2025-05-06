import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def train_arima(df: pd.DataFrame, order=(5, 1, 0)):
    df = df.set_index("date")
    train, test = train_test_split_time_series(df)

    model = ARIMA(
        train["level"], order=order
    )
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))

    np.save(
        PROCESSED_DATA_DIR / "arima_forecast.npy",
        forecast.values
    )
    return fitted, forecast, test

if __name__ == "__main__":
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    train_arima(df)