import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def train_rf(df: pd.DataFrame):
    df = df.dropna()
    train, test = train_test_split_time_series(df)
    features = ["lag_1", "lag_7", "lag_30", "elevation"]

    X_train = train[features]
    y_train = train["level"]
    X_test = test[features]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    joblib.dump(
        model,
        PROCESSED_DATA_DIR / "rf_model.pkl"
    )
    np.save(
        PROCESSED_DATA_DIR / "rf_preds.npy",
        preds
    )
    return model, preds, test

if __name__ == "__main__":
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    train_rf(df)