import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def train_gp(df: pd.DataFrame):
    df = df.dropna()
    train, test = train_test_split_time_series(df)
    features = ["lag_1", "lag_7", "lag_30", "elevation"]
    X_train = train[features].values
    y_train = train["level"].values

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10
    )
    gp.fit(X_train, y_train)

    X_test = test[features].values
    preds, std = gp.predict(X_test, return_std=True)

    joblib.dump(
        gp,
        PROCESSED_DATA_DIR / "gp_model.pkl"
    )
    np.save(
        PROCESSED_DATA_DIR / "gp_preds.npy",
        preds
    )
    np.save(
        PROCESSED_DATA_DIR / "gp_std.npy",
        std
    )
    return gp, preds, std, test

if __name__ == "__main__":
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    train_gp(df)