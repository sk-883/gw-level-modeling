import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
from src.models.utils import train_test_split_time_series
from src.config import PROCESSED_DATA_DIR

def build_and_fit_bayesian_model(df: pd.DataFrame):
    df = df.dropna()
    train, test = train_test_split_time_series(df)

    X_train = train[["lag_1", "lag_7", "lag_30", "elevation"]].values
    y_train = train["level"].values

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal(
            "beta", mu=0, sigma=10, shape=X_train.shape[1]
        )
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + pm.math.dot(X_train, beta)
        y_obs = pm.Normal(
            "y_obs", mu=mu, sigma=sigma, observed=y_train
        )
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    # posterior predictive
    ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
    pred_mean = ppc["y_obs"].mean(axis=0)

    # save trace and preds
    az.to_netcdf(
        trace,
        PROCESSED_DATA_DIR / "bayesian_trace.nc"
    )
    np.save(
        PROCESSED_DATA_DIR / "bayesian_pred.npy",
        pred_mean
    )
    return model, trace, pred_mean, test

if __name__ == "__main__":
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        parse_dates=["date"]
    )
    build_and_fit_bayesian_model(df)