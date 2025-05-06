# Groundwater Level Prediction

This repository contains code and resources for predicting groundwater levels using a variety of statistical and machine learning models. The goal is to benchmark **Bayesian Linear Regression**, **ARIMA/SARIMA**, **Random Forest**, and **Gaussian Process Regression** on time-series and geospatial features, providing both point forecasts and uncertainty estimates.

##  Features

* **Data Preprocessing**: Load raw CSV/GeoJSON files, clean missing values, and merge geospatial data using GeoPandas.
* **Feature Engineering**: Create temporal lag features, rolling statistics, and spatial covariates (elevation, soil type).
* **Modeling Suite**:

  * **Bayesian Regression** (via PyMC3 or CmdStanPy)
  * **ARIMA/SARIMA** (via statsmodels)
  * **Random Forest** (via Scikit-learn)
  * **Gaussian Process** Regression (via Scikit-learn)
* **Evaluation**: Compute RMSE, MAE, R², log-likelihood, and compile a comparative metrics table.
* **Visualization**: Plot actual vs. predicted series with uncertainty bands and map forecast errors on GIS maps.

##  Repository Structure

```
groundwater-prediction/
├── data/
│   ├── raw/                   # Original downloaded CSV/GeoJSON files
│   └── processed/             # Cleaned & merged datasets ready for modeling
│
├── notebooks/                 # Exploratory analysis & prototype models
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_prototyping.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Paths, parameters (e.g. date ranges, hyperparams)
│   ├── data_preprocessing.py  # load_raw(), clean(), merge_geospatial(), save_processed()
│   ├── feature_engineering.py # create_lag_features(), add_geospatial_columns()
│   ├── models/
│   │   ├── bayesian_regression.py   # build_and_fit_bayesian_model()
│   │   ├── arima_model.py           # train_arima(), forecast_arima()
│   │   ├── random_forest.py         # train_rf(), predict_rf()
│   │   ├── gaussian_process.py      # train_gp(), predict_gp()
│   │   └── utils.py                 # train_test_split_time_series(), common utilities
│   ├── evaluation.py          # compute_metrics(), compare_models()
│   └── visualize.py           # plot_time_series(), plot_prediction_intervals(), map_uncertainty()
│
├── requirements.txt           # pandas, geopandas, pymc3, statsmodels, scikit-learn, matplotlib
├── setup.py                   # Optional: package installation
└── README.md                  # This file
```

##  Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/groundwater-prediction.git
   cd groundwater-prediction
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:

   * Place raw CSV/GeoJSON files in `data/raw/`.
   * Run preprocessing to generate cleaned data:

     ```bash
     python -m src.data_preprocessing
     ```

##  Running the Pipeline

1. **Feature Engineering**:

   ```bash
   python -m src.feature_engineering
   ```

2. **Train & Evaluate Models**:

   ```bash
   python -m src.models.bayesian_regression
   python -m src.models.arima_model
   python -m src.models.random_forest
   python -m src.models.gaussian_process
   python -m src.evaluation
   ```

3. **Generate Visualizations**:

   ```bash
   python -m src.visualize
   ```

## 🧪 Notebooks

Use the Jupyter notebooks in `notebooks/` for interactive exploration and prototyping. They demonstrate data exploration, feature creation, and quick model experiments.

##  Results & Interpretation

See the `evaluation.py` output for a comparative table of model performance metrics. Use GIS maps and time-series plots to analyze spatial and temporal forecast accuracy.

##  Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, new models, or improvements.

##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Developed by **sk_883** *
