from setuptools import setup, find_packages

setup(
    name="groundwater_prediction",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "geopandas",
        "numpy",
        "scikit-learn",
        "statsmodels",
        "pymc3",
        "arviz",
        "matplotlib",
        "joblib",
    ],
    entry_points={
        "console_scripts": [
            "gw-preprocess=src.data_preprocessing:main",
            "gw-feature=src.feature_engineering:main",
            "gw-evaluate=src.evaluation:main",
        ],
    },
)