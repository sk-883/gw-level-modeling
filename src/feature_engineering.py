import pandas as pd
import geopandas as gpd
from src.config import PROCESSED_DATA_DIR


def create_lag_features(df: pd.DataFrame, lags=[1, 7, 30]) -> pd.DataFrame:
    df = df.sort_values(["well_id", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("well_id")["level"].shift(lag)
    return df


def add_geospatial_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Example: extract elevation if stored in Z dimension
    if gdf.geometry.has_z:
        gdf["elevation"] = gdf.geometry.z
    else:
        # else assign dummy or drop
        gdf["elevation"] = 0
    return gdf


def main():
    # load the processed files
    gdf = gpd.read_file(
        PROCESSED_DATA_DIR / "wells_processed.geojson"
    )
    df = pd.DataFrame(gdf.drop(columns="geometry"))

    df_lag = create_lag_features(df)
    gdf_lag = gdf.merge(
        df_lag.drop(columns=["well_id", "date"]),
        left_index=True,
        right_index=True
    )
    gdf_feat = add_geospatial_columns(gdf_lag)

    # save back
    gdf_feat.to_file(
        PROCESSED_DATA_DIR / "wells_features.geojson",
        driver="GeoJSON"
    )
    gdf_feat.drop(columns="geometry").to_csv(
        PROCESSED_DATA_DIR / "well_levels_features.csv",
        index=False
    )

if __name__ == "__main__":
    main()