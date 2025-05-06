import pandas as pd
import geopandas as gpd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_well_data():
    return pd.read_csv(
        RAW_DATA_DIR / "well_levels.csv",
        parse_dates=["date"]
    )

def load_raw_geo_data():
    return gpd.read_file(RAW_DATA_DIR / "wells.geojson")

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["level"])
    # add any further cleaning steps here
    return df

def merge_geospatial(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    merged = gdf.merge(df, on="well_id")
    return merged

def save_processed(merged: gpd.GeoDataFrame):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # save as GeoJSON
    merged.to_file(
        PROCESSED_DATA_DIR / "wells_processed.geojson",
        driver="GeoJSON"
    )
    # also save flat CSV for modeling
    merged.drop(columns="geometry").to_csv(
        PROCESSED_DATA_DIR / "well_levels_processed.csv",
        index=False
    )

if __name__ == "__main__":
    df = load_raw_well_data()
    gdf = load_raw_geo_data()
    df = clean(df)
    merged = merge_geospatial(df, gdf)
    save_processed(merged)