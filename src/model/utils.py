import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_time_series(
    df: pd.DataFrame,
    date_col: str = "date",
    test_size: float = 0.2
):
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test