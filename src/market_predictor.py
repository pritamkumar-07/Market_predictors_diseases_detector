from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "PreviousPrice",
    "Quantity",
    "Month",
    "Year",
    "CropEncoded",
    "MarketEncoded",
    "SeasonEncoded",
]


def load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def build_encoders(df: pd.DataFrame):
    crop_map = {v: i for i, v in enumerate(sorted(df["Crop"].unique()))}
    market_map = {v: i for i, v in enumerate(sorted(df["Market"].unique()))}
    season_map = {v: i for i, v in enumerate(sorted(df["Season"].unique()))}
    return crop_map, market_map, season_map


def build_market_features(
    df: pd.DataFrame,
    crop_map=None,
    market_map=None,
    season_map=None,
) -> pd.DataFrame:
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"])

    if crop_map is None:
        crop_map = {v: i for i, v in enumerate(sorted(x["Crop"].unique()))}
    if market_map is None:
        market_map = {v: i for i, v in enumerate(sorted(x["Market"].unique()))}
    if season_map is None:
        season_map = {v: i for i, v in enumerate(sorted(x["Season"].unique()))}

    x["Month"] = x["Date"].dt.month
    x["Year"] = x["Date"].dt.year
    x["CropEncoded"] = x["Crop"].map(crop_map).fillna(-1)
    x["MarketEncoded"] = x["Market"].map(market_map).fillna(-1)
    x["SeasonEncoded"] = x["Season"].map(season_map).fillna(-1)

    return x[FEATURE_COLUMNS].astype(float)


def predict_market_price(model, X: pd.DataFrame) -> float:
    return float(model.predict(X)[0])


def market_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[["Price", "Quantity"]]
    result = numeric.describe().T
    result["median"] = numeric.median()
    result["variance"] = numeric.var()
    result["std_dev"] = numeric.std()
    return result[["count", "mean", "median", "std_dev", "variance", "min", "max"]]
