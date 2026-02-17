"""Feature engineering: build gold demand features."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nyc_taxi")


def build_demand_zone_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Build demand aggregation per zone per hour with features.

    Args:
        df: Silver-quality taxi trip data with pickup datetime and location.

    Returns:
        Gold demand DataFrame with lag/rolling features.
    """
    logger.info("Building gold demand features (zone-hour aggregation)")

    # Ensure datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # Extract time components
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    df["hour_of_day"] = df["tpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["tpep_pickup_datetime"].dt.month

    # Aggregate demand per zone per hour
    demand = (
        df.groupby(["PULocationID", "pickup_hour"])
        .agg(
            demand=("VendorID", "count"),
            avg_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_duration_min=("trip_distance", lambda x: len(x)),  # placeholder
        )
        .reset_index()
    )

    # Compute actual avg duration if possible
    if "tpep_dropoff_datetime" in df.columns:
        df["duration_minutes"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        )
        dur_agg = (
            df.groupby(["PULocationID", "pickup_hour"])["duration_minutes"]
            .mean()
            .reset_index()
        )
        demand = demand.merge(dur_agg, on=["PULocationID", "pickup_hour"], how="left")
        demand.drop("avg_duration_min", axis=1, inplace=True)
        demand.rename(columns={"duration_minutes": "avg_duration_min"}, inplace=True)

    # Time features
    demand["hour_of_day"] = demand["pickup_hour"].dt.hour
    demand["day_of_week"] = demand["pickup_hour"].dt.dayofweek
    demand["is_weekend"] = (demand["day_of_week"] >= 5).astype(int)
    demand["month"] = demand["pickup_hour"].dt.month
    demand["hour_of_week"] = demand["day_of_week"] * 24 + demand["hour_of_day"]

    # Sort for lag features
    demand = demand.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)

    # Lag features per zone
    for lag in [1, 6, 24]:
        demand[f"demand_lag_{lag}"] = demand.groupby("PULocationID")["demand"].shift(lag)

    # Rolling features per zone
    for window in [6, 24]:
        demand[f"demand_roll_mean_{window}"] = (
            demand.groupby("PULocationID")["demand"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        demand[f"demand_roll_std_{window}"] = (
            demand.groupby("PULocationID")["demand"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Fill NaN lags with 0
    demand = demand.fillna(0)

    # Rename for clarity
    demand.rename(columns={"PULocationID": "zone_id"}, inplace=True)

    logger.info(f"Gold demand features built: {len(demand)} rows, "
                f"{len(demand.columns)} columns, "
                f"{demand['zone_id'].nunique()} zones")
    return demand


def get_feature_columns() -> list[str]:
    """Return the feature columns for modeling."""
    return [
        "hour_of_day", "day_of_week", "is_weekend", "month", "hour_of_week",
        "avg_distance", "avg_fare", "avg_duration_min",
        "demand_lag_1", "demand_lag_6", "demand_lag_24",
        "demand_roll_mean_6", "demand_roll_mean_24",
        "demand_roll_std_6", "demand_roll_std_24",
    ]
