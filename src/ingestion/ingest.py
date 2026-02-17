"""Data ingestion: Kaggle download or synthetic fallback."""
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("nyc_taxi")

SCHEMA_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]


def try_kaggle_download(dataset: str, dest_dir: str) -> str | None:
    """Attempt to download from Kaggle. Returns CSV path or None."""
    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=dest_dir, unzip=True)
        # Find the downloaded CSV
        csvs = list(Path(dest_dir).glob("*.csv"))
        if csvs:
            logger.info(f"Kaggle download successful: {csvs[0]}")
            return str(csvs[0])
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
    return None


def generate_synthetic_dataset(
    n_rows: int = 200_000,
    start_date: str = "2023-01-01",
    end_date: str = "2023-06-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic NYC taxi-like dataset."""
    logger.info(f"Generating synthetic dataset: {n_rows} rows, {start_date} to {end_date}")
    rng = np.random.default_rng(seed)

    date_range_seconds = (
        pd.Timestamp(end_date) - pd.Timestamp(start_date)
    ).total_seconds()

    pickup_offsets = rng.uniform(0, date_range_seconds, size=n_rows)
    pickups = pd.Timestamp(start_date) + pd.to_timedelta(pickup_offsets, unit="s")
    pickups = pickups.round("s")

    # Duration in seconds (1 min to 2 hours, with realistic distribution)
    durations = rng.lognormal(mean=6.5, sigma=0.8, size=n_rows).clip(60, 7200)
    dropoffs = pickups + pd.to_timedelta(durations, unit="s")

    distances = rng.lognormal(mean=1.2, sigma=0.9, size=n_rows).clip(0.1, 100).round(2)
    fares = (2.50 + distances * 2.50 + durations / 60 * 0.50).round(2)

    # Inject some noise / bad data for quality testing
    passenger_counts = rng.choice([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 9], size=n_rows)
    vendor_ids = rng.choice([1, 2], size=n_rows)
    pu_locations = rng.integers(1, 264, size=n_rows)
    do_locations = rng.integers(1, 264, size=n_rows)
    payment_types = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.6, 0.3, 0.05, 0.05])
    ratecodes = rng.choice([1, 2, 3, 4, 5, 6], size=n_rows, p=[0.85, 0.05, 0.03, 0.03, 0.02, 0.02])

    tips = np.where(payment_types == 1, (fares * rng.uniform(0, 0.3, n_rows)).round(2), 0.0)
    extras = rng.choice([0.0, 0.5, 1.0, 2.5], size=n_rows)
    mta_tax = np.full(n_rows, 0.5)
    tolls = np.where(rng.random(n_rows) < 0.1, rng.uniform(2, 15, n_rows).round(2), 0.0)
    surcharge = np.full(n_rows, 0.3)
    totals = (fares + tips + extras + mta_tax + tolls + surcharge).round(2)

    df = pd.DataFrame({
        "VendorID": vendor_ids,
        "tpep_pickup_datetime": pickups,
        "tpep_dropoff_datetime": dropoffs,
        "passenger_count": passenger_counts,
        "trip_distance": distances,
        "RatecodeID": ratecodes,
        "store_and_fwd_flag": rng.choice(["Y", "N"], size=n_rows, p=[0.05, 0.95]),
        "PULocationID": pu_locations,
        "DOLocationID": do_locations,
        "payment_type": payment_types,
        "fare_amount": fares,
        "extra": extras,
        "mta_tax": mta_tax,
        "tip_amount": tips,
        "tolls_amount": tolls,
        "improvement_surcharge": surcharge,
        "total_amount": totals,
    })

    # Inject specific quality issues for testing repair logic
    n_bad = int(n_rows * 0.02)
    bad_idx = rng.choice(n_rows, size=n_bad, replace=False)

    # Swap pickup/dropoff for some rows
    swap_idx = bad_idx[: n_bad // 5]
    df.loc[swap_idx, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = (
        df.loc[swap_idx, ["tpep_dropoff_datetime", "tpep_pickup_datetime"]].values
    )

    # Negative distances
    neg_idx = bad_idx[n_bad // 5: 2 * n_bad // 5]
    df.loc[neg_idx, "trip_distance"] = -df.loc[neg_idx, "trip_distance"]

    # Negative fares
    neg_fare_idx = bad_idx[2 * n_bad // 5: 3 * n_bad // 5]
    df.loc[neg_fare_idx, "fare_amount"] = -rng.uniform(1, 50, len(neg_fare_idx)).round(2)

    # Zero duration with positive distance
    zero_dur_idx = bad_idx[3 * n_bad // 5: 4 * n_bad // 5]
    df.loc[zero_dur_idx, "tpep_dropoff_datetime"] = df.loc[zero_dur_idx, "tpep_pickup_datetime"]

    # Insert some nulls
    null_idx = bad_idx[4 * n_bad // 5:]
    df.loc[null_idx, "passenger_count"] = np.nan

    logger.info(f"Synthetic dataset generated: {len(df)} rows with ~{n_bad} injected issues")
    return df


def ingest(config: dict) -> pd.DataFrame:
    """Main ingestion entry point. Try Kaggle, fallback to synthetic."""
    pipeline_cfg = config
    max_rows = pipeline_cfg.get("pipeline", {}).get("max_rows", 200_000)
    date_start = pipeline_cfg.get("pipeline", {}).get("date_range", {}).get("start", "2023-01-01")
    date_end = pipeline_cfg.get("pipeline", {}).get("date_range", {}).get("end", "2023-06-30")

    raw_dir = str(Path(__file__).resolve().parents[2] / "data" / "raw")
    os.makedirs(raw_dir, exist_ok=True)

    dataset_name = pipeline_cfg.get("ingestion", {}).get(
        "kaggle_dataset", "elemento/nyc-yellow-taxi-trip-data"
    )

    # Try Kaggle first
    csv_path = try_kaggle_download(dataset_name, raw_dir)
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        logger.info(f"Loaded {len(df)} rows from Kaggle")
        return df

    # Fallback to synthetic
    df = generate_synthetic_dataset(
        n_rows=min(max_rows, 200_000),
        start_date=date_start,
        end_date=date_end,
    )
    return df
