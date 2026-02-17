"""Targeted repair: ActiveClean-inspired data repair with budget control."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nyc_taxi")


class TargetedRepairer:
    """Apply targeted repairs to taxi trip data with logging."""

    def __init__(self, config: dict):
        self.cfg = config.get("repair", {})
        self.budget = self.cfg.get("targeted_repair_budget", 0.3)
        self.quality_cfg = config.get("quality", {})
        self.repair_log = []

    def _log_repair(self, rule_name: str, affected_count: int, columns: list,
                    example_before: dict = None, example_after: dict = None):
        self.repair_log.append({
            "rule": rule_name,
            "affected_rows": int(affected_count),
            "columns_touched": columns,
            "example_before": example_before or {},
            "example_after": example_after or {},
        })

    def repair_datetime_swap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix pickup >= dropoff by swapping."""
        mask = df["tpep_pickup_datetime"] >= df["tpep_dropoff_datetime"]
        count = int(mask.sum())
        if count > 0:
            example_before = df.loc[mask, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]].iloc[0].to_dict()
            example_before = {k: str(v) for k, v in example_before.items()}

            df.loc[mask, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = (
                df.loc[mask, ["tpep_dropoff_datetime", "tpep_pickup_datetime"]].values
            )
            # For zero-duration, add 5 minutes
            still_equal = df["tpep_pickup_datetime"] == df["tpep_dropoff_datetime"]
            df.loc[still_equal, "tpep_dropoff_datetime"] += pd.Timedelta(minutes=5)

            example_after = df.loc[mask, ["tpep_pickup_datetime", "tpep_dropoff_datetime"]].iloc[0].to_dict()
            example_after = {k: str(v) for k, v in example_after.items()}

            self._log_repair("swap_datetime", count,
                             ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                             example_before, example_after)
        logger.info(f"Repair swap_datetime: {count} rows fixed")
        return df

    def repair_passenger_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clamp passenger_count to [1, 6], impute nulls with median."""
        min_p = self.quality_cfg.get("min_passenger", 1)
        max_p = self.quality_cfg.get("max_passenger", 6)

        invalid = df["passenger_count"].isna() | (df["passenger_count"] < min_p) | (df["passenger_count"] > max_p)
        count = int(invalid.sum())
        if count > 0:
            example_before = {"passenger_count": str(df.loc[invalid, "passenger_count"].iloc[0])}
            median_val = df.loc[~invalid, "passenger_count"].median()
            if pd.isna(median_val):
                median_val = 1
            df.loc[invalid, "passenger_count"] = int(median_val)
            example_after = {"passenger_count": str(int(median_val))}
            self._log_repair("clamp_passenger", count, ["passenger_count"],
                             example_before, example_after)
        logger.info(f"Repair clamp_passenger: {count} rows fixed")
        return df

    def repair_negative_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix negative trip distances by taking absolute value."""
        mask = df["trip_distance"] < 0
        count = int(mask.sum())
        if count > 0:
            example_before = {"trip_distance": float(df.loc[mask, "trip_distance"].iloc[0])}
            df.loc[mask, "trip_distance"] = df.loc[mask, "trip_distance"].abs()
            example_after = {"trip_distance": float(df.loc[mask, "trip_distance"].iloc[0])}
            self._log_repair("abs_distance", count, ["trip_distance"],
                             example_before, example_after)
        logger.info(f"Repair abs_distance: {count} rows fixed")
        return df

    def repair_fare_winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize fare_amount: fix negatives and cap outliers."""
        # Fix negatives
        neg_mask = df["fare_amount"] < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            example_before = {"fare_amount": float(df.loc[neg_mask, "fare_amount"].iloc[0])}
            df.loc[neg_mask, "fare_amount"] = df.loc[neg_mask, "fare_amount"].abs()
            example_after = {"fare_amount": float(df.loc[neg_mask, "fare_amount"].iloc[0])}
            self._log_repair("fix_negative_fare", neg_count, ["fare_amount"],
                             example_before, example_after)

        # Winsorize upper outliers
        q1 = df["fare_amount"].quantile(0.25)
        q3 = df["fare_amount"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        outlier_mask = df["fare_amount"] > upper
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            example_before = {"fare_amount": float(df.loc[outlier_mask, "fare_amount"].iloc[0])}
            df.loc[outlier_mask, "fare_amount"] = upper
            example_after = {"fare_amount": float(upper)}
            self._log_repair("winsorize_fare", outlier_count, ["fare_amount"],
                             example_before, example_after)

        # Also fix total_amount negatives
        neg_total = df["total_amount"] < 0
        nt_count = int(neg_total.sum())
        if nt_count > 0:
            df.loc[neg_total, "total_amount"] = df.loc[neg_total, "total_amount"].abs()
            self._log_repair("fix_negative_total", nt_count, ["total_amount"])

        logger.info(f"Repair fare: {neg_count} negatives, {outlier_count} outliers winsorized")
        return df

    def repair_zero_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with zero duration but positive distance."""
        duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
        mask = (duration <= 0) & (df["trip_distance"] > 0)
        count = int(mask.sum())
        if count > 0:
            self._log_repair("drop_zero_duration", count,
                             ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"])
            df = df[~mask].reset_index(drop=True)
        logger.info(f"Repair drop_zero_duration: {count} rows dropped")
        return df

    def apply_all_repairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all targeted repairs within budget."""
        self.repair_log = []
        original_len = len(df)

        df = self.repair_datetime_swap(df)
        df = self.repair_passenger_count(df)
        df = self.repair_negative_distance(df)
        df = self.repair_fare_winsorize(df)
        df = self.repair_zero_duration(df)

        total_affected = sum(r["affected_rows"] for r in self.repair_log)
        logger.info(f"Total repairs: {total_affected} row-fixes across {len(self.repair_log)} rules. "
                     f"Rows: {original_len} -> {len(df)}")
        return df

    def save_repair_log(self, output_path: str):
        """Save repair log to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.repair_log, f, indent=2, default=str)
        logger.info(f"Repair log saved to {output_path}")
