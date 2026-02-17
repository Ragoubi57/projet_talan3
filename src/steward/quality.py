"""Data quality checks and validation (AI Steward)."""
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("nyc_taxi")


class DataQualityChecker:
    """Run quality checks on taxi trip data and produce reports."""

    def __init__(self, config: dict):
        self.cfg = config.get("quality", {})
        self.null_threshold = self.cfg.get("null_threshold", 0.05)
        self.iqr_factor = self.cfg.get("outlier_iqr_factor", 3.0)
        self.min_passenger = self.cfg.get("min_passenger", 1)
        self.max_passenger = self.cfg.get("max_passenger", 6)
        self.max_fare = self.cfg.get("max_fare", 500.0)
        self.max_distance = self.cfg.get("max_distance", 200.0)
        self.max_duration_hours = self.cfg.get("max_duration_hours", 12)
        self.results = []

    def _add_result(self, check_name: str, passed: bool, failing_count: int,
                    total: int, details: str = ""):
        self.results.append({
            "check": check_name,
            "passed": passed,
            "failing_rows": int(failing_count),
            "total_rows": int(total),
            "fail_pct": round(failing_count / max(total, 1) * 100, 2),
            "details": details,
        })

    def check_schema(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist."""
        required = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "fare_amount", "total_amount",
            "PULocationID", "DOLocationID",
        ]
        missing = [c for c in required if c not in df.columns]
        self._add_result("schema_validation", len(missing) == 0, len(missing),
                         len(required), f"Missing: {missing}" if missing else "All required columns present")
        return len(missing) == 0

    def check_nulls(self, df: pd.DataFrame):
        """Check null percentages per column."""
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = null_count / len(df)
            passed = null_pct <= self.null_threshold
            self._add_result(f"null_check_{col}", passed, null_count, len(df),
                             f"{null_pct:.2%} nulls (threshold: {self.null_threshold:.0%})")

    def check_temporal_sanity(self, df: pd.DataFrame):
        """Pickup must be before dropoff."""
        if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
            mask = df["tpep_pickup_datetime"] >= df["tpep_dropoff_datetime"]
            failing = int(mask.sum())
            self._add_result("temporal_sanity", failing == 0, failing, len(df),
                             "pickup_datetime must be < dropoff_datetime")

    def check_ranges(self, df: pd.DataFrame):
        """Check value ranges for key columns."""
        # Passenger count
        if "passenger_count" in df.columns:
            valid = df["passenger_count"].dropna()
            bad = ((valid < self.min_passenger) | (valid > self.max_passenger)).sum()
            self._add_result("passenger_count_range", int(bad) == 0, int(bad), len(df),
                             f"Expected [{self.min_passenger}, {self.max_passenger}]")

        # Trip distance
        if "trip_distance" in df.columns:
            bad = (df["trip_distance"] < 0).sum()
            self._add_result("trip_distance_non_negative", int(bad) == 0, int(bad), len(df))

            extreme = (df["trip_distance"] > self.max_distance).sum()
            self._add_result("trip_distance_max", int(extreme) == 0, int(extreme), len(df),
                             f"Max allowed: {self.max_distance}")

        # Fare
        if "fare_amount" in df.columns:
            bad = (df["fare_amount"] < 0).sum()
            self._add_result("fare_non_negative", int(bad) == 0, int(bad), len(df))

            extreme = (df["fare_amount"] > self.max_fare).sum()
            self._add_result("fare_max", int(extreme) == 0, int(extreme), len(df),
                             f"Max allowed: {self.max_fare}")

        # Total amount
        if "total_amount" in df.columns:
            bad = (df["total_amount"] < 0).sum()
            self._add_result("total_amount_non_negative", int(bad) == 0, int(bad), len(df))

    def check_duration_distance_sanity(self, df: pd.DataFrame):
        """Duration=0 with distance>0 or vice versa."""
        if all(c in df.columns for c in ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"]):
            duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
            zero_dur_pos_dist = ((duration == 0) & (df["trip_distance"] > 0)).sum()
            self._add_result("zero_duration_positive_distance", int(zero_dur_pos_dist) == 0,
                             int(zero_dur_pos_dist), len(df))

    def check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate rows."""
        dups = df.duplicated().sum()
        self._add_result("no_duplicates", int(dups) == 0, int(dups), len(df))

    def check_outliers_iqr(self, df: pd.DataFrame, col: str):
        """Detect extreme outliers via IQR."""
        if col not in df.columns:
            return
        vals = df[col].dropna()
        if len(vals) == 0:
            return
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        outliers = ((vals < lower) | (vals > upper)).sum()
        self._add_result(f"iqr_outlier_{col}", int(outliers) == 0, int(outliers), len(df),
                         f"IQR bounds: [{lower:.2f}, {upper:.2f}]")

    def run_all_checks(self, df: pd.DataFrame) -> list[dict]:
        """Run all quality checks and return results."""
        self.results = []
        self.check_schema(df)
        self.check_nulls(df)
        self.check_temporal_sanity(df)
        self.check_ranges(df)
        self.check_duration_distance_sanity(df)
        self.check_duplicates(df)
        for col in ["fare_amount", "trip_distance", "total_amount"]:
            self.check_outliers_iqr(df, col)

        total_checks = len(self.results)
        failed = sum(1 for r in self.results if not r["passed"])
        total_failing_rows = sum(r["failing_rows"] for r in self.results if not r["passed"])
        logger.info(f"Quality checks: {total_checks - failed}/{total_checks} passed, "
                     f"{total_failing_rows} total failing row-checks")
        return self.results

    def failing_row_count(self, df: pd.DataFrame) -> int:
        """Count unique rows failing at least one check."""
        failing_mask = pd.Series(False, index=df.index)

        if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
            failing_mask |= df["tpep_pickup_datetime"] >= df["tpep_dropoff_datetime"]

        if "passenger_count" in df.columns:
            pc = df["passenger_count"]
            failing_mask |= pc.isna() | (pc < self.min_passenger) | (pc > self.max_passenger)

        if "trip_distance" in df.columns:
            failing_mask |= df["trip_distance"] < 0

        if "fare_amount" in df.columns:
            failing_mask |= df["fare_amount"] < 0

        if "total_amount" in df.columns:
            failing_mask |= df["total_amount"] < 0

        if all(c in df.columns for c in ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"]):
            duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
            failing_mask |= (duration == 0) & (df["trip_distance"] > 0)

        return int(failing_mask.sum())

    def generate_report(self, output_path: str):
        """Generate a human-readable quality report."""
        report_lines = [
            "# Data Quality Report",
            f"\n**Generated:** {datetime.now().isoformat()}",
            f"\n**Total checks:** {len(self.results)}",
            f"**Passed:** {sum(1 for r in self.results if r['passed'])}",
            f"**Failed:** {sum(1 for r in self.results if not r['passed'])}",
            "\n## Check Details\n",
            "| Check | Status | Failing Rows | Fail % | Details |",
            "|-------|--------|-------------|--------|---------|",
        ]
        for r in self.results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            report_lines.append(
                f"| {r['check']} | {status} | {r['failing_rows']:,} | {r['fail_pct']}% | {r['details']} |"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))
        logger.info(f"Quality report written to {output_path}")


def compute_quality_pct(df: pd.DataFrame, config: dict) -> float:
    """Compute percentage of rows failing checks."""
    checker = DataQualityChecker(config)
    failing = checker.failing_row_count(df)
    return round(failing / max(len(df), 1) * 100, 2)
