"""Drift monitoring: detect data/model drift between batches."""
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("nyc_taxi")


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-8
    ref_counts, bin_edges = np.histogram(reference, bins=n_bins)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pcts = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pcts = cur_counts / max(cur_counts.sum(), 1) + eps

    return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple:
    """Kolmogorov-Smirnov test."""
    from scipy import stats
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


class DriftMonitor:
    """Monitor data drift between reference and current batches."""

    def __init__(self, config: dict):
        self.cfg = config.get("monitoring", {})
        self.psi_threshold = self.cfg.get("psi_threshold", 0.2)
        self.ks_threshold = self.cfg.get("ks_threshold", 0.05)
        self.drift_method = self.cfg.get("drift_method", "psi")
        self.results = {}

    def check_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
                    columns: list[str] = None) -> dict:
        """Check drift for specified columns between reference and current data."""
        if columns is None:
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
            columns = [c for c in numeric_cols if c in current_df.columns]

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "method": self.drift_method,
            "columns": {},
            "overall_drift_detected": False,
            "retrain_recommended": False,
        }

        drift_count = 0
        for col in columns:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            col_result = {}
            if self.drift_method == "psi":
                psi_val = psi(ref_vals, cur_vals)
                drifted = psi_val > self.psi_threshold
                col_result = {
                    "psi": round(psi_val, 4),
                    "threshold": self.psi_threshold,
                    "drift_detected": drifted,
                }
            else:
                ks_stat, p_val = ks_test(ref_vals, cur_vals)
                drifted = p_val < self.ks_threshold
                col_result = {
                    "ks_statistic": round(ks_stat, 4),
                    "p_value": round(p_val, 4),
                    "threshold": self.ks_threshold,
                    "drift_detected": drifted,
                }

            col_result["ref_mean"] = round(float(ref_vals.mean()), 4)
            col_result["cur_mean"] = round(float(cur_vals.mean()), 4)
            col_result["ref_std"] = round(float(ref_vals.std()), 4)
            col_result["cur_std"] = round(float(cur_vals.std()), 4)

            self.results["columns"][col] = col_result
            if drifted:
                drift_count += 1

        self.results["drifted_columns"] = drift_count
        self.results["total_columns"] = len(columns)
        self.results["overall_drift_detected"] = drift_count > 0
        self.results["retrain_recommended"] = (
            drift_count / max(len(columns), 1) > 0.3
        )

        logger.info(f"Drift check: {drift_count}/{len(columns)} columns drifted. "
                     f"Retrain recommended: {self.results['retrain_recommended']}")
        return self.results

    def simulate_drift(self, df: pd.DataFrame, drift_fraction: float = 0.2) -> pd.DataFrame:
        """Simulate a 'later batch' with drift for testing."""
        n = len(df)
        split = int(n * (1 - drift_fraction))
        current = df.iloc[split:].copy()

        # Add some drift
        for col in current.select_dtypes(include=[np.number]).columns:
            if col in ("zone_id", "cluster"):
                continue
            noise_scale = current[col].std() * 0.3
            current[col] = current[col] + np.random.normal(0, max(noise_scale, 0.1), len(current))

        logger.info(f"Simulated drift batch: {len(current)} rows from last {drift_fraction:.0%}")
        return current

    def save_report(self, output_path: str):
        """Save drift report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_path.replace(".md", ".json") if output_path.endswith(".md") else output_path
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Markdown report
        md_path = output_path if output_path.endswith(".md") else output_path.replace(".json", ".md")
        lines = [
            "# Drift Monitoring Report",
            f"\n**Timestamp:** {self.results.get('timestamp', 'N/A')}",
            f"**Method:** {self.results.get('method', 'N/A')}",
            f"**Drifted columns:** {self.results.get('drifted_columns', 0)}/{self.results.get('total_columns', 0)}",
            f"**Overall drift:** {'⚠️ YES' if self.results.get('overall_drift_detected') else '✅ NO'}",
            f"**Retrain recommended:** {'⚠️ YES' if self.results.get('retrain_recommended') else '✅ NO'}",
            "\n## Column Details\n",
            "| Column | Drift? | Ref Mean | Cur Mean | Statistic |",
            "|--------|--------|----------|----------|-----------|",
        ]

        for col, info in self.results.get("columns", {}).items():
            drift_flag = "⚠️" if info.get("drift_detected") else "✅"
            stat = info.get("psi", info.get("ks_statistic", "N/A"))
            lines.append(
                f"| {col} | {drift_flag} | {info.get('ref_mean', 'N/A')} | "
                f"{info.get('cur_mean', 'N/A')} | {stat} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Drift report saved to {md_path}")
