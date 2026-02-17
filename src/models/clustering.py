"""Clustering: segment trips or OD-time aggregates."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("nyc_taxi")


def cluster_trips(df: pd.DataFrame, config: dict = None) -> tuple:
    """Cluster trips/aggregates using KMeans (+ HDBSCAN if available).

    Returns: (df_with_clusters, cluster_summary, labels)
    """
    config = config or {}
    cluster_cfg = config.get("clustering", {})
    n_clusters = cluster_cfg.get("n_clusters", 5)
    feature_names = cluster_cfg.get("features", [
        "trip_distance", "fare_amount", "duration_minutes", "hour_of_day", "day_of_week"
    ])

    # Prepare features
    available = [c for c in feature_names if c in df.columns]
    if not available:
        # Fallback features from gold demand table
        available = [c for c in ["avg_distance", "avg_fare", "hour_of_day", "day_of_week", "demand"]
                     if c in df.columns]

    if not available:
        logger.warning("No clustering features available")
        return df, {}, np.array([])

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["cluster"] = labels

    # Try HDBSCAN if available
    hdbscan_labels = None
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        hdbscan_labels = clusterer.fit_predict(X_scaled)
        df["cluster_hdbscan"] = hdbscan_labels
        logger.info(f"HDBSCAN found {len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)} clusters")
    except ImportError:
        logger.info("HDBSCAN not available, using KMeans only")

    # Cluster interpretation
    cluster_summary = {}
    for c in range(n_clusters):
        mask = labels == c
        summary = {
            "size": int(mask.sum()),
            "pct": round(mask.sum() / len(df) * 100, 1),
        }
        for feat in available:
            vals = df.loc[mask, feat]
            summary[f"mean_{feat}"] = round(float(vals.mean()), 2)
            summary[f"median_{feat}"] = round(float(vals.median()), 2)

        # Label suggestion based on features
        summary["label"] = _suggest_label(summary, available)
        cluster_summary[f"cluster_{c}"] = summary

    logger.info(f"KMeans clustering: {n_clusters} clusters on {len(available)} features")
    return df, cluster_summary, labels


def _suggest_label(summary: dict, features: list) -> str:
    """Suggest a human-readable label for a cluster."""
    labels = []

    if "mean_avg_distance" in summary or "mean_trip_distance" in summary:
        dist_key = "mean_avg_distance" if "mean_avg_distance" in summary else "mean_trip_distance"
        if summary[dist_key] > 10:
            labels.append("Long-haul")
        elif summary[dist_key] < 2:
            labels.append("Short-hop")
        else:
            labels.append("Medium-range")

    if "mean_hour_of_day" in summary:
        hour = summary["mean_hour_of_day"]
        if 6 <= hour < 10:
            labels.append("Morning-rush")
        elif 16 <= hour < 20:
            labels.append("Evening-rush")
        elif 22 <= hour or hour < 6:
            labels.append("Late-night")
        else:
            labels.append("Midday")

    if "mean_demand" in summary:
        if summary["mean_demand"] > 50:
            labels.append("High-demand")
        elif summary["mean_demand"] < 5:
            labels.append("Low-demand")

    return " / ".join(labels) if labels else f"Segment (n={summary['size']})"


def save_cluster_report(cluster_summary: dict, output_path: str):
    """Save cluster interpretation report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Clustering Report",
        f"\n## Cluster Summary ({len(cluster_summary)} clusters)\n",
    ]

    for name, info in cluster_summary.items():
        lines.append(f"### {name}: {info.get('label', 'Unknown')}")
        lines.append(f"- **Size:** {info['size']} ({info['pct']}%)")
        for k, v in info.items():
            if k not in ("size", "pct", "label"):
                lines.append(f"- **{k}:** {v}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    # Also save JSON
    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(cluster_summary, f, indent=2)

    logger.info(f"Cluster report saved to {output_path}")
