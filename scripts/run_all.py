#!/usr/bin/env python3
"""Main pipeline orchestrator: runs the full NYC Taxi AI data value chain."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import get_config, get_storage_config, setup_logging, ensure_dir


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("NYC Taxi AI Data Value Chain – Pipeline Start")
    logger.info("=" * 60)

    config = get_config("pipeline")
    storage_cfg = get_storage_config()
    reports_dir = str(PROJECT_ROOT / "reports")
    shared_dir = os.environ.get("SHARED_DATA_DIR", str(PROJECT_ROOT / "shared_data"))
    ensure_dir(reports_dir)
    ensure_dir(shared_dir)

    timings = {}
    metrics_all = {}
    t_total_start = time.time()

    # ── Stage 1: Ingest → Bronze ──
    try:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1: Data Ingestion")
        logger.info("=" * 50)
        t0 = time.time()
        from src.ingestion.ingest import ingest
        raw_df = ingest(config)
        timings["ingestion"] = round(time.time() - t0, 2)
        throughput = len(raw_df) / max(timings["ingestion"], 0.01)
        logger.info(f"Ingested {len(raw_df)} rows in {timings['ingestion']}s "
                     f"({throughput:.0f} rows/s)")
        metrics_all["ingestion_rows"] = len(raw_df)
        metrics_all["ingestion_throughput_rows_per_sec"] = round(throughput, 0)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # ── Stage 2: Write Bronze to Iceberg ──
    spark = None
    try:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2: Bronze → Iceberg")
        logger.info("=" * 50)
        t0 = time.time()
        from src.storage_iceberg.iceberg_io import get_spark_session, write_bronze
        spark = get_spark_session(storage_cfg)
        write_bronze(spark, raw_df, storage_cfg)
        timings["bronze_write"] = round(time.time() - t0, 2)
    except Exception as e:
        logger.warning(f"Iceberg bronze write failed: {e}. Continuing with in-memory data.")
        timings["bronze_write"] = 0

    # ── Stage 3: Quality checks on raw data ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 3: Data Quality Checks (Pre-Repair)")
    logger.info("=" * 50)
    t0 = time.time()
    from src.steward.quality import DataQualityChecker, compute_quality_pct
    checker = DataQualityChecker(config)
    quality_results_before = checker.run_all_checks(raw_df)
    fail_pct_before = compute_quality_pct(raw_df, config)
    checker.generate_report(os.path.join(reports_dir, "quality_report_before.md"))
    timings["quality_before"] = round(time.time() - t0, 2)
    metrics_all["fail_pct_before_repair"] = fail_pct_before
    logger.info(f"Quality fail % (before repair): {fail_pct_before}%")

    # ── Stage 4: LLM Suggestions ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 4: LLM Semantic Suggestions")
    logger.info("=" * 50)
    from src.steward.llm_suggestions import LLMSuggestionEngine
    llm_engine = LLMSuggestionEngine(config)
    llm_engine.generate_suggestions(quality_results_before)
    llm_engine.save_suggestions(
        os.path.join(reports_dir, "llm_suggestions.json"),
        os.path.join(reports_dir, "llm_suggestions.md"),
    )

    # ── Stage 5: Build Silver (validated + standardized) ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 5: Silver – Validate & Standardize")
    logger.info("=" * 50)
    t0 = time.time()
    silver_df = raw_df.copy()
    # Standardize dtypes
    silver_df["tpep_pickup_datetime"] = pd.to_datetime(silver_df["tpep_pickup_datetime"])
    silver_df["tpep_dropoff_datetime"] = pd.to_datetime(silver_df["tpep_dropoff_datetime"])
    for col in ["passenger_count", "VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type"]:
        if col in silver_df.columns:
            silver_df[col] = pd.to_numeric(silver_df[col], errors="coerce")
    timings["silver_validate"] = round(time.time() - t0, 2)

    # ── Stage 6: Build Gold features (baseline) ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 6: Gold Features (Baseline)")
    logger.info("=" * 50)
    t0 = time.time()
    from src.features.engineering import build_demand_zone_hour, get_feature_columns
    gold_df_baseline = build_demand_zone_hour(silver_df)
    timings["gold_baseline"] = round(time.time() - t0, 2)

    # Write gold to Iceberg if spark available
    if spark:
        try:
            from src.storage_iceberg.iceberg_io import write_gold, write_silver
            write_silver(spark, silver_df, storage_cfg)
            write_gold(spark, gold_df_baseline, storage_cfg)
        except Exception as e:
            logger.warning(f"Iceberg gold write failed: {e}")

    # Materialize to parquet for Python model training
    gold_parquet_path = os.path.join(shared_dir, "gold_demand.parquet")
    gold_df_baseline.to_parquet(gold_parquet_path, index=False)
    logger.info(f"Gold features materialized to {gold_parquet_path}")

    # ── Stage 7: Baseline Forecast Model ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 7: Baseline Forecast Model")
    logger.info("=" * 50)
    t0 = time.time()
    from src.models.forecast import train_forecast_model, log_to_mlflow, save_metrics_json
    feature_cols = get_feature_columns()
    model_baseline, metrics_baseline, preds_baseline = train_forecast_model(
        gold_df_baseline, feature_cols, config=config
    )
    timings["baseline_model"] = round(time.time() - t0, 2)
    metrics_all["baseline_mae"] = metrics_baseline["mae"]
    metrics_all["baseline_rmse"] = metrics_baseline["rmse"]
    metrics_all["baseline_smape"] = metrics_baseline["smape"]

    # Log to MLflow
    log_to_mlflow(model_baseline, metrics_baseline, config, stage="baseline")

    # ── Stage 8: Targeted Repair Loop ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 8: Targeted Repair Loop")
    logger.info("=" * 50)
    t0 = time.time()
    from src.steward.repair import TargetedRepairer
    repairer = TargetedRepairer(config)
    repaired_df = repairer.apply_all_repairs(silver_df.copy())
    repairer.save_repair_log(os.path.join(reports_dir, "repair_patch.json"))
    timings["repair"] = round(time.time() - t0, 2)

    # Quality after repair
    fail_pct_after = compute_quality_pct(repaired_df, config)
    metrics_all["fail_pct_after_repair"] = fail_pct_after
    logger.info(f"Quality fail % (after repair): {fail_pct_after}% (was {fail_pct_before}%)")

    # Rebuild gold from repaired data
    gold_df_repaired = build_demand_zone_hour(repaired_df)
    gold_repaired_path = os.path.join(shared_dir, "gold_demand_repaired.parquet")
    gold_df_repaired.to_parquet(gold_repaired_path, index=False)

    # Write repaired to Iceberg
    if spark:
        try:
            from src.storage_iceberg.iceberg_io import write_silver, write_gold
            write_silver(spark, repaired_df, storage_cfg)
            write_gold(spark, gold_df_repaired, storage_cfg)
        except Exception as e:
            logger.warning(f"Iceberg write (repaired) failed: {e}")

    # ── Stage 9: Retrain on repaired data ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 9: Retrain Forecast (Post-Repair)")
    logger.info("=" * 50)
    t0 = time.time()
    model_repaired, metrics_repaired, preds_repaired = train_forecast_model(
        gold_df_repaired, feature_cols, config=config
    )
    timings["retrain_model"] = round(time.time() - t0, 2)
    metrics_all["repaired_mae"] = metrics_repaired["mae"]
    metrics_all["repaired_rmse"] = metrics_repaired["rmse"]
    metrics_all["repaired_smape"] = metrics_repaired["smape"]

    log_to_mlflow(model_repaired, metrics_repaired, config, stage="post_repair")

    # Quality report after repair
    checker_after = DataQualityChecker(config)
    checker_after.run_all_checks(repaired_df)
    checker_after.generate_report(os.path.join(reports_dir, "quality_report.md"))

    # ── Stage 10: Clustering ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 10: Clustering & Segmentation")
    logger.info("=" * 50)
    t0 = time.time()
    from src.models.clustering import cluster_trips, save_cluster_report
    clustered_df, cluster_summary, cluster_labels = cluster_trips(gold_df_repaired, config)
    save_cluster_report(cluster_summary, os.path.join(reports_dir, "cluster_report.md"))
    timings["clustering"] = round(time.time() - t0, 2)

    # ── Stage 11: Drift Monitoring ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 11: Drift Monitoring")
    logger.info("=" * 50)
    t0 = time.time()
    from src.monitoring.drift import DriftMonitor
    monitor = DriftMonitor(config)
    drift_batch = monitor.simulate_drift(gold_df_repaired)
    drift_features = [c for c in ["demand", "avg_distance", "avg_fare", "hour_of_day"]
                      if c in gold_df_repaired.columns]
    drift_results = monitor.check_drift(gold_df_repaired, drift_batch, drift_features)
    monitor.save_report(os.path.join(reports_dir, "drift_report.md"))
    timings["drift_monitoring"] = round(time.time() - t0, 2)
    metrics_all["drift_detected"] = drift_results.get("overall_drift_detected", False)
    metrics_all["retrain_recommended"] = drift_results.get("retrain_recommended", False)

    # ── Stage 12: Generate reports & artifacts ──
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 12: Generate Reports & Artifacts")
    logger.info("=" * 50)

    # Model report
    model_report_lines = [
        "# Model Report",
        "\n## Forecasting",
        f"\n### Baseline Model ({metrics_baseline['model']})",
        f"- MAE: {metrics_baseline['mae']}",
        f"- RMSE: {metrics_baseline['rmse']}",
        f"- sMAPE: {metrics_baseline['smape']}%",
        f"- Train size: {metrics_baseline['train_size']}",
        f"- Test size: {metrics_baseline['test_size']}",
        f"\n### Post-Repair Model ({metrics_repaired['model']})",
        f"- MAE: {metrics_repaired['mae']}",
        f"- RMSE: {metrics_repaired['rmse']}",
        f"- sMAPE: {metrics_repaired['smape']}%",
        f"\n### Improvement",
        f"- MAE: {metrics_baseline['mae']} → {metrics_repaired['mae']} "
        f"({'↓' if metrics_repaired['mae'] <= metrics_baseline['mae'] else '↑'} "
        f"{abs(metrics_baseline['mae'] - metrics_repaired['mae']):.4f})",
        f"- Quality fail %: {fail_pct_before}% → {fail_pct_after}%",
        "\n## Clustering",
        f"- Algorithm: KMeans (k={config.get('clustering', {}).get('n_clusters', 5)})",
        f"- Clusters: {len(cluster_summary)}",
    ]
    with open(os.path.join(reports_dir, "model_report.md"), "w") as f:
        f.write("\n".join(model_report_lines))

    # Catalog
    catalog = {
        "tables": {
            "bronze.trips": {"rows": len(raw_df), "layer": "bronze"},
            "silver.trips": {"rows": len(repaired_df), "layer": "silver"},
            "gold.demand_zone_hour": {"rows": len(gold_df_repaired), "layer": "gold"},
        },
        "models": {
            "forecast_baseline": metrics_baseline,
            "forecast_repaired": metrics_repaired,
        },
        "artifacts": [
            "reports/quality_report.md",
            "reports/repair_patch.json",
            "reports/model_report.md",
            "reports/drift_report.md",
            "reports/cluster_report.md",
            "reports/llm_suggestions.md",
        ],
    }
    with open(os.path.join(reports_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2, default=str)

    # Lineage
    lineage_mmd = """graph LR
    A[Raw Data / Kaggle / Synthetic] -->|ingest| B[Bronze Iceberg]
    B -->|validate + standardize| C[Silver Iceberg]
    C -->|quality checks| D{Quality Gate}
    D -->|pass| E[Gold Features]
    D -->|fail| F[Targeted Repair]
    F -->|rebuild| C
    E -->|train| G[Forecast Model]
    E -->|cluster| H[Clustering]
    G -->|track| I[MLflow]
    E -->|drift check| J[Drift Monitor]
"""
    with open(os.path.join(reports_dir, "lineage.mmd"), "w") as f:
        f.write(lineage_mmd)

    # Total timing
    total_time = round(time.time() - t_total_start, 2)
    timings["total"] = total_time

    # Final metrics
    metrics_all["timings"] = timings
    save_metrics_json(metrics_all, os.path.join(reports_dir, "metrics.json"))

    # Stop Spark
    if spark:
        try:
            spark.stop()
        except Exception:
            pass

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total runtime: {total_time}s")
    logger.info(f"Quality: {fail_pct_before}% → {fail_pct_after}% failing rows")
    logger.info(f"Forecast MAE: {metrics_baseline['mae']} → {metrics_repaired['mae']}")
    logger.info(f"Reports: {reports_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import pandas as pd  # ensure pandas is available at top level
    main()
