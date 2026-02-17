"""Iceberg storage: write/read bronze, silver, gold tables via PySpark."""
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("nyc_taxi")


def get_spark_session(storage_cfg: dict):
    """Create a PySpark session configured for Iceberg + MinIO."""
    from pyspark.sql import SparkSession

    s3 = storage_cfg.get("s3", {})
    catalog = storage_cfg.get("catalog", {})

    spark = (
        SparkSession.builder.appName(
            storage_cfg.get("spark", {}).get("app_name", "nyc_taxi_iceberg")
        )
        .master(storage_cfg.get("spark", {}).get("master", "local[*]"))
        .config(
            "spark.jars.packages",
            ",".join(storage_cfg.get("spark", {}).get("packages", [])),
        )
        .config("spark.sql.catalog.nyc_taxi", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nyc_taxi.type", "rest")
        .config("spark.sql.catalog.nyc_taxi.uri", catalog.get("uri", "http://iceberg-rest:8181"))
        .config("spark.sql.catalog.nyc_taxi.warehouse", catalog.get("warehouse", "s3://warehouse/"))
        .config("spark.sql.catalog.nyc_taxi.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.nyc_taxi.s3.endpoint", s3.get("endpoint", "http://minio:9000"))
        .config("spark.sql.catalog.nyc_taxi.s3.path-style-access", "true")
        .config("spark.hadoop.fs.s3a.endpoint", s3.get("endpoint", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.access.key", s3.get("access_key", "minioadmin"))
        .config("spark.hadoop.fs.s3a.secret.key", s3.get("secret_key", "minioadmin"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.defaultCatalog", "nyc_taxi")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created with Iceberg catalog")
    return spark


def create_namespace(spark, namespace: str):
    """Create an Iceberg namespace if it doesn't exist."""
    try:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS nyc_taxi.{namespace}")
        logger.info(f"Namespace nyc_taxi.{namespace} ready")
    except Exception as e:
        logger.warning(f"Namespace creation note: {e}")


def write_bronze(spark, df: pd.DataFrame, storage_cfg: dict):
    """Write raw data to Iceberg bronze table."""
    ns = storage_cfg.get("namespaces", {}).get("bronze", "bronze")
    ns_name = ns.split(".")[-1] if "." in ns else ns
    create_namespace(spark, ns_name)

    sdf = spark.createDataFrame(df)
    table_name = f"nyc_taxi.{ns_name}.trips"

    sdf.writeTo(table_name).using("iceberg").createOrReplace()
    count = spark.table(table_name).count()
    logger.info(f"Bronze table '{table_name}' written: {count} rows")
    return table_name


def write_silver(spark, df: pd.DataFrame, storage_cfg: dict):
    """Write validated/cleaned data to Iceberg silver table."""
    ns = storage_cfg.get("namespaces", {}).get("silver", "silver")
    ns_name = ns.split(".")[-1] if "." in ns else ns
    create_namespace(spark, ns_name)

    sdf = spark.createDataFrame(df)
    table_name = f"nyc_taxi.{ns_name}.trips"

    sdf.writeTo(table_name).using("iceberg").createOrReplace()
    count = spark.table(table_name).count()
    logger.info(f"Silver table '{table_name}' written: {count} rows")
    return table_name


def write_gold(spark, df: pd.DataFrame, storage_cfg: dict, table_suffix: str = "demand_zone_hour"):
    """Write gold aggregated data to Iceberg gold table."""
    ns = storage_cfg.get("namespaces", {}).get("gold", "gold")
    ns_name = ns.split(".")[-1] if "." in ns else ns
    create_namespace(spark, ns_name)

    sdf = spark.createDataFrame(df)
    table_name = f"nyc_taxi.{ns_name}.{table_suffix}"

    sdf.writeTo(table_name).using("iceberg").createOrReplace()
    count = spark.table(table_name).count()
    logger.info(f"Gold table '{table_name}' written: {count} rows")
    return table_name


def read_table(spark, layer: str, storage_cfg: dict, table: str = "trips") -> pd.DataFrame:
    """Read an Iceberg table back as a Pandas DataFrame."""
    ns = storage_cfg.get("namespaces", {}).get(layer, layer)
    ns_name = ns.split(".")[-1] if "." in ns else ns
    table_name = f"nyc_taxi.{ns_name}.{table}"
    sdf = spark.table(table_name)
    return sdf.toPandas()


def materialize_to_parquet(spark, layer: str, storage_cfg: dict,
                           output_path: str, table: str = "demand_zone_hour"):
    """Export an Iceberg table to Parquet for fast Python access."""
    ns = storage_cfg.get("namespaces", {}).get(layer, layer)
    ns_name = ns.split(".")[-1] if "." in ns else ns
    table_name = f"nyc_taxi.{ns_name}.{table}"

    sdf = spark.table(table_name)
    pdf = sdf.toPandas()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(output_path, index=False)
    logger.info(f"Materialized {table_name} to {output_path}: {len(pdf)} rows")
    return pdf
