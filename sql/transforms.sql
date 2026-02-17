-- SQL transforms for Iceberg tables
-- These can be executed via Spark SQL or DuckDB

-- Bronze to Silver: validate and standardize
CREATE OR REPLACE TABLE nyc_taxi.silver.trips AS
SELECT
    CAST(VendorID AS INT) AS vendor_id,
    tpep_pickup_datetime AS pickup_datetime,
    tpep_dropoff_datetime AS dropoff_datetime,
    CAST(passenger_count AS INT) AS passenger_count,
    CAST(trip_distance AS DOUBLE) AS trip_distance,
    CAST(RatecodeID AS INT) AS ratecode_id,
    store_and_fwd_flag,
    CAST(PULocationID AS INT) AS pu_location_id,
    CAST(DOLocationID AS INT) AS do_location_id,
    CAST(payment_type AS INT) AS payment_type,
    CAST(fare_amount AS DOUBLE) AS fare_amount,
    CAST(extra AS DOUBLE) AS extra,
    CAST(mta_tax AS DOUBLE) AS mta_tax,
    CAST(tip_amount AS DOUBLE) AS tip_amount,
    CAST(tolls_amount AS DOUBLE) AS tolls_amount,
    CAST(improvement_surcharge AS DOUBLE) AS improvement_surcharge,
    CAST(total_amount AS DOUBLE) AS total_amount
FROM nyc_taxi.bronze.trips
WHERE tpep_pickup_datetime < tpep_dropoff_datetime
  AND passenger_count BETWEEN 1 AND 6
  AND trip_distance >= 0
  AND fare_amount >= 0;

-- Silver to Gold: demand aggregation per zone per hour
CREATE OR REPLACE TABLE nyc_taxi.gold.demand_zone_hour AS
SELECT
    pu_location_id AS zone_id,
    DATE_TRUNC('hour', pickup_datetime) AS pickup_hour,
    COUNT(*) AS demand,
    AVG(trip_distance) AS avg_distance,
    AVG(fare_amount) AS avg_fare,
    AVG(TIMESTAMPDIFF(MINUTE, pickup_datetime, dropoff_datetime)) AS avg_duration_min,
    HOUR(pickup_datetime) AS hour_of_day,
    DAYOFWEEK(pickup_datetime) AS day_of_week,
    CASE WHEN DAYOFWEEK(pickup_datetime) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    MONTH(pickup_datetime) AS month
FROM nyc_taxi.silver.trips
GROUP BY pu_location_id, DATE_TRUNC('hour', pickup_datetime),
         HOUR(pickup_datetime), DAYOFWEEK(pickup_datetime), MONTH(pickup_datetime);
