# NYC Yellow Taxi – AI-Enhanced Data Value Chain

Production-quality AI data pipeline using NYC Yellow Taxi trip data with Apache Iceberg, MinIO, Spark, MLflow, and Docker.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│    MinIO     │  Iceberg     │   MLflow +   │   Pipeline     │
│  (S3 Store)  │  REST Catalog│   Postgres   │  (Python+Spark)│
│  :9000/:9001 │    :8181     │    :5050     │                │
└──────────────┴──────────────┴──────────────┴────────────────┘

Data Plane:       Bronze → Silver → Gold (Iceberg tables on MinIO)
Steward Plane:    Quality checks → Targeted repair → LLM suggestions
ML Product Layer: Forecasting → Clustering → Drift monitoring → MLflow
```

## Quick Start

### 1. Start all services
```bash
docker compose up -d --build
```

### 2. Run the full pipeline
```bash
docker compose run --rm pipeline python scripts/run_all.py
```

### 3. Access services
| Service        | URL                          |
|---------------|------------------------------|
| MinIO Console | http://localhost:9001         |
| MLflow UI     | http://localhost:5050         |
| Iceberg REST  | http://localhost:8181/v1/config |

MinIO credentials: `minioadmin` / `minioadmin`

## Project Structure

```
├── config/
│   ├── storage.yaml          # Iceberg + MinIO + Spark config
│   └── pipeline.yaml         # Pipeline parameters
├── scripts/
│   └── run_all.py            # Main orchestrator
├── src/
│   ├── ingestion/            # Kaggle download + synthetic fallback
│   ├── storage_iceberg/      # Iceberg table I/O via PySpark
│   ├── steward/              # Quality checks, repair, LLM suggestions
│   ├── features/             # Feature engineering (gold demand table)
│   ├── models/               # Forecasting + clustering
│   ├── monitoring/           # Drift detection (PSI/KS)
│   └── utils/                # Config loading, logging, helpers
├── sql/                      # SQL transforms (Spark SQL)
├── docker/
│   ├── Dockerfile.pipeline   # Pipeline container
│   └── entrypoint.sh         # MinIO init script
├── docker-compose.yml        # Full stack definition
├── docs/
│   ├── architecture.mmd      # Architecture diagram (Mermaid)
│   ├── sequence.mmd          # Pipeline sequence diagram
│   └── governance.mmd        # Quality + repair governance
├── reports/                  # Generated artifacts
│   ├── quality_report.md
│   ├── repair_patch.json
│   ├── model_report.md
│   ├── drift_report.md
│   ├── cluster_report.md
│   ├── catalog.json
│   ├── lineage.mmd
│   ├── metrics.json
│   ├── llm_suggestions.md
│   └── llm_suggestions.json
├── logs/
│   └── run.log
├── requirements.txt
└── requirements-extra.txt
```

## Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Ingest** | Download from Kaggle or generate synthetic dataset |
| 2 | **Bronze** | Write raw data to Iceberg table on MinIO |
| 3 | **Quality** | Run schema/null/range/temporal/outlier/duplicate checks |
| 4 | **LLM Suggestions** | Generate repair suggestions (offline stub) |
| 5 | **Silver** | Validate and standardize schema/types |
| 6 | **Gold** | Build demand features (zone-hour aggregation + lags) |
| 7 | **Baseline Model** | Train forecast model, log to MLflow |
| 8 | **Targeted Repair** | Apply 5 repair rules within budget |
| 9 | **Retrain** | Rebuild gold + retrain on repaired data |
| 10 | **Clustering** | KMeans segmentation with interpretation |
| 11 | **Drift Monitor** | Simulate batch drift, generate report |
| 12 | **Reports** | Generate all artifacts |

## Targeted Repair Rules

| Rule | Description | Fix |
|------|-------------|-----|
| swap_datetime | pickup >= dropoff | Swap timestamps |
| clamp_passenger | count not in [1,6] | Clamp + impute |
| abs_distance | negative distance | Take absolute value |
| winsorize_fare | negative/extreme fare | Abs + IQR cap |
| drop_zero_duration | 0 duration + positive distance | Drop row |

## Success Metrics (computed before/after)

- **% rows failing checks**: decreased after repair
- **Forecast MAE/RMSE/sMAPE**: compared baseline vs post-repair
- **Runtime per stage**: logged in metrics.json
- **Ingestion throughput**: rows/second

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KAGGLE_USERNAME` | Kaggle API username | (optional) |
| `KAGGLE_KEY` | Kaggle API key | (optional) |
| `AWS_ACCESS_KEY_ID` | MinIO access key | minioadmin |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | minioadmin |

## Technology Stack

- **Storage**: Apache Iceberg + MinIO (S3-compatible)
- **Catalog**: Iceberg REST Catalog
- **Processing**: PySpark 3.5
- **ML**: LightGBM, scikit-learn, KMeans
- **Tracking**: MLflow + PostgreSQL
- **Quality**: Custom quality-as-code checks
- **Monitoring**: PSI/KS drift tests
- **Orchestration**: Python (run_all.py)
- **Infrastructure**: Docker Compose