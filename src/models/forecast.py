"""Forecasting model: predict taxi demand per zone per hour."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("nyc_taxi")


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def train_forecast_model(df: pd.DataFrame, feature_cols: list[str],
                         target: str = "demand", config: dict = None):
    """Train a demand forecasting model with time-based split.

    Returns: (model, metrics_dict, predictions_df)
    """
    config = config or {}
    test_frac = config.get("forecasting", {}).get("test_fraction", 0.2)

    # Sort by time
    df = df.sort_values("pickup_hour").reset_index(drop=True)

    # Filter valid feature columns
    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        raise ValueError("No feature columns available for training")

    X = df[available_features].values
    y = df[target].values

    # Time-based split
    split_idx = int(len(df) * (1 - test_frac))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Try LightGBM first, then sklearn
    model = None
    model_name = "unknown"
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        model_name = "LightGBM"
        logger.info("Trained LightGBM forecast model")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        model_name = "GradientBoosting"
        logger.info("Trained sklearn GradientBoosting forecast model")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # demand can't be negative

    # Metrics
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    smape_val = smape(y_test, y_pred)

    metrics = {
        "model": model_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape": round(smape_val, 4),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "features": available_features,
    }

    preds_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
    })

    logger.info(f"Forecast metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, sMAPE={smape_val:.2f}%")
    return model, metrics, preds_df


def log_to_mlflow(model, metrics: dict, config: dict, stage: str = "baseline"):
    """Log model and metrics to MLflow."""
    try:
        import mlflow

        tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000")
        experiment_name = config.get("mlflow", {}).get("experiment_name", "nyc_taxi_forecast")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"forecast_{stage}"):
            mlflow.log_params({
                "model_type": metrics.get("model", "unknown"),
                "stage": stage,
                "train_size": metrics.get("train_size", 0),
                "test_size": metrics.get("test_size", 0),
            })
            mlflow.log_metrics({
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "smape": metrics["smape"],
            })
            # Log model
            try:
                import lightgbm
                mlflow.lightgbm.log_model(model, "model")
            except (ImportError, Exception):
                mlflow.sklearn.log_model(model, "model")

        logger.info(f"MLflow: logged {stage} run to {experiment_name}")
        return True
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}. Using JSON fallback.")
        return False


def save_metrics_json(metrics: dict, output_path: str):
    """Save metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {output_path}")
