"""Utility helpers: config loading, logging, timing."""
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from functools import wraps

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config(name: str = "pipeline") -> dict:
    """Load a named config from config/ directory."""
    cfg_path = PROJECT_ROOT / "config" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return load_yaml(str(cfg_path))


def get_storage_config() -> dict:
    return get_config("storage")


def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure project-wide logging."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_file or str(log_dir / "run.log")

    logger = logging.getLogger("nyc_taxi")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def timed(stage_name: str):
    """Decorator to time a pipeline stage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("nyc_taxi")
            logger.info(f"▶ Stage [{stage_name}] starting...")
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            logger.info(f"✔ Stage [{stage_name}] completed in {elapsed:.1f}s")
            return result, elapsed
        return wrapper
    return decorator


def ensure_dir(path: str):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
