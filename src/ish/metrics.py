import logging
import os
from typing import Any, Dict, Optional

try:
    from prometheus_client import Counter, Gauge, start_http_server
except ImportError:  # pragma: no cover
    Counter = None
    Gauge = None
    start_http_server = None

LOG_FORMAT = "%(asctime)s %(levelname)s %(module)s %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
LOG_LEVEL_ENV = "ISH_LOG_LEVEL"
LOG_LEVELS_ENV = "ISH_LOG_LEVELS"
METRICS_PORT_ENV = "ISH_METRICS_PORT"
METRICS_ADDR_ENV = "ISH_METRICS_ADDR"
_METRICS_SERVER_STARTED = False

MESSAGES_MOVED_COUNTER = (
    Counter("ish_messages_moved_total", "Total number of messages moved") if Counter else None
)
MESSAGES_SKIPPED_COUNTER = (
    Counter("ish_messages_skipped_total", "Total number of messages skipped") if Counter else None
)
TRAINING_EMBEDDINGS_GAUGE = (
    Gauge("ish_training_embeddings_total", "Embeddings used during last training run") if Gauge else None
)
TRAINING_ACCURACY_GAUGE = (
    Gauge("ish_training_accuracy", "Classifier accuracy from last training run") if Gauge else None
)
TRAINING_DURATION_GAUGE = (
    Gauge("ish_training_duration_seconds", "Seconds spent training the classifier") if Gauge else None
)
TRAINING_CLASSIFICATION_GAUGE = (
    Gauge(
        "ish_training_classification_metric",
        "Classification metrics per label and metric",
        labelnames=("label", "metric"),
    )
    if Gauge
    else None
)
CLASSIFY_FOLDER_EMBEDDINGS_GAUGE = (
    Gauge(
        "ish_classify_folder_embeddings_total",
        "Number of embeddings considered per folder during classification",
        labelnames=("folder",),
    )
    if Gauge
    else None
)
DB_SIZE_GAUGE = (
    Gauge("ish_cache_db_size_bytes", "Size of the cache sqlite DB on disk in bytes") if Gauge else None
)


def configure_logging(level: int = DEFAULT_LOG_LEVEL) -> None:
    """Configure root logging once and honor env overrides."""
    root_logger = logging.getLogger()
    formatter = logging.Formatter(LOG_FORMAT)
    if root_logger.handlers:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(level=level, format=LOG_FORMAT)
        root_logger = logging.getLogger()
    _apply_env_logging_overrides(root_logger, formatter)


def _apply_env_logging_overrides(root_logger: logging.Logger, formatter: logging.Formatter) -> None:
    env_level = _parse_log_level(os.environ.get(LOG_LEVEL_ENV))
    if env_level is not None:
        root_logger.setLevel(env_level)
        for handler in root_logger.handlers:
            handler.setLevel(env_level)
            handler.setFormatter(formatter)

    for logger_name, level in _parse_named_log_levels(os.environ.get(LOG_LEVELS_ENV)).items():
        if not logger_name:
            continue
        logging.getLogger(logger_name).setLevel(level)


def _parse_log_level(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    name = value.upper()
    return logging._nameToLevel.get(name)  # type: ignore[attr-defined]


def _parse_named_log_levels(raw: Optional[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        if not item.strip() or "=" not in item:
            continue
        name, level_str = item.split("=", 1)
        parsed = _parse_log_level(level_str)
        if parsed is not None:
            overrides[name.strip()] = parsed
    return overrides


def start_metrics_server_if_configured() -> None:
    """Start Prometheus metrics exporter if configured via env vars."""
    global _METRICS_SERVER_STARTED
    if _METRICS_SERVER_STARTED or start_http_server is None:
        return
    port_raw = os.environ.get(METRICS_PORT_ENV, "9100")
    if not port_raw:
        return
    try:
        port = int(port_raw)
    except ValueError:
        logging.getLogger("ish").warning("Invalid metrics port provided: %s", port_raw)
        return
    addr = os.environ.get(METRICS_ADDR_ENV, "0.0.0.0")
    try:
        start_http_server(port, addr=addr)
        _METRICS_SERVER_STARTED = True
        logging.getLogger("ish").info("Prometheus metrics listening on %s:%s", addr, port)
    except OSError as exc:
        logging.getLogger("ish").error("Failed to start metrics server on %s:%s: %s", addr, port, exc)


def increment_moved(count: int) -> None:
    if count <= 0 or MESSAGES_MOVED_COUNTER is None:
        return
    MESSAGES_MOVED_COUNTER.inc(count)


def increment_skipped(count: int = 1) -> None:
    if count <= 0 or MESSAGES_SKIPPED_COUNTER is None:
        return
    MESSAGES_SKIPPED_COUNTER.inc(count)


def record_training_stats(embeddings: int, accuracy: float, duration: float) -> None:
    if TRAINING_EMBEDDINGS_GAUGE is not None:
        TRAINING_EMBEDDINGS_GAUGE.set(embeddings)
    if TRAINING_ACCURACY_GAUGE is not None:
        TRAINING_ACCURACY_GAUGE.set(accuracy)
    if TRAINING_DURATION_GAUGE is not None:
        TRAINING_DURATION_GAUGE.set(duration)


def record_classification_metrics(report: Dict[str, Any]) -> None:
    if TRAINING_CLASSIFICATION_GAUGE is None:
        return
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                TRAINING_CLASSIFICATION_GAUGE.labels(str(label), metric_name).set(value)
        else:
            TRAINING_CLASSIFICATION_GAUGE.labels(str(label), "score").set(metrics)


def record_folder_embedding_count(folder: str, count: int) -> None:
    if CLASSIFY_FOLDER_EMBEDDINGS_GAUGE is None:
        return
    CLASSIFY_FOLDER_EMBEDDINGS_GAUGE.labels(folder).set(count)


def record_db_size(path: str) -> None:
    if DB_SIZE_GAUGE is None:
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    DB_SIZE_GAUGE.set(size)
