"""MLflow helpers for GSPO IE training.

Default tracking server: http://<INTERNAL_IP>:5000
VERL also reads MLFLOW_TRACKING_URI from the environment.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_TRACKING_URI = "http://<INTERNAL_IP>:5000"
DEFAULT_EXPERIMENT = "gspo-ie-qwen3-0.6b"


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Configure tracking URI + experiment. Returns experiment name used."""
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    exp = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    return exp


@contextmanager
def mlflow_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> Iterator[Any]:
    """Context manager for a single MLflow run (prep / eval / SFT)."""
    import mlflow

    setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        if params:
            # MLflow params must be scalars/strings; flatten lightly
            flat = {}
            for k, v in params.items():
                if isinstance(v, (dict, list)):
                    flat[k] = str(v)[:250]
                else:
                    flat[k] = v
            mlflow.log_params(flat)
        yield run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    import mlflow

    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    import mlflow

    path = Path(path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    elif path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path or path.name)


def log_dict_as_json(obj: Any, filename: str = "artifact.json") -> None:
    import json
    import tempfile

    import mlflow

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        mlflow.log_artifact(str(p))


def verl_logger_flags(enable_mlflow: bool = True) -> str:
    """Hydra-style logger list for VERL trainer.logger."""
    if enable_mlflow:
        return '["console","mlflow"]'
    return '["console"]'


def export_env_for_verl(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, str]:
    """Env vars VERL Tracking reads (see verl/utils/tracking.py)."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    exp = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
    env = {
        "MLFLOW_TRACKING_URI": uri,
        "MLFLOW_EXPERIMENT_NAME": exp,
    }
    os.environ.update(env)
    return env
