"""S3 snapshot ingestion, feature preparation, and employment model training."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(ARTIFACTS_DIR / "raw")))
RUN_CONTEXT_PATH = ARTIFACTS_DIR / "source_run.json"
PROCESSED_PATH = ARTIFACTS_DIR / "processed" / "employment_features.parquet"
TRAIN_PATH = ARTIFACTS_DIR / "splits" / "train.parquet"
TEST_PATH = ARTIFACTS_DIR / "splits" / "test.parquet"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = ARTIFACTS_DIR / "models" / "employment_model.joblib"
QUALITY_GATE = float(os.getenv("MIN_ROC_AUC", "0.70"))

TABLES = ("enrollees", "enrollee_education", "work_experience", "training_hours", "employment", "city_dev_index")
FEATURE_COLUMNS = [
    "city", "gender", "enrolled_university", "education_level", "major_discipline",
    "relevent_experience", "experience", "company_size", "company_type", "last_new_job",
    "training_hours", "city_development_index", "experience_years", "last_new_job_years",
]
CATEGORICAL_COLUMNS = FEATURE_COLUMNS[:10]
NUMERIC_COLUMNS = FEATURE_COLUMNS[10:]
DERIVED_COLUMNS = {"experience_years", "last_new_job_years"}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _normalise_run_id(value: str) -> str:
    return value.removeprefix("run_id=").strip("/")


def _s3_client():
    import boto3

    return boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))


def _resolve_run_id(client: Any, bucket: str, prefix: str) -> str:
    configured = _normalise_run_id(os.getenv("S3_TRAINING_RUN_ID", ""))
    if configured:
        return configured
    root = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    candidates: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            name = item["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if name.startswith("run_id="):
                candidates.append(_normalise_run_id(name))
    if not candidates:
        raise FileNotFoundError(f"No run_id=<timestamp> folders found under s3://{bucket}/{root}")
    return max(candidates)


def sync_snapshot_from_s3() -> dict[str, Any]:
    """Download one complete CSV snapshot created by the export notebook."""
    bucket = os.getenv("S3_EXPORT_BUCKET", "")
    if not bucket:
        raise ValueError("S3_EXPORT_BUCKET is required")
    prefix = os.getenv("S3_EXPORT_PREFIX", "raw/supabase-pooler-export").strip("/")
    client = _s3_client()
    run_id = _resolve_run_id(client, bucket, prefix)
    run_prefix = "/".join(part for part in [prefix, f"run_id={run_id}"] if part)
    raw_run_dir = RAW_DATA_DIR / f"run_id={run_id}"
    raw_run_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, str]] = []
    for table in TABLES:
        key = f"{run_prefix}/public/{table}/data.csv"
        target = raw_run_dir / f"{table}.csv"
        try:
            client.download_file(bucket, key, str(target))
        except Exception as error:
            raise FileNotFoundError(f"Missing required export: s3://{bucket}/{key}") from error
        downloaded.append({"table": table, "s3_uri": f"s3://{bucket}/{key}", "local_path": str(target)})
    report = {
        "status": "downloaded",
        "run_id": run_id,
        "bucket": bucket,
        "prefix": run_prefix,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": downloaded,
    }
    _write_json(RUN_CONTEXT_PATH, report)
    _write_json(ARTIFACTS_DIR / "ingestion_report.json", report)
    return report


def _active_raw_dir() -> Path:
    if not RUN_CONTEXT_PATH.exists():
        raise FileNotFoundError("No S3 snapshot selected; run sync_snapshot_from_s3 first")
    run_id = json.loads(RUN_CONTEXT_PATH.read_text(encoding="utf-8"))["run_id"]
    return RAW_DATA_DIR / f"run_id={run_id}"


def _read_training_csv(directory: Path, table: str) -> pd.DataFrame:
    """Read one exported table and discard pipeline audit fields before joins.

    The Supabase import script can add ``imported_at`` to every source table.
    It describes when a row entered Supabase, rather than a model feature, and
    would otherwise collide repeatedly during the six-table merge.
    """
    frame = pd.read_csv(directory / f"{table}.csv")
    audit_columns = {"imported_at", "exported_at", "dms_commit_timestamp"}
    return frame.drop(columns=[column for column in audit_columns if column in frame.columns])


def _read_sources() -> pd.DataFrame:
    directory = _active_raw_dir()
    profile = _read_training_csv(directory, "enrollees").drop(columns=["full_name"], errors="ignore")
    education = _read_training_csv(directory, "enrollee_education")
    experience = _read_training_csv(directory, "work_experience")
    training = _read_training_csv(directory, "training_hours")
    employment = _read_training_csv(directory, "employment").rename(columns={"employed": "label"})
    city = _read_training_csv(directory, "city_dev_index").rename(
        columns={"City": "city", "City Development Index": "city_development_index"}
    )
    city = city[["city", "city_development_index"]]
    return profile.merge(education, on="enrollee_id", how="left").merge(
        experience, on="enrollee_id", how="left"
    ).merge(training, on="enrollee_id", how="left").merge(
        city, on="city", how="left"
    ).merge(employment, on="enrollee_id", how="inner")


def validate_source_data() -> dict[str, Any]:
    frame = _read_sources()
    required = (set(FEATURE_COLUMNS) - DERIVED_COLUMNS) | {"enrollee_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty or not frame.enrollee_id.is_unique:
        raise ValueError("Source data must contain unique, non-empty enrollee records")
    labels = set(frame.label.dropna().unique())
    if not labels.issubset({0, 1, 0.0, 1.0}) or frame.label.nunique() != 2:
        raise ValueError("employment label must contain both binary classes 0 and 1")
    report = {
        "status": "passed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "positive_rate": float(frame.label.mean()),
        "run_id": json.loads(RUN_CONTEXT_PATH.read_text(encoding="utf-8"))["run_id"],
    }
    _write_json(ARTIFACTS_DIR / "data_quality.json", report)
    return report


def prepare_features() -> dict[str, Any]:
    frame = _read_sources()
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").fillna("unknown")
    for column in ["training_hours", "city_development_index"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].median())
    frame["experience_years"] = pd.to_numeric(
        frame.experience.astype("string").replace({"<1": "0", ">20": "21"}), errors="coerce"
    ).fillna(-1)
    frame["last_new_job_years"] = pd.to_numeric(
        frame.last_new_job.astype("string").replace({"never": "0", ">4": "5"}), errors="coerce"
    ).fillna(-1)
    frame["label"] = frame.label.astype(int)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[["enrollee_id", *FEATURE_COLUMNS, "label"]].to_parquet(PROCESSED_PATH, index=False)
    report = {"rows": int(len(frame)), "features": len(FEATURE_COLUMNS), "output": str(PROCESSED_PATH)}
    _write_json(ARTIFACTS_DIR / "processing_summary.json", report)
    return report


def split_data() -> dict[str, Any]:
    frame = pd.read_parquet(PROCESSED_PATH)
    train, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.label)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    return {"train_rows": int(len(train)), "test_rows": int(len(test)), "random_state": 42}


def train_model() -> dict[str, Any]:
    import mlflow
    import mlflow.sklearn

    train, test = pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
    ])
    model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "supabase-s3-employment-training"))
    with mlflow.start_run() as run:
        model.fit(train[FEATURE_COLUMNS], train.label)
        predictions = model.predict(test[FEATURE_COLUMNS])
        probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        metrics = {
            "run_id": run.info.run_id,
            "roc_auc": float(roc_auc_score(test.label, probabilities)),
            "accuracy": float(accuracy_score(test.label, predictions)),
            "precision": float(precision_score(test.label, predictions)),
            "recall": float(recall_score(test.label, predictions)),
        }
        mlflow.log_params({"algorithm": "logistic_regression", "feature_count": len(FEATURE_COLUMNS), "source_run_id": json.loads(RUN_CONTEXT_PATH.read_text())["run_id"]})
        mlflow.log_metrics({key: value for key, value in metrics.items() if key != "run_id"})
        mlflow.sklearn.log_model(model, "model")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    _write_json(METRICS_PATH, metrics)
    return metrics


def publish_if_quality_passes() -> dict[str, Any]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics["roc_auc"] < QUALITY_GATE:
        raise RuntimeError(f"Quality gate failed: ROC-AUC {metrics['roc_auc']:.4f} < {QUALITY_GATE:.4f}")
    release = {"status": "published", "published_at": datetime.now(timezone.utc).isoformat(), "model_path": str(MODEL_PATH), "quality_gate": QUALITY_GATE, **metrics}
    _write_json(ARTIFACTS_DIR / "production" / "model_manifest.json", release)
    return release
