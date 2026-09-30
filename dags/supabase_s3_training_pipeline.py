"""Airflow DAG: a versioned Supabase CSV snapshot on S3 to an MLflow model."""
from datetime import datetime

from airflow.decorators import dag
from airflow.operators.bash import BashOperator


@dag(
    dag_id="supabase_s3_employment_training",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["supabase", "s3", "training-course-data", "model-training"],
)
def supabase_s3_employment_training():
    def stage(task_id: str, command: str) -> BashOperator:
        return BashOperator(task_id=task_id, bash_command=f"cd /opt/airflow && python -m app.cli {command}")

    ingest = stage("ingest_selected_s3_snapshot", "ingest")
    validate = stage("validate_source_data", "validate")
    prepare = stage("prepare_feature_dataset", "prepare")
    split = stage("create_train_test_split", "split")
    train = stage("train_and_track_mlflow", "train")
    publish = stage("quality_gate_and_publish", "publish")
    ingest >> validate >> prepare >> split >> train >> publish


supabase_s3_employment_training()
