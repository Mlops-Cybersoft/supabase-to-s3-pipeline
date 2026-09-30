"""Run an individual Airflow pipeline stage from a BashOperator."""
import argparse
import json

from app import pipeline


STAGES = {
    "ingest": pipeline.sync_snapshot_from_s3,
    "validate": pipeline.validate_source_data,
    "prepare": pipeline.prepare_features,
    "split": pipeline.split_data,
    "train": pipeline.train_model,
    "publish": pipeline.publish_if_quality_passes,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    stage = parser.parse_args().stage
    print(json.dumps(STAGES[stage]()))


if __name__ == "__main__":
    main()
