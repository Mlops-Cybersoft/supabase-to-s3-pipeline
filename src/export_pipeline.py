"""Export selected Supabase tables through Transaction Pooler to CSV files on S3."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import boto3
import psycopg
from dotenv import load_dotenv
from psycopg import sql


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TableSpec:
    schema: str
    table: str
    primary_key: str

    def validate(self) -> None:
        for label, value in (("schema", self.schema), ("table", self.table), ("primary_key", self.primary_key)):
            if not SAFE_IDENTIFIER.fullmatch(value):
                raise ValueError(f"Invalid {label} identifier: {value!r}")


@dataclass(frozen=True)
class Settings:
    pooler_url: str
    bucket: str
    prefix: str
    region: str
    tables: list[TableSpec]
    chunk_size: int = 5_000
    server_side_encryption: str | None = "AES256"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        pooler_url = os.getenv("SUPABASE_TRANSACTION_POOLER_URL", "")
        bucket = os.getenv("S3_EXPORT_BUCKET", "")
        raw_tables = os.getenv("EXPORT_TABLES_JSON", "")
        if not pooler_url or not bucket or not raw_tables:
            raise ValueError("Set SUPABASE_TRANSACTION_POOLER_URL, S3_EXPORT_BUCKET, and EXPORT_TABLES_JSON")
        try:
            tables = [TableSpec(**item) for item in json.loads(raw_tables)]
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("EXPORT_TABLES_JSON must be a JSON array of schema/table/primary_key objects") from error
        settings = cls(
            pooler_url=pooler_url,
            bucket=bucket,
            prefix=os.getenv("S3_EXPORT_PREFIX", "raw/supabase-pooler-export").strip("/"),
            region=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
            tables=tables,
            chunk_size=int(os.getenv("EXPORT_CHUNK_SIZE", "5000")),
            server_side_encryption=os.getenv("S3_SERVER_SIDE_ENCRYPTION", "AES256") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlparse(self.pooler_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError("SUPABASE_TRANSACTION_POOLER_URL must be a PostgreSQL URL")
        if parsed.port != 6543:
            raise ValueError("This project intentionally requires Supabase Transaction Pooler port 6543")
        if not self.bucket or self.chunk_size < 1:
            raise ValueError("S3 bucket must be set and EXPORT_CHUNK_SIZE must be positive")
        if not self.tables:
            raise ValueError("At least one table must be configured")
        for table in self.tables:
            table.validate()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, bytes):
        return value.hex()
    return value


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_table(settings: Settings, table: TableSpec, run_id: str, temp_dir: Path) -> dict[str, Any]:
    """Fetch one table in batches, write a local CSV, then upload it to S3."""
    table.validate()
    file_path = temp_dir / f"{table.schema}__{table.table}.csv"
    query = sql.SQL("SELECT * FROM {}.{} ORDER BY {}").format(
        sql.Identifier(table.schema), sql.Identifier(table.table), sql.Identifier(table.primary_key)
    )
    row_count = 0
    with psycopg.connect(settings.pooler_url, autocommit=True) as connection:
        with connection.cursor() as cursor, file_path.open("w", newline="", encoding="utf-8") as output:
            cursor.execute(query)
            headers = [column.name for column in cursor.description]
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(headers)
            while rows := cursor.fetchmany(settings.chunk_size):
                writer.writerows([[_csv_value(value) for value in row] for row in rows])
                row_count += len(rows)

    key = "/".join(part for part in [settings.prefix, f"run_id={run_id}", table.schema, table.table, "data.csv"] if part)
    extra_args: dict[str, str] = {"ContentType": "text/csv"}
    if settings.server_side_encryption:
        extra_args["ServerSideEncryption"] = settings.server_side_encryption
    s3 = boto3.client("s3", region_name=settings.region)
    s3.upload_file(str(file_path), settings.bucket, key, ExtraArgs=extra_args)
    return {
        "schema": table.schema,
        "table": table.table,
        "primary_key": table.primary_key,
        "rows": row_count,
        "s3_uri": f"s3://{settings.bucket}/{key}",
        "sha256": _sha256(file_path),
    }


def export_all_tables(settings: Settings | None = None) -> dict[str, Any]:
    """Export each configured table and publish one run manifest to S3."""
    settings = settings or Settings.from_env()
    run_id = _run_id()
    with tempfile.TemporaryDirectory(prefix="supabase-s3-export-") as temporary_directory:
        reports = [export_table(settings, table, run_id, Path(temporary_directory)) for table in settings.tables]
    manifest_key = "/".join(part for part in [settings.prefix, f"run_id={run_id}", "manifest.json"] if part)
    manifest_uri = f"s3://{settings.bucket}/{manifest_key}"
    manifest = {
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": "supabase-transaction-pooler",
        "tables": reports,
        "manifest_s3_uri": manifest_uri,
    }
    put_args: dict[str, Any] = {
        "Bucket": settings.bucket,
        "Key": manifest_key,
        "Body": json.dumps(manifest, indent=2).encode("utf-8"),
        "ContentType": "application/json",
    }
    if settings.server_side_encryption:
        put_args["ServerSideEncryption"] = settings.server_side_encryption
    boto3.client("s3", region_name=settings.region).put_object(**put_args)
    return manifest
