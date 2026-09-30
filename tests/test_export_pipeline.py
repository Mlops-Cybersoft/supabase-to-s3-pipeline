import os

import pytest

from src.export_pipeline import Settings, TableSpec


def test_table_spec_rejects_unsafe_identifier():
    with pytest.raises(ValueError, match="Invalid table"):
        TableSpec(schema="public", table="users; DROP TABLE users", primary_key="id").validate()


def test_settings_require_transaction_pooler(monkeypatch):
    monkeypatch.setenv("SUPABASE_TRANSACTION_POOLER_URL", "postgresql://user:pass@pooler.supabase.com:5432/postgres")
    monkeypatch.setenv("S3_EXPORT_BUCKET", "lake")
    monkeypatch.setenv("EXPORT_TABLES_JSON", '[{"schema":"public","table":"employment","primary_key":"enrollee_id"}]')
    with pytest.raises(ValueError, match="port 6543"):
        Settings.from_env()


def test_settings_loads_table_config(monkeypatch):
    monkeypatch.setenv("SUPABASE_TRANSACTION_POOLER_URL", "postgresql://user:pass@pooler.supabase.com:6543/postgres?sslmode=require")
    monkeypatch.setenv("S3_EXPORT_BUCKET", "lake")
    monkeypatch.setenv("EXPORT_TABLES_JSON", '[{"schema":"public","table":"employment","primary_key":"enrollee_id"}]')
    settings = Settings.from_env()
    assert settings.tables == [TableSpec(schema="public", table="employment", primary_key="enrollee_id")]
