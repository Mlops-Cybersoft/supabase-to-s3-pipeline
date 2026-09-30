"""Create the MLflow PostgreSQL database without resetting Docker volumes."""
import os

import psycopg2


connection = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    user=os.getenv("POSTGRES_USER", "airflow"),
    password=os.getenv("POSTGRES_PASSWORD", "airflow"),
    dbname="postgres",
)
connection.autocommit = True
with connection.cursor() as cursor:
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'mlflow'")
    if not cursor.fetchone():
        cursor.execute("CREATE DATABASE mlflow")
connection.close()
