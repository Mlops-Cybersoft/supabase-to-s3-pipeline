FROM apache/airflow:2.10.4-python3.11

USER airflow
COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
COPY scripts /opt/airflow/scripts
