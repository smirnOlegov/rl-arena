# syntax=docker/dockerfile:1.7

# Tracking server: официальный образ + драйвер Postgres (backend store)
# и boto3 для S3-совместимого artifact store.
FROM ghcr.io/mlflow/mlflow:v3.17.0

RUN pip install --no-cache-dir "psycopg[binary]==3.2.10" "boto3==1.40.61"

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=10 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)"]
