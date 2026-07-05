FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow>=3,<4" psycopg2-binary boto3
EXPOSE 5000
