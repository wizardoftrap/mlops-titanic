FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas scikit-learn mlflow uvicorn pydantic fastapi

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=https://opportunistically-unneedful-seymour.ngrok-free.dev

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
