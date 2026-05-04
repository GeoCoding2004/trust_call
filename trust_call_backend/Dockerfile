FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY trust_call_backend/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

COPY trust_call_backend ./trust_call_backend
COPY configs ./configs

ENV PYTHONUNBUFFERED=1
ENV RAWNET_URL=http://rawnet-service:8000/predict
ENV DISTILBERT_URL=http://distilbert-service:8002/predict

EXPOSE 8080

CMD ["uvicorn", "trust_call_backend.server:app", "--host", "0.0.0.0", "--port", "8080"]
