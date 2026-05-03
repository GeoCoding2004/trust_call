FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

COPY main.py model.py ./
RUN mkdir -p /models
ENV RAWNET_MODEL_PATH=/models/fine_tuned_DF_model.pth

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
