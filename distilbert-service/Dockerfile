FROM python:3.10-slim

WORKDIR /app

# Install dependencies required for transformers and FastAPI
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

# Copy the actual service code
COPY . .

# Prevent Hugging Face tokenizers from throwing parallel processing warnings
ENV TOKENIZERS_PARALLELISM=false

# Expose the API port required by the gateway (8002)
EXPOSE 8002

# Run the uvicorn server on container boot (this triggers the "Warm Start" lifespan event)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
