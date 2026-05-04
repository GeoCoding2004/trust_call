# DistilBERT Semantic Auditor

This microservice provides semantic risk scoring for scrubbed text using a DistilBERT-based pipeline. It supports:
- A heuristic demo mode for early integration
- An easy swap to a fine-tuned DistilBERT checkpoint later

## API Contract

**POST** `/predict`

Request body:
```json
{
  "scrubbed_text": "string"
}
```

Response body:
```json
{
  "semantic_score": 0.0,
  "label": "benign",
  "flagged_phrases": ["urgent", "wire transfer"],
  "model_name": "distilbert-base-uncased (heuristic)"
}
```

## Run Locally

From `distilbert-service/`:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8002
```

## Example curl

```bash
curl -X POST "http://127.0.0.1:8002/predict" \
  -H "Content-Type: application/json" \
  -d "{\"scrubbed_text\":\"Urgent: verify your account and send money now.\"}"
```

## Swap in a Fine-Tuned Checkpoint

Set environment variables before launching:
- `DISTILBERT_MODEL_NAME=/path/to/checkpoint-or-hf-repo`
- `DISTILBERT_USE_CLASSIFIER=true`

Optional:
- `DISTILBERT_USE_EMBEDDINGS=false` to disable embedding-based adjustment
- `DISTILBERT_USE_CUDA=true` to enable GPU if available

## Local Test Client

```bash
python test_client.py
```
