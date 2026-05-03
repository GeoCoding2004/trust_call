# Evaluation Results Template

> **Instructions:** Fill in every `NOT_MEASURED` field with real numbers collected from a live evaluation run before submission. Do not leave placeholders or enter fabricated values. If a metric is not yet measured, write `NOT_MEASURED` and explain why.

Evaluation date: `2026-05-03`
Evaluated by: `Not yet measured before submission.`
Commit SHA: `Not captured yet.`

---

## IEP1: RawNet2 Signal Auditor

Dataset: `NOT_MEASURED (e.g., ASVspoof 2019 LA eval set, N samples)`
Model weights file: `fine_tuned_DF_model.pth`
Device: `NOT_MEASURED (CPU / GPU)`

| Metric | Value |
|---|---|
| Accuracy | NOT_MEASURED |
| Precision (spoof) | NOT_MEASURED |
| Recall (spoof) | NOT_MEASURED |
| F1 Score (spoof) | NOT_MEASURED |
| ROC-AUC | NOT_MEASURED |
| Equal Error Rate (EER) | NOT_MEASURED |
| False Positive Rate | NOT_MEASURED |
| False Negative Rate | NOT_MEASURED |
| Latency p50 (ms) | NOT_MEASURED |
| Latency p95 (ms) | NOT_MEASURED |

Notes: `NOT_MEASURED`

How to reproduce:
```bash
cd rawnet-service
python evaluate_rawnet.py --data_dir /path/to/dataset --output results/rawnet_eval.json
```

---

## IEP2: DistilBERT Semantic Auditor

Dataset: `NOT_MEASURED (e.g., N scam transcripts, N benign transcripts)`
Model: `./custom_scam_model` (fine-tuned DistilBERT)

| Metric | Value |
|---|---|
| Accuracy | NOT_MEASURED |
| Precision (suspicious) | NOT_MEASURED |
| Recall (suspicious) | NOT_MEASURED |
| F1 Score (suspicious) | NOT_MEASURED |
| ROC-AUC | NOT_MEASURED |
| False Positive Rate | NOT_MEASURED |
| False Negative Rate | NOT_MEASURED |
| Latency p50 (ms) | NOT_MEASURED |
| Latency p95 (ms) | NOT_MEASURED |
| Neural model availability | NOT_MEASURED (% requests served by neural vs heuristic fallback) |

Notes: `NOT_MEASURED`

How to reproduce:
```bash
pytest distilbert-service/test_main.py -v
# For a full eval set, extend distilbert-service/test_main.py with a parametrized dataset loader
```

---

## IEP3: ECAPA-TDNN Identity Auditor

Dataset: `NOT_MEASURED (e.g., VoxCeleb1 eval pairs, N genuine, N impostor)`
Threshold policy: `configs/iep3_identity.json`

| Metric | Value |
|---|---|
| Equal Error Rate (EER) | NOT_MEASURED |
| TAR @ FAR=1% | NOT_MEASURED |
| TAR @ FAR=0.1% | NOT_MEASURED |
| Precision (accept) | NOT_MEASURED |
| Recall (accept) | NOT_MEASURED |
| F1 Score (accept) | NOT_MEASURED |
| Genuine pair mean cosine similarity | NOT_MEASURED |
| Impostor pair mean cosine similarity | NOT_MEASURED |
| Latency p50 (ms) | NOT_MEASURED |
| Latency p95 (ms) | NOT_MEASURED |

Notes: `NOT_MEASURED`

How to reproduce:
```bash
python scripts/evaluate_iep3.py --data_dir data/evaluation/iep3
python scripts/plot_iep3_metrics.py
```

---

## EEP Late Fusion (End-to-End)

Dataset: `NOT_MEASURED (N labeled call recordings)`
Evaluation method: `NOT_MEASURED`

| Metric | Value |
|---|---|
| System accuracy | NOT_MEASURED |
| Threat precision | NOT_MEASURED |
| Threat recall | NOT_MEASURED |
| False alarm rate | NOT_MEASURED |
| Miss rate | NOT_MEASURED |
| End-to-end latency p50 (ms) | NOT_MEASURED |
| End-to-end latency p95 (ms) | NOT_MEASURED |
| IEP1 timeout rate | NOT_MEASURED |
| IEP2 timeout rate | NOT_MEASURED |
| Degraded response rate | NOT_MEASURED |

Notes: `NOT_MEASURED`

---

## Baseline Comparison

If applicable, compare against a keyword-only baseline (the regex heuristic in `distilbert-service/main.py`).

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Heuristic baseline | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Fine-tuned DistilBERT | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| Full Trust-Call fusion | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

---

## Infrastructure Metrics

| Metric | Value |
|---|---|
| RawNet container startup time (s) | NOT_MEASURED |
| DistilBERT container startup time (s) | NOT_MEASURED |
| Backend startup time (s) | NOT_MEASURED |
| Peak memory usage per service (MB) | NOT_MEASURED |
| Docker image sizes (MB) | NOT_MEASURED |
