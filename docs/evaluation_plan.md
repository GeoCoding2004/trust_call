# Evaluation Plan

This document defines the evaluation methodology for each AI component of Trust-Call and the end-to-end fusion pipeline.

> **Data honesty note:** Current evaluation uses limited demo / synthetic data. Full benchmark requires externally sourced labeled datasets. Do not fabricate results. Fill `docs/evaluation_results_template.md` with real numbers before submission.

---

## IEP1: RawNet2 Signal Auditor (Deepfake Detection)

### What is being evaluated
The RawNet2 model’s ability to classify audio as real (human) or spoofed (AI-generated / deepfake).

### Recommended dataset
- ASVspoof 2019 LA evaluation set (genuine + spoofed utterances)
- Any TTS/voice-conversion samples for adversarial testing

### Metrics
| Metric | Description |
|---|---|
| Accuracy | (TP + TN) / total samples |
| Precision | TP / (TP + FP) for spoof class |
| Recall | TP / (TP + FN) for spoof class |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve |
| Equal Error Rate (EER) | Point where FPR == FNR (standard ASVspoof metric) |
| FPR (False Positive Rate) | Real audio incorrectly labelled as spoof |
| FNR (False Negative Rate) | Spoofed audio incorrectly labelled as real |
| Latency p50 / p95 | Inference time in milliseconds at median and 95th percentile |

### How to run
```bash
cd rawnet-service
python evaluate_rawnet.py --data_dir /path/to/asvspoof/eval --output results/rawnet_eval.json
```
See `rawnet-service/evaluate_rawnet.py` for full argument documentation.

### Existing evaluation artifacts
- `rawnet-service/ASVspoof_Baseline_cm.png` – confusion matrix from baseline model
- `rawnet-service/Fine_Tuned_HuggingFace_cm.png` – confusion matrix from fine-tuned model
- MLflow runs: `rawnet-service/mlruns/`

---

## IEP2: DistilBERT Semantic Auditor (Scam Detection)

### What is being evaluated
The fine-tuned DistilBERT classifier’s ability to label call transcript segments as `suspicious` or `benign`.

### Recommended dataset
- Labeled call transcript snippets (scam / non-scam)
- `semantic_model_training/` contains the training pipeline and any included data samples

### Metrics
| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| Precision | Precision for `suspicious` class |
| Recall | Recall for `suspicious` class |
| F1 Score | F1 for `suspicious` class |
| ROC-AUC | AUC if probability outputs are available |
| FPR | Benign calls flagged as suspicious |
| FNR | Scam calls missed |
| Latency p50 / p95 | Inference latency in milliseconds |
| Fallback rate | % of predictions served by heuristic vs neural model |

### How to run
See `semantic_model_training/` for training and evaluation scripts. To test the deployed service:
```bash
pytest distilbert-service/test_main.py -v
```

---

## IEP3: ECAPA-TDNN Identity Auditor (Speaker Verification)

### What is being evaluated
The speaker verification system’s ability to accept a genuine speaker and reject impostors.

### Recommended dataset
- VoxCeleb1 evaluation pairs, or
- Custom recorded genuine/impostor audio pairs
- `data/evaluation/iep3/README.md` contains dataset preparation instructions

### Metrics
| Metric | Description |
|---|---|
| EER (Equal Error Rate) | Threshold where FAR == FRR |
| TAR @ FAR=1% | True acceptance rate at 1% false acceptance |
| TAR @ FAR=0.1% | True acceptance rate at 0.1% false acceptance |
| Precision | Accepted verifications that are genuine |
| Recall | Genuine speakers correctly accepted |
| F1 Score | F1 for accept class |
| Cosine similarity distribution | Mean/std for genuine vs impostor pairs |
| Latency p50 / p95 | Embedding extraction + comparison latency |
| TOFU candidate embedding quality | RMS / peak / active-ratio distribution |

### How to run
```bash
python scripts/evaluate_iep3.py --help
python scripts/evaluate_iep3.py --data_dir data/evaluation/iep3
python scripts/plot_iep3_metrics.py
```

---

## EEP Late Fusion (End-to-End)

### What is being evaluated
The combined system’s accuracy in producing a correct final threat/safe/review decision when given live or recorded audio.

### Metrics
| Metric | Description |
|---|---|
| System accuracy | End-to-end correct decisions / total calls |
| Threat precision | THREAT DETECTED calls that were true threats |
| Threat recall | True threat calls correctly flagged |
| False alarm rate | Safe calls incorrectly marked THREAT DETECTED |
| Miss rate | Threat calls incorrectly marked SAFE |
| Latency p50 / p95 | Time from audio chunk to telemetry update |
| Timeout rate | % of IEP1/IEP2 calls that timed out and fell back |
| Degraded response rate | % of fusions with one or more services unavailable |

### How to collect
Run the full pipeline on a labeled set of recorded calls and compare `fusion_status` output to ground truth labels. No automated script exists yet – see `docs/evaluation_results_template.md` for the template.

---

## Latency Budget

| Stage | Target p50 | Target p95 |
|---|---|---|
| WebRTC audio chunk arrival (3 s chunk) | ≈ 3000 ms (real time) | 3100 ms |
| Whisper transcription (tiny.en, CPU) | < 1000 ms | < 2000 ms |
| RawNet inference (4 s audio, CPU) | < 500 ms | < 1000 ms |
| DistilBERT inference (CPU) | < 200 ms | < 500 ms |
| ECAPA embedding + cosine compare | < 300 ms | < 600 ms |
| Total fusion to WebSocket broadcast | < 2000 ms | < 4000 ms |

---

## Notes on Honest Reporting

- Do not use results from a test set that overlaps with training data.
- The heuristic fallback in IEP2 will produce weaker metrics than the neural model; report them separately.
- For IEP3, report results on both enrolled and TOFU-candidate speakers.
- Confusion matrices from `rawnet-service/` are from training/validation; label them clearly as such.


Note: RawNet evaluation/inference requires supplying model weights via `RAWNET_MODEL_PATH`; CI images intentionally omit `.pth` files.
