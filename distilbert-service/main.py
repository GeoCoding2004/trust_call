import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from prometheus_fastapi_instrumentator import Instrumentator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =====================================================================
# --- OLD CODE (Regex Heuristic Baseline) ---
# Keeping this commented out for easy rollback if needed
# DEFAULT_MODEL_NAME = os.getenv("DISTILBERT_MODEL_NAME", "distilbert-base-uncased")
# USE_CLASSIFIER = os.getenv("DISTILBERT_USE_CLASSIFIER", "false").lower() in {"1", "true", "yes"}
# =====================================================================

# --- NEW CODE (Fine-Tuned Neural Network) ---
DEFAULT_MODEL_NAME = os.getenv("DISTILBERT_MODEL_NAME", "./custom_scam_model")
#USE_CLASSIFIER = os.getenv("DISTILBERT_USE_CLASSIFIER", "true").lower() in {"1", "true", "yes"}
USE_CLASSIFIER = os.getenv("DISTILBERT_USE_CLASSIFIER", "false").lower() in {"1", "true", "yes"}

USE_EMBEDDINGS = os.getenv("DISTILBERT_USE_EMBEDDINGS", "true").lower() in {"1", "true", "yes"}
USE_CUDA = os.getenv("DISTILBERT_USE_CUDA", "false").lower() in {"1", "true", "yes"}

MIN_TEXT_CHARS = 3
DEFAULT_MAX_LENGTH = 512

SUSPICIOUS_PATTERNS: List[Tuple[str, str]] = [
    (r"\burgent\b", "urgent"),
    (r"\bverify(?: your)? account\b", "verify account"),
    (r"\bbank details?\b", "bank details"),
    (r"\botp\b|\bone[- ]time password\b", "OTP"),
    (r"\bsecurity code\b", "security code"),
    (r"\bwire transfer\b", "wire transfer"),
    (r"\bdo not tell (?:anyone|anybody)\b", "do not tell anyone"),
    (r"\bact now\b", "act now"),
    (r"\baccount locked\b", "account locked"),
    (r"\bsend money\b", "send money"),
    (r"\bconfirm identity\b", "confirm identity"),
]

PHRASE_WEIGHTS = {
    "urgent": 0.18,
    "verify account": 0.14,
    "bank details": 0.16,
    "OTP": 0.18,
    "security code": 0.16,
    "wire transfer": 0.2,
    "do not tell anyone": 0.2,
    "act now": 0.14,
    "account locked": 0.16,
    "send money": 0.2,
    "confirm identity": 0.14,
}

SEMANTIC_SCORE_BUCKETS = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TEXT_LENGTH_BUCKETS = (0, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000)
PHRASE_COUNT_BUCKETS = (0, 1, 2, 3, 5, 8, 13, 21)

trust_call_distilbert_model_ready = Gauge(
    "trust_call_distilbert_model_ready",
    "1 if DistilBERT tokenizer and classifier are loaded.",
)
trust_call_distilbert_model_mode = Gauge(
    "trust_call_distilbert_model_mode",
    "One-hot gauge indicating whether classifier or heuristic mode is active.",
    labelnames=("mode",),
)
trust_call_distilbert_predictions_total = Counter(
    "trust_call_distilbert_predictions_total",
    "Semantic predictions by label and mode.",
    labelnames=("label", "mode"),
)
trust_call_distilbert_errors_total = Counter(
    "trust_call_distilbert_errors_total",
    "DistilBERT errors by type.",
    labelnames=("error_type",),
)
trust_call_distilbert_fallback_total = Counter(
    "trust_call_distilbert_fallback_total",
    "Fallback or heuristic behaviors used by the semantic service.",
    labelnames=("reason",),
)
trust_call_distilbert_inference_latency_seconds = Histogram(
    "trust_call_distilbert_inference_latency_seconds",
    "Latency of semantic prediction requests.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
trust_call_distilbert_semantic_score = Histogram(
    "trust_call_distilbert_semantic_score",
    "Distribution of DistilBERT scam-risk scores.",
    buckets=SEMANTIC_SCORE_BUCKETS,
)
trust_call_distilbert_text_length_chars = Histogram(
    "trust_call_distilbert_text_length_chars",
    "Length of semantic service text inputs.",
    buckets=TEXT_LENGTH_BUCKETS,
)
trust_call_distilbert_empty_text_total = Counter(
    "trust_call_distilbert_empty_text_total",
    "Empty or insufficient semantic inputs.",
)
trust_call_distilbert_flagged_keyword_groups_total = Counter(
    "trust_call_distilbert_flagged_keyword_groups_total",
    "Counts of flagged keyword groups without exposing raw phrase content.",
    labelnames=("keyword_group",),
)
trust_call_distilbert_flagged_phrase_count = Histogram(
    "trust_call_distilbert_flagged_phrase_count",
    "Number of suspicious phrase groups observed in a request.",
    buckets=PHRASE_COUNT_BUCKETS,
)

HEURISTIC_FALLBACK_RECORDED = False


class TextInput(BaseModel):
    scrubbed_text: str


@dataclass
class ModelBundle:
    model_name: str
    tokenizer: Optional[Any]
    model: Optional[torch.nn.Module]
    max_length: int
    use_classifier: bool
    use_embeddings: bool
    device: torch.device
    load_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.tokenizer is not None and self.model is not None


MODEL_BUNDLE: Optional[ModelBundle] = None


def _set_model_mode(mode: str) -> None:
    trust_call_distilbert_model_mode.labels(mode="classifier").set(1 if mode == "classifier" else 0)
    trust_call_distilbert_model_mode.labels(mode="heuristic").set(1 if mode == "heuristic" else 0)


def classify_flagged_phrase_group(phrase: str) -> str:
    normalized = phrase.strip().lower()
    if normalized in {"urgent", "act now", "account locked"}:
        return "urgency"
    if normalized in {"send money", "wire transfer"}:
        return "money"
    if normalized in {"bank details", "security code", "otp"}:
        return "credentials"
    if normalized in {"verify account", "confirm identity"}:
        return "verification"
    if normalized in {"do not tell anyone"}:
        return "authority"
    if "payment" in normalized:
        return "payment"
    return "unknown"


def _metrics_label_for_response(label: str) -> str:
    mapping = {
        "benign": "safe",
        "safe": "safe",
        "suspicious": "suspicious",
        "scam": "scam",
        "insufficient_text": "insufficient_text",
    }
    return mapping.get(label, "error")


def _safe_model_max_length(tokenizer: Any) -> int:
    max_length = getattr(tokenizer, "model_max_length", DEFAULT_MAX_LENGTH)
    if not isinstance(max_length, int) or max_length > 4096:
        return DEFAULT_MAX_LENGTH
    return max_length


def load_model_bundle() -> ModelBundle:
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    tokenizer = None
    model = None
    load_error = None
    max_length = DEFAULT_MAX_LENGTH

    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        max_length = _safe_model_max_length(tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME)
        model.config.output_hidden_states = USE_EMBEDDINGS
        model.to(device)
        model.eval()
    except Exception as exc:
        load_error = str(exc)

    return ModelBundle(
        model_name=DEFAULT_MODEL_NAME,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        use_classifier=USE_CLASSIFIER,
        use_embeddings=USE_EMBEDDINGS,
        device=device,
        load_error=load_error,
    )


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def extract_flagged_phrases(text: str) -> List[str]:
    flagged: List[str] = []
    for pattern, label in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            flagged.append(label)
    return flagged


def _embedding_adjustment(bundle: ModelBundle, text: str) -> float:
    if not bundle.ready or bundle.use_classifier or not bundle.use_embeddings:
        return 0.0
    try:
        inputs = bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=bundle.max_length,
        )
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bundle.model(**inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return 0.0
        cls_embedding = hidden_states[-1][:, 0, :]
        norm_value = float(torch.linalg.norm(cls_embedding).item())
        return min(0.05, norm_value / 1000.0)
    except Exception:
        return 0.0


def compute_semantic_score(text: str, flagged: List[str], bundle: ModelBundle) -> Tuple[float, str, str]:
    if bundle.ready and bundle.use_classifier:
        inputs = bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=bundle.max_length,
        )
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bundle.model(**inputs)
        logits = outputs.logits.squeeze(0)
        if logits.numel() == 1:
            score = float(torch.sigmoid(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            score = float(probs[1].item()) if probs.numel() >= 2 else float(probs.max().item())
        label = "suspicious" if score >= 0.6 else "benign"
        return score, label, "classifier"

    score = 0.05
    for phrase in flagged:
        score += PHRASE_WEIGHTS.get(phrase, 0.1)
    score += min(len(text) / 200.0, 1.0) * 0.1
    score += _embedding_adjustment(bundle, text)
    score = min(score, 1.0)
    label = "suspicious" if score >= 0.6 else "benign"
    return score, label, "heuristic"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL_BUNDLE
    global HEURISTIC_FALLBACK_RECORDED
    MODEL_BUNDLE = load_model_bundle()
    if MODEL_BUNDLE.ready and MODEL_BUNDLE.use_classifier:
        trust_call_distilbert_model_ready.set(1)
        _set_model_mode("classifier")
    else:
        trust_call_distilbert_model_ready.set(0)
        _set_model_mode("heuristic")
        if MODEL_BUNDLE.load_error:
            trust_call_distilbert_errors_total.labels(error_type="model_load_failed").inc()
        if not HEURISTIC_FALLBACK_RECORDED:
            trust_call_distilbert_fallback_total.labels(reason="classifier_not_ready").inc()
            HEURISTIC_FALLBACK_RECORDED = True
    yield


app = FastAPI(title="DistilBERT Semantic Auditor", version="1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    trust_call_distilbert_errors_total.labels(error_type="validation_error").inc()
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.post("/predict")
async def predict_semantic_risk(payload: TextInput):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model bundle not initialized")

    started = time.perf_counter()
    try:
        normalized = normalize_text(payload.scrubbed_text)
        trust_call_distilbert_text_length_chars.observe(len(normalized))
        if len(normalized) < MIN_TEXT_CHARS:
            trust_call_distilbert_empty_text_total.inc()
            trust_call_distilbert_fallback_total.labels(reason="empty_text").inc()
            trust_call_distilbert_predictions_total.labels(
                label="insufficient_text",
                mode="none",
            ).inc()
            return {
                "semantic_score": 0.0,
                "label": "insufficient_text",
                "flagged_phrases": [],
                "model_name": MODEL_BUNDLE.model_name,
            }

        flagged = extract_flagged_phrases(normalized)
        score, label, mode = compute_semantic_score(normalized, flagged, MODEL_BUNDLE)
        trust_call_distilbert_semantic_score.observe(score)
        trust_call_distilbert_flagged_phrase_count.observe(len(flagged))
        if mode == "heuristic":
            trust_call_distilbert_fallback_total.labels(reason="heuristic_mode").inc()
        for phrase in flagged:
            trust_call_distilbert_flagged_keyword_groups_total.labels(
                keyword_group=classify_flagged_phrase_group(phrase)
            ).inc()
        trust_call_distilbert_predictions_total.labels(
            label=_metrics_label_for_response(label),
            mode=mode,
        ).inc()

        model_name = MODEL_BUNDLE.model_name
        if mode == "heuristic":
            model_name = f"{model_name} (heuristic)"

        return {
            "semantic_score": round(score, 4),
            "label": label,
            "flagged_phrases": flagged,
            "model_name": model_name,
        }
    except ValueError as exc:
        trust_call_distilbert_errors_total.labels(error_type="validation_error").inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        trust_call_distilbert_predictions_total.labels(label="error", mode="none").inc()
        trust_call_distilbert_errors_total.labels(error_type="prediction_failed").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error during semantic analysis")
    finally:
        trust_call_distilbert_inference_latency_seconds.observe(time.perf_counter() - started)
