from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trust_call_backend.identity_auditor import ECAPASpeakerEmbedder
from trust_call_backend.identity_config import IdentityAuditorConfig, load_identity_auditor_config


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}


@dataclass(frozen=True)
class AudioSample:
    speaker_id: str
    path: Path


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    label: str
    enroll_speaker: str
    verify_speaker: str
    enroll_files: str
    verify_file: str
    similarity: float
    match_confidence: float
    identity_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate IEP3 speaker-verification thresholds on a speaker-labeled "
            "dataset. Expected layout: dataset_root/<speaker_id>/**/*.wav"
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root folder containing one subdirectory per speaker.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "trust_call_backend" / "state" / "evaluation" / "iep3",
        help="Directory where trials.csv and summary.json will be written.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional IEP3 config JSON. Defaults to configs/iep3_identity.json.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional cap for quick subset evaluations.",
    )
    parser.add_argument(
        "--max-files-per-speaker",
        type=int,
        default=5,
        help="Maximum audio files to use per speaker after deterministic shuffling.",
    )
    parser.add_argument(
        "--enrollment-files",
        type=int,
        default=1,
        help="Number of files to average into each speaker enrollment vector.",
    )
    parser.add_argument(
        "--max-impostor-trials-per-speaker",
        type=int,
        default=20,
        help="Maximum different-speaker trials generated per enrolled speaker.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="False accept rate target used for the recommended match threshold.",
    )
    parser.add_argument(
        "--target-miss-as-mismatch",
        type=float,
        default=0.05,
        help=(
            "Same-speaker miss rate target used for the recommended review threshold. "
            "Scores below this threshold become mismatch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for deterministic sampling.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Optional model device override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if args.max_files_per_speaker < 2:
        raise SystemExit("--max-files-per-speaker must be at least 2")
    if args.enrollment_files < 1:
        raise SystemExit("--enrollment-files must be at least 1")
    if args.enrollment_files >= args.max_files_per_speaker:
        raise SystemExit("--enrollment-files must be lower than --max-files-per-speaker")
    if not 0.0 < args.target_far < 1.0:
        raise SystemExit("--target-far must be in the interval (0, 1)")
    if not 0.0 < args.target_miss_as_mismatch < 1.0:
        raise SystemExit("--target-miss-as-mismatch must be in the interval (0, 1)")

    rng = random.Random(args.seed)
    config = load_identity_auditor_config(args.config)
    speakers = load_speaker_samples(
        dataset_root=dataset_root,
        max_speakers=args.max_speakers,
        max_files_per_speaker=args.max_files_per_speaker,
        rng=rng,
    )
    if len(speakers) < 2:
        raise SystemExit("At least two speakers with enough audio files are required")

    embedder = ECAPASpeakerEmbedder(
        config=config,
        savedir=config.resolve_model_savedir(),
        device=args.device,
    )

    embeddings = extract_embeddings(speakers=speakers, embedder=embedder)
    trials = build_trials(
        speakers=speakers,
        embeddings=embeddings,
        enrollment_files=args.enrollment_files,
        max_impostor_trials_per_speaker=args.max_impostor_trials_per_speaker,
        rng=rng,
    )
    if not trials:
        raise SystemExit("No trials were generated from the dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_trials_csv(output_dir / "trials.csv", trials)
    summary = build_summary(
        trials=trials,
        config=config,
        dataset_root=dataset_root,
        args=args,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(trials)} trials to {output_dir / 'trials.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(
        "Recommended thresholds: "
        f"match={summary['recommendations']['match_threshold']:.6f}, "
        f"review={summary['recommendations']['review_threshold']:.6f}"
    )
    return 0


def load_speaker_samples(
    dataset_root: Path,
    max_speakers: int | None,
    max_files_per_speaker: int,
    rng: random.Random,
) -> dict[str, list[AudioSample]]:
    speakers: dict[str, list[AudioSample]] = {}
    for speaker_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        files = [
            path
            for path in sorted(speaker_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if len(files) < 2:
            continue
        rng.shuffle(files)
        selected = files[:max_files_per_speaker]
        speakers[speaker_dir.name] = [
            AudioSample(speaker_id=speaker_dir.name, path=path) for path in selected
        ]

    speaker_ids = sorted(speakers)
    if max_speakers is not None:
        speaker_ids = speaker_ids[:max_speakers]
    return {speaker_id: speakers[speaker_id] for speaker_id in speaker_ids}


def extract_embeddings(
    speakers: dict[str, list[AudioSample]],
    embedder: ECAPASpeakerEmbedder,
) -> dict[Path, np.ndarray]:
    embeddings: dict[Path, np.ndarray] = {}
    total = sum(len(samples) for samples in speakers.values())
    completed = 0
    for samples in speakers.values():
        for sample in samples:
            audio, sample_rate = sf.read(sample.path, dtype="float32")
            embedding, _ = embedder.extract(audio, int(sample_rate))
            embeddings[sample.path] = embedding
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Embedded {completed}/{total} files")
    return embeddings


def build_trials(
    speakers: dict[str, list[AudioSample]],
    embeddings: dict[Path, np.ndarray],
    enrollment_files: int,
    max_impostor_trials_per_speaker: int,
    rng: random.Random,
) -> list[TrialResult]:
    enrollment_vectors: dict[str, np.ndarray] = {}
    enrollment_paths: dict[str, list[Path]] = {}
    verification_samples: dict[str, list[AudioSample]] = {}

    for speaker_id, samples in speakers.items():
        enroll_samples = samples[:enrollment_files]
        verify_samples = samples[enrollment_files:]
        if not verify_samples:
            continue
        vectors = [embeddings[sample.path] for sample in enroll_samples]
        enrollment_vectors[speaker_id] = l2_normalize(np.mean(vectors, axis=0))
        enrollment_paths[speaker_id] = [sample.path for sample in enroll_samples]
        verification_samples[speaker_id] = verify_samples

    trials: list[TrialResult] = []
    trial_index = 1
    speaker_ids = sorted(enrollment_vectors)

    for speaker_id in speaker_ids:
        enroll_vector = enrollment_vectors[speaker_id]
        for sample in verification_samples[speaker_id]:
            trials.append(
                score_trial(
                    trial_index=trial_index,
                    label="same",
                    enroll_speaker=speaker_id,
                    verify_speaker=speaker_id,
                    enroll_files=enrollment_paths[speaker_id],
                    verify_file=sample.path,
                    enroll_vector=enroll_vector,
                    verify_vector=embeddings[sample.path],
                )
            )
            trial_index += 1

        impostor_pool = [
            sample
            for other_speaker, samples in verification_samples.items()
            if other_speaker != speaker_id
            for sample in samples
        ]
        rng.shuffle(impostor_pool)
        for sample in impostor_pool[:max_impostor_trials_per_speaker]:
            trials.append(
                score_trial(
                    trial_index=trial_index,
                    label="different",
                    enroll_speaker=speaker_id,
                    verify_speaker=sample.speaker_id,
                    enroll_files=enrollment_paths[speaker_id],
                    verify_file=sample.path,
                    enroll_vector=enroll_vector,
                    verify_vector=embeddings[sample.path],
                )
            )
            trial_index += 1

    return trials


def score_trial(
    trial_index: int,
    label: str,
    enroll_speaker: str,
    verify_speaker: str,
    enroll_files: Iterable[Path],
    verify_file: Path,
    enroll_vector: np.ndarray,
    verify_vector: np.ndarray,
) -> TrialResult:
    similarity = float(np.dot(enroll_vector, verify_vector))
    match_confidence = float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
    return TrialResult(
        trial_id=f"trial_{trial_index:06d}",
        label=label,
        enroll_speaker=enroll_speaker,
        verify_speaker=verify_speaker,
        enroll_files=";".join(str(path) for path in enroll_files),
        verify_file=str(verify_file),
        similarity=round(similarity, 8),
        match_confidence=round(match_confidence, 8),
        identity_score=round(1.0 - match_confidence, 8),
    )


def write_trials_csv(path: Path, trials: list[TrialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(trials[0]).keys()))
        writer.writeheader()
        for trial in trials:
            writer.writerow(asdict(trial))


def build_summary(
    trials: list[TrialResult],
    config: IdentityAuditorConfig,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict:
    same_scores = np.array(
        [trial.match_confidence for trial in trials if trial.label == "same"], dtype=np.float32
    )
    different_scores = np.array(
        [trial.match_confidence for trial in trials if trial.label == "different"],
        dtype=np.float32,
    )
    if same_scores.size == 0 or different_scores.size == 0:
        raise SystemExit("Both same-speaker and different-speaker trials are required")

    thresholds = np.linspace(0.0, 1.0, num=1001, dtype=np.float32)
    operating_points = [compute_operating_point(float(t), same_scores, different_scores) for t in thresholds]
    eer_point = min(
        operating_points,
        key=lambda point: (abs(point["false_accept_rate"] - point["false_reject_rate"]), point["threshold"]),
    )

    match_candidates = [
        point for point in operating_points if point["false_accept_rate"] <= args.target_far
    ]
    if match_candidates:
        match_point = min(
            match_candidates,
            key=lambda point: (point["false_reject_rate"], -point["threshold"]),
        )
    else:
        match_point = max(operating_points, key=lambda point: point["threshold"])

    sorted_same = np.sort(same_scores)
    review_threshold = float(np.quantile(sorted_same, args.target_miss_as_mismatch))
    review_threshold = min(review_threshold, match_point["threshold"] - 0.001)
    review_threshold = max(review_threshold, 0.0)
    review_point = compute_operating_point(review_threshold, same_scores, different_scores)

    current_match_point = compute_operating_point(
        config.match_threshold, same_scores, different_scores
    )
    current_review_point = compute_operating_point(
        config.review_threshold, same_scores, different_scores
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "sample_counts": {
            "same_trials": int(same_scores.size),
            "different_trials": int(different_scores.size),
            "total_trials": len(trials),
        },
        "config": config.to_dict(),
        "run_options": {
            "max_speakers": args.max_speakers,
            "max_files_per_speaker": args.max_files_per_speaker,
            "enrollment_files": args.enrollment_files,
            "max_impostor_trials_per_speaker": args.max_impostor_trials_per_speaker,
            "target_far": args.target_far,
            "target_miss_as_mismatch": args.target_miss_as_mismatch,
            "seed": args.seed,
            "device": args.device,
        },
        "score_distribution": {
            "same": distribution(same_scores),
            "different": distribution(different_scores),
        },
        "current_thresholds": {
            "match_threshold": config.match_threshold,
            "review_threshold": config.review_threshold,
            "match_threshold_metrics": current_match_point,
            "review_threshold_metrics": current_review_point,
        },
        "recommendations": {
            "match_threshold": round(float(match_point["threshold"]), 6),
            "review_threshold": round(float(review_threshold), 6),
            "match_threshold_metrics": match_point,
            "review_threshold_metrics": review_point,
            "eer": round(
                float((eer_point["false_accept_rate"] + eer_point["false_reject_rate"]) / 2.0),
                6,
            ),
            "eer_threshold": round(float(eer_point["threshold"]), 6),
        },
    }


def compute_operating_point(
    threshold: float,
    same_scores: np.ndarray,
    different_scores: np.ndarray,
) -> dict[str, float]:
    false_reject_rate = float(np.mean(same_scores < threshold))
    false_accept_rate = float(np.mean(different_scores >= threshold))
    return {
        "threshold": round(float(threshold), 6),
        "false_accept_rate": round(false_accept_rate, 6),
        "false_reject_rate": round(false_reject_rate, 6),
    }


def distribution(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(np.min(scores)), 6),
        "p05": round(float(np.quantile(scores, 0.05)), 6),
        "p25": round(float(np.quantile(scores, 0.25)), 6),
        "median": round(float(np.median(scores)), 6),
        "p75": round(float(np.quantile(scores, 0.75)), 6),
        "p95": round(float(np.quantile(scores, 0.95)), 6),
        "max": round(float(np.max(scores)), 6),
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
    }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero_norm_embedding")
    return (vector / norm).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
