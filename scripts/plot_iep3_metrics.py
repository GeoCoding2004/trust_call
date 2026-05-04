from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot IEP3 evaluation metrics from trials.csv and summary.json."
    )
    parser.add_argument(
        "--evaluation-dir",
        required=True,
        type=Path,
        help="Directory containing trials.csv and summary.json from evaluate_iep3.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to <evaluation-dir>/plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_dir / "plots"
    )

    trials_path = evaluation_dir / "trials.csv"
    summary_path = evaluation_dir / "summary.json"
    if not trials_path.is_file():
        raise SystemExit(f"Missing trials file: {trials_path}")
    if not summary_path.is_file():
        raise SystemExit(f"Missing summary file: {summary_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    same_scores, different_scores = load_scores(trials_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_score_histogram(
        plt=plt,
        same_scores=same_scores,
        different_scores=different_scores,
        summary=summary,
        bins=args.bins,
        output_path=output_dir / "score_distribution.png",
    )
    plot_error_rates(
        plt=plt,
        same_scores=same_scores,
        different_scores=different_scores,
        summary=summary,
        output_path=output_dir / "threshold_error_rates.png",
    )

    print(f"Wrote plots to {output_dir}")
    return 0


def load_scores(trials_path: Path) -> tuple[list[float], list[float]]:
    same_scores: list[float] = []
    different_scores: list[float] = []
    with trials_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score = float(row["match_confidence"])
            if row["label"] == "same":
                same_scores.append(score)
            elif row["label"] == "different":
                different_scores.append(score)

    if not same_scores or not different_scores:
        raise SystemExit("Both same-speaker and different-speaker rows are required")
    return same_scores, different_scores


def plot_score_histogram(
    *,
    plt,
    same_scores: list[float],
    different_scores: list[float],
    summary: dict,
    bins: int,
    output_path: Path,
) -> None:
    match_threshold = summary["recommendations"]["match_threshold"]
    review_threshold = summary["recommendations"]["review_threshold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        different_scores,
        bins=bins,
        alpha=0.68,
        label="Different speaker",
        color="#C44E52",
        density=True,
    )
    ax.hist(
        same_scores,
        bins=bins,
        alpha=0.68,
        label="Same speaker",
        color="#4C72B0",
        density=True,
    )
    ax.axvline(match_threshold, color="#1B1B1B", linestyle="-", linewidth=2, label="Match threshold")
    ax.axvline(review_threshold, color="#5F5F5F", linestyle="--", linewidth=2, label="Review threshold")
    ax.set_title("IEP3 Speaker Verification Score Distribution")
    ax.set_xlabel("Match confidence")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_rates(
    *,
    plt,
    same_scores: list[float],
    different_scores: list[float],
    summary: dict,
    output_path: Path,
) -> None:
    thresholds = [index / 1000 for index in range(1001)]
    false_accept_rates = [
        sum(score >= threshold for score in different_scores) / len(different_scores)
        for threshold in thresholds
    ]
    false_reject_rates = [
        sum(score < threshold for score in same_scores) / len(same_scores)
        for threshold in thresholds
    ]

    match_threshold = summary["recommendations"]["match_threshold"]
    review_threshold = summary["recommendations"]["review_threshold"]
    eer_threshold = summary["recommendations"]["eer_threshold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, false_accept_rates, label="False accept rate", color="#C44E52")
    ax.plot(thresholds, false_reject_rates, label="False reject rate", color="#4C72B0")
    ax.axvline(match_threshold, color="#1B1B1B", linestyle="-", linewidth=2, label="Match threshold")
    ax.axvline(review_threshold, color="#5F5F5F", linestyle="--", linewidth=2, label="Review threshold")
    ax.axvline(eer_threshold, color="#55A868", linestyle=":", linewidth=2, label="EER threshold")
    ax.set_title("IEP3 Threshold Tradeoff")
    ax.set_xlabel("Match confidence threshold")
    ax.set_ylabel("Error rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
