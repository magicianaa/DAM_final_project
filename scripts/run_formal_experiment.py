from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.evaluation.cross_validate import cross_validate_models
from src.evaluation.metrics import relevant_by_user
from src.pipeline import PipelineResult, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal MovieLens experiment suite used by the final report."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-users", type=int, default=300)
    parser.add_argument("--max-ratings", type=int, default=500_000)
    parser.add_argument("--max-tags", type=int, default=100_000)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-movies", type=int, default=None)
    parser.add_argument("--max-matrix-cells", type=int, default=25_000_000)
    parser.add_argument("--min-user-ratings", type=int, default=1)
    parser.add_argument("--min-movie-ratings", type=int, default=1)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--skip-cross-validation", action="store_true")
    return parser.parse_args()


def make_full_ratings(result: PipelineResult) -> pd.DataFrame:
    full = pd.concat([result.features.ratings, result.test_ratings], ignore_index=True)
    sort_cols = [c for c in ["userId", "timestamp", "movieId"] if c in full.columns]
    return full.sort_values(sort_cols).reset_index(drop=True)


def build_data_profile(result: PipelineResult, full_ratings: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    features = result.features
    relevant = relevant_by_user(result.test_ratings)
    user_count = int(full_ratings["userId"].nunique())
    movie_count = int(features.movies["movieId"].nunique())
    matrix_cells = user_count * movie_count
    sparsity = 1.0 - (len(full_ratings) / matrix_cells) if matrix_cells else 0.0

    rows = [
        ("dataset_mode", "sample" if args.sample else "MovieLens"),
        ("top_k", args.k),
        ("ratings_total_after_filtering", len(full_ratings)),
        ("train_ratings", len(features.ratings)),
        ("test_ratings", len(result.test_ratings)),
        ("users", user_count),
        ("movies", movie_count),
        ("rating_matrix_sparsity", round(sparsity, 6)),
        ("rating_mean", round(float(full_ratings["rating"].mean()), 4) if not full_ratings.empty else 0.0),
        ("rating_min", round(float(full_ratings["rating"].min()), 4) if not full_ratings.empty else 0.0),
        ("rating_max", round(float(full_ratings["rating"].max()), 4) if not full_ratings.empty else 0.0),
        ("test_users_with_relevant_items", len(relevant)),
        ("genre_feature_count", features.genre_features.shape[1]),
        ("tag_text_feature_count", features.tag_features.shape[1]),
        ("content_feature_count", features.content_features.shape[1]),
        ("min_user_ratings", args.min_user_ratings),
        ("min_movie_ratings", args.min_movie_ratings),
        ("max_eval_users", args.max_eval_users or "all"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def save_table(df: pd.DataFrame, stem: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    markdown = df.to_markdown(index=False) if not df.empty else "_No rows generated._"
    (REPORTS_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")


def build_reproducible_command(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/run_formal_experiment.py"]
    if args.sample:
        parts.append("--sample")
    if args.data_dir:
        parts.extend(["--data-dir", str(args.data_dir)])
    if args.download:
        parts.append("--download")
    parts.extend(
        [
            "--k",
            str(args.k),
            "--max-eval-users",
            str(args.max_eval_users),
            "--min-user-ratings",
            str(args.min_user_ratings),
            "--min-movie-ratings",
            str(args.min_movie_ratings),
            "--cv-folds",
            str(args.cv_folds),
        ]
    )
    if args.max_ratings != 500_000:
        parts.extend(["--max-ratings", str(args.max_ratings)])
    if args.max_tags != 100_000:
        parts.extend(["--max-tags", str(args.max_tags)])
    if args.max_users:
        parts.extend(["--max-users", str(args.max_users)])
    if args.max_movies:
        parts.extend(["--max-movies", str(args.max_movies)])
    if args.max_matrix_cells != 25_000_000:
        parts.extend(["--max-matrix-cells", str(args.max_matrix_cells)])
    if args.skip_cross_validation:
        parts.append("--skip-cross-validation")
    return " ".join(parts)


def write_formal_summary(
    args: argparse.Namespace,
    result: PipelineResult,
    data_profile: pd.DataFrame,
    cross_validation: pd.DataFrame,
) -> None:
    lines = [
        "# Formal Experiment Summary",
        "",
        "## Research Questions",
        "",
        "- RQ1: How do popularity, collaborative filtering, matrix factorization, content-based, and hybrid recommenders compare on Top-N movie recommendation?",
        "- RQ2: Does adding content features improve the accuracy-diversity-coverage trade-off?",
        "- RQ3: Which models behave better for low-, medium-, and high-activity users?",
        "- RQ4: Do stronger models also introduce stronger popularity bias?",
        "",
        "## Reproducible Command",
        "",
        "```powershell",
        build_reproducible_command(args),
        "```",
        "",
        "## Dataset Profile",
        "",
        data_profile.to_markdown(index=False),
        "",
        "## Main Result",
        "",
        result.evaluation.to_markdown(index=False) if not result.evaluation.empty else "_No model comparison rows generated._",
        "",
        "## Stability Check",
        "",
    ]
    if cross_validation.empty:
        lines.append("_Cross-validation was skipped or no valid fold was generated._")
    else:
        lines.append(cross_validation.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Report Artifacts",
            "",
            "- `reports/model_comparison.*`: main Top-N model comparison.",
            "- `reports/ablation_results.*`: contribution of collaborative, content, and popularity components.",
            "- `reports/user_segment_results.*`: low-, medium-, and high-activity user analysis.",
            "- `reports/popularity_bias.*`: popularity-bias diagnostics.",
            "- `reports/significance_results.*`: pairwise per-user NDCG@K significance tests.",
            "- `reports/cross_validation_results.*`: k-fold stability check.",
        ]
    )
    (REPORTS_DIR / "formal_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        data_dir=args.data_dir,
        download=args.download,
        sample=args.sample,
        k=args.k,
        max_eval_users=args.max_eval_users,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
        max_ratings=args.max_ratings,
        max_tags=args.max_tags,
        max_users=args.max_users,
        max_movies=args.max_movies,
        max_matrix_cells=args.max_matrix_cells,
    )

    full_ratings = make_full_ratings(result)
    data_profile = build_data_profile(result, full_ratings, args)
    save_table(data_profile, "data_profile")

    if args.skip_cross_validation:
        cross_validation = pd.DataFrame()
    else:
        movies = result.features.movies.drop(columns=["tag_text"], errors="ignore")
        cross_validation = cross_validate_models(
            ratings=full_ratings,
            movies=movies,
            tags=result.features.tags,
            n_splits=args.cv_folds,
            k=args.k,
            max_eval_users=args.max_eval_users,
        )
        save_table(cross_validation, "cross_validation_results")

    write_formal_summary(args, result, data_profile, cross_validation)
    print(f"Formal experiment artifacts written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
