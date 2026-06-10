from __future__ import annotations

import argparse
import sys

from src.config import configure_logging
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens hybrid recommendation pipeline")
    parser.add_argument("--data-dir", default=None, help="Path containing MovieLens ratings.csv/movies.csv/tags.csv/links.csv")
    parser.add_argument("--download", action="store_true", help="Download MovieLens latest-small before running")
    parser.add_argument("--sample", action="store_true", help="Use built-in tiny sample data for smoke tests")
    parser.add_argument("--k", type=int, default=10, help="Top-K recommendation/evaluation size")
    parser.add_argument("--max-eval-users", type=int, default=None, help="Limit evaluated users for quick runs")
    parser.add_argument("--max-ratings", type=int, default=500_000, help="Read at most this many ratings from large MovieLens CSVs; use 0 for no read cap")
    parser.add_argument("--max-tags", type=int, default=100_000, help="Read at most this many tags from large MovieLens CSVs; use 0 for no read cap")
    parser.add_argument("--max-users", type=int, default=None, help="Keep only the most active N users before building the rating matrix")
    parser.add_argument("--max-movies", type=int, default=None, help="Keep only the most rated N movies before building the rating matrix")
    parser.add_argument("--max-matrix-cells", type=int, default=25_000_000, help="Safety cap for dense user-movie matrix cells")
    parser.add_argument("--min-user-ratings", type=int, default=1)
    parser.add_argument("--min-movie-ratings", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Suppress info-level logs")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    log = configure_logging(level=20 if not args.quiet else 30)
    log.info("Starting MovieLens recommendation pipeline")
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
    log.info("Pipeline completed.")
    log.info(f"Training ratings: {len(result.features.ratings):,}; movies: {result.features.movies['movieId'].nunique():,}; users: {result.features.ratings['userId'].nunique():,}")
    log.info("Evaluation:\n%s", result.evaluation.to_string(index=False))
    log.info("Sample hybrid recommendations:\n%s", result.sample_recommendations[["title", "score", "reason"]].head(args.k).to_string(index=False))


if __name__ == "__main__":
    main()
