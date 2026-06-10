from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all recommenders and save reports.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--max-ratings", type=int, default=500_000)
    parser.add_argument("--max-tags", type=int, default=100_000)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-movies", type=int, default=None)
    parser.add_argument("--max-matrix-cells", type=int, default=25_000_000)
    parser.add_argument("--min-user-ratings", type=int, default=1)
    parser.add_argument("--min-movie-ratings", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
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
    print(result.evaluation.to_string(index=False))
