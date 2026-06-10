from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MOVIELENS_FULL_DIR, MOVIELENS_SMALL_DIR, RAW_DIR
from src.data.download import REQUIRED_FILES, has_movielens_files


def resolve_movielens_dir(path: str | Path | None = None) -> Path:
    if path is None:
        for candidate in [MOVIELENS_FULL_DIR, MOVIELENS_SMALL_DIR]:
            if has_movielens_files(candidate):
                return candidate
        for candidate in sorted(RAW_DIR.iterdir() if RAW_DIR.exists() else [], key=lambda p: p.name):
            if candidate.is_dir() and has_movielens_files(candidate):
                return candidate
        base = MOVIELENS_SMALL_DIR
    else:
        base = Path(path)
    if has_movielens_files(base):
        return base
    nested = base / "ml-latest-small"
    if has_movielens_files(nested):
        return nested
    nested = base / "ml-32m"
    if has_movielens_files(nested):
        return nested
    missing = ", ".join(REQUIRED_FILES)
    raise FileNotFoundError(
        f"MovieLens files not found in {base}. Expected: {missing}. "
        "Run `python main.py --download` or manually unzip a MovieLens dataset under data/raw."
    )


def _nrows(limit: int | None) -> int | None:
    return limit if limit and limit > 0 else None


def load_movielens(
    path: str | Path | None = None,
    max_ratings: int | None = None,
    max_tags: int | None = None,
) -> dict[str, pd.DataFrame]:
    data_dir = resolve_movielens_dir(path)
    return {
        "ratings": pd.read_csv(data_dir / "ratings.csv", nrows=_nrows(max_ratings)),
        "movies": pd.read_csv(data_dir / "movies.csv"),
        "tags": pd.read_csv(data_dir / "tags.csv", nrows=_nrows(max_tags)),
        "links": pd.read_csv(data_dir / "links.csv"),
    }


def make_sample_movielens() -> dict[str, pd.DataFrame]:
    """Small deterministic dataset used for smoke tests when raw data is absent."""
    movies = pd.DataFrame(
        [
            (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
            (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
            (3, "Heat (1995)", "Action|Crime|Thriller"),
            (4, "Sabrina (1995)", "Comedy|Romance"),
            (5, "GoldenEye (1995)", "Action|Adventure|Thriller"),
            (6, "Sense and Sensibility (1995)", "Drama|Romance"),
            (7, "Seven (a.k.a. Se7en) (1995)", "Mystery|Thriller"),
            (8, "Usual Suspects, The (1995)", "Crime|Mystery|Thriller"),
            (9, "Apollo 13 (1995)", "Adventure|Drama|IMAX"),
            (10, "Babe (1995)", "Children|Drama"),
            (11, "Matrix, The (1999)", "Action|Sci-Fi|Thriller"),
            (12, "Finding Nemo (2003)", "Adventure|Animation|Children|Comedy"),
        ],
        columns=["movieId", "title", "genres"],
    )
    rows = [
        (1, 1, 5.0, 100), (1, 2, 4.0, 101), (1, 12, 4.5, 102), (1, 4, 2.0, 103),
        (2, 3, 4.5, 100), (2, 5, 4.0, 101), (2, 7, 4.0, 102), (2, 11, 5.0, 103),
        (3, 4, 4.0, 100), (3, 6, 5.0, 101), (3, 10, 4.5, 102), (3, 1, 3.5, 103),
        (4, 3, 3.5, 100), (4, 7, 5.0, 101), (4, 8, 4.5, 102), (4, 11, 4.0, 103),
        (5, 1, 4.5, 100), (5, 9, 4.0, 101), (5, 10, 4.0, 102), (5, 12, 5.0, 103),
        (6, 2, 4.0, 100), (6, 5, 4.5, 101), (6, 9, 3.5, 102), (6, 11, 5.0, 103),
    ]
    tags = pd.DataFrame(
        [
            (1, 1, "pixar", 100), (1, 12, "fish animation", 101),
            (2, 11, "sci-fi action", 100), (4, 8, "twist mystery", 101),
            (3, 6, "period romance", 100), (5, 9, "space nasa", 100),
        ],
        columns=["userId", "movieId", "tag", "timestamp"],
    )
    links = pd.DataFrame({"movieId": movies["movieId"], "imdbId": range(1, 13), "tmdbId": range(101, 113)})
    return {"ratings": pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]), "movies": movies, "tags": tags, "links": links}
