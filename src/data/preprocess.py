from __future__ import annotations

import re

import pandas as pd


def clean_ratings(ratings: pd.DataFrame, min_user_ratings: int = 1, min_movie_ratings: int = 1) -> pd.DataFrame:
    df = ratings.copy()
    df = df.dropna(subset=["userId", "movieId", "rating"])
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    df = df.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")

    if min_user_ratings > 1:
        keep_users = df["userId"].value_counts()
        df = df[df["userId"].isin(keep_users[keep_users >= min_user_ratings].index)]
    if min_movie_ratings > 1:
        keep_movies = df["movieId"].value_counts()
        df = df[df["movieId"].isin(keep_movies[keep_movies >= min_movie_ratings].index)]
    return df.reset_index(drop=True)


def limit_interaction_scope(
    ratings: pd.DataFrame,
    max_users: int | None = None,
    max_movies: int | None = None,
    max_matrix_cells: int | None = 25_000_000,
) -> pd.DataFrame:
    """Limit the dense user-movie matrix size while preserving active entities.

    Full MovieLens releases can contain tens of millions of ratings. The current
    MVP recommenders intentionally use a dense matrix for clarity, so this helper
    keeps the experiment tractable by selecting the most active users/movies.
    """
    df = ratings.copy()
    if df.empty:
        return df

    user_counts = df["userId"].value_counts()
    movie_counts = df["movieId"].value_counts()

    if max_matrix_cells and max_matrix_cells > 0:
        n_users = len(user_counts)
        n_movies = len(movie_counts)
        if n_users * n_movies > max_matrix_cells:
            if max_users is None:
                max_users = max(1, min(n_users, int(max_matrix_cells ** 0.5)))
            if max_movies is None:
                max_movies = max(1, min(n_movies, max_matrix_cells // max_users))

    if max_users and max_users > 0 and len(user_counts) > max_users:
        keep_users = set(user_counts.head(max_users).index)
        df = df[df["userId"].isin(keep_users)]
    if max_movies and max_movies > 0:
        movie_counts = df["movieId"].value_counts()
        if len(movie_counts) > max_movies:
            keep_movies = set(movie_counts.head(max_movies).index)
            df = df[df["movieId"].isin(keep_movies)]

    return df.reset_index(drop=True)


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    df = movies.copy()
    df = df.dropna(subset=["movieId", "title"]).drop_duplicates("movieId")
    df["movieId"] = df["movieId"].astype(int)
    df["genres"] = df["genres"].fillna("(no genres listed)")
    df["year"] = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].str.replace(r"\s+\(\d{4}\)$", "", regex=True)
    return df.reset_index(drop=True)


def clean_tags(tags: pd.DataFrame | None) -> pd.DataFrame:
    if tags is None or tags.empty:
        return pd.DataFrame(columns=["movieId", "tag_text"])
    df = tags.copy().dropna(subset=["movieId", "tag"])
    df["movieId"] = df["movieId"].astype(int)
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    return (
        df.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(sorted(set(t for t in s if t))))
        .reset_index(name="tag_text")
    )


def _extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def leave_one_out_split(
    ratings: pd.DataFrame,
    min_interactions: int = 3,
    relevant_threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for _, group in ratings.sort_values("timestamp").groupby("userId"):
        if len(group) < min_interactions:
            train_parts.append(group)
            continue
        relevant = group[group["rating"] >= relevant_threshold]
        if relevant.empty:
            train_parts.append(group)
            continue
        test_row = relevant.tail(1)
        test_parts.append(test_row)
        train_parts.append(group.drop(test_row.index))
    train = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test
