from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_scores(scores: pd.Series) -> pd.Series:
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    min_v = scores.min()
    max_v = scores.max()
    if max_v == min_v:
        return pd.Series(np.zeros(len(scores)), index=scores.index)
    return (scores - min_v) / (max_v - min_v)


def seen_movies(ratings: pd.DataFrame, user_id: int) -> set[int]:
    return set(ratings.loc[ratings["userId"] == user_id, "movieId"].astype(int))


def top_n_frame(
    scores: pd.Series,
    movies: pd.DataFrame,
    k: int,
    model_name: str,
    train_ratings: pd.DataFrame,
    user_id: int | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    scores = scores.copy()
    if user_id is not None and exclude_seen:
        scores = scores.drop(index=list(seen_movies(train_ratings, user_id)), errors="ignore")
    top = scores.sort_values(ascending=False).head(k).reset_index()
    top.columns = ["movieId", "score"]
    out = top.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    out["model"] = model_name
    return out[["model", "movieId", "title", "genres", "score"]]
