from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data.feature_engineering import FeatureStore
from src.models.utils import normalize_scores


def recommend_for_new_user(
    features: FeatureStore,
    genres: list[str] | None = None,
    liked_movie_ids: list[int] | None = None,
    keywords: str = "",
    k: int = 10,
) -> pd.DataFrame:
    genres = genres or []
    liked_movie_ids = liked_movie_ids or []
    genre_cols = [f"genre__{g}" for g in genres if f"genre__{g}" in features.genre_features.columns]
    score = pd.Series(0.0, index=features.movies["movieId"])

    if genre_cols:
        genre_score = features.genre_features[genre_cols].sum(axis=1)
        score = score.add(normalize_scores(genre_score), fill_value=0.0)

    if liked_movie_ids:
        liked_vecs = features.content_features.reindex(liked_movie_ids).dropna(how="all")
        if not liked_vecs.empty:
            profile = liked_vecs.mean(axis=0).to_numpy().reshape(1, -1)
            sim = cosine_similarity(profile, features.content_features.to_numpy())[0]
            score = score.add(pd.Series(sim, index=features.content_features.index), fill_value=0.0)
            score = score.drop(index=liked_movie_ids, errors="ignore")

    if keywords.strip():
        words = set(keywords.lower().split())
        text = (features.movies["title"].fillna("") + " " + features.movies["genres"].fillna("") + " " + features.movies["tag_text"].fillna("")).str.lower()
        keyword_score = text.apply(lambda x: sum(1 for w in words if w in x))
        score = score.add(normalize_scores(pd.Series(keyword_score.to_numpy(), index=features.movies["movieId"])), fill_value=0.0)

    score = score.add(0.35 * normalize_scores(features.popularity["weighted_rating"]), fill_value=0.0)
    top = score.sort_values(ascending=False).head(k).reset_index()
    top.columns = ["movieId", "score"]
    out = top.merge(features.movies[["movieId", "title", "genres"]], on="movieId", how="left")
    out["model"] = "cold_start"
    out["reason"] = out.apply(lambda r: cold_start_reason(r, genres, liked_movie_ids, keywords), axis=1)
    return out[["model", "movieId", "title", "genres", "score", "reason"]]


def cold_start_reason(row: pd.Series, genres: list[str], liked_movie_ids: list[int], keywords: str) -> str:
    reasons = []
    matched_genres = [g for g in genres if g in str(row.get("genres", "")).split("|")]
    if matched_genres:
        reasons.append("matches preferred genres: " + ", ".join(matched_genres[:3]))
    if liked_movie_ids:
        reasons.append("similar to selected favorite movies")
    if keywords.strip():
        reasons.append("matches keyword preference")
    reasons.append("also has strong popularity/rating signal")
    return "; ".join(reasons)
