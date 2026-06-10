from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data.feature_engineering import FeatureStore
from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class ColdStartRecommender(BaseRecommender):
    name = "cold_start"

    def __init__(self, popularity_weight: float = 0.35) -> None:
        super().__init__()
        self.popularity_weight = popularity_weight

    def fit(self, features):
        super().fit(features)
        return self

    def score(self, user_id: int) -> pd.Series:
        raise NotImplementedError(
            "ColdStartRecommender requires explicit genre/movie/keyword preferences. "
            "Use recommend_for_new_user() instead."
        )

    def recommend_for_new_user(
        self,
        genres: list[str] | None = None,
        liked_movie_ids: list[int] | None = None,
        keywords: str = "",
        k: int = 10,
    ) -> pd.DataFrame:
        assert self.features is not None
        genres = genres or []
        liked_movie_ids = liked_movie_ids or []
        genre_cols = [f"genre__{g}" for g in genres if f"genre__{g}" in self.features.genre_features.columns]
        score = pd.Series(0.0, index=self.features.movies["movieId"])

        if genre_cols:
            genre_score = self.features.genre_features[genre_cols].sum(axis=1)
            score = score.add(normalize_scores(genre_score), fill_value=0.0)

        if liked_movie_ids:
            liked_vecs = self.features.content_features.reindex(liked_movie_ids).dropna(how="all")
            if not liked_vecs.empty:
                profile = liked_vecs.mean(axis=0).to_numpy().reshape(1, -1)
                sim = cosine_similarity(profile, self.features.content_features.to_numpy())[0]
                score = score.add(pd.Series(sim, index=self.features.content_features.index), fill_value=0.0)
                score = score.drop(index=liked_movie_ids, errors="ignore")

        if keywords.strip():
            words = set(keywords.lower().split())
            text = (
                self.features.movies["title"].fillna("")
                + " "
                + self.features.movies["genres"].fillna("")
                + " "
                + self.features.movies["tag_text"].fillna("")
            ).str.lower()
            keyword_score = text.apply(lambda x: sum(1 for w in words if w in x))
            score = score.add(
                normalize_scores(pd.Series(keyword_score.to_numpy(), index=self.features.movies["movieId"])),
                fill_value=0.0,
            )

        score = score.add(
            self.popularity_weight * normalize_scores(self.features.popularity["weighted_rating"]),
            fill_value=0.0,
        )
        top = score.sort_values(ascending=False).head(k).reset_index()
        top.columns = ["movieId", "score"]
        out = top.merge(self.features.movies[["movieId", "title", "genres"]], on="movieId", how="left")
        out["model"] = "cold_start"
        out["reason"] = out.apply(lambda r: _cold_start_reason(r, genres, liked_movie_ids, keywords), axis=1)
        return out[["model", "movieId", "title", "genres", "score", "reason"]]


def _cold_start_reason(row: pd.Series, genres: list[str], liked_movie_ids: list[int], keywords: str) -> str:
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


def recommend_for_new_user(
    features: FeatureStore,
    genres: list[str] | None = None,
    liked_movie_ids: list[int] | None = None,
    keywords: str = "",
    k: int = 10,
) -> pd.DataFrame:
    """Convenience wrapper for backward compatibility."""
    model = ColdStartRecommender()
    model.fit(features)
    return model.recommend_for_new_user(
        genres=genres,
        liked_movie_ids=liked_movie_ids,
        keywords=keywords,
        k=k,
    )
