from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore
from src.models.hybrid import HybridRecommender


def explain_recommendations(
    recommendations: pd.DataFrame,
    features: FeatureStore,
    user_id: int | None = None,
    model=None,
) -> pd.DataFrame:
    out = recommendations.copy()
    out["reason"] = [explain_row(row, features, user_id, model) for _, row in out.iterrows()]
    return out


def explain_row(row: pd.Series, features: FeatureStore, user_id: int | None, model=None) -> str:
    movie_id = int(row["movieId"])
    genres = str(row.get("genres", "")).split("|")
    reasons = []

    if user_id is not None and user_id in set(features.ratings["userId"]):
        user_high = features.ratings[(features.ratings["userId"] == user_id) & (features.ratings["rating"] >= 4.0)]
        liked_movies = features.movies[features.movies["movieId"].isin(user_high["movieId"])]
        liked_genres: set[str] = set()
        for value in liked_movies["genres"].dropna():
            liked_genres.update(value.split("|"))
        overlap = [g for g in genres if g in liked_genres and g != "(no genres listed)"]
        if overlap:
            reasons.append("genre overlap with your high-rated movies: " + ", ".join(overlap[:3]))

    pop = features.popularity.loc[movie_id] if movie_id in features.popularity.index else None
    if pop is not None and pop["rating_count"] > features.popularity["rating_count"].median():
        reasons.append(f"popular and well-rated ({pop['rating_count']:.0f} ratings, avg {pop['rating_mean']:.2f})")

    if isinstance(model, HybridRecommender) and user_id is not None:
        parts = model.score_components(user_id).loc[movie_id]
        top_source = parts.sort_values(ascending=False).index[0].replace("_score", "")
        reasons.append(f"hybrid score is mainly driven by {top_source}")

    if not reasons:
        reasons.append("recommended by the selected model based on available ratings/content signals")
    return "; ".join(reasons)
