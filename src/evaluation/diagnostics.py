from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore
from src.evaluation.metrics import relevant_by_user


def popularity_bias_for_model(
    model,
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    top_quantile: float = 0.80,
) -> dict[str, float | str | int]:
    relevant = relevant_by_user(test_ratings)
    train_users = set(features.ratings["userId"])
    users = [u for u in relevant if u in train_users]
    if max_users:
        users = users[:max_users]

    popularity = features.popularity
    count_threshold = popularity["rating_count"].quantile(top_quantile) if not popularity.empty else 0.0
    rows = []
    rec_catalog = set()
    for user_id in users:
        recs = model.recommend(user_id, k=k, exclude_seen=True)
        rec_catalog.update(recs["movieId"].astype(int).tolist())
        joined = recs[["movieId"]].merge(
            popularity[["rating_count", "rating_mean", "weighted_rating"]].reset_index(),
            on="movieId",
            how="left",
        )
        rows.append(
            {
                "avg_rating_count": joined["rating_count"].fillna(0).mean(),
                "avg_rating_mean": joined["rating_mean"].fillna(0).mean(),
                "avg_weighted_rating": joined["weighted_rating"].fillna(0).mean(),
                "popular_item_ratio": (joined["rating_count"].fillna(0) >= count_threshold).mean(),
            }
        )

    if not rows:
        return {
            "model": model.name,
            "avg_rating_count": 0.0,
            "avg_rating_mean": 0.0,
            "avg_weighted_rating": 0.0,
            "popular_item_ratio": 0.0,
            "recommended_catalog_coverage": 0.0,
            "evaluated_users": 0,
        }

    df = pd.DataFrame(rows)
    return {
        "model": model.name,
        "avg_rating_count": df["avg_rating_count"].mean(),
        "avg_rating_mean": df["avg_rating_mean"].mean(),
        "avg_weighted_rating": df["avg_weighted_rating"].mean(),
        "popular_item_ratio": df["popular_item_ratio"].mean(),
        "recommended_catalog_coverage": len(rec_catalog) / max(features.movies["movieId"].nunique(), 1),
        "evaluated_users": len(users),
    }


def popularity_bias_report(
    models: dict[str, object],
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
) -> pd.DataFrame:
    rows = [popularity_bias_for_model(model, features, test_ratings, k=k, max_users=max_users) for model in models.values()]
    return pd.DataFrame(rows).sort_values(["popular_item_ratio", "avg_rating_count"], ascending=False)
