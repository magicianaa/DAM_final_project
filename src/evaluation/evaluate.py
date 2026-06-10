from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore
from src.evaluation.metrics import (
    coverage,
    diversity,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_by_user,
)


def evaluate_model(
    model,
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    user_ids: list[int] | None = None,
) -> dict[str, float | str]:
    relevant = relevant_by_user(test_ratings)
    allowed_users = set(user_ids) if user_ids is not None else None
    train_users = set(features.ratings["userId"])
    users = [u for u in relevant if u in train_users and (allowed_users is None or u in allowed_users)]
    if max_users:
        users = users[:max_users]

    rows = []
    all_recs = []
    for user_id in users:
        rec_df = model.recommend(user_id, k=k, exclude_seen=True)
        recs = rec_df["movieId"].astype(int).tolist()
        all_recs.append(recs)
        truth = relevant[user_id]
        rows.append(
            {
                "precision": precision_at_k(recs, truth, k),
                "recall": recall_at_k(recs, truth, k),
                "ndcg": ndcg_at_k(recs, truth, k),
                "hit_rate": hit_rate_at_k(recs, truth, k),
                "diversity": diversity(recs, features),
            }
        )

    if not rows:
        return {
            "model": model.name,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "evaluated_users": 0,
        }

    df = pd.DataFrame(rows)
    return {
        "model": model.name,
        "precision_at_k": df["precision"].mean(),
        "recall_at_k": df["recall"].mean(),
        "ndcg_at_k": df["ndcg"].mean(),
        "hit_rate_at_k": df["hit_rate"].mean(),
        "coverage": coverage(all_recs, features.movies["movieId"].nunique()),
        "diversity": df["diversity"].mean(),
        "evaluated_users": len(users),
        }


def evaluate_models(
    models: dict[str, object],
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    user_ids: list[int] | None = None,
) -> pd.DataFrame:
    results = []
    for model in models.values():
        results.append(evaluate_model(model, features, test_ratings, k=k, max_users=max_users, user_ids=user_ids))
    return pd.DataFrame(results).sort_values(["ndcg_at_k", "recall_at_k"], ascending=False)
