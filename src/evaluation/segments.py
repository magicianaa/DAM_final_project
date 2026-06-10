from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore
from src.evaluation.evaluate import evaluate_model


def build_user_activity_segments(features: FeatureStore) -> pd.DataFrame:
    counts = features.ratings.groupby("userId").size().sort_values()
    if counts.empty:
        return pd.DataFrame(columns=["userId", "activity_segment", "train_rating_count"])

    n = len(counts)
    rows = []
    for idx, (user_id, count) in enumerate(counts.items()):
        rank = idx / max(n, 1)
        if rank < 1 / 3:
            segment = "low_activity"
        elif rank < 2 / 3:
            segment = "medium_activity"
        else:
            segment = "high_activity"
        rows.append({"userId": int(user_id), "activity_segment": segment, "train_rating_count": int(count)})
    return pd.DataFrame(rows)


def evaluate_user_segments(
    models: dict[str, object],
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users_per_segment: int | None = None,
) -> pd.DataFrame:
    segments = build_user_activity_segments(features)
    results = []
    for segment, group in segments.groupby("activity_segment"):
        user_ids = group["userId"].astype(int).tolist()
        for model in models.values():
            row = evaluate_model(
                model,
                features,
                test_ratings,
                k=k,
                max_users=max_users_per_segment,
                user_ids=user_ids,
            )
            row["activity_segment"] = segment
            row["segment_train_users"] = len(user_ids)
            results.append(row)
    if not results:
        return pd.DataFrame()
    cols = ["activity_segment", "model", "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k", "coverage", "diversity", "evaluated_users", "segment_train_users"]
    return pd.DataFrame(results)[cols].sort_values(["activity_segment", "ndcg_at_k"], ascending=[True, False])
