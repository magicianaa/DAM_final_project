from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore, build_feature_store
from src.data.preprocess import clean_movies, clean_ratings, clean_tags, k_fold_split
from src.evaluation.evaluate import evaluate_model
from src.models import build_default_models


def cross_validate_models(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    n_splits: int = 5,
    k: int = 10,
    max_eval_users: int | None = None,
) -> pd.DataFrame:
    folds = k_fold_split(ratings, n_splits=n_splits)

    model_names: list[str] | None = None
    fold_records: dict[str, list[dict[str, float]]] = {}

    for fold_idx, (train_ratings, test_ratings) in enumerate(folds):
        if train_ratings.empty or test_ratings.empty:
            continue

        fold_movies = movies[movies["movieId"].isin(train_ratings["movieId"].unique())].copy()
        fold_tags = tags[tags["movieId"].isin(train_ratings["movieId"].unique())].copy() if not tags.empty else tags

        features = build_feature_store(fold_movies, train_ratings, fold_tags)

        models = build_default_models()
        for model in models.values():
            model.fit(features)

        if model_names is None:
            model_names = list(models.keys())
            fold_records = {name: [] for name in model_names}

        for name in model_names:
            result = evaluate_model(models[name], features, test_ratings, k=k, max_users=max_eval_users)
            fold_records[name].append({k: v for k, v in result.items() if k != "model"})

    if model_names is None:
        return pd.DataFrame()

    # Build summary DataFrame
    metric_names = [
        "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k",
        "map_at_k", "mrr_at_k", "coverage", "diversity",
    ]
    rows = []
    for name in model_names:
        records = fold_records[name]
        if not records:
            continue
        row: dict[str, float | str] = {"model": name}
        for metric in metric_names:
            values = [r[metric] for r in records if metric in r and r.get("evaluated_users", 0) > 0]
            if values:
                row[f"{metric}_mean"] = float(pd.Series(values).mean())
                row[f"{metric}_std"] = float(pd.Series(values).std())
            else:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0
        row["n_folds_completed"] = len([r for r in records if r.get("evaluated_users", 0) > 0])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ndcg_at_k_mean", ascending=False)
