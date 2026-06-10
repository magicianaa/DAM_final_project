from __future__ import annotations

import itertools

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from src.data.feature_engineering import FeatureStore
from src.evaluation.metrics import ndcg_at_k, relevant_by_user


def collect_per_user_ndcg(
    model,
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
) -> pd.Series:
    relevant = relevant_by_user(test_ratings)
    train_users = set(features.ratings["userId"])
    users = [u for u in relevant if u in train_users]
    if max_users:
        users = users[:max_users]

    scores: dict[int, float] = {}
    for user_id in users:
        rec_df = model.recommend(user_id, k=k, exclude_seen=True)
        recs = rec_df["movieId"].astype(int).tolist()
        scores[user_id] = ndcg_at_k(recs, relevant[user_id], k)

    return pd.Series(scores, name=model.name)


def paired_ttest(
    scores_a: pd.Series,
    scores_b: pd.Series,
    alternative: str = "two-sided",
) -> dict:
    common_users = scores_a.index.intersection(scores_b.index)
    if len(common_users) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "mean_difference": 0.0, "significant": False}
    a = scores_a[common_users]
    b = scores_b[common_users]
    result = ttest_rel(a, b, alternative=alternative)
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_difference": float(a.mean() - b.mean()),
        "significant": bool(result.pvalue < 0.05),
    }


def wilcoxon_signed_rank_test(
    scores_a: pd.Series,
    scores_b: pd.Series,
    alternative: str = "two-sided",
) -> dict:
    common_users = scores_a.index.intersection(scores_b.index)
    if len(common_users) < 5:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    a = scores_a[common_users]
    b = scores_b[common_users]
    diff = a.values - b.values
    if (diff == 0).all():
        return {"statistic": float("nan"), "p_value": 1.0, "significant": False}
    try:
        result = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "significant": bool(result.pvalue < 0.05),
        }
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}


def model_significance_report(
    models: dict[str, object],
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    test: str = "wilcoxon",
) -> pd.DataFrame:
    per_user_scores = {}
    for name, model in models.items():
        per_user_scores[name] = collect_per_user_ndcg(model, features, test_ratings, k=k, max_users=max_users)

    test_func = wilcoxon_signed_rank_test if test == "wilcoxon" else paired_ttest

    columns = [
        "model_a",
        "model_b",
        "mean_ndcg_a",
        "mean_ndcg_b",
        "diff",
        "test",
        "p_value",
        "significant",
    ]
    rows = []
    for name_a, name_b in itertools.combinations(per_user_scores, 2):
        result = test_func(per_user_scores[name_a], per_user_scores[name_b])
        mean_a = per_user_scores[name_a].mean()
        mean_b = per_user_scores[name_b].mean()
        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "mean_ndcg_a": mean_a,
            "mean_ndcg_b": mean_b,
            "diff": mean_a - mean_b,
            "test": test,
            "p_value": result["p_value"],
            "significant": result["significant"],
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("p_value", na_position="last")
