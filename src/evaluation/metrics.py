from __future__ import annotations

import itertools
from math import log2

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data.feature_engineering import FeatureStore


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for idx, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    unique_items = set(itertools.chain.from_iterable(all_recommendations))
    return len(unique_items) / catalog_size


def diversity(recommended: list[int], features: FeatureStore) -> float:
    ids = [movie_id for movie_id in recommended if movie_id in features.content_features.index]
    if len(ids) < 2:
        return 0.0
    arr = features.content_features.loc[ids].to_numpy()
    sim = cosine_similarity(arr)
    pairs = sim[np.triu_indices_from(sim, k=1)]
    return float(1.0 - np.mean(pairs)) if len(pairs) else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0 or not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def mean_average_precision_at_k(
    all_recommendations: list[list[int]],
    all_relevant: list[set[int]],
    k: int,
) -> float:
    if not all_recommendations:
        return 0.0
    ap_scores = [
        average_precision_at_k(recs, rel, k)
        for recs, rel in zip(all_recommendations, all_relevant)
    ]
    return float(np.mean(ap_scores))


def mean_reciprocal_rank(
    all_recommendations: list[list[int]],
    all_relevant: list[set[int]],
    k: int,
) -> float:
    ranks = []
    for recs, rel in zip(all_recommendations, all_relevant):
        for rank, movie_id in enumerate(recs[:k], start=1):
            if movie_id in rel:
                ranks.append(1.0 / rank)
                break
        else:
            ranks.append(0.0)
    return float(np.mean(ranks))


def relevant_by_user(test_ratings: pd.DataFrame, threshold: float = 4.0) -> dict[int, set[int]]:
    relevant = test_ratings[test_ratings["rating"] >= threshold]
    return relevant.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int))).to_dict()
