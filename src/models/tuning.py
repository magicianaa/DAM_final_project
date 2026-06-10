from __future__ import annotations

import json
import itertools

import pandas as pd

from src.config import REPORTS_DIR
from src.data.feature_engineering import FeatureStore
from src.evaluation.evaluate import evaluate_model
from src.models.hybrid import HybridRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.user_cf import UserBasedCFRecommender


def grid_search_hybrid_weights(
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    step: float = 0.1,
) -> pd.DataFrame:
    candidates = []
    steps = int(1.0 / step)
    for cf_steps in range(steps + 1):
        for ct_steps in range(steps + 1 - cf_steps):
            pop_steps = steps - cf_steps - ct_steps
            cw = round(cf_steps * step, 10)
            tw = round(ct_steps * step, 10)
            pw = round(pop_steps * step, 10)
            if abs(cw + tw + pw - 1.0) > 1e-9:
                continue
            if cw == 1.0 or tw == 1.0 or pw == 1.0:
                continue
            candidates.append((cw, tw, pw))

    rows = []
    for cf_weight, content_weight, popularity_weight in candidates:
        model = HybridRecommender(
            cf_weight=cf_weight,
            content_weight=content_weight,
            popularity_weight=popularity_weight,
        )
        model.fit(features)
        result = evaluate_model(model, features, test_ratings, k=k, max_users=max_users)
        rows.append({
            "cf_weight": cf_weight,
            "content_weight": content_weight,
            "popularity_weight": popularity_weight,
            "ndcg_at_k": result["ndcg_at_k"],
            "map_at_k": result["map_at_k"],
            "precision_at_k": result["precision_at_k"],
            "recall_at_k": result["recall_at_k"],
        })

    return pd.DataFrame(rows).sort_values("ndcg_at_k", ascending=False)


def grid_search_user_cf_n_neighbors(
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    neighbor_values: list[int] | None = None,
) -> pd.DataFrame:
    if neighbor_values is None:
        neighbor_values = [5, 10, 20, 30, 50, 100, 200]

    rows = []
    for n_neighbors in neighbor_values:
        model = UserBasedCFRecommender(n_neighbors=n_neighbors)
        model.fit(features)
        result = evaluate_model(model, features, test_ratings, k=k, max_users=max_users)
        rows.append({
            "n_neighbors": n_neighbors,
            "ndcg_at_k": result["ndcg_at_k"],
            "map_at_k": result["map_at_k"],
            "precision_at_k": result["precision_at_k"],
            "recall_at_k": result["recall_at_k"],
        })

    return pd.DataFrame(rows).sort_values("ndcg_at_k", ascending=False)


def grid_search_svd_components(
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    component_values: list[int] | None = None,
) -> pd.DataFrame:
    if component_values is None:
        component_values = [5, 10, 20, 50, 100]

    rows = []
    for n_components in component_values:
        model = MatrixFactorizationRecommender(n_components=n_components)
        model.fit(features)
        result = evaluate_model(model, features, test_ratings, k=k, max_users=max_users)
        rows.append({
            "n_components": n_components,
            "ndcg_at_k": result["ndcg_at_k"],
            "map_at_k": result["map_at_k"],
            "precision_at_k": result["precision_at_k"],
            "recall_at_k": result["recall_at_k"],
        })

    return pd.DataFrame(rows).sort_values("ndcg_at_k", ascending=False)


def tune_all_models(
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
) -> dict:
    hybrid_results = grid_search_hybrid_weights(features, test_ratings, k=k, max_users=max_users)
    best_hybrid = hybrid_results.iloc[0] if not hybrid_results.empty else None

    cf_results = grid_search_user_cf_n_neighbors(features, test_ratings, k=k, max_users=max_users)
    best_cf = cf_results.iloc[0] if not cf_results.empty else None

    svd_results = grid_search_svd_components(features, test_ratings, k=k, max_users=max_users)
    best_svd = svd_results.iloc[0] if not svd_results.empty else None

    best_params = {}
    if best_hybrid is not None:
        best_params["hybrid"] = {
            "cf_weight": float(best_hybrid["cf_weight"]),
            "content_weight": float(best_hybrid["content_weight"]),
            "popularity_weight": float(best_hybrid["popularity_weight"]),
            "best_ndcg": float(best_hybrid["ndcg_at_k"]),
        }
    if best_cf is not None:
        best_params["user_cf"] = {
            "n_neighbors": int(best_cf["n_neighbors"]),
            "best_ndcg": float(best_cf["ndcg_at_k"]),
        }
    if best_svd is not None:
        best_params["svd"] = {
            "n_components": int(best_svd["n_components"]),
            "best_ndcg": float(best_svd["ndcg_at_k"]),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    return best_params
