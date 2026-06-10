from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import FeatureStore
from src.evaluation.evaluate import evaluate_models
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.item_cf import ItemBasedCFRecommender
from src.models.popularity import PopularityRecommender


def run_ablation_experiments(
    features: FeatureStore,
    test_ratings: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
) -> pd.DataFrame:
    """Compare rating, content, popularity, and hybrid score components."""
    models = {
        "popularity_only": PopularityRecommender(),
        "cf_only_item": ItemBasedCFRecommender(),
        "content_only": ContentBasedRecommender(),
        "cf_content_no_pop": HybridRecommender(cf_weight=0.55, content_weight=0.45, popularity_weight=0.0),
        "cf_pop_no_content": HybridRecommender(cf_weight=0.70, content_weight=0.0, popularity_weight=0.30),
        "content_pop_no_cf": HybridRecommender(cf_weight=0.0, content_weight=0.70, popularity_weight=0.30),
        "hybrid_default": HybridRecommender(cf_weight=0.45, content_weight=0.35, popularity_weight=0.20),
    }
    for name, model in models.items():
        model.name = name
        model.fit(features)
    results = evaluate_models(models, features, test_ratings, k=k, max_users=max_users)
    results = results.rename(columns={"model": "ablation_setting"})
    return results
