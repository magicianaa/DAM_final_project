from __future__ import annotations

from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.item_cf import ItemBasedCFRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.popularity import PopularityRecommender
from src.models.user_cf import UserBasedCFRecommender


def build_default_models() -> dict[str, object]:
    return {
        "popularity": PopularityRecommender(),
        "user_cf": UserBasedCFRecommender(),
        "item_cf": ItemBasedCFRecommender(),
        "matrix_factorization": MatrixFactorizationRecommender(),
        "content_based": ContentBasedRecommender(),
        "hybrid": HybridRecommender(),
    }
