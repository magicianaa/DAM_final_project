from __future__ import annotations

import pandas as pd

from src.models.base import BaseRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.item_cf import ItemBasedCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.utils import normalize_scores


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(
        self,
        cf_weight: float = 0.45,
        content_weight: float = 0.35,
        popularity_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.cf = ItemBasedCFRecommender()
        self.content = ContentBasedRecommender()
        self.popularity = PopularityRecommender()

    def fit(self, features):
        super().fit(features)
        self.cf.fit(features)
        self.content.fit(features)
        self.popularity.fit(features)
        return self

    def score_components(self, user_id: int) -> pd.DataFrame:
        assert self.features is not None
        idx = self.features.movies["movieId"]
        return pd.DataFrame(
            {
                "cf_score": self.cf.score(user_id).reindex(idx).fillna(0.0),
                "content_score": self.content.score(user_id).reindex(idx).fillna(0.0),
                "popularity_score": self.popularity.score(user_id).reindex(idx).fillna(0.0),
            },
            index=idx,
        )

    def score(self, user_id: int) -> pd.Series:
        parts = self.score_components(user_id)
        final = (
            self.cf_weight * parts["cf_score"]
            + self.content_weight * parts["content_score"]
            + self.popularity_weight * parts["popularity_score"]
        )
        return normalize_scores(final)
