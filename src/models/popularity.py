from __future__ import annotations

import pandas as pd

from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def score(self, user_id: int | None = None) -> pd.Series:
        assert self.features is not None
        return normalize_scores(self.features.popularity["weighted_rating"])
