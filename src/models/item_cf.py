from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class ItemBasedCFRecommender(BaseRecommender):
    name = "item_cf"

    def __init__(self, n_similar: int = 40, min_positive_rating: float = 3.5) -> None:
        super().__init__()
        self.n_similar = n_similar
        self.min_positive_rating = min_positive_rating
        self.item_similarity: pd.DataFrame | None = None

    def fit(self, features):
        super().fit(features)
        matrix = features.rating_matrix.T
        sim = cosine_similarity(matrix)
        self.item_similarity = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
        return self

    def score(self, user_id: int) -> pd.Series:
        assert self.features is not None
        if user_id not in self.features.rating_matrix.index or self.item_similarity is None:
            return normalize_scores(self.features.popularity["weighted_rating"])
        user_ratings = self.features.rating_matrix.loc[user_id]
        liked = user_ratings[user_ratings >= self.min_positive_rating]
        if liked.empty:
            return normalize_scores(self.features.popularity["weighted_rating"])
        sims = self.item_similarity.reindex(index=liked.index).fillna(0.0)
        raw = sims.T.dot(liked) / (sims.abs().sum(axis=0) + 1e-9)
        return normalize_scores(raw.reindex(self.features.movies["movieId"]).fillna(0.0))
