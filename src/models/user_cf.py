from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class UserBasedCFRecommender(BaseRecommender):
    name = "user_cf"

    def __init__(self, n_neighbors: int = 30) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.user_similarity: pd.DataFrame | None = None
        self.user_means: pd.Series | None = None

    def fit(self, features):
        super().fit(features)
        matrix = features.rating_matrix
        self.user_means = matrix.replace(0, np.nan).mean(axis=1).fillna(0.0)
        centered = matrix.sub(self.user_means, axis=0).where(matrix.ne(0), 0.0)
        sim = cosine_similarity(centered)
        self.user_similarity = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
        return self

    def score(self, user_id: int) -> pd.Series:
        assert self.features is not None
        if user_id not in self.features.rating_matrix.index or self.user_similarity is None:
            return normalize_scores(self.features.popularity["weighted_rating"])
        sims = self.user_similarity.loc[user_id].drop(index=user_id, errors="ignore")
        sims = sims[sims > 0].sort_values(ascending=False).head(self.n_neighbors)
        if sims.empty:
            return normalize_scores(self.features.popularity["weighted_rating"])
        neighbor_ratings = self.features.rating_matrix.loc[sims.index]
        weighted = neighbor_ratings.T.dot(sims) / (np.abs(sims).sum() + 1e-9)
        return normalize_scores(weighted.reindex(self.features.movies["movieId"]).fillna(0.0))
