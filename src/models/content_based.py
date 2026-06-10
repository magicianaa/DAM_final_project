from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"

    def __init__(self, min_positive_rating: float = 3.5) -> None:
        super().__init__()
        self.min_positive_rating = min_positive_rating
        self.content_similarity: pd.DataFrame | None = None

    def fit(self, features):
        super().fit(features)
        sim = cosine_similarity(features.content_features.fillna(0.0))
        idx = features.content_features.index
        self.content_similarity = pd.DataFrame(sim, index=idx, columns=idx)
        return self

    def user_profile_scores(self, user_id: int) -> pd.Series:
        assert self.features is not None
        if user_id not in set(self.features.ratings["userId"]):
            return normalize_scores(self.features.popularity["weighted_rating"])
        user_ratings = self.features.ratings[self.features.ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= self.min_positive_rating]
        if liked.empty:
            liked = user_ratings.sort_values("rating", ascending=False).head(3)
        feature_matrix = self.features.content_features
        liked_features = feature_matrix.reindex(liked["movieId"]).fillna(0.0)
        weights = liked["rating"].to_numpy(dtype=float)
        if liked_features.empty or weights.sum() == 0:
            return normalize_scores(self.features.popularity["weighted_rating"])
        profile = np.average(liked_features.to_numpy(), axis=0, weights=weights)
        scores = cosine_similarity(profile.reshape(1, -1), feature_matrix.to_numpy())[0]
        return normalize_scores(pd.Series(scores, index=feature_matrix.index))

    def score(self, user_id: int) -> pd.Series:
        return self.user_profile_scores(user_id)
