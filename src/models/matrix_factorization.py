from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.models.base import BaseRecommender
from src.models.utils import normalize_scores


class MatrixFactorizationRecommender(BaseRecommender):
    name = "matrix_factorization"

    def __init__(self, n_components: int = 20, random_state: int = 42) -> None:
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.predictions: pd.DataFrame | None = None

    def fit(self, features):
        super().fit(features)
        matrix = features.rating_matrix
        if min(matrix.shape) <= 2:
            self.predictions = matrix.copy()
            return self
        n_components = min(self.n_components, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        user_factors = svd.fit_transform(matrix)
        reconstructed = np.dot(user_factors, svd.components_)
        self.predictions = pd.DataFrame(reconstructed, index=matrix.index, columns=matrix.columns)
        return self

    def score(self, user_id: int) -> pd.Series:
        assert self.features is not None
        if self.predictions is None or user_id not in self.predictions.index:
            return normalize_scores(self.features.popularity["weighted_rating"])
        return normalize_scores(self.predictions.loc[user_id].reindex(self.features.movies["movieId"]).fillna(0.0))
