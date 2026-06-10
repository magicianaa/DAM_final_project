from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data.feature_engineering import FeatureStore


class BaseRecommender(ABC):
    name = "base"

    def __init__(self) -> None:
        self.features: FeatureStore | None = None

    def fit(self, features: FeatureStore) -> "BaseRecommender":
        self.features = features
        return self

    @abstractmethod
    def score(self, user_id: int) -> pd.Series:
        """Return a movieId-indexed score series."""

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        from src.models.utils import top_n_frame

        assert self.features is not None
        return top_n_frame(
            self.score(user_id),
            self.features.movies,
            k,
            self.name,
            self.features.ratings,
            user_id=user_id,
            exclude_seen=exclude_seen,
        )
