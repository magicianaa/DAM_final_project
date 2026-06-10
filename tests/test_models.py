from __future__ import annotations

import pandas as pd
import pytest

from src.data.feature_engineering import build_feature_store
from src.models.base import BaseRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.item_cf import ItemBasedCFRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.popularity import PopularityRecommender
from src.models.user_cf import UserBasedCFRecommender
from src.models.cold_start import ColdStartRecommender, recommend_for_new_user


@pytest.fixture
def sample_features():
    from src.data.preprocess import clean_movies, clean_ratings

    ratings = clean_ratings(pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3],
        "movieId": [10, 20, 30, 10, 40, 20],
        "rating": [4.0, 3.5, 5.0, 2.0, 4.5, 3.0],
        "timestamp": range(6),
    }))
    movies = clean_movies(pd.DataFrame({
        "movieId": [10, 20, 30, 40],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Matrix, The (1999)",
            "Inception (2010)",
        ],
        "genres": [
            "Animation|Children",
            "Adventure|Children",
            "Action|Sci-Fi",
            "Action|Sci-Fi|Thriller",
        ],
    }))
    tags = pd.DataFrame(columns=["movieId", "tag_text"])
    return build_feature_store(movies, ratings, tags)


class TestBaseRecommender:
    def test_fit_returns_self(self, sample_features):
        class Dummy(BaseRecommender):
            name = "dummy"

            def score(self, user_id):
                return pd.Series(dtype=float)

        model = Dummy()
        result = model.fit(sample_features)
        assert result is model
        assert model.features is sample_features

    def test_recommend_returns_dataframe(self, sample_features):
        class Dummy(BaseRecommender):
            name = "dummy"

            def score(self, user_id):
                return pd.Series(0.5, index=sample_features.movies["movieId"])

        model = Dummy()
        model.fit(sample_features)
        result = model.recommend(1, k=3, exclude_seen=True)
        assert isinstance(result, pd.DataFrame)
        assert "movieId" in result.columns
        assert "title" in result.columns
        assert len(result) <= 3


class TestPopularityRecommender:
    def test_score_ignores_user(self, sample_features):
        model = PopularityRecommender()
        model.fit(sample_features)
        s1 = model.score(1)
        s2 = model.score(2)
        pd.testing.assert_series_equal(s1, s2)


class TestUserBasedCF:
    def test_fit_and_score(self, sample_features):
        model = UserBasedCFRecommender(n_neighbors=5)
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)
        assert not result.isna().all()


class TestItemBasedCF:
    def test_no_n_similar_param(self, sample_features):
        model = ItemBasedCFRecommender()
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)


class TestMatrixFactorization:
    def test_fit_and_score(self, sample_features):
        model = MatrixFactorizationRecommender(n_components=2)
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)


class TestContentBased:
    def test_fit_and_score(self, sample_features):
        model = ContentBasedRecommender()
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)


class TestHybridRecommender:
    def test_score_components(self, sample_features):
        model = HybridRecommender()
        model.fit(sample_features)
        parts = model.score_components(1)
        assert "cf_score" in parts.columns
        assert "content_score" in parts.columns
        assert "popularity_score" in parts.columns

    def test_score(self, sample_features):
        model = HybridRecommender()
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)
        assert result.max() <= 1.0 + 1e-9

    def test_custom_weights(self, sample_features):
        model = HybridRecommender(cf_weight=0.5, content_weight=0.3, popularity_weight=0.2)
        model.fit(sample_features)
        result = model.score(1)
        assert isinstance(result, pd.Series)


class TestColdStartRecommender:
    def test_class_based(self, sample_features):
        model = ColdStartRecommender()
        model.fit(sample_features)
        result = model.recommend_for_new_user(genres=["Action"], k=3)
        assert len(result) == 3
        assert "reason" in result.columns

    def test_legacy_api(self, sample_features):
        result = recommend_for_new_user(sample_features, genres=["Action"], k=3)
        assert len(result) == 3
        assert "reason" in result.columns

    def test_with_keywords(self, sample_features):
        model = ColdStartRecommender()
        model.fit(sample_features)
        result = model.recommend_for_new_user(keywords="space", k=2)
        assert len(result) == 2

    def test_score_raises(self, sample_features):
        model = ColdStartRecommender()
        model.fit(sample_features)
        with pytest.raises(NotImplementedError):
            model.score(1)
