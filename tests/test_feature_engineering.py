from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import (
    build_feature_store,
    build_genre_diversity_features,
    build_genre_features,
    build_popularity_features,
    build_rating_matrix,
    build_tag_features,
    build_year_features,
)


class TestBuildRatingMatrix:
    def test_shape(self):
        df = pd.DataFrame({
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [4.0, 3.0, 5.0, 2.0],
        })
        result = build_rating_matrix(df)
        assert result.shape == (2, 3)
        assert result.loc[1, 10] == 4.0
        assert result.loc[1, 30] == 0.0


class TestBuildGenreFeatures:
    def test_one_hot(self):
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["Action|Comedy", "Drama"],
        })
        result = build_genre_features(movies)
        assert "genre__Action" in result.columns
        assert result.loc[1, "genre__Action"] == 1.0
        assert result.loc[2, "genre__Action"] == 0.0


class TestBuildPopularityFeatures:
    def test_basic(self):
        ratings = pd.DataFrame({
            "userId": [1, 2, 3, 4],
            "movieId": [10, 10, 10, 20],
            "rating": [4.0, 5.0, 3.0, 5.0],
        })
        movies = pd.DataFrame({"movieId": [10, 20]})
        result = build_popularity_features(ratings, movies)
        assert "weighted_rating" in result.columns
        assert "popularity_score" in result.columns
        assert result.loc[10, "rating_count"] == 3


class TestBuildYearFeatures:
    def test_decade_onehots(self):
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A (1995)", "B (2005)"],
            "genres": ["Action", "Drama"],
            "year": [1995, 2005],
            "clean_title": ["A", "B"],
        })
        result = build_year_features(movies)
        assert "recency_score" in result.columns
        # should have decade dummies for 1990s and 2000s
        assert any("1990s" in c for c in result.columns)
        assert any("2000s" in c for c in result.columns)


class TestBuildGenreDiversityFeatures:
    def test_genre_count(self):
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres": ["Action|Comedy|Drama", "Action"],
        })
        result = build_genre_diversity_features(movies)
        assert result.loc[1, "genre_count"] == 3
        assert result.loc[2, "genre_count"] == 1


class TestBuildFeatureStore:
    def test_full_store(self):
        from src.data.preprocess import clean_movies, clean_ratings

        ratings = clean_ratings(pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [10, 20, 10],
            "rating": [4.0, 3.0, 5.0],
            "timestamp": [100, 200, 300],
        }))
        movies = clean_movies(pd.DataFrame({
            "movieId": [10, 20],
            "title": ["A (1995)", "B (2005)"],
            "genres": ["Action", "Drama"],
        }))
        tags = pd.DataFrame(columns=["movieId", "tag_text"])
        store = build_feature_store(movies, ratings, tags)
        assert store.rating_matrix.shape == (2, 2)
        assert store.content_features.shape[1] > 0
        assert store.year_features.shape[0] == 2
        assert store.genre_diversity_features.shape[0] == 2
        assert len(store.popularity) == 2
