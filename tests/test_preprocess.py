from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import (
    clean_movies,
    clean_ratings,
    clean_tags,
    k_fold_split,
    leave_one_out_split,
)


class TestCleanRatings:
    def test_basic(self):
        df = pd.DataFrame({
            "userId": [1, 2, 1],
            "movieId": [10, 20, 10],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": [100, 200, 300],
        })
        result = clean_ratings(df)
        assert len(result) == 2  # dedup keeps last for (userId=1, movieId=10)

    def test_drops_nulls(self):
        df = pd.DataFrame({
            "userId": [1, None, 1],
            "movieId": [10, 20, 10],
            "rating": [4.0, 3.5, None],
            "timestamp": [100, 200, 300],
        })
        result = clean_ratings(df)
        assert len(result) == 1

    def test_dedup_keeps_last(self):
        df = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [10, 10],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        })
        result = clean_ratings(df)
        assert len(result) == 1
        assert result["rating"].iloc[0] == 5.0


class TestCleanMovies:
    def test_extracts_year(self):
        df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": ["Animation|Children", "Adventure"],
        })
        result = clean_movies(df)
        assert result["year"].iloc[0] == 1995
        assert result["clean_title"].iloc[0] == "Toy Story"

    def test_fills_missing_genres(self):
        df = pd.DataFrame({
            "movieId": [1],
            "title": ["Test (2000)"],
            "genres": [None],
        })
        result = clean_movies(df)
        assert result["genres"].iloc[0] == "(no genres listed)"


class TestLeaveOneOutSplit:
    def test_holds_out_last_high_rating(self):
        df = pd.DataFrame({
            "userId": [1, 1, 1],
            "movieId": [10, 20, 30],
            "rating": [3.0, 4.5, 5.0],
            "timestamp": [100, 200, 300],
        })
        train, test = leave_one_out_split(df, min_interactions=3)
        assert len(train) == 2
        assert len(test) == 1
        assert test["movieId"].iloc[0] == 30

    def test_few_interactions_stay_in_train(self):
        df = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [10, 20],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        })
        train, test = leave_one_out_split(df, min_interactions=3)
        assert len(train) == 2
        assert len(test) == 0


class TestKFoldSplit:
    def test_n_splits(self):
        df = pd.DataFrame({
            "userId": [1, 1, 1, 1, 2, 2, 2, 2],
            "movieId": range(1, 9),
            "rating": [4.0] * 8,
            "timestamp": range(8),
        })
        folds = k_fold_split(df, n_splits=3)
        assert len(folds) == 3
        for train, test in folds:
            assert len(train) + len(test) == 8
            train_keys = set(zip(train["userId"], train["movieId"]))
            test_keys = set(zip(test["userId"], test["movieId"]))
            assert train_keys.isdisjoint(test_keys)

    def test_min_splits(self):
        df = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [10, 20],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        })
        with pytest.raises(ValueError):
            k_fold_split(df, n_splits=1)
