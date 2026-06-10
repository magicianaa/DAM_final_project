from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


@dataclass
class FeatureStore:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    rating_matrix: pd.DataFrame
    genre_features: pd.DataFrame
    tag_features: pd.DataFrame
    content_features: pd.DataFrame
    popularity: pd.DataFrame


def build_rating_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean").fillna(0.0)


def build_genre_features(movies: pd.DataFrame) -> pd.DataFrame:
    genre_lists = movies["genres"].fillna("").apply(lambda x: [g for g in x.split("|") if g and g != "(no genres listed)"])
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(genre_lists)
    return pd.DataFrame(arr, columns=[f"genre__{c}" for c in mlb.classes_], index=movies["movieId"]).astype(float)


def build_tag_features(movies: pd.DataFrame, tags: pd.DataFrame, max_features: int = 300) -> pd.DataFrame:
    tag_map = tags.set_index("movieId")["tag_text"] if not tags.empty else pd.Series(dtype=str)
    text = (
        movies["clean_title"].fillna(movies["title"]).astype(str)
        + " "
        + movies["genres"].fillna("").str.replace("|", " ", regex=False)
        + " "
        + movies["movieId"].map(tag_map).fillna("")
    )
    if text.str.strip().eq("").all():
        return pd.DataFrame(index=movies["movieId"])
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    arr = vectorizer.fit_transform(text)
    return pd.DataFrame(arr.toarray(), columns=[f"tag__{c}" for c in vectorizer.get_feature_names_out()], index=movies["movieId"])


def build_popularity_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).rename(columns={"count": "rating_count", "mean": "rating_mean"})
    stats = movies[["movieId"]].merge(stats, on="movieId", how="left").fillna({"rating_count": 0, "rating_mean": ratings["rating"].mean() if not ratings.empty else 0})
    global_mean = ratings["rating"].mean() if not ratings.empty else 0
    min_votes = max(float(stats["rating_count"].quantile(0.60)), 1.0)
    v = stats["rating_count"].astype(float)
    r = stats["rating_mean"].astype(float)
    stats["weighted_rating"] = (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * global_mean
    stats["popularity_score"] = MinMaxScaler().fit_transform(np.log1p(stats[["rating_count"]]))
    return stats.set_index("movieId")


def build_feature_store(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame, tag_max_features: int = 300) -> FeatureStore:
    rating_matrix = build_rating_matrix(ratings)
    genre_features = build_genre_features(movies)
    tag_features = build_tag_features(movies, tags, max_features=tag_max_features)
    content_features = pd.concat([genre_features, tag_features], axis=1).fillna(0.0)
    popularity = build_popularity_features(ratings, movies)
    merged_movies = movies.merge(tags, on="movieId", how="left")
    merged_movies["tag_text"] = merged_movies["tag_text"].fillna("")
    return FeatureStore(
        movies=merged_movies,
        ratings=ratings,
        tags=tags,
        rating_matrix=rating_matrix,
        genre_features=genre_features,
        tag_features=tag_features,
        content_features=content_features,
        popularity=popularity,
    )
