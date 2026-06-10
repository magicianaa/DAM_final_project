from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs
from src.data.download import download_movielens_latest_small
from src.data.feature_engineering import FeatureStore, build_feature_store
from src.data.load_data import load_movielens, make_sample_movielens
from src.data.preprocess import clean_movies, clean_ratings, clean_tags, leave_one_out_split, limit_interaction_scope
from src.evaluation.ablation import run_ablation_experiments
from src.evaluation.diagnostics import popularity_bias_report
from src.evaluation.evaluate import evaluate_models
from src.evaluation.segments import evaluate_user_segments
from src.evaluation.significance import model_significance_report
from src.models import build_default_models
from src.models.cold_start import recommend_for_new_user
from src.visualization.explain_recommendation import explain_recommendations
from src.visualization.plot_results import plot_metric_comparison


@dataclass
class PipelineResult:
    features: FeatureStore
    test_ratings: pd.DataFrame
    models: dict[str, object]
    evaluation: pd.DataFrame
    ablation: pd.DataFrame
    user_segments: pd.DataFrame
    popularity_bias: pd.DataFrame
    significance: pd.DataFrame
    sample_recommendations: pd.DataFrame
    cold_start_recommendations: pd.DataFrame


def prepare_features(
    data_dir: str | Path | None = None,
    download: bool = False,
    sample: bool = False,
    min_user_ratings: int = 1,
    min_movie_ratings: int = 1,
    max_ratings: int | None = None,
    max_tags: int | None = None,
    max_users: int | None = None,
    max_movies: int | None = None,
    max_matrix_cells: int | None = 25_000_000,
    tag_max_features: int = 300,
) -> tuple[FeatureStore, pd.DataFrame]:
    ensure_project_dirs()
    if download:
        download_movielens_latest_small()
    raw = make_sample_movielens() if sample else load_movielens(data_dir, max_ratings=max_ratings, max_tags=max_tags)
    movies = clean_movies(raw["movies"])
    ratings = clean_ratings(raw["ratings"], min_user_ratings=min_user_ratings, min_movie_ratings=min_movie_ratings)
    ratings = limit_interaction_scope(
        ratings,
        max_users=max_users,
        max_movies=max_movies,
        max_matrix_cells=max_matrix_cells,
    )
    if ratings.empty:
        raise ValueError(
            "No ratings remain after filtering. Lower --min-user-ratings/--min-movie-ratings "
            "or increase the loaded dataset size."
        )
    tags = clean_tags(raw.get("tags"))
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())].reset_index(drop=True)
    train_ratings, test_ratings = leave_one_out_split(ratings)
    features = build_feature_store(movies, train_ratings, tags, tag_max_features=tag_max_features)
    save_processed_artifacts(features, test_ratings)
    return features, test_ratings


def fit_models(features: FeatureStore) -> dict[str, object]:
    models = build_default_models()
    for model in models.values():
        model.fit(features)
    return models


def run_pipeline(
    data_dir: str | Path | None = None,
    download: bool = False,
    sample: bool = False,
    k: int = 10,
    max_eval_users: int | None = None,
    min_user_ratings: int = 1,
    min_movie_ratings: int = 1,
    max_ratings: int | None = None,
    max_tags: int | None = None,
    max_users: int | None = None,
    max_movies: int | None = None,
    max_matrix_cells: int | None = 25_000_000,
) -> PipelineResult:
    features, test_ratings = prepare_features(
        data_dir=data_dir,
        download=download,
        sample=sample,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
        max_ratings=max_ratings,
        max_tags=max_tags,
        max_users=max_users,
        max_movies=max_movies,
        max_matrix_cells=max_matrix_cells,
    )
    models = fit_models(features)
    evaluation = evaluate_models(models, features, test_ratings, k=k, max_users=max_eval_users)
    save_evaluation(evaluation)
    ablation = run_ablation_experiments(features, test_ratings, k=k, max_users=max_eval_users)
    user_segments = evaluate_user_segments(models, features, test_ratings, k=k, max_users_per_segment=max_eval_users)
    popularity_bias = popularity_bias_report(models, features, test_ratings, k=k, max_users=max_eval_users)
    significance = model_significance_report(models, features, test_ratings, k=k, max_users=max_eval_users)
    save_analysis_reports(evaluation, ablation, user_segments, popularity_bias, significance)

    user_id = int(features.ratings["userId"].iloc[0])
    sample_recs = models["hybrid"].recommend(user_id, k=k)
    sample_recs = explain_recommendations(sample_recs, features, user_id=user_id, model=models["hybrid"])
    sample_recs.to_csv(REPORTS_DIR / "sample_recommendations.csv", index=False, encoding="utf-8-sig")

    genres = available_genres(features)[:2]
    cold_recs = recommend_for_new_user(features, genres=genres, liked_movie_ids=[], keywords="", k=k)
    cold_recs.to_csv(REPORTS_DIR / "cold_start_recommendations.csv", index=False, encoding="utf-8-sig")
    plot_metric_comparison(evaluation, REPORTS_DIR / "model_comparison.png")
    return PipelineResult(
        features,
        test_ratings,
        models,
        evaluation,
        ablation,
        user_segments,
        popularity_bias,
        significance,
        sample_recs,
        cold_recs,
    )


def save_processed_artifacts(features: FeatureStore, test_ratings: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.movies.to_csv(PROCESSED_DIR / "movies_clean.csv", index=False, encoding="utf-8-sig")
    features.ratings.to_csv(PROCESSED_DIR / "ratings_train.csv", index=False, encoding="utf-8-sig")
    test_ratings.to_csv(PROCESSED_DIR / "ratings_test.csv", index=False, encoding="utf-8-sig")
    features.popularity.reset_index().to_csv(PROCESSED_DIR / "movie_popularity.csv", index=False, encoding="utf-8-sig")
    features.content_features.reset_index().to_csv(PROCESSED_DIR / "movie_content_features.csv", index=False, encoding="utf-8-sig")


def save_evaluation(evaluation: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(REPORTS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "model_comparison.md").write_text(evaluation.to_markdown(index=False), encoding="utf-8")


def save_analysis_reports(
    evaluation: pd.DataFrame,
    ablation: pd.DataFrame,
    user_segments: pd.DataFrame,
    popularity_bias: pd.DataFrame,
    significance: pd.DataFrame,
) -> None:
    save_report_table(ablation, "ablation_results")
    save_report_table(user_segments, "user_segment_results")
    save_report_table(popularity_bias, "popularity_bias")
    save_report_table(significance, "significance_results")
    write_experiment_summary(evaluation, ablation, user_segments, popularity_bias, significance)


def save_report_table(df: pd.DataFrame, stem: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    markdown = df.to_markdown(index=False) if not df.empty else "_No rows generated._"
    (REPORTS_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")


def write_experiment_summary(
    evaluation: pd.DataFrame,
    ablation: pd.DataFrame,
    user_segments: pd.DataFrame,
    popularity_bias: pd.DataFrame,
    significance: pd.DataFrame | None = None,
) -> None:
    lines = ["# Experiment Summary", ""]
    if not evaluation.empty:
        best = evaluation.sort_values("ndcg_at_k", ascending=False).iloc[0]
        best_map = evaluation.sort_values("map_at_k", ascending=False).iloc[0]
        best_mrr = evaluation.sort_values("mrr_at_k", ascending=False).iloc[0]
        coverage_best = evaluation.sort_values("coverage", ascending=False).iloc[0]
        diversity_best = evaluation.sort_values("diversity", ascending=False).iloc[0]
        lines.extend(
            [
                f"- Best NDCG model: `{best['model']}` (NDCG@K={best['ndcg_at_k']:.4f}, Recall@K={best['recall_at_k']:.4f}).",
                f"- Best MAP model: `{best_map['model']}` (MAP@K={best_map['map_at_k']:.4f}).",
                f"- Best MRR model: `{best_mrr['model']}` (MRR@K={best_mrr['mrr_at_k']:.4f}).",
                f"- Highest coverage model: `{coverage_best['model']}` (Coverage={coverage_best['coverage']:.4f}).",
                f"- Highest diversity model: `{diversity_best['model']}` (Diversity={diversity_best['diversity']:.4f}).",
            ]
        )
    if not ablation.empty:
        best_ablation = ablation.sort_values("ndcg_at_k", ascending=False).iloc[0]
        lines.append(
            f"- Best ablation setting: `{best_ablation['ablation_setting']}` "
            f"(NDCG@K={best_ablation['ndcg_at_k']:.4f})."
        )
    if not popularity_bias.empty:
        most_pop = popularity_bias.iloc[0]
        least_pop = popularity_bias.sort_values("popular_item_ratio", ascending=True).iloc[0]
        lines.extend(
            [
                f"- Strongest popularity bias: `{most_pop['model']}` "
                f"(popular item ratio={most_pop['popular_item_ratio']:.4f}).",
                f"- Weakest popularity bias: `{least_pop['model']}` "
                f"(popular item ratio={least_pop['popular_item_ratio']:.4f}).",
            ]
        )
    if not user_segments.empty:
        segment_lines = []
        for segment, group in user_segments.groupby("activity_segment"):
            best_segment = group.sort_values("ndcg_at_k", ascending=False).iloc[0]
            segment_lines.append(f"`{segment}` -> `{best_segment['model']}`")
        lines.append("- Best model by user activity segment: " + ", ".join(segment_lines) + ".")
    if significance is not None and not significance.empty:
        significant_pairs = significance[significance["significant"] == True]
        if not significant_pairs.empty:
            best_pair = significant_pairs.sort_values("p_value", na_position="last").iloc[0]
            lines.append(
                f"- Strongest significant NDCG@K difference: `{best_pair['model_a']}` vs "
                f"`{best_pair['model_b']}` (diff={best_pair['diff']:.4f}, p={best_pair['p_value']:.4g})."
            )
        else:
            lines.append("- No pairwise NDCG@K difference reached p<0.05 in the current evaluation subset.")
    lines.append("")
    lines.append("These observations are generated automatically from the latest pipeline run and should be interpreted together with the full CSV/Markdown tables.")
    (REPORTS_DIR / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def available_genres(features: FeatureStore) -> list[str]:
    genres = set()
    for value in features.movies["genres"].dropna():
        genres.update(g for g in value.split("|") if g and g != "(no genres listed)")
    return sorted(genres)
