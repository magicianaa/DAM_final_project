from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MOVIELENS_FULL_DIR, MOVIELENS_SMALL_DIR
from src.pipeline import available_genres, fit_models, prepare_features
from src.models.cold_start import recommend_for_new_user
from src.visualization.explain_recommendation import explain_recommendations


st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")


@st.cache_resource(show_spinner=True)
def load_demo(
    sample: bool,
    data_dir: str,
    min_user_ratings: int,
    min_movie_ratings: int,
    max_ratings: int,
    max_users: int,
    max_movies: int,
    max_matrix_cells: int,
):
    features, test_ratings = prepare_features(
        data_dir=data_dir or None,
        download=False,
        sample=sample,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
        max_ratings=max_ratings or None,
        max_users=max_users or None,
        max_movies=max_movies or None,
        max_matrix_cells=max_matrix_cells,
    )
    models = fit_models(features)
    return features, test_ratings, models


st.title("Hybrid Movie Recommendation System")
st.caption("融合协同过滤与内容特征的混合电影推荐系统 — 数据挖掘期末项目")

with st.sidebar:
    st.header("Data")
    default_dir = MOVIELENS_FULL_DIR if (MOVIELENS_FULL_DIR / "ratings.csv").exists() else MOVIELENS_SMALL_DIR
    sample = st.checkbox("Use built-in smoke-test sample", value=not (default_dir / "ratings.csv").exists())
    data_dir = st.text_input("MovieLens directory", value=str(default_dir))
    min_user_ratings = st.number_input("Min user ratings", min_value=1, max_value=50, value=1, step=1)
    min_movie_ratings = st.number_input("Min movie ratings", min_value=1, max_value=50, value=1, step=1)
    max_ratings = st.number_input("Max ratings to load", min_value=0, max_value=5_000_000, value=250_000, step=50_000)
    max_users = st.number_input("Max active users", min_value=0, max_value=20_000, value=1000, step=100)
    max_movies = st.number_input("Max rated movies", min_value=0, max_value=20_000, value=3000, step=100)
    max_matrix_cells = st.number_input("Max dense matrix cells", min_value=100_000, max_value=50_000_000, value=5_000_000, step=500_000)
    page = st.radio("View", ["Existing user", "Cold start", "Model comparison"], index=0)

try:
    features, test_ratings, models = load_demo(
        sample,
        data_dir,
        min_user_ratings,
        min_movie_ratings,
        max_ratings,
        max_users,
        max_movies,
        max_matrix_cells,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

if page == "Existing user":
    left, right = st.columns([1, 2])
    with left:
        user_ids = sorted(features.ratings["userId"].unique().tolist())
        user_id = st.selectbox("User ID", user_ids, index=0)
        model_name = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index("hybrid"))
        k = st.slider("Top N", 5, 30, 10)
        user_history = (
            features.ratings[features.ratings["userId"] == user_id]
            .merge(features.movies[["movieId", "title", "genres"]], on="movieId", how="left")
            .sort_values("rating", ascending=False)
        )
        st.subheader("User profile")
        st.dataframe(user_history[["title", "genres", "rating"]].head(12), use_container_width=True, hide_index=True)
    with right:
        model = models[model_name]
        recs = model.recommend(int(user_id), k=k)
        recs = explain_recommendations(recs, features, user_id=int(user_id), model=model)
        st.subheader("Recommendations")
        st.dataframe(recs[["title", "genres", "score", "reason"]], use_container_width=True, hide_index=True)

        # Genre distribution of recommendations
        all_genres = []
        for g in recs["genres"].dropna():
            all_genres.extend(str(g).split("|"))
        if all_genres:
            import collections
            genre_counts = collections.Counter(all_genres)
            st.subheader("Genre distribution of recommendations")
            st.bar_chart(
                pd.DataFrame({"count": genre_counts}).sort_values("count", ascending=True),
                use_container_width=True,
            )

elif page == "Cold start":
    genres = available_genres(features)
    selected_genres = st.multiselect("Preferred genres", genres, default=genres[:2])
    movie_options = features.movies.sort_values("title")[["movieId", "title"]]
    label_to_id = dict(zip(movie_options["title"], movie_options["movieId"]))
    selected_titles = st.multiselect("Favorite movies", list(label_to_id.keys()), default=[])
    keywords = st.text_input("Keywords or tags", value="")
    k = st.slider("Top N", 5, 30, 10, key="cold_k")
    liked_ids = [int(label_to_id[t]) for t in selected_titles]
    recs = recommend_for_new_user(features, selected_genres, liked_ids, keywords, k=k)
    st.subheader("Cold-start recommendations")
    st.dataframe(recs[["title", "genres", "score", "reason"]], use_container_width=True, hide_index=True)

else:
    st.subheader("Experiment results")
    tabs = st.tabs([
        "Data profile",
        "Models",
        "Ablation",
        "User segments",
        "Popularity bias",
        "Significance",
        "Cross validation",
        "Summary",
    ])

    def show_report_table(path: Path, chart_index: str | None = None):
        if not path.exists():
            st.info(f"Run the pipeline first to generate {path.name}.")
            return
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if chart_index and chart_index in df.columns:
            metric_cols = [c for c in df.columns if c.endswith("_at_k") or c in {"coverage", "diversity", "popular_item_ratio", "recommended_catalog_coverage"}]
            if metric_cols:
                chart_df = df.set_index(chart_index)[metric_cols].apply(pd.to_numeric, errors="coerce")
                chart_df = chart_df.dropna(axis=1, how="all")
                chart_df = chart_df.loc[:, chart_df.nunique(dropna=True) > 1]
                if not chart_df.empty:
                    st.bar_chart(chart_df)

    with tabs[0]:
        show_report_table(PROJECT_ROOT / "reports/data_profile.csv")
    with tabs[1]:
        show_report_table(PROJECT_ROOT / "reports/model_comparison.csv", chart_index="model")
        # Add a dedicated metrics comparison chart
        model_path = PROJECT_ROOT / "reports/model_comparison.csv"
        if model_path.exists():
            df = pd.read_csv(model_path)
            metric_cols = [c for c in df.columns if c.endswith("_at_k") or c in {"coverage", "diversity"}]
            if metric_cols:
                st.subheader("Metrics Comparison")
                metrics_df = df.set_index("model")[metric_cols]
                st.bar_chart(metrics_df.T, use_container_width=True)
    with tabs[2]:
        show_report_table(PROJECT_ROOT / "reports/ablation_results.csv", chart_index="ablation_setting")
    with tabs[3]:
        show_report_table(PROJECT_ROOT / "reports/user_segment_results.csv")
    with tabs[4]:
        show_report_table(PROJECT_ROOT / "reports/popularity_bias.csv", chart_index="model")
    with tabs[5]:
        show_report_table(PROJECT_ROOT / "reports/significance_results.csv")
    with tabs[6]:
        show_report_table(PROJECT_ROOT / "reports/cross_validation_results.csv", chart_index="model")
    with tabs[7]:
        summary_path = PROJECT_ROOT / "reports/formal_experiment_summary.md"
        fallback_summary_path = PROJECT_ROOT / "reports/experiment_summary.md"
        if summary_path.exists():
            st.markdown(summary_path.read_text(encoding="utf-8"))
        elif fallback_summary_path.exists():
            st.markdown(fallback_summary_path.read_text(encoding="utf-8"))
        else:
            st.info("Run the pipeline first to generate experiment_summary.md.")
