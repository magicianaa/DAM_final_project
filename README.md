# Hybrid Movie Recommendation System

This project is a data mining final project built around a MovieLens Top-N recommendation task. It combines user rating behavior with movie content features, then evaluates the accuracy, coverage, diversity, popularity bias, cold-start behavior, and explainability of several recommendation strategies.

The project is positioned as an end-to-end applied data mining workflow rather than a single recommender implementation. The formal research question is:

> Can a hybrid recommender that combines collaborative filtering, content similarity, and popularity signals achieve a better trade-off between recommendation accuracy, catalog coverage, diversity, and cold-start usability?

## Research Questions

- RQ1: How do popularity, user-based CF, item-based CF, matrix factorization, content-based, and hybrid recommenders compare on Top-N movie recommendation?
- RQ2: Does adding content information improve the accuracy-diversity-coverage trade-off?
- RQ3: Which models are more robust for low-, medium-, and high-activity users?
- RQ4: Do higher-scoring recommenders also show stronger popularity bias?
- RQ5: Can recommendation reasons make the demo easier to understand and defend in presentation?

## Project Structure

```text
data/raw/                 raw MovieLens files, not committed
data/processed/           generated cleaned data and features
docs/                     project documents and unrelated reference documents
src/data/                 download, loading, preprocessing, feature engineering
src/models/               popularity, user CF, item CF, SVD, content, hybrid, cold start
src/evaluation/           Precision@K, Recall@K, NDCG@K, HitRate, Coverage, Diversity
src/visualization/        result plots and recommendation explanations
app/streamlit_app.py      interactive demo
reports/                  generated CSV/Markdown/PNG outputs
main.py                   end-to-end pipeline entry point
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation

Preferred automatic download:

```powershell
python main.py --download --k 10 --max-eval-users 100
```

If network access fails, download MovieLens latest-small manually from:

https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Unzip it so these files exist:

```text
data/raw/ml-latest-small/ratings.csv
data/raw/ml-latest-small/movies.csv
data/raw/ml-latest-small/tags.csv
data/raw/ml-latest-small/links.csv
```

Full MovieLens data is also supported. If you downloaded MovieLens 32M, place it as:

```text
data/raw/ml-32m/ratings.csv
data/raw/ml-32m/movies.csv
data/raw/ml-32m/tags.csv
data/raw/ml-32m/links.csv
```

When no `--data-dir` is provided, the pipeline now auto-detects `data/raw/ml-32m` first, then `data/raw/ml-latest-small`.
For safety, the command-line defaults read at most 500,000 ratings and 100,000 tags from large CSV files. Use `--max-ratings 0 --max-tags 0` only when you intentionally want to read the full CSV files and have enough memory/time.

For a quick smoke test without external data:

```powershell
python main.py --sample --k 5
```

## Run Pipeline

```powershell
python main.py --data-dir data/raw/ml-latest-small --k 10 --max-eval-users 100
```

Recommended controlled run for MovieLens 32M:

```powershell
python main.py --data-dir data/raw/ml-32m --k 10 --max-ratings 500000 --max-users 2000 --max-movies 5000 --max-eval-users 100 --min-user-ratings 5 --min-movie-ratings 5
```

The current MVP uses a dense user-movie matrix for clarity. For full MovieLens 32M, keep `--max-ratings`, `--max-users`, `--max-movies`, or `--max-matrix-cells` enabled unless you have enough memory for a very large dense matrix.

Useful scale-control arguments:

```text
--max-ratings N        read at most N ratings from ratings.csv
--max-tags N           read at most N tags from tags.csv
--max-users N          keep the N most active users
--max-movies N         keep the N most rated movies
--max-matrix-cells N   safety cap for dense user-movie matrix size
--max-eval-users N     limit evaluation users for faster experiments
```

Outputs:

```text
data/processed/movies_clean.csv
data/processed/ratings_train.csv
data/processed/ratings_test.csv
data/processed/movie_popularity.csv
data/processed/movie_content_features.csv
reports/model_comparison.csv
reports/model_comparison.md
reports/model_comparison.png
reports/ablation_results.csv
reports/ablation_results.md
reports/user_segment_results.csv
reports/user_segment_results.md
reports/popularity_bias.csv
reports/popularity_bias.md
reports/significance_results.csv
reports/significance_results.md
reports/experiment_summary.md
reports/sample_recommendations.csv
reports/cold_start_recommendations.csv
```

## Formal Experiment Suite

Use this command for the final report and presentation artifacts:

```powershell
python scripts/run_formal_experiment.py --data-dir data/raw/ml-latest-small --k 10 --max-eval-users 300 --min-user-ratings 5 --min-movie-ratings 5 --cv-folds 3
```

If you are using the built-in smoke-test sample:

```powershell
python scripts/run_formal_experiment.py --sample --k 5 --max-eval-users 20 --cv-folds 3
```

The formal experiment suite writes:

```text
reports/data_profile.csv / .md              dataset scale and feature profile
reports/model_comparison.csv / .md          main Top-N model comparison
reports/ablation_results.csv / .md          CF/content/popularity contribution analysis
reports/user_segment_results.csv / .md      low-, medium-, high-activity user analysis
reports/popularity_bias.csv / .md           popularity-bias diagnostics
reports/significance_results.csv / .md      paired per-user NDCG@K significance tests
reports/cross_validation_results.csv / .md  k-fold stability check
reports/formal_experiment_summary.md        report-ready experiment summary
```

## Run Streamlit Demo

```powershell
streamlit run app/streamlit_app.py
```

The app includes:

- Existing user recommendation with selectable model and recommendation reasons.
- Cold-start recommendation based on preferred genres, favorite movies, and keywords.
- Model comparison, data profile, ablation, user-segment, popularity-bias, significance-test, cross-validation, and summary views after running the formal experiment suite.

## Iteration Analysis Outputs

The current P1 iteration adds reproducible experiment diagnostics:

- `ablation_results`: compares popularity-only, collaborative-only, content-only, and hybrid variants.
- `user_segment_results`: compares model behavior for low, medium, and high activity users.
- `popularity_bias`: measures average recommended item popularity and popular-item ratio.
- `significance_results`: compares pairwise per-user NDCG@K differences with a Wilcoxon signed-rank test.
- `cross_validation_results`: reports k-fold means and standard deviations for core metrics.
- `experiment_summary`: generates a short Markdown summary of best models and trade-offs.

Quick command:

```powershell
python main.py --sample --k 5
```

Small MovieLens command:

```powershell
python main.py --data-dir data/raw/ml-latest-small --k 5 --max-eval-users 20 --min-user-ratings 5 --min-movie-ratings 2 --max-ratings 50000 --max-users 300 --max-movies 1000
```

## Implemented Models

- Popularity Baseline
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization with `sklearn.decomposition.TruncatedSVD`
- Content-Based Recommendation using genres and tag/title TF-IDF features
- Hybrid Recommendation combining item CF, content similarity, and popularity

## Notes

Raw MovieLens data can be large and should remain in `data/raw/`. Generated processed CSV files are reproducible from `main.py`.
