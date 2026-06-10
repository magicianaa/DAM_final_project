# Processed Data

This directory stores reproducible intermediate data generated from MovieLens raw files. The files are not the source of truth; they can be regenerated from the raw dataset and the pipeline.

## Generate Processed Files

Smoke-test run:

```powershell
python main.py --sample --k 5
```

Formal experiment run:

```powershell
python scripts/run_formal_experiment.py --data-dir data/raw/ml-latest-small --k 10 --max-eval-users 300 --min-user-ratings 5 --min-movie-ratings 5 --cv-folds 3
```

## Generated Artifacts

- `movies_clean.csv`: cleaned movie metadata, including parsed year and normalized title fields.
- `ratings_train.csv`: training interactions after filtering and leave-one-out split.
- `ratings_test.csv`: held-out relevant interactions for Top-N evaluation.
- `movie_popularity.csv`: rating count, mean rating, weighted rating, and normalized popularity features.
- `movie_content_features.csv`: genre, tag/title TF-IDF, year, and genre-diversity features used by content and hybrid recommenders.

## Reproducibility Notes

- Raw data should remain under `data/raw/`.
- Processed files depend on CLI filters such as `--max-ratings`, `--max-users`, `--min-user-ratings`, and `--min-movie-ratings`.
- The formal report should state the exact command used to generate these files.
