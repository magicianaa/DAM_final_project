# Notebooks

These notebooks support the final report narrative for the MovieLens hybrid recommendation project. They are optional for running the production pipeline, but useful for explaining the data mining workflow in the report and presentation.

## Suggested Reading Order

1. `01_data_exploration.ipynb`
   - Inspect MovieLens users, movies, ratings, genres, tags, sparsity, and long-tail item distribution.
   - Generate evidence for the "Data and Preprocessing" section.

2. `02_baseline_experiments.ipynb`
   - Compare popularity and simple recommendation baselines.
   - Motivate why collaborative filtering, content-based, and hybrid models are needed.

3. `03_result_analysis.ipynb`
   - Interpret generated files under `reports/`.
   - Discuss model comparison, ablation, user activity segments, diversity, coverage, and popularity bias.

## Reproducible Pipeline First

The notebooks should not be the only way to reproduce results. For formal results, run:

```powershell
python scripts/run_formal_experiment.py --data-dir data/raw/ml-latest-small --k 10 --max-eval-users 300 --min-user-ratings 5 --min-movie-ratings 5 --cv-folds 3
```

The notebooks can then load `reports/*.csv` for visualization and interpretation.
