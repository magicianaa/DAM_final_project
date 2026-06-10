# Reports

Pipeline and formal experiment outputs are written here. These files are intended to feed the final report, PPT, and Streamlit demo.

## Main Evaluation

- `data_profile.csv` / `data_profile.md`: dataset size, train/test counts, sparsity, feature counts, and experiment settings.
- `model_comparison.csv` / `model_comparison.md`: main Top-N comparison across all implemented recommenders.
- `model_comparison.png`: visualization of core metrics.

## Analysis Tables

- `ablation_results.csv` / `ablation_results.md`: compares popularity-only, CF-only, content-only, and hybrid variants.
- `user_segment_results.csv` / `user_segment_results.md`: compares model behavior for low-, medium-, and high-activity users.
- `popularity_bias.csv` / `popularity_bias.md`: measures average recommended item popularity and popular-item ratio.
- `significance_results.csv` / `significance_results.md`: pairwise per-user NDCG@K significance tests.
- `cross_validation_results.csv` / `cross_validation_results.md`: k-fold mean/std metrics for stability checking.

## Demo Outputs

- `sample_recommendations.csv`: example hybrid recommendations for an existing user.
- `cold_start_recommendations.csv`: example recommendations for a new user preference profile.

## Summaries

- `experiment_summary.md`: automatically generated summary from the latest pipeline run.
- `formal_experiment_summary.md`: report-ready summary that includes research questions, reproducible command, data profile, main result, and stability check.

Recommended command for final report artifacts:

```powershell
python scripts/run_formal_experiment.py --data-dir data/raw/ml-latest-small --k 10 --max-eval-users 300 --min-user-ratings 5 --min-movie-ratings 5 --cv-folds 3
```
