from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(results: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return output_path
    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate_at_k", "coverage", "diversity"]
    ax = results.set_index("model")[metrics].plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("score")
    ax.set_title("Recommendation Model Comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
