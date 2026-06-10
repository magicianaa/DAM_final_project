from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_SMALL_DIR = RAW_DIR / "ml-latest-small"
MOVIELENS_FULL_DIR = RAW_DIR / "ml-32m"


def ensure_project_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, PROJECT_ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)
