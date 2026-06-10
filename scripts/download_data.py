from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_movielens_latest_small


if __name__ == "__main__":
    path = download_movielens_latest_small()
    print(f"MovieLens latest-small is ready at: {path}")
