from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config import MOVIELENS_SMALL_DIR, MOVIELENS_SMALL_URL, RAW_DIR


REQUIRED_FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")


def has_movielens_files(data_dir: Path = MOVIELENS_SMALL_DIR) -> bool:
    return all((data_dir / name).exists() for name in REQUIRED_FILES)


def download_movielens_latest_small(
    raw_dir: Path = RAW_DIR,
    url: str = MOVIELENS_SMALL_URL,
    force: bool = False,
) -> Path:
    """Download and extract MovieLens latest-small into data/raw."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_dir = raw_dir / "ml-latest-small"
    if has_movielens_files(target_dir) and not force:
        return target_dir

    zip_path = raw_dir / "ml-latest-small.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        if target_dir.exists() and force:
            shutil.rmtree(target_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    except Exception as exc:  # pragma: no cover - depends on network
        raise RuntimeError(
            "Could not download MovieLens latest-small. "
            f"Please download {url} manually, unzip it, and place ratings.csv, "
            "movies.csv, tags.csv, and links.csv under data/raw/ml-latest-small."
        ) from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()

    if not has_movielens_files(target_dir):
        raise RuntimeError(
            "MovieLens archive was downloaded, but required CSV files were not found."
        )
    return target_dir
