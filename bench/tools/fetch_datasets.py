"""Download the benchmark datasets to the paths the tracks read.

  python tools/fetch_datasets.py beam-1M longmemeval-s      # any of: beam-100K beam-500K beam-1M longmemeval-s

BEAM: Hugging Face dataset Mohammadta/BEAM (data/<split>-00000-of-00001.parquet) ->
data/beam/<split>.parquet. LongMemEval_S: xiaowu0162/longmemeval-cleaned
(longmemeval_s_cleaned.json) -> data/longmemeval_s_cleaned.json.
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

BENCH = Path(__file__).resolve().parents[1]


def fetch(name):
    if name.startswith("beam-"):
        split = name.split("-", 1)[1]
        source = hf_hub_download("Mohammadta/BEAM", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        target = BENCH / f"data/beam/{split}.parquet"
    elif name == "longmemeval-s":
        source = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
        target = BENCH / "data/longmemeval_s_cleaned.json"
    else:
        raise SystemExit(f"unknown dataset {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"{name} -> {target} ({target.stat().st_size // 2**20} MB)")


if __name__ == "__main__":
    for item in sys.argv[1:] or ["beam-1M", "longmemeval-s"]:
        fetch(item)
