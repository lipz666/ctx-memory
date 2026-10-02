"""Write a LOCA task config with only the first N seeds of each task (LOCA ships 5).

Usage: python bench/tools/loca_subset.py IN_CONFIG OUT_CONFIG --seeds 1
"""
import argparse
import json
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seeds", type=int, required=True)
    args = ap.parse_args()
    data = json.load(open(args.src))
    seen = Counter()
    kept = []
    for config in data["configurations"]:
        if seen[config["name"]] < args.seeds:
            kept.append(config)
        seen[config["name"]] += 1
    json.dump({**data, "configurations": kept}, open(args.dst, "w"), indent=1)
    print(f"{len(kept)} of {len(data['configurations'])} configurations -> {args.dst}")


if __name__ == "__main__":
    main()
