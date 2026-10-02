"""Estimate LOCA-bench token spend before running it.

Inputs:
  - the static prefix (tool definitions + task prompt) of each trajectory, measured by
    running LOCA against a mock upstream through `ccx serve` (no API spend): steps.jsonl;
  - Gemini-3-Flash per-length tool calls and trajectory lengths from the LOCA paper
    (arXiv 2602.07962, tables 4 and 5), as a stand-in for gemini-3.8-flash-high.

Model: context grows roughly linearly from the prefix P to the final trajectory length T
over N model calls, so one trajectory sends about N * (P + (T - P) / 2) input tokens.
N = tool calls + 1 is an upper bound (parallel tool calls share one model call).

Usage: python bench/tools/loca_estimate.py STEPS_JSONL [--lengths 8k,32k] [--seeds 5]
       [--output-per-call 1500] [--arms 1]
"""
import argparse
import json
import statistics

# LOCA paper, Gemini-3-Flash: average tool calls (table 5) and trajectory length (table 4).
PAPER = {
    "8k": (20.9, 19_795),
    "16k": (22.8, 23_786),
    "32k": (28.9, 35_327),
    "64k": (28.4, 46_226),
    "96k": (33.2, 73_698),
    "128k": (38.1, 101_427),
    "256k": (33.7, 141_735),
}
TASKS = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--lengths", default=",".join(PAPER))
    ap.add_argument("--seeds", type=int, default=5, help="seeds per task (LOCA has 5)")
    ap.add_argument("--output-per-call", type=int, default=1500,
                    help="output tokens per model call, thinking included")
    ap.add_argument("--arms", type=int, default=1, help="systems run on the same set")
    args = ap.parse_args()

    prefixes = [json.loads(l)["est"]["total"] for l in open(args.steps) if l.strip()]
    prefix = statistics.mean(prefixes)
    print(f"measured prefixes: n={len(prefixes)} mean={prefix:,.0f} "
          f"min={min(prefixes):,} max={max(prefixes):,} (chars/4 estimate)")
    trajectories = TASKS * args.seeds
    print(f"{'length':>6} {'calls':>6} {'final ctx':>10} {'in/traj':>10} "
          f"{'input':>14} {'output':>12}")
    total_in = total_out = 0
    for length in args.lengths.split(","):
        tools, final = PAPER[length]
        calls = tools + 1
        per_traj = calls * (prefix + max(final - prefix, 0) / 2)
        tin = per_traj * trajectories * args.arms
        tout = calls * args.output_per_call * trajectories * args.arms
        total_in += tin
        total_out += tout
        print(f"{length:>6} {calls:>6.1f} {final:>10,} {per_traj:>10,.0f} "
              f"{tin:>14,.0f} {tout:>12,.0f}")
    print(f"{'total':>6} {'':>6} {'':>10} {'':>10} {total_in:>14,.0f} {total_out:>12,.0f}"
          f"   ({trajectories} trajectories x {args.arms} arm(s) per length)")


if __name__ == "__main__":
    main()
