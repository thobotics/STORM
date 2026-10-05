"""Success-rate matrices of the RocketMan figures, from finished runs.

A run succeeds if its test MSE is at most 1.5e-3. Prints one matrix per graph:
rows = spatial distance k (8 at the top), columns = temporal distance i.

    python scripts/rocketman_matrix.py logs/rocketman
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import yaml

THRESHOLD = 1.5e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", nargs="?", default="logs/rocketman")
    args = parser.parse_args()

    runs = defaultdict(list)
    for res_file in glob.glob(os.path.join(args.log_dir, "**", "test_results.json"),
                              recursive=True):
        run_dir = os.path.dirname(res_file)
        cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
        d = cfg["dataset"]
        key = (d["graph"], cfg["model"]["hidden_size"], d["spatial_order"],
               d["temporal_distance"])
        runs[key].append(json.load(open(res_file))["test_loss"] <= THRESHOLD)

    for graph, hidden in sorted({k[:2] for k in runs}):
        print(f"\n{graph}, hidden_size={hidden}: success rate % (runs per cell)")
        print("k\\i " + "".join(f"{i:>10}" for i in range(9)))
        for k in range(8, -1, -1):
            cells = []
            for i in range(9):
                r = runs.get((graph, hidden, k, i))
                cells.append(f"{100 * sum(r) / len(r):5.0f} ({len(r):>2})" if r else "         -")
            print(f"{k:>3} " + "".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
