"""Paper 1 reproduction, part 1: the simulation study (their Tables 2-7).

    git clone https://github.com/TXMUNC/DDI_MDPI_Mathematics_2025 /content/p1repo
    pip install rdata
    python scripts/80_paper1_simulation.py --repo-dir /content/p1repo --mem 36g

3 latent scenarios x 4 beta settings x n in {5000, 10000} x m = d in {10, 20},
200 replications each = 9,600 independent simulations. They are spread over
the cores with Spark (sc.parallelize(...).map(...)): the paper says it kept
to small dimensions "due to computational constraints", and parallel
replications are the straightforward big-data answer to that.

Writes data/results/paper1/auc_ours.csv, comparison.csv and summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from d3arm import config
from d3arm.paper1.simulation import LATENT, METRICS, load_setting, make_beta, one_replication
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--repo-dir", required=True, help="clone of github.com/TXMUNC/DDI_MDPI_Mathematics_2025")
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--reps", type=int, default=200)
p.add_argument("--sizes", default="5000,10000")
p.add_argument("--dims", default="10,20")
args = p.parse_args()

spark = get_spark("paper1-simulation", args.cores, args.mem)
sc = spark.sparkContext
sizes = [int(v) for v in args.sizes.split(",")]
dims = [int(v) for v in args.dims.split(",")]
data = {(n, m): load_setting(Path(args.repo_dir), n, m) for n in sizes for m in dims}
bc = sc.broadcast(data)

tasks = [(lat, s, n, m, r) for lat in LATENT for s in (1, 2, 3, 4) for n in sizes for m in dims
         for r in range(args.reps)]


def run(task):
    lat, s, n, m, r = task
    x, names, idx = bc.value[(n, m)]
    beta = make_beta(names, m, m, s, idx)
    seed = 1_000_000 * LATENT[lat] + 20_000 * s + 100 * m + 10 * m + r + (0 if n == 5000 else 500_000)
    t = time.time()
    main, ddi = one_replication(x, names, m, beta, lat, seed)
    return [(lat, "main", s, n, m, r, *[main[k] for k in METRICS], time.time() - t),
            (lat, "ddi", s, n, m, r, *[ddi[k] for k in METRICS], 0.0)]


t0 = time.time()
rows = sc.parallelize(tasks, numSlices=max(64, sc.defaultParallelism * 4)).flatMap(run).collect()
wall = time.time() - t0
df = pd.DataFrame(rows, columns=["latent", "effect", "beta_setting", "n", "m", "rep", *METRICS, "seconds"])
cpu = df["seconds"].sum()

ours = (df.groupby(["latent", "effect", "beta_setting", "n", "m"])[METRICS].mean().reset_index())
# the paper reports no main-effect AUC for setting 2 and no DDI AUC for setting 1 (nothing to detect)
ours = ours[~((ours.effect == "main") & (ours.beta_setting == 2)) & ~((ours.effect == "ddi") & (ours.beta_setting == 1))]
out = config.RESULTS_DIR / "paper1"
out.mkdir(parents=True, exist_ok=True)
ours.round(4).to_csv(out / "auc_ours.csv", index=False)

ref = pd.read_csv(config.REFERENCE_DIR / "paper1_auc_tables.csv")
cmp = ref.merge(ours, on=["latent", "effect", "beta_setting", "n", "m"], suffixes=("_paper", "_ours"))
for k in METRICS:
    cmp[f"{k}_diff"] = cmp[f"{k}_ours"] - cmp[f"{k}_paper"]
cmp.round(4).to_csv(out / "comparison.csv", index=False)
diffs = cmp[[f"{k}_diff" for k in METRICS]].abs().values.ravel()

# the paper's headline: RR main-effect AUC drops ~30% with confounding vs no random effect, DDI ~15%
def mean_rr(lat, eff):
    return ours[(ours.latent == lat) & (ours.effect == eff)]["rr"].mean()
drop = {eff: round(100 * (1 - mean_rr("confounded", eff) / mean_rr("no_random_effect", eff)), 1)
        for eff in ("main", "ddi")}
ref_drop = {eff: round(100 * (1 - ref[(ref.latent == "confounded") & (ref.effect == eff)]["rr"].mean()
                              / ref[(ref.latent == "no_random_effect") & (ref.effect == eff)]["rr"].mean()), 1)
            for eff in ("main", "ddi")}

s = {"replications_per_setting": args.reps, "simulations": len(tasks), "table_cells_compared": int(len(diffs)),
     "mean_abs_auc_diff": round(float(diffs.mean()), 4), "median_abs_auc_diff": round(float(np.median(diffs)), 4),
     "max_abs_auc_diff": round(float(diffs.max()), 4),
     "cells_within_0.02": int((diffs <= 0.02).sum()), "cells_within_0.05": int((diffs <= 0.05).sum()),
     "rr_auc_drop_pct_ours": drop, "rr_auc_drop_pct_paper_tables": ref_drop,
     "wall_seconds": round(wall, 1), "sum_of_task_seconds": round(cpu, 1),
     "parallel_speedup": round(cpu / wall, 2)}
(out / "summary.json").write_text(json.dumps(s, indent=2))
for k, v in s.items():
    print(f"{k:32s} {v}")
pd.set_option("display.width", 220, "display.max_rows", 100)
print("\nours vs paper (RR and ROR shown; full table in comparison.csv):")
print(cmp[["latent", "effect", "beta_setting", "n", "m", "rr_paper", "rr_ours", "ror_paper", "ror_ours", "conf_paper", "conf_ours"]]
      .round(3).to_string(index=False))
spark.stop()
