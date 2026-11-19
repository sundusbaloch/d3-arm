"""Paper 1 at scale: their simulation with up to 1,000,000 reports.

    python scripts/83_paper1_scale.py --repo-dir /content/p1repo --mem 36g

The paper stopped at n = 10,000 "due to computational constraints". Here the
same model runs at n = 10k, 100k and 1M. Their drug matrix (n = 10,000, m = 20
vaccines) is resampled with replacement to the larger n, so the exposure
pattern is theirs; only the number of reports grows. The replications run in
parallel on Spark, as in script 80.

Question: does more data fix the damage that latent confounding does to the
indicators' AUC? (More data reduces noise; it cannot remove a bias.)

Writes data/results/paper1_scale/auc_by_n.csv, replications.csv and summary.json.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")          # one thread per Spark task, no oversubscription

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from d3arm import config
from d3arm.paper1.simulation import LATENT, METRICS, load_setting, make_beta, one_replication_resampled
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--repo-dir", required=True, help="clone of github.com/TXMUNC/DDI_MDPI_Mathematics_2025")
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--sizes", default="10000,100000,1000000")
p.add_argument("--reps", default="100,50,20", help="replications per size (same order as --sizes)")
p.add_argument("--base-n", type=int, default=10000, help="which of their matrices to resample (5000 or 10000)")
p.add_argument("--m", type=int, default=20, help="number of vaccines (10 or 20)")
p.add_argument("--settings", default="1,2,3,4")
args = p.parse_args()

sizes = [int(v) for v in args.sizes.split(",")]
reps = [int(v) for v in args.reps.split(",")]
if len(reps) != len(sizes):
    sys.exit("--reps needs one number per size")
settings = [int(v) for v in args.settings.split(",")]

spark = get_spark("paper1-scale", args.cores, args.mem)
sc = spark.sparkContext
x, names, idx = load_setting(Path(args.repo_dir), args.base_n, args.m)
bc = sc.broadcast((x, names, idx))
m = args.m

tasks = [(lat, s, n, r) for n, R in zip(sizes, reps) for lat in LATENT for s in settings for r in range(R)]
tasks.sort(key=lambda t: -t[2])            # big ones first, so no core waits at the end


def run(task):
    lat, s, n, r = task
    xx, nm, ix = bc.value
    beta = make_beta(nm, m, m, s, ix)
    seed = 1_000_000 * LATENT[lat] + 20_000 * s + 7 * n + r
    t = time.time()
    main, ddi = one_replication_resampled(xx, nm, m, beta, lat, seed, n)
    sec = time.time() - t
    return [(lat, "main", s, n, r, *[main[k] for k in METRICS], sec),
            (lat, "ddi", s, n, r, *[ddi[k] for k in METRICS], 0.0)]


t0 = time.time()
rows = sc.parallelize(tasks, numSlices=len(tasks)).flatMap(run).collect()
wall = time.time() - t0
df = pd.DataFrame(rows, columns=["latent", "effect", "beta_setting", "n", "rep", *METRICS, "seconds"])
# no main effects in setting 2 and no interactions in setting 1 (nothing to detect)
df = df[~((df.effect == "main") & (df.beta_setting == 2)) & ~((df.effect == "ddi") & (df.beta_setting == 1))]

out = config.RESULTS_DIR / "paper1_scale"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "replications.csv", index=False)
by_n = df.groupby(["latent", "effect", "n"])[METRICS].agg(["mean", "std"]).round(4)
by_n.columns = [f"{a}_{b}" for a, b in by_n.columns]
by_n = by_n.reset_index()
by_n.to_csv(out / "auc_by_n.csv", index=False)


def mean_auc(lat, eff, n, k="rr"):
    return df[(df.latent == lat) & (df.effect == eff) & (df.n == n)][k].mean()


s = {"base_matrix": f"n{args.base_n}_vax{m}", "settings": settings, "sizes": sizes, "replications": dict(zip(sizes, reps)),
     "tasks": len(tasks), "wall_seconds": round(wall, 1),
     "sum_of_task_seconds": round(df["seconds"].sum(), 1),
     "parallel_speedup": round(df["seconds"].sum() / wall, 2),
     "seconds_per_replication": {int(n): round(g["seconds"].mean(), 2) for n, g in df[df.effect == "main"].groupby("n")},
     "rr_auc": {}, "rr_auc_drop_pct_confounded_vs_none": {}}
for n in sizes:
    s["rr_auc"][n] = {f"{lat}_{eff}": round(mean_auc(lat, eff, n), 4) for lat in LATENT for eff in ("main", "ddi")}
    s["rr_auc_drop_pct_confounded_vs_none"][n] = {
        eff: round(100 * (1 - mean_auc("confounded", eff, n) / mean_auc("no_random_effect", eff, n)), 1)
        for eff in ("main", "ddi")}
(out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
print(json.dumps(s, indent=2, default=str))
pd.set_option("display.width", 200)
print("\nmean AUC by n (averaged over beta settings):")
print(df.groupby(["effect", "latent", "n"])[METRICS].mean().round(3).to_string())
spark.stop()
