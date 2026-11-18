"""Scaling experiments (no cluster, so we scale what we can on one machine):
  - cores:     local[1], local[2], local[4], local[8] on 100% of the data
  - data size: 10%, 25%, 50%, 100% of the transactions on all cores
for two workloads:
  son    SON frequent itemsets at 0.1% support (Python mapPartitions, 16 partitions)
  pairs  drug x AE counts at 5 reports + ROR/PRR for every pair (Spark SQL)

    python scripts/70_scaling.py --mem 36g
    python scripts/70_scaling.py --mem 36g --repeats 3 --workloads pairs

Every run is appended to data/results/scaling/runs.csv as soon as it finishes,
so a disconnect loses at most one run.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark import StorageLevel
from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.pairs import add_measures, drug_ae_counts
from d3arm.mining.son import son
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--mem", default="8g")
p.add_argument("--cores-list", default="1,2,4,8")
p.add_argument("--fractions", default="0.1,0.25,0.5")
p.add_argument("--max-cores", default="8")
p.add_argument("--workloads", default="pairs,son")
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--support", type=float, default=0.001)
p.add_argument("--son-partitions", type=int, default=16)
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

out = config.RESULTS_DIR / "scaling"
out.mkdir(parents=True, exist_ok=True)
log = out / "runs.csv"
fields = ["workload", "cores", "fraction", "repeat", "transactions", "load_seconds", "seconds", "detail"]
if not log.exists():
    with open(log, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()


def run_pairs(spark, fraction):
    tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
    if fraction < 1:
        tx = tx.sample(fraction=fraction, seed=7)
    t = time.time()
    counts, info = drug_ae_counts(tx, 5)
    n_sig = add_measures(counts).where("signal").count()
    return info["transactions"], 0.0, time.time() - t, f"signals={n_sig}"


def run_son(spark, fraction):
    tx = (spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
               .select(F.filter("items", lambda i: i.startswith("DRUG=") | i.startswith("AE=")).alias("items"))
               .where(F.size("items") > 0))
    if fraction < 1:
        tx = tx.sample(fraction=fraction, seed=7)
    t = time.time()
    baskets = (tx.rdd.map(lambda r: tuple(sorted(set(r[0])))).repartition(args.son_partitions)
                 .persist(StorageLevel.MEMORY_AND_DISK))
    n = baskets.count()
    load = time.time() - t
    t = time.time()
    freq, st = son(baskets, n, math.ceil(args.support * n))
    secs = time.time() - t
    baskets.unpersist()
    return n, load, secs, f"itemsets={len(freq)};candidates={st['candidates']};pass1={st['pass1_seconds']};pass2={st['pass2_seconds']}"


WORK = {"pairs": run_pairs, "son": run_son}
plan = [(c, 1.0) for c in args.cores_list.split(",")] + [(args.max_cores, float(f)) for f in args.fractions.split(",")]

for wl in args.workloads.split(","):
    for cores, fraction in plan:
        for rep in range(args.repeats):
            spark = get_spark(f"scaling-{wl}", cores, args.mem, shuffle_partitions=args.shuffle_partitions)
            n, load, secs, detail = WORK[wl](spark, fraction)
            spark.stop()
            row = {"workload": wl, "cores": cores, "fraction": fraction, "repeat": rep, "transactions": n,
                   "load_seconds": round(load, 1), "seconds": round(secs, 1), "detail": detail}
            with open(log, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
            print(row, flush=True)
