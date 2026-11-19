"""SON with PCY inside each partition vs SON with plain Apriori, same baskets.

    python scripts/32_son_pcy.py --mem 36g --min-support 0.01,0.005,0.001

PCY (Park, Chen & Yu): while pass 1 counts single items, it also hashes every
pair of every basket into a table of bucket counts. A pair is then counted
only if both items are frequent AND its bucket is frequent. Exact (a frequent
pair always lies in a frequent bucket), so the itemsets must be identical;
what changes is how many pairs each partition holds in memory.

Reports, per support level and for both versions: pass-1 time, total and
largest pair table, candidates, frequent itemsets, and whether the results are
identical. Writes data/results/son_pcy/summary.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark import StorageLevel
from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.son import son
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-support", default="0.01,0.005,0.001")
p.add_argument("--partitions", type=int, default=16)
p.add_argument("--buckets", type=int, default=1 << 20, help="PCY hash buckets per partition")
args = p.parse_args()

spark = get_spark("son-pcy", args.cores, args.mem)
t0 = time.time()
tx = (spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
           .select(F.filter("items", lambda i: i.startswith("DRUG=") | i.startswith("AE=")).alias("items"))
           .where(F.size("items") > 0))
baskets = (tx.rdd.map(lambda r: tuple(sorted(set(r[0]))))
             .repartition(args.partitions).persist(StorageLevel.MEMORY_AND_DISK))
n = baskets.count()
print(f"baskets: {n:,} in {args.partitions} partitions (loaded in {time.time() - t0:.1f}s)", flush=True)

out = config.RESULTS_DIR / "son_pcy"
out.mkdir(parents=True, exist_ok=True)
summary = {"transactions": n, "partitions": args.partitions, "pcy_buckets_per_partition": args.buckets, "runs": []}

for s in [float(x) for x in args.min_support.split(",")]:
    min_count = math.ceil(s * n)
    run = {"min_support": s, "min_count": min_count}
    results = {}
    for name, buckets in (("apriori", None), ("pcy", args.buckets)):
        t1 = time.time()
        freq, st = son(baskets, n, min_count, pcy_buckets=buckets)
        st["son_seconds"] = round(time.time() - t1, 1)
        st["frequent"] = len(freq)
        results[name] = freq
        run[name] = st
    a, b = results["apriori"], results["pcy"]
    run["identical"] = a == b
    pa, pb = run["apriori"]["pass1_pairs_counted"], run["pcy"]["pass1_pairs_counted"]
    run["pair_table_reduction_pct"] = round(100 * (1 - pb / pa), 1) if pa else None
    la, lb = run["apriori"]["pass1_largest_pair_table"], run["pcy"]["pass1_largest_pair_table"]
    run["largest_pair_table_reduction_pct"] = round(100 * (1 - lb / la), 1) if la else None
    summary["runs"].append(run)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(run), flush=True)

print("\nsupport | pairs held in pass 1 (Apriori -> PCY) | largest table per partition | pass 1 s (Apriori / PCY) | identical")
for r in summary["runs"]:
    print(f"{r['min_support']:<7} | {r['apriori']['pass1_pairs_counted']:,} -> {r['pcy']['pass1_pairs_counted']:,} "
          f"({r['pair_table_reduction_pct']}% fewer) | {r['apriori']['pass1_largest_pair_table']:,} -> "
          f"{r['pcy']['pass1_largest_pair_table']:,} | {r['apriori']['pass1_seconds']} / {r['pcy']['pass1_seconds']} "
          f"| {r['identical']}")
spark.stop()
