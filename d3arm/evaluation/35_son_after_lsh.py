"""SON on the data after removing LSH near-duplicates: do the frequent itemsets change?

    python scripts/35_son_after_lsh.py --mem 36g --min-support 0.01,0.005,0.001

Script 30 mined the Paper 2-deduplicated transactions, near-duplicates included.
This script drops the near-duplicate reports found by LSH (one report kept per
group, exactly as in step 2 of scripts/60_ablation.py) and runs SON again.

Two comparisons per support level:
  same_support : the usual setting, min count = ceil(s * n) on the smaller data
  same_count   : the old min count, so only the removed duplicates make a difference

Needs results/lsh/near_duplicate_pairs (script 50). Uses results/son/itemsets_s<s>
(script 30) as the "before" itemsets; if missing, SON is run on the old data here.
Writes data/results/son_after_lsh/summary.json and itemsets_s<s>/.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark import StorageLevel
from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.son import son
from d3arm.novelty.lsh import reports_to_drop
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--lsh-pairs", default="near_duplicate_pairs",
               help="near_duplicate_pairs (script 50) or near_duplicate_pairs_strict (script 52)")
p.add_argument("--min-support", default="0.01,0.005,0.001")
p.add_argument("--partitions", type=int, default=16)
args = p.parse_args()

spark = get_spark("son_after_lsh", args.cores, args.mem)
R = config.RESULTS_DIR
out = R / ("son_after_lsh" + args.lsh_pairs.replace("near_duplicate_pairs", ""))
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
verified = spark.read.parquet(str(R / "lsh" / args.lsh_pairs))
drop = reports_to_drop(spark, reports, verified).cache()


def to_baskets(df):
    items = (df.select(F.filter("items", lambda i: i.startswith("DRUG=") | i.startswith("AE=")).alias("items"))
               .where(F.size("items") > 0))
    return (items.rdd.map(lambda r: tuple(sorted(set(r[0]))))
                 .repartition(args.partitions).persist(StorageLevel.MEMORY_AND_DISK))


before = to_baskets(tx)
n1 = before.count()
after = to_baskets(tx.join(drop, "primaryid", "left_anti"))
n2 = after.count()
summary = {"transactions_before": n1, "transactions_after": n2, "removed": n1 - n2,
           "removed_pct": round(100 * (n1 - n2) / n1, 2), "partitions": args.partitions,
           "load_seconds": round(time.time() - t0, 1), "runs": []}
print(f"baskets before {n1:,}, after removing near-duplicates {n2:,} (-{n1 - n2:,})", flush=True)


def compare(old, new):
    lost = [k for k in old if k not in new]
    gained = [k for k in new if k not in old]
    common = [k for k in old if k in new]
    drops = [100 * (old[k] - new[k]) / old[k] for k in common]
    return {"itemsets_before": len(old), "itemsets_after": len(new), "kept": len(common),
            "lost": len(lost), "gained": len(gained),
            "identical_set": not (lost or gained),
            "count_drop_pct_median": round(statistics.median(drops), 2) if drops else None,
            "count_drop_pct_max": round(max(drops), 2) if drops else None}, lost


def show(items):
    return " + ".join(i.split("=", 1)[1] for i in items)


for s in [float(x) for x in args.min_support.split(",")]:
    run = {"min_support": s}
    c1 = math.ceil(s * n1)                       # same rounding as scripts 20 and 30
    old_dir = R / "son" / f"itemsets_s{s}"
    if old_dir.exists():
        old = {tuple(sorted(r["items"])): r["freq"] for r in spark.read.parquet(str(old_dir)).collect()}
        run["before_from"] = "results/son (script 30)"
    else:
        t1 = time.time()
        old, _ = son(before, n1, c1)
        run["before_from"] = f"computed here ({round(time.time() - t1, 1)} s)"

    c2 = math.ceil(s * n2)
    t1 = time.time()
    new, stats = son(after, n2, c2)
    run["son_seconds"] = round(time.time() - t1, 1)
    run["min_count_before"], run["min_count_after"] = c1, c2
    run["same_support"], lost_s = compare(old, new)

    t1 = time.time()
    new_c, _ = son(after, n2, c1)
    run["same_count_seconds"] = round(time.time() - t1, 1)
    run["same_count"], lost_c = compare(old, new_c)

    common = [k for k in old if k in new_c]
    biggest = sorted(common, key=lambda k: (old[k] - new_c[k]) / old[k], reverse=True)[:10]
    run["largest_count_drops"] = [{"itemset": show(k), "before": old[k], "after": new_c[k],
                                   "drop_pct": round(100 * (old[k] - new_c[k]) / old[k], 2)} for k in biggest]
    run["lost_at_same_count"] = [{"itemset": show(k), "before": old[k]} for k in
                                 sorted(lost_c, key=lambda k: -old[k])[:10]]

    rows = [(list(k), v) for k, v in new.items()]
    if rows:
        spark.createDataFrame(rows, "items array<string>, freq long") \
             .write.mode("overwrite").parquet(str(out / f"itemsets_s{s}"))
    summary["runs"].append(run)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    a, b = run["same_support"], run["same_count"]
    print(f"\nsupport {s}: min count {c1:,} -> {c2:,}; SON {run['son_seconds']} s")
    print(f"  same support: {a['itemsets_before']:,} -> {a['itemsets_after']:,} itemsets "
          f"(lost {a['lost']}, gained {a['gained']}); counts drop median {a['count_drop_pct_median']}%")
    print(f"  same count  : {b['itemsets_before']:,} -> {b['itemsets_after']:,} itemsets "
          f"(lost {b['lost']}, gained {b['gained']}); counts drop median {b['count_drop_pct_median']}%, "
          f"max {b['count_drop_pct_max']}%")
    for d in run["largest_count_drops"][:5]:
        print(f"    {d['drop_pct']:6.2f}%  {d['before']:>9,} -> {d['after']:>9,}  {d['itemset']}")
    for d in run["lost_at_same_count"][:5]:
        print(f"    lost: {d['before']:>9,}  {d['itemset']}")
    sys.stdout.flush()

print(f"\ntotal {round(time.time() - t0, 1)} s; summary in {out / 'summary.json'}")
spark.stop()
