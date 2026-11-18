"""Baseline: Spark MLlib FPGrowth (parallel FP-Growth) on the FAERS transactions.

This is the "off the shelf" frequent itemset miner we compare SON against:
same transactions, same minimum support, and the frequent itemsets must come
out identical (both are exact). What we compare is time and what each one
can give us besides the itemsets.

    python scripts/20_fpgrowth_baseline.py --mem 36g --min-support 0.01,0.005,0.001

By default only DRUG= and AE= items are used (SEX= and AGE= are dropped),
because the signals we want are drug -> AE.

Writes data/results/fpgrowth/summary.json and the itemsets for each support
level to data/results/fpgrowth/itemsets_s<support>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

from d3arm import config
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-support", default="0.01,0.005,0.001",
               help="comma-separated fractions of transactions")
p.add_argument("--min-confidence", type=float, default=0.05)
p.add_argument("--partitions", type=int, default=64)
p.add_argument("--with-demographics", action="store_true")
p.add_argument("--no-rules", action="store_true", help="only mine itemsets (rules re-run part of the mining)")
args = p.parse_args()

spark = get_spark("fpgrowth-baseline", args.cores, args.mem)
tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
if not args.with_demographics:
    tx = tx.withColumn("items", F.filter("items", lambda i: i.startswith("DRUG=") | i.startswith("AE=")))
tx = tx.where(F.size("items") > 0).select("items").repartition(args.partitions).cache()
n = tx.count()
print(f"transactions: {n:,}", flush=True)

out = config.RESULTS_DIR / "fpgrowth"
out.mkdir(parents=True, exist_ok=True)
summary = {"transactions": n, "partitions": args.partitions,
           "with_demographics": args.with_demographics, "runs": []}

for s in [float(x) for x in args.min_support.split(",")]:
    t0 = time.time()
    model = FPGrowth(itemsCol="items", minSupport=s, minConfidence=args.min_confidence,
                     numPartitions=args.partitions).fit(tx)
    fi = model.freqItemsets.withColumn("k", F.size("items")).cache()
    n_sets = fi.count()                                   # forces the mining to run
    t_mine = time.time() - t0
    by_k = {r["k"]: r["count"] for r in fi.groupBy("k").count().collect()}

    # drug -> AE rules: every antecedent item is a drug, the consequent is one AE
    # note: MLlib does not cache its own itemsets, so generating rules re-runs part
    # of the mining - rules_seconds is NOT a clean timing, only mine_seconds is
    t1 = time.time()
    n_rules = n_drug_ae = None
    if not args.no_rules:
        rules = model.associationRules.cache()
        drug_ae = rules.where(F.forall("antecedent", lambda i: i.startswith("DRUG="))
                              & F.forall("consequent", lambda i: i.startswith("AE=")))
        n_rules, n_drug_ae = rules.count(), drug_ae.count()
        rules.unpersist()
    t_rules = time.time() - t1

    fi.drop("k").write.mode("overwrite").parquet(str(out / f"itemsets_s{s}"))
    run = {"min_support": s, "min_count": int(-(-s * n // 1)), "itemsets": n_sets,
           "itemsets_by_size": dict(sorted(by_k.items())), "mine_seconds": round(t_mine, 1),
           "rules": n_rules, "drug_to_ae_rules": n_drug_ae, "rules_seconds": round(t_rules, 1)}
    summary["runs"].append(run)
    (out / f"summary{'_norules' if args.no_rules else ''}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(run), flush=True)
    print("most frequent pairs:")
    fi.where("k = 2").orderBy(F.desc("freq")).show(10, truncate=False)
    fi.unpersist()

spark.stop()
