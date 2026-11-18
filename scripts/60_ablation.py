"""Ablation: what does each novelty step change in the signals?

    python scripts/60_ablation.py --mem 36g

  step 1  Paper 2 dedup only                       (results/signals, script 31)
  step 2  + drop near-duplicates found by LSH      (keep one report per group)
  step 3  + keep only signals that are stable over time (time-window replay)

Needs results/lsh/near_duplicate_pairs (script 50) and results/windows/pairs (script 40).
Writes data/results/ablation/summary.json and esketamine_comparison.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.pairs import add_measures, drug_ae_counts
from d3arm.novelty.lsh import reports_to_drop
from d3arm.novelty.windows import replay, summarise_pairs
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-count", type=int, default=5)
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

spark = get_spark("ablation", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
R = config.RESULTS_DIR
out = R / "ablation"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
s = {}

# step 1: as in script 31
base = spark.read.parquet(str(R / "signals" / "drug_ae"))
s["step1_transactions"] = tx.count()
s["step1_pairs"] = base.count()
s["step1_signals"] = base.where("signal").count()

# step 2: drop LSH near-duplicates
verified = spark.read.parquet(str(R / "lsh" / "near_duplicate_pairs"))
drop = reports_to_drop(spark, reports, verified).cache()
s["step2_reports_dropped"] = drop.count()
tx2 = tx.join(drop, "primaryid", "left_anti").cache()
s["step2_transactions"] = tx2.count()
counts2, _ = drug_ae_counts(tx2, args.min_count)
sig2 = add_measures(counts2)
sig2.write.mode("overwrite").parquet(str(out / "signals_after_lsh"))
sig2 = spark.read.parquet(str(out / "signals_after_lsh"))
s["step2_pairs"] = sig2.count()
s["step2_signals"] = sig2.where("signal").count()
both = base.select("drug", "ae", F.col("signal").alias("s1")).join(
    sig2.select("drug", "ae", F.col("signal").alias("s2")), ["drug", "ae"], "full")
s["step2_signals_lost"] = both.where("s1 AND (s2 IS NULL OR NOT s2)").count()
s["step2_signals_gained"] = both.where("s2 AND (s1 IS NULL OR NOT s1)").count()

# step 3: stability over time, on the step-2 data
ev = replay(tx2, sig2.select("drug", "ae"))
ev.write.mode("overwrite").parquet(str(out / "evaluations_after_lsh"))
ev = spark.read.parquet(str(out / "evaluations_after_lsh"))
pairs3 = summarise_pairs(ev).join(sig2.select("drug", "ae", F.col("signal").alias("signal_full_data")), ["drug", "ae"])
stable = pairs3.where(F.col("signal_full_data") & F.col("first_signal").isNotNull() & (F.col("lost_after_first") == 0))
s["step3_stable_signals"] = stable.count()
s["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(s, indent=2))
for k, v in s.items():
    print(f"{k:28s} {v:>14,}" if isinstance(v, int) else f"{k:28s} {v:>14}")

# esketamine before / after
esk = lambda df, tag: (df.where(F.col("drug").rlike("ESKETAMINE"))
                         .select(F.regexp_replace("ae", "^AE=", "").alias("ae"), F.col("a").alias(f"a_{tag}"),
                                 F.round("ror", 2).alias(f"ror_{tag}"), F.col("signal").alias(f"signal_{tag}")))
cmp = esk(base, "step1").join(esk(sig2, "step2"), "ae", "left").orderBy(F.desc("a_step1"))
cmp.toPandas().to_csv(out / "esketamine_comparison.csv", index=False)
print("\nesketamine, top 15 AEs before and after removing near-duplicates:")
cmp.limit(15).show(15, truncate=False)
spark.stop()
