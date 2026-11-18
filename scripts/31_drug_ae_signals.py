"""Drug -> AE signals for every drug in FAERS, down to Paper 2's threshold (5 reports).

    python scripts/31_drug_ae_signals.py --mem 36g --min-count 5

Counts {drug}, {AE} and {drug, AE} with two A-priori passes (see
d3arm/mining/pairs.py), then ROR and PRR with 95% CIs for every pair.
Writes data/results/signals/drug_ae/ (parquet) and summary.json, and prints
esketamine's top pairs next to our Paper 2 reproduction as a cross-check.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.pairs import add_measures, drug_ae_counts
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-count", type=int, default=5)
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

spark = get_spark("drug-ae-signals", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
t0 = time.time()
tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
counts, info = drug_ae_counts(tx, args.min_count)
res = add_measures(counts)
out = config.RESULTS_DIR / "signals"
out.mkdir(parents=True, exist_ok=True)
res.write.mode("overwrite").parquet(str(out / "drug_ae"))
res = spark.read.parquet(str(out / "drug_ae"))

summary = {"min_count": args.min_count, **info,
           "drug_ae_pairs": res.count(),
           "signals_ror": res.where("signal").count(),
           "signals_prr_lower_gt_1": res.where("a >= 5 AND prr_lower > 1").count(),
           "drugs_with_a_signal": res.where("signal").select("drug").distinct().count(),
           "seconds": round(time.time() - t0, 1)}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
for k, v in summary.items():
    print(f"{k:28s} {v:>14,}" if isinstance(v, int) else f"{k:28s} {v:>14}")

# cross-check: esketamine (any role, all other reports as control) vs our Paper 2 run
pd.set_option("display.width", 200)
esk = (res.where(F.col("drug").rlike("ESKETAMINE"))
          .select(F.regexp_replace("ae", "^AE=", "").alias("pt"), "drug", "a", "ror", "ror_lower", "signal")
          .orderBy(F.desc("a")).limit(15).toPandas())
p2 = config.RESULTS_DIR / "paper2" / "ror_main.csv"
if p2.exists():
    ref = pd.read_csv(p2)[["pt", "a", "ror"]].rename(columns={"a": "a_paper2_repro", "ror": "ror_paper2_repro"})
    ref["pt"] = ref["pt"].str.upper()
    esk = esk.merge(ref, on="pt", how="left")
print("\nesketamine, top 15 AEs by report count:")
print(esk.round(2).to_string(index=False))
spark.stop()
