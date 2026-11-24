"""Stricter near-duplicates: an LSH pair (Jaccard >= 0.8) counts as the same case
only if its identifying fields agree (sex, country, age within 1 year, compatible
event dates). Script 50 used the token similarity alone, and the crash data
(script 93) showed that similar-looking reports are often different events.

    python scripts/52_lsh_same_case.py --mem 36g

Reads results/lsh/near_duplicate_pairs (script 50); writes
results/lsh/near_duplicate_pairs_strict (same columns) and same_case_summary.json.
Use it in scripts 35, 36, 37 and 60 with --lsh-pairs near_duplicate_pairs_strict.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.novelty.lsh import ids_to_drop, report_tokens, same_case_filter
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--age-tolerance", type=int, default=1)
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

spark = get_spark("lsh-same-case", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
R = config.RESULTS_DIR / "lsh"
t0 = time.time()
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
verified = spark.read.parquet(str(R / "near_duplicate_pairs"))
keep, s = same_case_filter(spark, reports, verified, args.age_tolerance)
keep.write.mode("overwrite").parquet(str(R / "near_duplicate_pairs_strict"))
keep = spark.read.parquet(str(R / "near_duplicate_pairs_strict"))
s["age_tolerance"] = args.age_tolerance
s["reports_dropped_all_pairs"] = ids_to_drop(spark, report_tokens(reports), verified).count()
s["reports_dropped_same_case_pairs"] = ids_to_drop(spark, report_tokens(reports), keep).count()
s["seconds"] = round(time.time() - t0, 1)
(R / "same_case_summary.json").write_text(json.dumps(s, indent=2))
for k, v in s.items():
    print(f"{k:34s} {v:>12,}" if isinstance(v, int) else f"{k:34s} {v:>12}")
spark.stop()
