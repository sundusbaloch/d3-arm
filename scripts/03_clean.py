"""Step 3: deduplication + normalisation.

    python scripts/03_clean.py --mem 8g                  # Paper 2's rule (default)
    python scripts/03_clean.py --mem 8g --dedup exact    # stricter report-level rule

Counts after each step go to data/logs/dedup_counts_<method>.json.
The cleaned reports always go to data/clean/reports, so the last run wins.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.faers.clean import clean_faers
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--shuffle-partitions", type=int, default=64)
p.add_argument("--dedup", choices=["paper2", "exact"], default="paper2")
args = p.parse_args()

spark = get_spark("faers-clean", args.cores, args.mem, args.shuffle_partitions)
t0 = time.time()
clean_faers(spark, config.PARQUET_DIR, config.CLEAN_DIR, config.LOG_DIR,
            drug_map=config.REFERENCE_DIR / "drug_map.csv", dedup=args.dedup)
print(f"done in {time.time() - t0:.1f}s")
spark.stop()
