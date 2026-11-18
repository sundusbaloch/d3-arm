"""Reports WITHOUT an age or an event date: flag groups with identical content.

LSH (script 50) only uses reports that have both an age and an event date,
because without them two different patients on the same drug with the same
reaction look exactly alike. So here we do NOT call anything a duplicate and
we do NOT remove anything. We only measure how much identical content there
is, and list the largest groups (e.g. a consumer campaign for one product).

    python scripts/51_mass_reporting.py --mem 36g

Writes data/results/mass_reporting/summary.json and top_groups.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.novelty.lsh import report_tokens
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--shuffle-partitions", type=int, default=200)
p.add_argument("--top", type=int, default=20)
args = p.parse_args()

spark = get_spark("mass-reporting", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
out = config.RESULTS_DIR / "mass_reporting"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
toks = report_tokens(reports, with_ids=False)
groups = toks.groupBy("tokens").agg(F.count("*").alias("n_reports")).cache()
big = groups.where("n_reports > 1")
s = {"reports_without_age_or_event_date": toks.count(),
     "distinct_token_sets": groups.count(),
     "groups_with_identical_content": big.count(),
     "reports_in_those_groups": big.agg(F.sum("n_reports")).first()[0] or 0,
     "groups_with_100_or_more_reports": groups.where("n_reports >= 100").count(),
     "reports_in_groups_of_100_or_more": groups.where("n_reports >= 100").agg(F.sum("n_reports")).first()[0] or 0}
s["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(s, indent=2))
for k, v in s.items():
    print(f"{k:36s} {v:>14,}" if isinstance(v, int) else f"{k:36s} {v:>14}")

top = groups.orderBy(F.desc("n_reports")).limit(args.top).toPandas()
top["tokens"] = top["tokens"].map(lambda t: " | ".join(t))
top.to_csv(out / "top_groups.csv", index=False)
print(f"\nlargest {min(10, len(top))} groups (identical drugs, AEs, indications, sex and country):")
for _, row in top.head(10).iterrows():
    text = row["tokens"]
    print(f"  {row['n_reports']:>8,}  {text[:150]}{' ...' if len(text) > 150 else ''}")
spark.stop()
