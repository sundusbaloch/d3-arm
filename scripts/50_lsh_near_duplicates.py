"""Novelty part 2: near-duplicate reports with MinHash-LSH (b bands x r rows).

    python scripts/50_lsh_near_duplicates.py --mem 36g
    python scripts/50_lsh_near_duplicates.py --mem 36g --bands 20 --rows 10 --out-name lsh_b20_r10

Writes data/results/lsh/summary.json and near_duplicate_pairs/.
Validation: inside the reports of one drug (default esketamine) we also find
all pairs by brute force and measure how many of them LSH found (recall).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.novelty.lsh import brute_force_pairs, candidate_probability, find_near_duplicates
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--bands", type=int, default=25)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--threshold", type=float, default=0.8)
p.add_argument("--max-bucket", type=int, default=500)
p.add_argument("--partitions", type=int, default=64)
p.add_argument("--shuffle-partitions", type=int, default=200)
p.add_argument("--validate-drug", default="ESKETAMINE")
p.add_argument("--validate-max", type=int, default=3000)
p.add_argument("--out-name", default="lsh", help="results folder name, e.g. lsh_b20_r10 for a tuning run")
args = p.parse_args()

spark = get_spark("lsh", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
out = config.RESULTS_DIR / args.out_name
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
b, r, th = args.bands, args.rows, args.threshold
verified, groups, toks, s = find_near_duplicates(spark, reports, b, r, th, args.max_bucket, args.partitions)
verified.write.mode("overwrite").parquet(str(out / "near_duplicate_pairs"))
s = {"bands": b, "rows": r, "hash_functions": b * r, "threshold": th,
     "s_curve_threshold": round((1 / b) ** (1 / r), 3), "clean_reports": reports.count(), **s}
s["lsh_seconds"] = round(time.time() - t0, 1)
s["p_candidate_at_jaccard"] = {str(j): round(candidate_probability(j, b, r), 4) for j in (0.5, 0.6, 0.7, 0.8, 0.9)}

# ---- recall check by brute force inside one drug's reports
t1 = time.time()
tag = f"DRUG={args.validate_drug}"
sub = {row["set_id"]: row["tokens"] for row in groups.where(F.array_contains("tokens", tag)).collect()}
if len(sub) > args.validate_max:
    keep = set(random.Random(1).sample(sorted(sub), args.validate_max))
    sub = {k: v for k, v in sub.items() if k in keep}
truth = brute_force_pairs(sub, th)
ids = list(sub)
found = {(row["id1"], row["id2"]) for row in verified.where(F.col("id1").isin(ids) & F.col("id2").isin(ids)).collect()}
s["validation"] = {"drug": args.validate_drug, "token_sets": len(sub), "true_pairs_brute_force": len(truth),
                   "found_by_lsh": len(found & truth), "extra_found": len(found - truth),
                   "recall": round(len(found & truth) / len(truth), 4) if truth else None,
                   "seconds": round(time.time() - t1, 1)}
s["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(s, indent=2))

for k, v in s.items():
    if isinstance(v, dict):
        print(f"{k}: {json.dumps(v)}")
    else:
        print(f"{k:34s} {v:>14,}" if isinstance(v, int) else f"{k:34s} {v:>14}")

print("\nlargest groups of reports with identical content:")
for row in groups.orderBy(F.desc("n_reports")).limit(5).collect():
    print(f"  {row['n_reports']:>7,} reports: {', '.join(row['tokens'][:8])}{' ...' if len(row['tokens']) > 8 else ''}")
print("\nexample near-duplicate pairs (Jaccard < 1):")
ex = verified.where("jaccard < 1").orderBy(F.desc("jaccard")).limit(3).collect()
tok = {row["set_id"]: set(row["tokens"]) for row in
       groups.where(F.col("set_id").isin([e["id1"] for e in ex] + [e["id2"] for e in ex])).collect()}
for e in ex:
    a, bb = tok[e["id1"]], tok[e["id2"]]
    print(f"  J={e['jaccard']:.2f}  only in first: {sorted(a - bb)}  only in second: {sorted(bb - a)}")
spark.stop()
