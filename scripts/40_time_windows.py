"""Novelty part 1: replay FAERS quarter by quarter (time to signal, stability)
and a Bloom filter that raises an alert only for signals not seen before.

    python scripts/40_time_windows.py --mem 36g

Needs data/results/signals/drug_ae (script 31). Writes data/results/windows/:
    pairs/            one row per drug-AE pair: first_signal, stable, ...
    summary.json      counts, time-to-signal table, Bloom filter results
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from d3arm import config
from d3arm.novelty.bloom import Bloom, bloom_size, expected_fp
from d3arm.novelty.windows import BASE, quarter_label, replay, summarise_pairs
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--shuffle-partitions", type=int, default=200)
p.add_argument("--bloom-p", default="0.05,0.01,0.001", help="target false-positive rates to try")
args = p.parse_args()

spark = get_spark("time-windows", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
out = config.RESULTS_DIR / "windows"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
final = spark.read.parquet(str(config.RESULTS_DIR / "signals" / "drug_ae")) \
             .select("drug", "ae", F.col("signal").alias("signal_full_data"))
ev = replay(tx, final.select("drug", "ae"))
ev.write.mode("overwrite").parquet(str(out / "evaluations"))
ev = spark.read.parquet(str(out / "evaluations"))
pairs = summarise_pairs(ev).join(final, ["drug", "ae"])
pairs = pairs.withColumn("stable", F.col("signal_full_data") & F.col("first_signal").isNotNull()
                         & (F.col("lost_after_first") == 0))
pairs.write.mode("overwrite").parquet(str(out / "pairs"))
pairs = spark.read.parquet(str(out / "pairs")).cache()
t_replay = time.time() - t0

s = {"pairs_followed": pairs.count(),
     "evaluations": ev.count(),
     "signals_full_data": pairs.where("signal_full_data").count(),
     "ever_signalled": pairs.where("first_signal IS NOT NULL").count(),
     "stable_signals": pairs.where("stable").count(),
     "signal_full_data_but_lost_at_some_point": pairs.where("signal_full_data AND lost_after_first > 0").count(),
     "signalled_once_but_not_at_end": pairs.where("first_signal IS NOT NULL AND NOT signal_full_data").count(),
     "replay_seconds": round(t_replay, 1)}

tts = {quarter_label(r["first_signal"]): r["count"] for r in
       pairs.where("first_signal IS NOT NULL").groupBy("first_signal").count().orderBy("first_signal").collect()}
s["new_signals_per_quarter"] = tts

# ---- Bloom filter stream: quarter by quarter, alert only for pairs not seen before
t1 = time.time()
stream = (ev.where("signal")
            .join(pairs.select("drug", "ae", "first_signal"), ["drug", "ae"])
            .select("qi", F.xxhash64("drug", "ae").alias("h1"),
                    F.xxhash64("ae", "drug", F.lit("d3arm")).alias("h2"),
                    (F.col("qi") == F.col("first_signal")).alias("is_new")))  # exact answer, to grade the Bloom filter
# millions of rows: going through a parquet file is much faster than toPandas()
stream.write.mode("overwrite").parquet(str(out / "stream"))
stream = pd.read_parquet(out / "stream").sort_values("qi", kind="stable")
n_expected = max(int(stream["is_new"].sum()), 1)   # distinct pairs that ever signal (tiny test data may have none)
bloom_runs = []
for target in [float(x) for x in args.bloom_p.split(",")]:
    m, k = bloom_size(n_expected, target)
    bf = Bloom(m, k)
    missed = alerts = truly_new = 0
    for qi, g in stream.groupby("qi", sort=True):
        h1, h2 = g["h1"].to_numpy(), g["h2"].to_numpy()
        seen = bf.contains(h1, h2)
        new_exact = g["is_new"].to_numpy()
        truly_new += int(new_exact.sum())
        alerts += int((~seen).sum())
        missed += int((new_exact & seen).sum())          # Bloom said "seen", but it is new: false positive
        bf.add(h1, h2)
    bloom_runs.append({"target_fp": target, "m_bits": m, "k": k, "size_kb": round(bf.size_bytes / 1024, 1),
                       "truly_new_signals": truly_new, "alerts_raised": alerts,
                       "missed_new_signals": missed, "measured_fp": round(missed / max(truly_new, 1), 5),
                       "expected_fp": round(expected_fp(n_expected, m, k), 5)})
s["bloom"] = bloom_runs
s["exact_set_of_64bit_keys_kb"] = round(n_expected * 8 / 1024, 1)
s["bloom_seconds"] = round(time.time() - t1, 1)
s["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(s, indent=2))

for k_, v in s.items():
    if k_ not in ("new_signals_per_quarter", "bloom"):
        print(f"{k_:42s} {v:>12,}" if isinstance(v, int) else f"{k_:42s} {v:>12}")
print("\nnew signals per quarter (first time the rule fires):")
print("  " + ", ".join(f"{q}: {n:,}" for q, n in tts.items()))
print("\nBloom filter:")
for r in bloom_runs:
    print("  ", json.dumps(r))

print("\nesketamine: when did each signal first appear?")
(pairs.where(F.col("drug").rlike("ESKETAMINE") & F.col("signal_full_data"))
      .orderBy(F.desc("a_end")).limit(15)
      .select(F.regexp_replace("ae", "^AE=", "").alias("ae"), "a_end", "quarters_eval", "first_signal",
              "lost_after_first", "stable")
      .toPandas()
      .assign(first_signal=lambda d: d["first_signal"].map(lambda q: quarter_label(int(q)) if q == q else "-"))
      .pipe(lambda d: print(d.to_string(index=False))))
spark.stop()
