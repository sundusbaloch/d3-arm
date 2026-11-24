"""Paper 4's AV crash data through the whole D3-ARM pipeline.

    python scripts/93_crash_pipeline.py --mem 8g

Stages (the same functions as for FAERS; see d3arm/crash/pipeline.py):
  1. prepare      1,171 crash reports (CA DMV + NHTSA SGO, 2014Q4-2024Q1), normalised
  2. deduplicate  MinHash-LSH on each crash's token set (reporter, city, state, month,
                  time, vehicle, road, weather, movements ...), Jaccard >= 0.8;
                  all 685k pairs are also checked by brute force, so recall is exact.
                  A similar pair counts as the same crash only if the incident month
                  agrees, and the time when both have one (robotaxi crashes in one city
                  look alike: high Jaccard alone is not enough)
  3. mine         (a) SON + Paper 4's rules (support 0.01, confidence 0.7, lift >= 1),
                      before and after removing near-duplicates;
                  (b) disproportionality signals: circumstance -> outcome
                      (injury severity, crash type), a >= 5 and lower 95% CI of ROR > 1
  4. track        quarterly replay: first quarter each signal appears, stable or not;
                  Bloom filter alerts for new signals
No FAERS data needed. Writes data/results/crash_pipeline/.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pyspark.sql import functions as F

from d3arm import config
from d3arm.crash.arm import make_rules
from d3arm.crash.av import COLUMNS, baskets
from d3arm.crash.pipeline import load_full, tokens, transactions
from d3arm.mining.pairs import add_measures, drug_ae_counts
from d3arm.mining.son import son
from d3arm.novelty.bloom import Bloom, bloom_size
from d3arm.novelty.lsh import brute_force_pairs, ids_to_drop, near_duplicates
from d3arm.novelty.windows import quarter_label, replay, summarise_pairs
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--bands", type=int, default=25)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--threshold", type=float, default=0.8)
p.add_argument("--min-count", type=int, default=5)
p.add_argument("--xlsx", default=None, help="crash table (default: data/reference/paper4/av_crashes.xlsx)")
args = p.parse_args()

spark = get_spark("crash-pipeline", args.cores, args.mem, shuffle_partitions=8)
sc = spark.sparkContext
out = config.RESULTS_DIR / "crash_pipeline"
out.mkdir(parents=True, exist_ok=True)
ref = pd.read_csv(config.REFERENCE_DIR / "paper4_tables.csv")
t0 = time.time()

# ---- 1. prepare
df = load_full(args.xlsx) if args.xlsx else load_full()
n = len(df)
s = {"reports": n, "quarters": int(df["window"].nunique()),
     "first_quarter": df["window"].min(), "last_quarter": df["window"].max()}
print(f"1. prepare: {n:,} crash reports, {s['quarters']} quarters ({s['first_quarter']} - {s['last_quarter']})", flush=True)

# ---- 2. deduplicate
tok = tokens(df)
toks = spark.createDataFrame(tok, "primaryid long, tokens array<string>").cache()
verified, groups, st = near_duplicates(spark, toks, args.bands, args.rows, args.threshold, 500, 4)
lsh_pairs = {(int(r["id1"]), int(r["id2"])) for r in verified.collect()}
sets = {int(r["set_id"]): r["tokens"] for r in groups.collect()}
bf = brute_force_pairs(sets, args.threshold)
by_id = dict(tok)
key = lambda t, prefix: next((x for x in t if x.startswith(prefix)), None)


def same_crash(a, b):
    """similar AND the identifying fields agree: same incident month, and the same time when both have one"""
    ta, tb = by_id[a], by_id[b]
    if key(ta, "Month=") != key(tb, "Month="):
        return False
    ka, kb = key(ta, "Time="), key(tb, "Time=")
    return ka is None or kb is None or ka == kb


confirmed = {p for p in lsh_pairs if same_crash(*p)}
conf_df = spark.createDataFrame(sorted(confirmed) or [(0, 0)], "id1 long, id2 long").where("id1 != id2")
drop = {int(r["primaryid"]) for r in ids_to_drop(spark, toks, conf_df).collect()}
s["dedup"] = {**st, "brute_force_pairs": len(bf), "lsh_found": len(lsh_pairs & bf),
              "recall_vs_brute_force": round(len(lsh_pairs & bf) / len(bf), 3) if bf else None,
              "all_pairs": n * (n - 1) // 2,
              "similar_pairs_different_month_or_time": len(lsh_pairs - confirmed),
              "confirmed_same_crash_pairs": len(confirmed),
              "reports_removed": len(drop), "reports_removed_pct": round(100 * len(drop) / n, 1)}


def example(a, b):
    ta, tb = set(by_id[a]), set(by_id[b])
    return {"ids": [a, b], "jaccard": round(len(ta & tb) / len(ta | tb), 3),
            "only_first": sorted(ta - tb), "only_second": sorted(tb - ta)}


s["dedup"]["examples_confirmed"] = [example(*p) for p in sorted(confirmed)[:8]]
s["dedup"]["examples_rejected"] = [example(*p) for p in sorted(lsh_pairs - confirmed)[:8]]
print(f"\n2. deduplicate: {st['identical_groups']} groups of identical crashes ({st['reports_in_identical_groups']} reports); "
      f"LSH {st['candidate_pairs']} candidates -> {st['near_duplicate_pairs']} pairs with Jaccard >= {args.threshold}; "
      f"brute force {len(bf)} pairs, LSH found {len(lsh_pairs & bf)}")
print(f"   of these, {len(lsh_pairs - confirmed)} differ in incident month or time (different crashes that look alike), "
      f"{len(confirmed)} are the same crash; reports removed: {len(drop)} ({s['dedup']['reports_removed_pct']}%)")
for e in s["dedup"]["examples_confirmed"][:4]:
    print(f"   same crash {e['ids']} J={e['jaccard']}: {e['only_first']} vs {e['only_second']}")
for e in s["dedup"]["examples_rejected"][:3]:
    print(f"   look-alike {e['ids']} J={e['jaccard']}: {e['only_first']} vs {e['only_second']}")
sys.stdout.flush()


# ---- 3a. SON + Paper 4's rules, before and after
def found(table, rules):
    rows = ref[ref.table == table]
    trip = list(zip(rules["support"], rules["confidence"], rules["lift"]))
    return int(sum(any(abs(a - x) <= 6e-4 and abs(b - y) <= 6e-4 and abs(c - z) <= 6e-4 for x, y, z in trip)
                   for a, b, c in zip(rows.support, rows.confidence, rows.lift))), len(rows)


def rules_for(frame, label):
    b = baskets(frame[list(COLUMNS.values())])
    mc = math.ceil(0.01 * len(b) - 1e-9)
    t = time.time()
    freq, _ = son(sc.parallelize(b, 8), len(b), mc, max_k=4)
    res = {"reports": len(b), "min_count": mc, "itemsets": len(freq), "son_seconds": round(time.time() - t, 1)}
    keys = {}
    for k in (2, 3, 4):
        r = make_rules({i: f for i, f in freq.items() if len(i) <= k}, len(b), min_conf=0.7, min_lift=1.0)
        r = r[r["k"] == k]
        res[f"{k}_item_rules"] = len(r)
        if len(b) == n:   # the paper's values only apply to the full table (support changes with n)
            hit, tot = found(str(k), r)
            res[f"table{k}_rows_found"] = f"{hit}/{tot}"
        keys[k] = {(a, c) for a, c in zip(r["antecedent"], r["consequent"])}
    print(f"   {label}: {len(b)} reports, {len(freq)} itemsets; rules 2/3/4 items: "
          f"{res['2_item_rules']}/{res['3_item_rules']}/{res['4_item_rules']}"
          + (f"; paper rows found {res['table2_rows_found']}, {res['table3_rows_found']}, {res['table4_rows_found']}"
             if len(b) == n else ""))
    return res, keys


print("\n3a. SON + Paper 4's rules (support 0.01, confidence 0.7, lift >= 1)")
kept = df[~df["primaryid"].isin(drop)]
r_before, k_before = rules_for(df, "all reports")
r_after, k_after = rules_for(kept, "after LSH dedup")
s["rules"] = {"before": r_before, "after": r_after,
              "changed": {k: {"lost": len(k_before[k] - k_after[k]), "new": len(k_after[k] - k_before[k])} for k in (2, 3, 4)}}
print("   rules lost / new after dedup:", s["rules"]["changed"])

# ---- 3b. disproportionality signals: circumstance -> outcome
schema = "primaryid long, window string, items array<string>"
tx_all = spark.createDataFrame(transactions(df), schema).cache()
tx_kept = tx_all.where(~F.col("primaryid").isin(list(drop))) if drop else tx_all


def signals(tx):
    c, _ = drug_ae_counts(tx, args.min_count)
    return add_measures(c).toPandas()


sig_b, sig_a = signals(tx_all), signals(tx_kept)
clean = lambda x: x.split("=", 1)[1]
for t in (sig_b, sig_a):
    t["circumstance"], t["outcome"] = t["drug"].map(clean), t["ae"].map(clean)
sb = set(zip(sig_b[sig_b.signal].drug, sig_b[sig_b.signal].ae))
sa = set(zip(sig_a[sig_a.signal].drug, sig_a[sig_a.signal].ae))
s["signals"] = {"pairs_tested_before": len(sig_b), "signals_before": len(sb),
                "pairs_tested_after": len(sig_a), "signals_after": len(sa),
                "lost_after_dedup": len(sb - sa), "new_after_dedup": len(sa - sb)}
cols = ["circumstance", "outcome", "a", "n_drug", "ror", "ror_lower"]
top = {}
for outcome in sorted(sig_a["outcome"].unique()):
    g = sig_a[(sig_a.outcome == outcome) & sig_a.signal].sort_values("ror_lower", ascending=False)
    top[outcome] = g[cols].head(8).round(3).to_dict("records")
s["signals"]["top_after_dedup"] = top
sig_a.to_csv(out / "signals_after_dedup.csv", index=False)
print(f"\n3b. signals (a >= {args.min_count}, lower CI of ROR > 1): {len(sb)} before, {len(sa)} after dedup "
      f"(lost {len(sb - sa)}, new {len(sa - sb)}) out of {len(sig_a)} pairs tested")
for outcome, rows in top.items():
    if rows:
        print(f"   {outcome}:")
        for r in rows[:5]:
            print(f"      {r['circumstance']:55s} a={r['a']:4d}  ROR {r['ror']:7.2f}  lower {r['ror_lower']:6.2f}")

# support/confidence rules for the same outcomes, for comparison
b = baskets(kept[list(COLUMNS.values())])
freq, _ = son(sc.parallelize(b, 8), len(b), math.ceil(0.01 * len(b) - 1e-9), max_k=2)
s["signals"]["support_confidence_rules_to_outcomes"] = {}
for col in ("Highest injury severity alleged", "ACCIDENT TYPE"):
    r = make_rules(freq, len(b), min_conf=0.7, min_lift=1.0, target=f"{col}=")
    r = r[r["antecedent"].map(len) == 1]
    s["signals"]["support_confidence_rules_to_outcomes"][col] = {
        "rules": len(r), "outcomes": sorted({c[0] for c in r["consequent"]})}
print("   support/confidence rules (1 antecedent) to the same outcomes:",
      {k: (v["rules"], v["outcomes"]) for k, v in s["signals"]["support_confidence_rules_to_outcomes"].items()})
sys.stdout.flush()

# ---- 4. track over time
pairs = spark.createDataFrame(sig_a[["drug", "ae"]])
ev = replay(tx_kept, pairs).cache()
summ = summarise_pairs(ev).toPandas()
st_sig = summ[summ.signal_at_end]
s["time"] = {"pairs_followed": len(summ), "signals_at_end": len(st_sig),
             "stable": int(summ.stable.sum()),
             "fired_but_not_signal_at_end": int((summ.first_signal.notna() & ~summ.signal_at_end).sum()),
             "first_signal_per_year": {}}
first = summ[summ.first_signal.notna()].first_signal.astype(int).map(quarter_label)
s["time"]["first_signal_per_year"] = first.str[:4].value_counts().sort_index().to_dict()
ks = summ[summ.first_signal.notna()].copy()
ks["first"] = ks.first_signal.astype(int).map(quarter_label)
ks["circumstance"], ks["outcome"] = ks["drug"].map(clean), ks["ae"].map(clean)
s["time"]["severe_injury_signals"] = ks[ks.outcome == "Highest injury severity alleged=fatality/serious"][
    ["circumstance", "first", "stable", "a_end"]].sort_values("first").to_dict("records")
print(f"\n4. time: {len(st_sig)} signals at the end, {s['time']['stable']} stable, "
      f"{s['time']['fired_but_not_signal_at_end']} fired and did not last")
print("   first signal per year:", s["time"]["first_signal_per_year"])
for r in s["time"]["severe_injury_signals"][:10]:
    print(f"   fatality/serious <- {r['circumstance']:45s} first {r['first']}  stable {r['stable']}  a={r['a_end']}")

stream = (ev.where("signal").join(spark.createDataFrame(summ[["drug", "ae", "first_signal"]].dropna()), ["drug", "ae"])
            .select("qi", F.xxhash64("drug", "ae").alias("h1"), F.xxhash64("ae", "drug", F.lit("d3arm")).alias("h2"),
                    (F.col("qi") == F.col("first_signal")).alias("is_new")).toPandas().sort_values("qi", kind="stable"))
n_new = max(int(stream["is_new"].sum()), 1)
m, k = bloom_size(n_new, 0.01)
bfil = Bloom(m, k)
missed = alerts = 0
for qi, g in stream.groupby("qi", sort=True):
    seen = bfil.contains(g["h1"].to_numpy(), g["h2"].to_numpy())
    alerts += int((~seen).sum())
    missed += int((g["is_new"].to_numpy() & seen).sum())
    bfil.add(g["h1"].to_numpy(), g["h2"].to_numpy())
s["time"]["bloom_1pct"] = {"new_signals": n_new, "m_bits": m, "k": k, "alerts": alerts, "missed": missed,
                           "size_bytes": bfil.size_bytes, "exact_set_bytes": n_new * 8}
print(f"   Bloom filter (1%): {n_new} new signals, {alerts} alerts, {missed} missed, "
      f"{bfil.size_bytes} bytes vs {n_new * 8} bytes exact")

s["seconds"] = round(time.time() - t0, 1)
(out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
print(f"\ntotal {s['seconds']} s; results in {out}")
spark.stop()
