"""Paper 4 reproduction: association rules for AV crashes (Ding et al. 2025, Tables 2-7).

    python scripts/91_paper4_av_rules.py --mem 8g

Data: the authors' own table (data/reference/paper4/av_crashes.xlsx, 1,171 reports).
Settings from the paper and its code: min confidence 0.7, lift >= 1,
  - all crashes: support 0.01, rules of exactly 2, 3 and 4 items, one consequent (Tables 2-4)
  - per crash type (head-on, rear-end, side/others): only the 808 reports with a known
    crash type, the SV impact area left out, support 0.005, consequent = the crash type (Tables 5-7)
Frequent itemsets come from our SON on Spark; the same analysis is also run with
mlxtend (the authors' library) to show both give identical rules.

Each published table row (support, confidence, lift, 3 decimals) is looked up
among our rules. Writes data/results/paper4/ (top rules per table, summary.json).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from d3arm import config
from d3arm.crash.arm import make_rules, mlxtend_rules, show
from d3arm.crash.av import baskets, load
from d3arm.mining.son import son
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--partitions", type=int, default=8)
p.add_argument("--skip-mlxtend", action="store_true")
args = p.parse_args()

spark = get_spark("paper4-av", args.cores, args.mem)
sc = spark.sparkContext
out = config.RESULTS_DIR / "paper4"
out.mkdir(parents=True, exist_ok=True)
ref = pd.read_csv(config.REFERENCE_DIR / "paper4_tables.csv")


def son_itemsets(b, min_support, max_k=None):
    n = len(b)
    mc = math.ceil(min_support * n - 1e-9)          # mlxtend keeps support >= min_support
    rdd = sc.parallelize(b, args.partitions).cache()
    t = time.time()
    freq, _ = son(rdd, n, mc, max_k=max_k)
    rdd.unpersist()
    return freq, mc, time.time() - t


def found(table, rules):
    """how many of the paper's rows exist among our rules (support, confidence, lift equal to 3 decimals)"""
    rows = ref[ref.table == table]
    trip = list(zip(rules["support"], rules["confidence"], rules["lift"]))   # paper values are rounded to 3 decimals
    hit = [any(abs(s - a) <= 6e-4 and abs(c - b) <= 6e-4 and abs(l - d) <= 6e-4 for a, b, d in trip)
           for s, c, l in zip(rows.support, rows.confidence, rows.lift)]
    return int(sum(hit)), len(rows)


def same_as_mlxtend(b, s, ours, k=None, target=None):
    if args.skip_mlxtend:
        return None, None
    t = time.time()
    _, mx = mlxtend_rules(b, s, 0.7, 1.0, max_len=k)
    sec = round(time.time() - t, 1)
    mx = mx[mx["consequents"].map(len) == 1]
    if k:
        mx = mx[mx["antecedents"].map(len) == k - 1]
    if target:
        mx = mx[mx["consequents"].map(lambda c: next(iter(c)) == target)]
    key = lambda a, c: (tuple(sorted(a)), tuple(sorted(c)))
    a = {key(x, y) for x, y in zip(ours["antecedent"], ours["consequent"])}
    m = {key(x, y) for x, y in zip(mx["antecedents"], mx["consequents"])}
    return a == m, sec


df = load()
summary = {"reports": len(df), "tables": {}}
print(f"AV crash reports: {len(df):,}")

# ---------- Tables 2-4: all crashes
b = baskets(df)
freq, mc, sec = son_itemsets(b, 0.01, max_k=4)
summary["all_crashes"] = {"min_support": 0.01, "min_count": mc, "itemsets_up_to_4": len(freq), "son_seconds": round(sec, 1)}
for k, table in ((2, "2"), (3, "3"), (4, "4")):
    r = make_rules({i: f for i, f in freq.items() if len(i) <= k}, len(b), min_conf=0.7, min_lift=1.0)
    r = r[r["k"] == k].reset_index(drop=True)
    same, mx_sec = same_as_mlxtend(b, 0.01, r, k=k)
    hit, tot = found(table, r)
    summary["tables"][table] = {"rules": len(r), "paper_rows_found": f"{hit}/{tot}",
                                "identical_to_mlxtend": same, "mlxtend_seconds": mx_sec}
    r.head(30).to_csv(out / f"table{table}_top30.csv", index=False)
    print(f"\nTable {table} ({k}-item rules): {len(r)} rules, paper rows found {hit}/{tot}, same as mlxtend: {same}")
    print(show(r, 20 if k == 2 else 10))

# ---------- Tables 5-7: per crash type
known = df[df["ACCIDENT TYPE"].notna()]
cols = [c for c in df.columns if c != "SV-area"]
b2 = baskets(known, cols)
freq2, mc2, sec2 = son_itemsets(b2, 0.005)
summary["crash_types"] = {"reports_with_known_type": len(known), "min_support": 0.005, "min_count": mc2,
                          "itemsets": len(freq2), "son_seconds": round(sec2, 1),
                          "type_counts": known["ACCIDENT TYPE"].value_counts().to_dict()}
print(f"\ncrash-type analysis: {len(known)} reports, {len(freq2):,} frequent itemsets ({sec2:.1f}s)")
for table, label in (("5_head_on", "head on"), ("6_rear_end", "rear end"), ("7_side", "others")):
    target = f"ACCIDENT TYPE={label}"
    r = make_rules(freq2, len(b2), min_conf=0.7, min_lift=1.0, target=target)
    r = r[r["antecedent"].map(len) > 0].reset_index(drop=True)
    same, mx_sec = same_as_mlxtend(b2, 0.005, r, target=target)
    hit, tot = found(table, r)
    summary["tables"][table] = {"rules": len(r), "paper_rows_found": f"{hit}/{tot}",
                                "identical_to_mlxtend": same, "mlxtend_seconds": mx_sec}
    r.head(30).to_csv(out / f"table{table}_top30.csv", index=False)
    print(f"\nTable {table}: {len(r)} rules, paper rows found {hit}/{tot}, same as mlxtend: {same}")
    print(show(r, 10))

(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\n" + json.dumps(summary["tables"], indent=1))
spark.stop()
