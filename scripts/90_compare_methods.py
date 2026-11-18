"""All four papers' methods and our pipeline on the SAME data (FAERS 2014Q3-2024Q4).

  Paper 1  modified Apriori: drug-AE pairs with >= 3 reports, ranked by a secondary
           indicator; we use ROR >= 2 (the highest threshold in their simulation code)
  Paper 2  ROR signal rule: a >= 5 and lower 95% CI of ROR > 1
  Paper 3  target-constrained Apriori: consequent fixed to the outcome (here an AE),
           support >= 0.2%, confidence >= 65%, lift > 1, redundant rules pruned
           (their restrained-driver setting; unrestrained: 3% / 50%)
  Paper 4  plain Apriori (mlxtend in the paper): support >= 1%, confidence >= 0.7, lift >= 1
  Ours     Paper 2's rule + LSH near-duplicates removed + stable over time

Papers 3 and 4 are rule miners for crash data; on FAERS their rules come from
the frequent itemsets (we reuse SON's exact itemsets at 0.1% support, which
contain every itemset these thresholds can produce).

For each method: how many drug -> AE results, and how many of esketamine's
signals from our Paper 2 reproduction it recovers (recall on a reference
set the domain paper itself produced).

    python scripts/90_compare_methods.py --mem 36g
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.pairs import add_measures, drug_ae_counts
from d3arm.novelty.windows import summarise_pairs
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--itemsets", default="itemsets_s0.001", help="SON output folder under results/son")
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

spark = get_spark("compare-methods", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
R = config.RESULTS_DIR
out = R / "comparison"
out.mkdir(parents=True, exist_ok=True)
ESK = "DRUG=ESKETAMINE"

ref = pd.read_csv(R / "paper2" / "ror_main.csv")
ref_pts = set(ref.loc[ref["signal"], "pt"].str.upper())          # esketamine signals, Paper 2 reproduction


def esk_recall(pts):
    pts = {x.upper() for x in pts}
    recall = round(len(pts & ref_pts) / len(ref_pts), 3) if ref_pts else None
    return {"esketamine_aes_found": len(pts), "recall_of_paper2_esketamine_signals": recall}


rows = []

# ---- Paper 1: pairs with >= 3 reports, ROR >= 2
t = time.time()
tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
c3, _ = drug_ae_counts(tx, 3)
p1 = add_measures(c3).where(F.col("ror") >= 2).cache()
n_p1 = p1.count()
esk1 = [r["ae"][3:] for r in p1.where(F.col("drug") == ESK).select("ae").collect()]
rows.append({"method": "Paper 1: modified Apriori (minsup 3, ROR >= 2)", "drug_ae_results": n_p1,
             **esk_recall(esk1), "seconds": round(time.time() - t, 1)})
print(rows[-1], flush=True)

# ---- Paper 2: a >= 5 and lower CI > 1 (script 31)
sig = spark.read.parquet(str(R / "signals" / "drug_ae"))
esk2 = [r["ae"][3:] for r in sig.where((F.col("drug") == ESK) & F.col("signal")).select("ae").collect()]
rows.append({"method": "Paper 2: ROR a >= 5, lower CI > 1", "drug_ae_results": sig.where("signal").count(),
             **esk_recall(esk2), "seconds": json.loads((R / "signals" / "summary.json").read_text())["seconds"]})
print(rows[-1], flush=True)

# ---- Papers 3 and 4: rules from the exact frequent itemsets
son_sum = json.loads((R / "son" / "summary.json").read_text())
N = son_sum["transactions"]
freq = {tuple(sorted(r["items"])): r["freq"] for r in spark.read.parquet(str(R / "son" / args.itemsets)).collect()}
rules = []
for items, f in freq.items():
    if len(items) < 2:
        continue
    for y in items:                                  # one item as consequent, the rest as antecedent
        x = tuple(i for i in items if i != y)
        conf = f / freq[x]
        rules.append({"antecedent": x, "consequent": y, "support": f / N, "conf": conf,
                      "lift": conf / (freq[(y,)] / N)})
rules = pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "conf", "lift"])


def kind(row):
    a = "DRUG" if all(i.startswith("DRUG=") for i in row["antecedent"]) else (
        "AE" if all(i.startswith("AE=") for i in row["antecedent"]) else "MIXED")
    return f"{a}->{row['consequent'].split('=')[0]}"


rules["type"] = [kind(r) for _, r in rules.iterrows()]
rules.to_pickle(out / "all_rules_from_son_itemsets.pkl")


def prune_redundant(r):
    """drop X -> Y if some smaller X' -> Y (X' inside X) in the same set has at least the same confidence"""
    keep = []
    best = {(row.antecedent, row.consequent): row.conf for row in r.itertuples()}
    for row in r.itertuples():
        x = row.antecedent
        sub = any(best.get((s, row.consequent), -1) >= row.conf
                  for k in range(1, len(x)) for s in combinations(x, k))
        keep.append(not sub)
    return r.loc[pd.Series(keep, index=r.index, dtype=bool)]


settings = [
    ("Paper 3: target Apriori (sup 0.2%, conf 65%, lift > 1)", 0.002, 0.65, 1.0, True),
    ("Paper 3: target Apriori (sup 3%, conf 50%, lift > 1)", 0.03, 0.50, 1.0, True),
    ("Paper 4: Apriori (sup 1%, conf 0.7, lift >= 1)", 0.01, 0.70, 1.0, False),
    ("Paper 4: Apriori (sup 0.5%, conf 0.7, lift >= 1)", 0.005, 0.70, 1.0, False),
]
type_counts = {}
for name, s, c, l, target in settings:
    r = rules[(rules["support"] >= s) & (rules["conf"] >= c) & (rules["lift"] > l if target else rules["lift"] >= l)]
    if target:                                       # consequent fixed to the outcome: an AE
        r = prune_redundant(r[r["consequent"].str.startswith("AE=")])
    type_counts[name] = r["type"].value_counts().to_dict()
    da = r[r["type"] == "DRUG->AE"]
    esk = [y[3:] for x, y in zip(da["antecedent"], da["consequent"]) if ESK in x]
    rows.append({"method": name, "drug_ae_results": len(da), "all_rules": len(r), **esk_recall(esk),
                 "seconds": son_sum["runs"][-1]["son_seconds"] if son_sum["runs"] else None})
    print(rows[-1], flush=True)
esk_freq = freq.get((ESK,), 0)

# ---- ours: Paper 2's rule after LSH dedup, kept only if stable over time
t = time.time()
sig2 = spark.read.parquet(str(R / "ablation" / "signals_after_lsh"))
ev = spark.read.parquet(str(R / "ablation" / "evaluations_after_lsh"))
st = (summarise_pairs(ev).join(sig2.select("drug", "ae", "signal"), ["drug", "ae"])
      .where(F.col("signal") & F.col("first_signal").isNotNull() & (F.col("lost_after_first") == 0)).cache())
esk_n = [r["ae"][3:] for r in st.where(F.col("drug") == ESK).select("ae").collect()]
rows.append({"method": "Ours: Paper 2 rule + LSH dedup + stable over time", "drug_ae_results": st.count(),
             **esk_recall(esk_n), "seconds": round(time.time() - t, 1)})
print(rows[-1], flush=True)

table = pd.DataFrame(rows)
table.to_csv(out / "methods_on_faers.csv", index=False)
summary = {"transactions": N, "reference_esketamine_signals": len(ref_pts),
           "esketamine_report_count_in_transactions": esk_freq or "below the 0.1% itemset threshold",
           "rule_types_by_setting": type_counts, "methods": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
pd.set_option("display.width", 220, "display.max_colwidth", 60)
print("\nall methods on FAERS 2014Q3-2024Q4:")
print(table.to_string(index=False))
print("\nrule types per Paper 3/4 setting:", json.dumps(type_counts, indent=1))
spark.stop()
