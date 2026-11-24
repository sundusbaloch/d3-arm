"""Ground truth: how well does each method separate known adverse drug reactions
from known non-reactions? Reference: the time-indexed reference standard of
Harpaz et al. 2014 (62 positive and 75 negative drug-event pairs).

    python scripts/36_ground_truth_harpaz.py --ref /content/drive/MyDrive/d3arm_data/reference/timeIndexedReferenceStandard.xls --mem 36g

Methods compared (all on the same clean FAERS transactions):
  Paper 1 rule          a >= 3 and ROR >= 2
  Paper 2 rule          a >= 5 and lower 95% CI of ROR > 1
  Paper 2 + stable      Paper 2 rule, and never lost after it first fired (quarterly replay)
  + LSH dedup           Paper 2 rule after removing the LSH near-duplicates
  D3-ARM                LSH dedup + stable
for two event definitions: narrow MedDRA terms only, and narrow + broad terms.

Needs results/lsh/near_duplicate_pairs (script 50). Writes data/results/ground_truth/.
The reference file is not in the repository: download it from figshare
(doi 10.6084/m9.figshare.1133904). Reading .xls needs the xlrd package.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.evaluation.ground_truth import (auc, load_reference, mcnemar_exact, metrics, pair_table,
                                           quarterly_counts)
from d3arm.novelty.lsh import reports_to_drop
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True, help="timeIndexedReferenceStandard.xls (or .xlsx with the same sheets)")
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--lsh-pairs", default="near_duplicate_pairs",
               help="near_duplicate_pairs (script 50) or near_duplicate_pairs_strict (script 52)")
p.add_argument("--shuffle-partitions", type=int, default=200)
args = p.parse_args()

spark = get_spark("ground_truth", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
R = config.RESULTS_DIR
out = R / ("ground_truth" + args.lsh_pairs.replace("near_duplicate_pairs", ""))
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

pairs, events, drugs = load_reference(args.ref)
print(f"reference: {len(pairs)} pairs ({int(pairs.truth.sum())} positive), "
      f"{pairs.drug.nunique()} drugs, {pairs.event.nunique()} events", flush=True)

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
verified = spark.read.parquet(str(R / "lsh" / args.lsh_pairs))
drop = reports_to_drop(spark, reports, verified).cache()
tx_lsh = tx.join(drop, "primaryid", "left_anti")

summary = {"reference": {"pairs": len(pairs), "positive": int(pairs.truth.sum()),
                         "negative": int((pairs.truth == 0).sum()), "drugs": int(pairs.drug.nunique()),
                         "events": int(pairs.event.nunique())},
           "reports_removed_by_lsh": drop.count(), "definitions": {}}

for label, levels in [("narrow", ["narrow"]), ("narrow_broad", ["narrow", "broad"])]:
    n_q, nd, ne, a, seen = quarterly_counts(spark, tx, pairs, events, drugs, levels)
    t_all = pair_table(pairs, n_q, nd, ne, a)
    n_q2, nd2, ne2, a2, _ = quarterly_counts(spark, tx_lsh, pairs, events, drugs, levels)
    t_lsh = pair_table(pairs, n_q2, nd2, ne2, a2)

    truth = t_all["truth"].to_numpy() == 1
    methods = {
        "Paper 1 rule (a >= 3, ROR >= 2)": t_all["paper1_rule"],
        "Paper 2 rule (a >= 5, lower CI > 1)": t_all["signal"],
        "Paper 2 rule + stable": t_all["stable"],
        "+ LSH dedup": t_lsh["signal"],
        "D3-ARM (LSH dedup + stable)": t_lsh["stable"],
    }
    res = {name: metrics(truth, flag) for name, flag in methods.items()}
    aucs = {"ROR, Paper 2 data": auc(truth, t_all["ror"]),
            "lower CI of ROR, Paper 2 data": auc(truth, t_all["ror_lower"]),
            "lower CI of ROR, after LSH": auc(truth, t_lsh["ror_lower"])}
    comp = mcnemar_exact(t_all["signal"], t_lsh["stable"], truth)

    both = t_all.merge(t_lsh, on=["drug", "event", "truth"], suffixes=("", "_lsh"))
    both.to_csv(out / f"harpaz_pairs_{label}.csv", index=False)
    summary["definitions"][label] = {
        "pairs_with_no_report": int((t_all["a"] == 0).sum()),
        "pairs_with_a_below_5": int((t_all["a"] < 5).sum()),
        "methods": res, "auc": aucs, "paper2_vs_d3arm_mcnemar": comp,
        "drugs_reports": {r["drug"]: int(r["reports"]) for r in seen.to_dict("records")}}

    print(f"\n=== event definition: {label} ===")
    print(f"pairs with no report: {summary['definitions'][label]['pairs_with_no_report']}, "
          f"with a < 5: {summary['definitions'][label]['pairs_with_a_below_5']}")
    print(f"{'method':38s} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  sens   spec   PPV")
    for name, m in res.items():
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{name:38s} {m['tp']:3d} {m['fp']:3d} {m['tn']:3d} {m['fn']:3d}  "
              f"{f(m['sensitivity'])}  {f(m['specificity'])}  {f(m['ppv'])}")
    print("AUC:", aucs)
    print("Paper 2 rule vs D3-ARM (correct calls only one of them got):", comp)
    sys.stdout.flush()

missing = sorted(set(drugs.drug) - set(summary["definitions"]["narrow"]["drugs_reports"]))
summary["drugs_not_found"] = missing
print("\nreference drugs with no FAERS report:", missing or "none")
summary["seconds"] = round(time.time() - t0, 1)
(out / "harpaz_summary.json").write_text(json.dumps(summary, indent=2))
print(f"total {summary['seconds']} s; results in {out}")
spark.stop()
