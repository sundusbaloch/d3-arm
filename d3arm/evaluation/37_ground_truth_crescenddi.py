"""Ground truth from CRESCENDDI (Kontsioti et al. 2022): drug-drug interaction
triplets and single-drug ADRs with negative controls.

    python scripts/37_ground_truth_crescenddi.py --ref /content/drive/MyDrive/d3arm_data/reference/crescenddi --mem 36g

--ref is the folder with the five "Data Record" .xlsx files from figshare
(collection doi 10.6084/m9.figshare.c.5481408, CC BY 4.0; not in the repository).

Part 1, DDI triplets (10,286 positive, 4,544 negative), the ladder of scripts 33-34:
  combination signal -> interaction signal (+ CRR lower CI > 1) -> + Bonferroni
  -> + not co-packaged -> + CRR >= 2, on the Paper 2 data and after LSH dedup.
Part 2, single drug-event pairs (Data Record 3), the methods of script 36:
  Paper 1 rule, Paper 2 rule, Paper 2 + stable, + LSH dedup, D3-ARM (LSH + stable).

Needs results/lsh/near_duplicate_pairs (script 50). Writes data/results/ground_truth/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.evaluation.crescenddi import load, single_counts, single_table, triplet_counts, triplet_measures
from d3arm.evaluation.ground_truth import auc, mcnemar_exact, metrics
from d3arm.novelty.lsh import reports_to_drop
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True, help="folder with the CRESCENDDI Data Record 1-3 .xlsx files")
p.add_argument("--m", type=int, default=0,
               help="number of triples tested, for Bonferroni (default: results/triples/summary.json, else 31,761,534)")
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--lsh-pairs", default="near_duplicate_pairs",
               help="near_duplicate_pairs (script 50) or near_duplicate_pairs_strict (script 52)")
p.add_argument("--shuffle-partitions", type=int, default=200)
p.add_argument("--skip-single", action="store_true", help="only the DDI triplets")
args = p.parse_args()

spark = get_spark("crescenddi", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
R = config.RESULTS_DIR
out = R / ("ground_truth" + args.lsh_pairs.replace("near_duplicate_pairs", ""))
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

trip, single, drugs = load(args.ref)
m = args.m
if not m:
    f = R / "triples" / "summary.json"
    m = json.loads(f.read_text())["triples"] if f.exists() else 31_761_534
print(f"triplets: {len(trip):,} ({int(trip.truth.sum()):,} positive); single pairs: {len(single):,} "
      f"({int(single.truth.sum()):,} positive); Bonferroni m = {m:,}", flush=True)

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
verified = spark.read.parquet(str(R / "lsh" / args.lsh_pairs))
drop = reports_to_drop(spark, reports, verified).cache()
tx_lsh = tx.join(drop, "primaryid", "left_anti")
summary = {"bonferroni_m": m, "triplets": {"total": len(trip), "positive": int(trip.truth.sum())},
           "single": {"total": len(single), "positive": int(single.truth.sum())}}


def report(title, truth, methods, scores, extra=None):
    res = {name: metrics(truth, flag) for name, flag in methods.items()}
    aucs = {name: auc(truth, s) for name, s in scores.items()}
    print(f"\n=== {title} ===")
    print(f"{'method':44s} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}   sens   spec    PPV")
    for name, r in res.items():
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{name:44s} {r['tp']:6d} {r['fp']:6d} {r['tn']:6d} {r['fn']:6d}  {f(r['sensitivity'])}  "
              f"{f(r['specificity'])}  {f(r['ppv'])}")
    print("AUC:", aucs)
    if extra:
        print(extra)
    sys.stdout.flush()
    return {"methods": res, "auc": aucs}


# ---- part 1: DDI triplets
c_all, seen = triplet_counts(spark, tx, trip, drugs)
c_lsh, _ = triplet_counts(spark, tx_lsh, trip, drugs)
t_all = triplet_measures(spark, c_all, m)
t_lsh = triplet_measures(spark, c_lsh, m)
key = ["d1", "d2", "ae"]
t_all = trip.rename(columns={"event": "ae"}).merge(t_all, on=key)
t_lsh = trip.rename(columns={"event": "ae"}).merge(t_lsh, on=key)
truth = t_all["truth"].to_numpy() == 1
assert (t_lsh["truth"].to_numpy() == t_all["truth"].to_numpy()).all()
ddi = report("DDI triplets", truth, {
    "combination signal (a11 >= 5, ROR lower > 1)": t_all["comb_signal"],
    "interaction signal (+ CRR lower > 1)": t_all["interaction_signal"],
    "+ Bonferroni": t_all["step1_bonferroni"],
    "+ not co-packaged": t_all["step2_not_kit"],
    "+ CRR >= 2": t_all["step3_crr2"],
    "interaction signal after LSH dedup": t_lsh["interaction_signal"],
    "all checks after LSH dedup": t_lsh["step3_crr2"]},
    {"CRR (Paper 2 data)": t_all["crr"], "lower CI of CRR": t_all["crr_lower"],
     "lower CI of combination ROR": t_all["ror_comb_lower"]})
ddi["coverage"] = {"triplets_with_a11_0": int((t_all["a11"] == 0).sum()),
                   "triplets_with_a11_below_5": int((t_all["a11"] < 5).sum()),
                   "triplets_with_a11_at_least_5": int((t_all["a11"] >= 5).sum())}
print("coverage:", ddi["coverage"])
summary["triplets"].update(ddi)
t_all.merge(t_lsh[key + ["a11", "interaction_signal", "step3_crr2"]], on=key, suffixes=("", "_lsh")) \
     .to_csv(out / "crescenddi_triplets.csv", index=False)
found = set(seen["drug"])
summary["ddi_drugs_not_found"] = sorted((set(trip.d1) | set(trip.d2)) - found)
print(f"DDI reference drugs with no FAERS report: {len(summary['ddi_drugs_not_found'])}",
      summary["ddi_drugs_not_found"][:40])

# ---- part 2: single drug-event pairs
if not args.skip_single:
    s_all = single_table(single, *single_counts(spark, tx, single, drugs))
    s_lsh = single_table(single, *single_counts(spark, tx_lsh, single, drugs))
    truth = s_all["truth"].to_numpy() == 1
    sg = report("single drug-event pairs", truth, {
        "Paper 1 rule (a >= 3, ROR >= 2)": s_all["paper1_rule"],
        "Paper 2 rule (a >= 5, lower CI > 1)": s_all["signal"],
        "Paper 2 rule + stable": s_all["stable"],
        "+ LSH dedup": s_lsh["signal"],
        "D3-ARM (LSH dedup + stable)": s_lsh["stable"]},
        {"ROR, Paper 2 data": s_all["ror"], "lower CI of ROR, Paper 2 data": s_all["ror_lower"],
         "lower CI of ROR, after LSH": s_lsh["ror_lower"]},
        extra=f"Paper 2 rule vs D3-ARM: {mcnemar_exact(s_all['signal'], s_lsh['stable'], truth)}")
    sg["paper2_vs_d3arm_mcnemar"] = mcnemar_exact(s_all["signal"], s_lsh["stable"], truth)
    sg["coverage"] = {"pairs_with_a_0": int((s_all["a_end"] == 0).sum()),
                      "pairs_with_a_at_least_5": int((s_all["a_end"] >= 5).sum())}
    print("coverage:", sg["coverage"])
    summary["single"].update(sg)
    s_all.merge(s_lsh[["drug", "event", "a_end", "signal", "stable"]], on=["drug", "event"], suffixes=("", "_lsh")) \
         .to_csv(out / "crescenddi_single.csv", index=False)

summary["seconds"] = round(time.time() - t0, 1)
(out / "crescenddi_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\ntotal {summary['seconds']} s; results in {out}")
spark.stop()
