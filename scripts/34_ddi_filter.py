"""Make the drug-drug interaction signals of script 33 specific enough to use.

    python scripts/34_ddi_filter.py --mem 36g

Script 33 flags 15.9M of 31.8M triples as "interaction signals": 31.8M
tests at 95% confidence each, plus artefacts. This script adds three checks,
one at a time, and shows how many signals survive each and whether the
well-known interactions survive:
  1. multiple testing: Bonferroni over all triples tested, i.e. the lower bounds of
     both the combination ROR and the CRR must stay above 1 at z = Phi^-1(1 - 0.05 / m)
  2. co-packaged products: drop pairs whose two "drugs" almost never appear apart
     (n(d1,d2) / min(n(d1), n(d2)) >= --kit-share; e.g. a skincare kit)
  3. effect size: CRR >= --min-crr (the pair at least doubles the AE rate of the worse drug alone)
Reads data/results/triples/drug_drug_ae (script 33); writes data/results/triples/filtered/ and filter_summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.triples import specificity_filters, z_bonferroni
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--kit-share", type=float, default=0.8)
p.add_argument("--min-crr", type=float, default=2.0)
args = p.parse_args()

spark = get_spark("ddi-filter", args.cores, args.mem)
t0 = time.time()
R = config.RESULTS_DIR / "triples"
tri = spark.read.parquet(str(R / "drug_drug_ae"))
m = tri.count()
z = z_bonferroni(m, args.alpha)
t = specificity_filters(tri, m, args.alpha, args.kit_share, args.min_crr).cache()

steps = [("0. script 33 rule (95% CIs)", F.col("interaction_signal")),
         ("1. + Bonferroni", F.col("interaction_signal") & F.col("bonf_ok")),
         ("2. + not a co-packaged product", F.col("interaction_signal") & F.col("bonf_ok") & F.col("not_kit")),
         ("3. + CRR >= %.1f" % args.min_crr, F.col("interaction_signal") & F.col("bonf_ok") & F.col("not_kit") & F.col("big_crr"))]
known = {
    "aspirin + warfarin -> GI haemorrhage": ("WARFARIN", "ASPIRIN|ACETYLSALICYLIC", "^AE=GASTROINTESTINAL HAEMORRHAGE$"),
    "clarithromycin + simvastatin -> rhabdomyolysis": ("SIMVASTATIN", "CLARITHROMYCIN", "^AE=RHABDOMYOLYSIS$"),
    "amiodarone + simvastatin -> rhabdomyolysis": ("SIMVASTATIN", "AMIODARONE", "^AE=RHABDOMYOLYSIS$"),
    "tramadol + sertraline -> serotonin syndrome": ("TRAMADOL", "SERTRALINE", "^AE=SEROTONIN SYNDROME$"),
    "tramadol + fluoxetine -> serotonin syndrome": ("TRAMADOL", "FLUOXETINE", "^AE=SEROTONIN SYNDROME$"),
    "tramadol + paroxetine -> serotonin syndrome": ("TRAMADOL", "PAROXETINE", "^AE=SEROTONIN SYNDROME$"),
    "tramadol + citalopram -> serotonin syndrome": ("TRAMADOL", "^DRUG=CITALOPRAM", "^AE=SEROTONIN SYNDROME$"),
    "tramadol + escitalopram -> serotonin syndrome": ("TRAMADOL", "ESCITALOPRAM", "^AE=SEROTONIN SYNDROME$"),
}


def is_known(a, b, ae):
    return (((F.col("d1").rlike(a) & F.col("d2").rlike(b)) | (F.col("d1").rlike(b) & F.col("d2").rlike(a)))
            & F.col("ae").rlike(ae))


summary = {"triples_tested": m, "alpha": args.alpha, "bonferroni_z": round(z, 3),
           "kit_share": args.kit_share, "min_crr": args.min_crr, "steps": [], "known_interactions": {}}
for name, cond in steps:
    s = t.where(cond)
    row = {"step": name, "signals": s.count(), "drug_pairs": s.select("d1", "d2").distinct().count()}
    summary["steps"].append(row)
    print(f"{name:34s} {row['signals']:>12,} signals  {row['drug_pairs']:>9,} drug pairs", flush=True)
for label, (a, b, ae) in known.items():
    k = t.where(is_known(a, b, ae))
    summary["known_interactions"][label] = [
        {"rows": k.where(cond).count()} for _, cond in steps]
kept = {lab: [x["rows"] > 0 for x in v] for lab, v in summary["known_interactions"].items()}
summary["known_kept_per_step"] = [sum(v[i] for v in kept.values()) for i in range(len(steps))]
print(f"\nknown interactions still flagged after each step (of {len(known)}):", summary["known_kept_per_step"])
for lab, v in kept.items():
    print(f"  {lab:48s} " + " ".join("yes" if x else " - " for x in v))

kits = (t.where(~F.col("not_kit")).select("d1", "d2", "n11", "n_d1", "n_d2", "together_share").distinct()
         .orderBy(F.desc("n11")).limit(8).toPandas())
print("\nlargest pairs treated as one co-packaged product:")
print(kits.round(3).to_string(index=False))
summary["co_packaged_pairs"] = t.where(~F.col("not_kit")).select("d1", "d2").distinct().count()

final = t.where(steps[-1][1])
final.write.mode("overwrite").parquet(str(R / "filtered"))
cols = ["d1", "d2", "ae", "a11", "n11", "ror_comb", "crr", "crr_lower_bonf"]
top = final.orderBy(F.desc("a11")).limit(15).select(*cols).toPandas()
for c in ("d1", "d2", "ae"):
    top[c] = top[c].str.replace(r"^(DRUG|AE)=", "", regex=True)
print("\nafter all checks, signals with the most reports:")
print(top.round(3).to_string(index=False))
esk = (final.where(F.col("d1").rlike("ESKETAMINE") | F.col("d2").rlike("ESKETAMINE"))
            .orderBy(F.desc("a11")).limit(10).select(*cols).toPandas())
summary["esketamine_signals_after_all_checks"] = int(final.where(F.col("d1").rlike("ESKETAMINE") | F.col("d2").rlike("ESKETAMINE")).count())
print("\nesketamine after all checks:", summary["esketamine_signals_after_all_checks"])
if len(esk):
    for c in ("d1", "d2", "ae"):
        esk[c] = esk[c].str.replace(r"^(DRUG|AE)=", "", regex=True)
    print(esk.round(3).to_string(index=False))
summary["seconds"] = round(time.time() - t0, 1)
(R / "filter_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\ntook {summary['seconds']}s")
spark.stop()
