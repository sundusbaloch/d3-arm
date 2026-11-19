"""Drug-drug interaction triples (drug 1, drug 2, AE) on all of FAERS.

    python scripts/33_ddi_triples.py --mem 36g

Counts every (d1, d2, AE) with >= --min-count reports using A-priori passes
(frequent drugs -> frequent drug pairs -> triples), then computes
  - the combination ROR (both drugs vs all other reports) and Paper 2's rule,
  - the combination risk ratio CRR = rate(both) / max(rate(d1 only), rate(d2 only)) with a 95% CI,
  - interaction signal = combination signal AND lower CI of CRR > 1.
See d3arm/mining/triples.py for the formulas.

Also prints a few well-known interactions as a sanity check (not a validation),
esketamine's strongest interaction candidates, and how big a Bloom filter for
all triples would be compared with an exact set of 64-bit keys.

Writes data/results/triples/ (parquet) and summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.mining.triples import add_interaction_measures, triple_counts, workload
from d3arm.novelty.bloom import bloom_size
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--min-count", type=int, default=5)
p.add_argument("--max-drugs", type=int, default=0, help="leave out reports with more drugs (0 = keep all)")
p.add_argument("--shuffle-partitions", type=int, default=400)
p.add_argument("--fraction", type=float, default=1.0, help="sample of the reports, for a quick trial run")
args = p.parse_args()

spark = get_spark("ddi-triples", args.cores, args.mem, shuffle_partitions=args.shuffle_partitions)
out = config.RESULTS_DIR / "triples"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
if args.fraction < 1:
    tx = tx.sample(fraction=args.fraction, seed=42)
w = workload(tx)
print("workload before pruning:", json.dumps(w), flush=True)

c, s = triple_counts(tx, args.min_count, args.max_drugs or None)
tri = add_interaction_measures(c)
tri.write.mode("overwrite").parquet(str(out / "drug_drug_ae"))
tri = spark.read.parquet(str(out / "drug_drug_ae")).cache()
s.update(w)
s.update({"min_count": args.min_count, "max_drugs": args.max_drugs or None, "fraction": args.fraction,
          "triples": tri.count(),
          "combination_signals": tri.where("comb_signal").count(),
          "triples_with_crr": tri.where(F.col("crr").isNotNull()).count(),
          "interaction_signals": tri.where("interaction_signal").count(),
          "drug_pairs_with_an_interaction_signal": tri.where("interaction_signal").select("d1", "d2").distinct().count(),
          "seconds": round(time.time() - t0, 1)})

# memory for "have we seen this triple before?": exact set vs Bloom filter
n_keys = max(s["triples"], 1)
m, k = bloom_size(n_keys, 0.01)
s["exact_set_of_64bit_keys_mb"] = round(n_keys * 8 / 2 ** 20, 1)
s["bloom_1pct_mb"] = round(m / 8 / 2 ** 20, 1)
s["bloom_1pct_k"] = k
(out / "summary.json").write_text(json.dumps(s, indent=2))
for key, v in s.items():
    print(f"{key:40s} {v:>14,}" if isinstance(v, int) and not isinstance(v, bool) else f"{key:40s} {v!s:>14}")

cols = ["d1", "d2", "ae", "a11", "n11", "ror_comb", "ror_comb_lower", "rate_both", "rate_d1_only",
        "rate_d2_only", "crr", "crr_lower", "interaction_signal"]


def show(df, title, n=10):
    print(f"\n{title}")
    pdf = (df.orderBy(F.desc("a11")).limit(n).select(*cols).toPandas())
    if pdf.empty:
        print("  (none)")
        return []
    for col in ("d1", "d2", "ae"):
        pdf[col] = pdf[col].str.replace(r"^(DRUG|AE)=", "", regex=True)
    print(pdf.round(3).to_string(index=False))
    return pdf.to_dict("records")


def either_order(a, b):
    return ((F.col("d1").rlike(a) & F.col("d2").rlike(b)) | (F.col("d1").rlike(b) & F.col("d2").rlike(a)))


# well-known interactions: sanity check only (normalised names, so match by pattern)
known = {
    "warfarin + aspirin -> bleeding": (either_order("WARFARIN", "ASPIRIN|ACETYLSALICYLIC"), "HAEMORRHAGE"),
    "simvastatin + amiodarone/clarithromycin -> rhabdomyolysis":
        (either_order("SIMVASTATIN", "AMIODARONE|CLARITHROMYCIN"), "RHABDOMYOLYSIS"),
    "tramadol + SSRI -> serotonin syndrome":
        (either_order("TRAMADOL", "SERTRALINE|FLUOXETINE|PAROXETINE|CITALOPRAM|ESCITALOPRAM"), "SEROTONIN SYNDROME"),
}
s["known_interaction_checks"] = {}
for name, (cond, ae) in known.items():
    rows = show(tri.where(cond & F.col("ae").rlike(ae)), f"sanity check: {name}", 5)
    s["known_interaction_checks"][name] = {
        "triples_found": len(rows), "interaction_signals": sum(bool(r["interaction_signal"]) for r in rows)}

s["esketamine_interaction_signals"] = tri.where(
    (F.col("d1").rlike("ESKETAMINE") | F.col("d2").rlike("ESKETAMINE")) & F.col("interaction_signal")).count()
show(tri.where((F.col("d1").rlike("ESKETAMINE") | F.col("d2").rlike("ESKETAMINE")) & F.col("interaction_signal")),
     "esketamine: interaction signals with the most reports")
show(tri.where("interaction_signal"), "all drugs: interaction signals with the most reports")
(out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
print(f"\ntook {(time.time() - t0) / 60:.1f} min")
spark.stop()
