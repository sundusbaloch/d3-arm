"""Ground truth from CRESCENDDI (Kontsioti et al. 2022, Scientific Data 9:72,
figshare collection doi 10.6084/m9.figshare.c.5481408, CC BY 4.0).

Two parts of the reference set are used:
  Data Records 1 and 2  drug-drug-event triplets: 10,286 positive and 4,544 negative controls
  Data Record 3         single-drug ADRs (positive; "Indication" rows left out) and negative controls

Events are MedDRA preferred terms, drugs are ingredient names. A report is
exposed to a drug if one of its drug items is (or contains) that ingredient,
with the same matching rule as d3arm.evaluation.ground_truth.

The triplets are scored with exactly the measures of scripts 33 and 34
(d3arm.mining.triples): combination ROR, CRR, Bonferroni, co-packaged check, CRR >= 2.
The single-drug pairs are scored like script 36, with the quarterly replay
done in pandas (vectorised) because there are about 90,000 pairs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from d3arm.evaluation.ground_truth import _component_matches, _event_matches, _key
from d3arm.mining.triples import add_interaction_measures, specificity_filters
from d3arm.novelty.windows import quarter_index, quarter_label

# FAERS (US, ingredient-level) names for reference names that differ
SYNONYMS = {
    "VALPROATE": ["VALPROIC ACID", "DIVALPROEX", "VALPROATE SODIUM"],
    "PARACETAMOL": ["ACETAMINOPHEN"],
    "SALBUTAMOL": ["ALBUTEROL"],
    "ADRENALINE": ["EPINEPHRINE"],
    "NORADRENALINE": ["NOREPINEPHRINE"],
    "GLYCERYL TRINITRATE": ["NITROGLYCERIN"],
    "CICLOSPORIN": ["CYCLOSPORINE"],
    "COLECALCIFEROL": ["CHOLECALCIFEROL"],
    "LEVOTHYROXINE": ["LEVOTHYROXINE SODIUM"],
}


def _find(folder, number):
    hits = sorted(Path(folder).glob(f"*Data*Record*{number}*.xlsx"))
    if not hits:
        raise FileNotFoundError(f"Data Record {number} (.xlsx) not found in {folder}")
    return hits[0]


def load(folder):
    """-> triplets (d1, d2, event, truth) with d1 < d2, single (drug, event, truth), drug names (drug, name)"""
    pos = pd.read_excel(_find(folder, 1))
    neg = pd.read_excel(_find(folder, 2))
    rows = []
    for df, truth in [(pos, 1), (neg, 0)]:
        for d1, d2, e in zip(df["DRUG_1_CONCEPT_NAME"], df["DRUG_2_CONCEPT_NAME"], df["EVENT_CONCEPT_NAME"]):
            a, b = sorted([_key(d1), _key(d2)])
            if a != b:
                rows.append((a, b, _key(e), truth))
    trip = pd.DataFrame(rows, columns=["d1", "d2", "event", "truth"]).drop_duplicates(["d1", "d2", "event"])

    x = pd.read_excel(_find(folder, 3), sheet_name=None)
    tabs = list(x.values())
    p = tabs[0]
    p = p[p["EVENT_TYPE"].astype(str).str.strip().str.lower() == "adverse event"]
    single = pd.concat([pd.DataFrame({"drug": p["DRUG_CONCEPT_NAME"].map(_key),
                                      "event": p["EVENT_CONCEPT_NAME"].map(_key), "truth": 1}),
                        pd.DataFrame({"drug": tabs[1]["DRUG_CONCEPT_NAME"].map(_key),
                                      "event": tabs[1]["EVENT_CONCEPT_NAME"].map(_key), "truth": 0})])
    # a pair listed as both positive and negative is dropped (should not happen)
    both = single.groupby(["drug", "event"])["truth"].nunique()
    single = single.drop_duplicates(["drug", "event", "truth"])
    single = single[~single.set_index(["drug", "event"]).index.isin(both[both > 1].index)]

    names = sorted(set(trip["d1"]) | set(trip["d2"]) | set(single["drug"]))
    drugs = pd.DataFrame([(d, n) for d in names for n in [d] + SYNONYMS.get(d, [])], columns=["drug", "name"])
    return trip.reset_index(drop=True), single.reset_index(drop=True), drugs


def _events_pdf(*frames):
    ev = sorted(set().union(*[set(f["event"]) for f in frames]))
    return pd.DataFrame({"event": ev, "pt": ev, "level": "narrow"})


# ------------------------------------------------------------------ triplets
def triplet_counts(spark, tx, trip, drugs):
    """counts for every reference triplet, in the columns d3arm.mining.triples expects"""
    tx = tx.select("primaryid", "window", "items").cache()
    N = tx.count()
    need = set(trip["d1"]) | set(trip["d2"])
    dm = _component_matches(tx, drugs[drugs["drug"].isin(need)], spark).select("primaryid", "drug").distinct().cache()
    em = _event_matches(tx, _events_pdf(trip), spark, ["narrow"]).select("primaryid", "event").distinct().cache()
    ref = spark.createDataFrame(trip[["d1", "d2", "event"]])
    nd = dm.groupBy("drug").count()
    ne = em.groupBy("event").count()
    pairs = (dm.alias("x").join(dm.alias("y"), "primaryid")
               .where(F.col("x.drug") < F.col("y.drug"))
               .select("primaryid", F.col("x.drug").alias("d1"), F.col("y.drug").alias("d2"))
               .join(F.broadcast(ref.select("d1", "d2").distinct()), ["d1", "d2"], "left_semi")).cache()
    n11 = pairs.groupBy("d1", "d2").count().withColumnRenamed("count", "n11")
    a11 = (pairs.join(em, "primaryid").withColumnRenamed("event", "event")
                .join(F.broadcast(ref), ["d1", "d2", "event"], "left_semi")
                .groupBy("d1", "d2", "event").count().withColumnRenamed("count", "a11"))
    de = (ref.select(F.col("d1").alias("drug"), "event").union(ref.select(F.col("d2").alias("drug"), "event"))
             .distinct())
    nde = (dm.join(em, "primaryid").join(F.broadcast(de), ["drug", "event"], "left_semi")
             .groupBy("drug", "event").count())
    out = (ref.join(a11, ["d1", "d2", "event"], "left").join(n11, ["d1", "d2"], "left")
              .join(nde.select(F.col("drug").alias("d1"), "event", F.col("count").alias("n_d1_ae")), ["d1", "event"], "left")
              .join(nde.select(F.col("drug").alias("d2"), "event", F.col("count").alias("n_d2_ae")), ["d2", "event"], "left")
              .join(nd.select(F.col("drug").alias("d1"), F.col("count").alias("n_d1")), "d1", "left")
              .join(nd.select(F.col("drug").alias("d2"), F.col("count").alias("n_d2")), "d2", "left")
              .join(ne.select("event", F.col("count").alias("n_ae")), "event", "left")
              .fillna(0, ["a11", "n11", "n_d1_ae", "n_d2_ae", "n_d1", "n_d2", "n_ae"])
              .withColumn("N", F.lit(N).cast("long"))
              .withColumnRenamed("event", "ae"))
    seen = dm.groupBy("drug").count().toPandas()
    res = out.toPandas()
    for d in (pairs, dm, em, tx):
        d.unpersist()
    return res, seen


def triplet_measures(spark, counts, m, kit_share=0.8, min_crr=2.0):
    """the measures and checks of scripts 33 and 34, on the reference triplets"""
    # only triplets reported at least once can be scored (zeros would divide by zero in Spark's ANSI mode)
    seen = counts[counts["a11"] > 0]
    flags = ["comb_signal", "interaction_signal", "bonf_ok", "not_kit", "big_crr"]
    if len(seen):
        s = spark.createDataFrame(seen)
        s = specificity_filters(add_interaction_measures(s), m, kit_share=kit_share, min_crr=min_crr)
        scored = s.toPandas()
        t = counts.merge(scored.drop(columns=[c for c in counts.columns if c not in ("d1", "d2", "ae")]),
                         on=["d1", "d2", "ae"], how="left")
    else:
        t = counts.copy()
        for c in ["ror_comb", "ror_comb_lower", "crr", "crr_lower"]:
            t[c] = np.nan
    for c in flags:
        t[c] = t[c].fillna(False).astype(bool) if c in t else False
    t["step1_bonferroni"] = t["interaction_signal"] & t["bonf_ok"]
    t["step2_not_kit"] = t["step1_bonferroni"] & t["not_kit"]
    t["step3_crr2"] = t["step2_not_kit"] & t["big_crr"]
    return t


# ------------------------------------------------------------------ single drug-event pairs
def single_counts(spark, tx, single, drugs):
    """new reports per quarter: N, per drug, per event, and per reference pair (pandas)"""
    tx = tx.select("primaryid", "window", "items").cache()
    n_q = tx.groupBy(quarter_index(F.col("window")).alias("qi")).count().toPandas().rename(columns={"count": "N"})
    dm = _component_matches(tx, drugs[drugs["drug"].isin(set(single["drug"]))], spark).cache()
    em = _event_matches(tx, _events_pdf(single), spark, ["narrow"]).cache()
    nd = dm.groupBy("drug", "qi").count().toPandas().rename(columns={"count": "n_drug"})
    ne = em.groupBy("event", "qi").count().toPandas().rename(columns={"count": "n_event"})
    ref = spark.createDataFrame(single[["drug", "event"]])
    a = (dm.join(em.select("primaryid", "event"), "primaryid")
           .join(F.broadcast(ref), ["drug", "event"], "left_semi")
           .groupBy("drug", "event", "qi").count().toPandas().rename(columns={"count": "a"}))
    for d in (dm, em, tx):
        d.unpersist()
    return n_q, nd, ne, a


def _ror_lower(a, nd, ne, n, z=1.96):
    a, nd, ne, n = (np.asarray(v, float) for v in (a, nd, ne, n))
    b, c, d = nd - a, ne - a, n - nd - ne + a
    ok = (a > 0) & (b > 0) & (c > 0) & (d > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ror = np.where(ok, (a * d) / (b * c), np.nan)
        se = np.where(ok, np.sqrt(1 / a + 1 / b + 1 / c + 1 / d), np.nan)
        low = np.exp(np.log(ror) - z * se)
    return ror, low


def single_table(single, n_q, nd, ne, a):
    """vectorised version of ground_truth.pair_table: final signal, Paper 1 rule, first signal, stable"""
    quarters = sorted(n_q["qi"])
    Ncum = n_q.set_index("qi")["N"].reindex(quarters, fill_value=0).cumsum()
    D = nd.pivot_table(index="drug", columns="qi", values="n_drug", fill_value=0).reindex(columns=quarters, fill_value=0).cumsum(axis=1)
    E = ne.pivot_table(index="event", columns="qi", values="n_event", fill_value=0).reindex(columns=quarters, fill_value=0).cumsum(axis=1)
    h = a.sort_values(["drug", "event", "qi"]).copy()
    h["a_cum"] = h.groupby(["drug", "event"])["a"].cumsum()
    dlong = D.stack().rename("nd").reset_index()
    elong = E.stack().rename("ne").reset_index()
    h = h.merge(dlong, on=["drug", "qi"], how="left").merge(elong, on=["event", "qi"], how="left")
    h["N"] = h["qi"].map(Ncum)
    _, low = _ror_lower(h["a_cum"], h["nd"], h["ne"], h["N"])
    h["sig"] = (h["a_cum"] >= 5) & (low > 1)
    first = h[h["sig"]].groupby(["drug", "event"])["qi"].min().rename("first_qi")
    h = h.merge(first, on=["drug", "event"], how="left")
    lost = h[h["first_qi"].notna() & (h["qi"] > h["first_qi"]) & ~h["sig"]].groupby(["drug", "event"]).size().rename("lost")

    t = single.copy()
    tot_a = a.groupby(["drug", "event"])["a"].sum().rename("a_end")
    t = t.merge(tot_a, on=["drug", "event"], how="left").fillna({"a_end": 0})
    t["n_drug"] = t["drug"].map(D.iloc[:, -1]).fillna(0) if len(quarters) else 0
    t["n_event"] = t["event"].map(E.iloc[:, -1]).fillna(0) if len(quarters) else 0
    ror, low = _ror_lower(t["a_end"], t["n_drug"], t["n_event"], Ncum.iloc[-1] if len(quarters) else 0)
    t["ror"], t["ror_lower"] = ror, low
    t["signal"] = (t["a_end"] >= 5) & (t["ror_lower"] > 1)
    t["paper1_rule"] = (t["a_end"] >= 3) & (t["ror"] >= 2)
    t = t.merge(first, on=["drug", "event"], how="left").merge(lost, on=["drug", "event"], how="left")
    t["lost"] = t["lost"].fillna(0)
    t["stable"] = t["signal"] & t["first_qi"].notna() & (t["lost"] == 0)
    t["first_signal"] = t["first_qi"].map(lambda q: quarter_label(int(q)) if pd.notna(q) else None)
    return t.drop(columns=["first_qi"])
