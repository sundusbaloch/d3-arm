"""Ground-truth evaluation against the time-indexed reference standard of
Harpaz et al. (2014, Scientific Data 1:140043, figshare doi 10.6084/m9.figshare.1133904).

The reference standard lists 137 drug-event pairs (62 positive, 75 negative).
An "event" is a group of MedDRA preferred terms (narrow, plus broad terms), so a
report counts as having the event if ANY of its reactions is in the group, and a
report counts as exposed if ANY of its drugs is (or contains) the reference
ingredient. Counts are per report, never per reaction.

For every pair we build the 2x2 table quarter by quarter (cumulative), so the
same counts give the full-data signal, the time to signal and the stability flag
(signal at the end and never lost after it first fired, as in scripts/40).
"""
import math

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from d3arm.novelty.windows import BASE, quarter_index, quarter_label

# names in FAERS that differ from the reference name (ingredient level)
SYNONYMS = {"URSODEOXYCHOLATE": ["URSODIOL", "URSODEOXYCHOLIC ACID"]}


def _key(s):
    return " ".join(str(s).strip().upper().split())


def load_reference(path):
    """-> pairs (drug, event, truth), event definitions (event, pt, level), drug names (drug, name)"""
    x = pd.read_excel(path, sheet_name=None)
    ref = x["Reference Standard"].rename(columns=str.strip)
    pairs = pd.DataFrame({"drug": ref["DRUG_CONCEPT_NAME"].map(_key),
                          "event": ref["EVENT_CONCEPT_NAME"].map(_key),
                          "truth": ref["GROUND_TRUTH"].astype(int)})
    ev = x["Event Definitions"].rename(columns=str.strip)
    ev["EVENT_CONCEPT_NAME"] = ev["EVENT_CONCEPT_NAME"].ffill()
    events = pd.DataFrame({"event": ev["EVENT_CONCEPT_NAME"].map(_key),
                           "pt": ev["MEDDRA_PT"].map(_key),
                           "level": ev["DEFINITION_LEVEL"].astype(str).str.strip().str.lower()}).drop_duplicates()
    missing = set(pairs["event"]) - set(events["event"])
    if missing:
        raise ValueError(f"events without a definition: {sorted(missing)}")
    names = [(d, n) for d in sorted(pairs["drug"].unique()) for n in [d] + SYNONYMS.get(d, [])]
    drugs = pd.DataFrame(names, columns=["drug", "name"])
    return pairs, events, drugs


def _component_matches(tx, drugs_pdf, spark):
    """(primaryid, qi, drug): report exposed to a reference drug. A FAERS drug item
    matches if one of its ' + ' components equals the name or starts with 'name '
    (salt or ester words left after normalisation, e.g. LISDEXAMFETAMINE DIMESYLATE)."""
    names = spark.createDataFrame(drugs_pdf.assign(first=drugs_pdf["name"].str.split(" ").str[0]))
    comp = (tx.select("primaryid", quarter_index(F.col("window")).alias("qi"),
                      F.explode(F.filter("items", lambda i: i.startswith("DRUG="))).alias("item"))
              .select("primaryid", "qi", F.explode(F.split(F.substring("item", 6, 10_000), r" \+ ")).alias("comp"))
              .withColumn("comp", F.trim("comp"))
              .withColumn("first", F.split("comp", " ")[0]))
    m = comp.join(F.broadcast(names), "first")
    hit = (F.col("comp") == F.col("name")) | F.col("comp").startswith(F.concat(F.col("name"), F.lit(" ")))
    return m.where(hit).select("primaryid", "qi", "drug").distinct()


def _event_matches(tx, events_pdf, spark, levels):
    ev = spark.createDataFrame(events_pdf[events_pdf["level"].isin(levels)][["event", "pt"]].drop_duplicates())
    ae = tx.select("primaryid", quarter_index(F.col("window")).alias("qi"),
                   F.explode(F.filter("items", lambda i: i.startswith("AE="))).alias("item")) \
           .select("primaryid", "qi", F.substring("item", 4, 10_000).alias("pt"))
    return ae.join(F.broadcast(ev), "pt").select("primaryid", "qi", "event").distinct()


def quarterly_counts(spark, tx, pairs, events, drugs, levels):
    """New reports per quarter: N, n_drug, n_event and a for every reference pair (pandas)."""
    tx = tx.select("primaryid", "window", "items").cache()
    n_q = tx.groupBy(quarter_index(F.col("window")).alias("qi")).count().toPandas().rename(columns={"count": "N"})
    dm = _component_matches(tx, drugs, spark).cache()
    em = _event_matches(tx, events, spark, levels).cache()
    nd = dm.groupBy("drug", "qi").count().toPandas().rename(columns={"count": "n_drug"})
    ne = em.groupBy("event", "qi").count().toPandas().rename(columns={"count": "n_event"})
    p = spark.createDataFrame(pairs[["drug", "event"]])
    a = (dm.join(em.select("primaryid", "event"), "primaryid")
           .join(F.broadcast(p), ["drug", "event"], "left_semi")
           .groupBy("drug", "event", "qi").count().toPandas().rename(columns={"count": "a"}))
    names_seen = dm.groupBy("drug").agg(F.countDistinct("primaryid").alias("reports")).toPandas()
    for d in (dm, em, tx):
        d.unpersist()
    return n_q, nd, ne, a, names_seen


def ror_ci(a, n_d, n_e, n, z=1.96):
    b, c, d = n_d - a, n_e - a, n - n_d - n_e + a
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return float("nan"), float("nan")
    ror = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return ror, math.exp(math.log(ror) - z * se)


def pair_table(pairs, n_q, nd, ne, a):
    """One row per reference pair: final counts, ROR, lower CI, first signal, stability."""
    quarters = sorted(n_q["qi"])
    N = n_q.set_index("qi")["N"].reindex(quarters, fill_value=0).cumsum()
    rows = []
    for r in pairs.itertuples(index=False):
        cd = nd[nd["drug"] == r.drug].set_index("qi")["n_drug"].reindex(quarters, fill_value=0).cumsum()
        ce = ne[ne["event"] == r.event].set_index("qi")["n_event"].reindex(quarters, fill_value=0).cumsum()
        na = a[(a["drug"] == r.drug) & (a["event"] == r.event)].set_index("qi")["a"].reindex(quarters, fill_value=0)
        ca = na.cumsum()
        history = []                      # (qi, signal) at quarters where new reports of the pair arrived
        for q in quarters:
            if na[q] > 0:
                ror, low = ror_ci(ca[q], cd[q], ce[q], N[q])
                history.append((q, bool(ca[q] >= 5 and low > 1)))
        ror, low = ror_ci(ca.iloc[-1], cd.iloc[-1], ce.iloc[-1], N.iloc[-1]) if quarters else (np.nan, np.nan)
        a_end = int(ca.iloc[-1]) if quarters else 0
        sig = bool(a_end >= 5 and low > 1)
        first = next((q for q, s in history if s), None)
        lost = sum(1 for q, s in history if first is not None and q > first and not s)
        rows.append({"drug": r.drug, "event": r.event, "truth": r.truth, "a": a_end,
                     "n_drug": int(cd.iloc[-1]) if quarters else 0, "n_event": int(ce.iloc[-1]) if quarters else 0,
                     "ror": ror, "ror_lower": low, "signal": sig,
                     "paper1_rule": bool(a_end >= 3 and ror >= 2),
                     "first_signal": quarter_label(first) if first is not None else None,
                     "stable": bool(sig and first is not None and lost == 0)})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(mid - half, 3), round(mid + half, 3)]


def metrics(truth, flag):
    truth = np.asarray(truth, bool)
    flag = np.asarray(flag, bool)
    tp, fn = int((flag & truth).sum()), int((~flag & truth).sum())
    fp, tn = int((flag & ~truth).sum()), int((~flag & ~truth).sum())
    out = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    for name, k, n in [("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp),
                       ("ppv", tp, tp + fp), ("npv", tn, tn + fn)]:
        out[name] = round(k / n, 3) if n else None
        out[name + "_ci"] = wilson(k, n)
    return out


def auc(truth, score):
    """Mann-Whitney AUC; missing scores count as the lowest score."""
    truth = np.asarray(truth, bool)
    s = pd.Series(score, dtype=float).fillna(-np.inf).to_numpy()
    pos, neg = s[truth], s[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return round(float((gt + 0.5 * eq) / (len(pos) * len(neg))), 3)


def mcnemar_exact(flag_a, flag_b, truth):
    """Discordant correct calls between two methods (b = only A right, c = only B right), exact two-sided p."""
    truth = np.asarray(truth, bool)
    ra = np.asarray(flag_a, bool) == truth
    rb = np.asarray(flag_b, bool) == truth
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    n = b + c
    if n == 0:
        return {"only_first_right": b, "only_second_right": c, "p": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"only_first_right": b, "only_second_right": c, "p": round(min(1.0, 2 * p), 4)}
