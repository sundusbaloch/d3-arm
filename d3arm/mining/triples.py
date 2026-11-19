"""Drug-drug interaction (DDI) triples on all of FAERS: (drug 1, drug 2, AE).

Counting, with the A-priori passes written as Spark joins:
  pass 1: frequent drugs and AEs (>= min_count reports)
  pass 2: frequent drug pairs {d1, d2} (only pairs of frequent drugs are built)
  pass 3: (d1, d2, AE) triples, built only from frequent drug pairs,
          kept if >= min_count reports.
Monotonicity guarantees nothing is lost: a triple with >= min_count reports
has every sub-pair and item at >= min_count too. The drug-AE pairs
{d1, AE} and {d2, AE} are then also frequent, so their counts can be joined in.

Measures for each triple. Reports fall into four groups:
    both drugs (11), d1 without d2 (10), d2 without d1 (01), neither (00)
    a11 = n(d1, d2, AE)                   n11 = n(d1, d2)
    a10 = n(d1, AE) - a11                 n10 = n(d1) - n11
    a01 = n(d2, AE) - a11                 n01 = n(d2) - n11
  1. combination signal (the combination treated as one "drug", like Paper 1's
     DDI scores): 2x2 table of both-drugs vs all other reports,
         a = a11, b = n11 - a11, c = n(AE) - a11, d = N - n11 - n(AE) + a11,
     signal = a11 >= 5 and lower 95% CI of the ROR > 1 (Paper 2's rule).
  2. combination risk ratio (CRR): is the AE reported more often with BOTH drugs
     than with either drug alone?
         CRR = r11 / max(r10, r01),   r = a / n (reporting rate of the AE)
     If all three groups are compared with the same reference group, this equals
     PRR11 / max(PRR10, PRR01). 95% CI on the log scale (ratio of two proportions):
         SE = sqrt(1/a11 - 1/n11 + 1/a_m - 1/n_m),  m = the single-drug group with the higher rate
  3. interaction signal = combination signal AND lower CI of CRR > 1.
A drug pair where one drug is (almost) never reported alone has no usable
single-drug rate: CRR is left empty (null) for it, never divided by zero.
"""
from pyspark.sql import functions as F

from d3arm.mining.pairs import split_items


def _drug_pairs(t):
    """(d1, d2, aes) rows, d1 < d2, for every pair of (frequent) drugs in a report.
    `drugs` must be sorted and distinct; slice(drugs, i + 2, ...) = the drugs after position i."""
    return (t.select(F.posexplode("drugs").alias("i", "d1"), "drugs", "aes")
             .select("d1", F.explode(F.expr("slice(drugs, i + 2, size(drugs))")).alias("d2"), "aes"))


def workload(tx):
    """how many pair and triple rows pass 2 and 3 will generate at most (before pruning by pairs)"""
    t = split_items(tx).select(F.size("drugs").alias("nd"), F.size("aes").alias("na"))
    pairs = (F.col("nd") * (F.col("nd") - 1) / 2).cast("long")
    r = t.agg(F.sum(pairs).alias("pair_rows"), F.sum(pairs * F.col("na")).alias("triple_rows_max"),
              F.max("nd").alias("max_drugs"), F.sum((F.col("nd") > 50).cast("int")).alias("reports_over_50_drugs")).first()
    return r.asDict()


def triple_counts(tx, min_count=5, max_drugs=None):
    """
    tx        : transactions (items with DRUG= / AE= prefixes)
    max_drugs : optional cap - reports with more drugs are left out of the WHOLE analysis
                (N, singles, pairs and triples all use the same reports, so the 2x2 tables stay consistent)
    Returns (triples DataFrame with all counts, stats dict).
    """
    t = tx.select("primaryid",
                  F.array_sort(F.array_distinct(F.filter("items", lambda i: i.startswith("DRUG=")))).alias("drugs"),
                  F.array_distinct(F.filter("items", lambda i: i.startswith("AE="))).alias("aes"))
    if max_drugs:
        t = t.where(F.size("drugs") <= max_drugs)
    t = t.cache()
    n = t.count()

    # pass 1
    drugs = (t.select(F.explode("drugs").alias("drug")).groupBy("drug").count()
              .where(F.col("count") >= min_count).withColumnRenamed("count", "n_drug")).cache()
    aes = (t.select(F.explode("aes").alias("ae")).groupBy("ae").count()
            .where(F.col("count") >= min_count).withColumnRenamed("count", "n_ae")).cache()
    keep_d = F.broadcast(drugs.select("drug"))
    # each report reduced to its frequent drugs (sorted); only reports with >= 2 of them can hold a pair.
    # AEs are not filtered here: an infrequent AE can never reach min_count in a triple, it just drops out.
    tf = (t.select("primaryid", F.explode("drugs").alias("drug"), "aes")
           .join(keep_d, "drug")
           .groupBy("primaryid").agg(F.array_sort(F.collect_list("drug")).alias("drugs"),
                                     F.first("aes").alias("aes"))
           .where(F.size("drugs") >= 2)
           .cache())

    # pass 2: frequent drug pairs
    pr = _drug_pairs(tf)
    pairs = (pr.groupBy("d1", "d2").count().where(F.col("count") >= min_count)
               .withColumnRenamed("count", "n11")).cache()

    # pass 3: triples, only from frequent pairs
    tri = (pr.join(pairs.select("d1", "d2"), ["d1", "d2"], "left_semi")
             .select("d1", "d2", F.explode("aes").alias("ae"))
             .groupBy("d1", "d2", "ae").count().where(F.col("count") >= min_count)
             .withColumnRenamed("count", "a11"))

    # drug-AE pair counts for the drugs and AEs that occur in a triple
    da = (t.select(F.explode("drugs").alias("drug"), "aes")
           .join(keep_d, "drug")
           .select("drug", F.explode("aes").alias("ae"))
           .groupBy("drug", "ae").count().where(F.col("count") >= min_count)   # n(d, AE) >= a11 >= min_count
           .withColumnRenamed("count", "a_drug_ae"))
    da1 = da.select(F.col("drug").alias("d1"), "ae", F.col("a_drug_ae").alias("n_d1_ae"))
    da2 = da.select(F.col("drug").alias("d2"), "ae", F.col("a_drug_ae").alias("n_d2_ae"))
    d1 = drugs.select(F.col("drug").alias("d1"), F.col("n_drug").alias("n_d1"))
    d2 = drugs.select(F.col("drug").alias("d2"), F.col("n_drug").alias("n_d2"))
    out = (tri.join(pairs, ["d1", "d2"])
              .join(da1, ["d1", "ae"]).join(da2, ["d2", "ae"])
              .join(F.broadcast(d1), "d1").join(F.broadcast(d2), "d2")
              .join(F.broadcast(aes), "ae")
              .withColumn("N", F.lit(n).cast("long"))
              .select("d1", "d2", "ae", "a11", "n11", "n_d1_ae", "n_d2_ae", "n_d1", "n_d2", "n_ae", "N"))
    stats = {"transactions": n, "frequent_drugs": drugs.count(), "frequent_aes": aes.count(),
             "reports_with_2_frequent_drugs": tf.count(), "frequent_drug_pairs": pairs.count()}
    return out, stats


def add_interaction_measures(c, z=1.96):
    a11, n11, nd1e, nd2e, nd1, nd2, ne, N = (F.col(x).cast("double") for x in
                                              ("a11", "n11", "n_d1_ae", "n_d2_ae", "n_d1", "n_d2", "n_ae", "N"))
    # 1. combination vs all other reports
    a, b, c_, d = a11, n11 - a11, ne - a11, N - n11 - ne + a11
    ok = (b > 0) & (c_ > 0) & (d > 0)
    ror = F.when(ok, (a * d) / (b * c_))
    se = F.when(ok, F.sqrt(1 / a + 1 / b + 1 / c_ + 1 / d))
    # 2. CRR: both drugs vs the single drug with the higher reporting rate
    a10, n10, a01, n01 = nd1e - a11, nd1 - n11, nd2e - a11, nd2 - n11
    r10 = F.when(n10 > 0, a10 / n10)
    r01 = F.when(n01 > 0, a01 / n01)
    use10 = r10.isNotNull() & (r01.isNull() | (r10 >= r01))
    a_m = F.when(use10, a10).otherwise(a01)
    n_m = F.when(use10, n10).otherwise(n01)
    ok2 = (a_m > 0) & (n_m > 0)
    crr = F.when(ok2, (a11 / n11) / (a_m / n_m))
    se2 = F.when(ok2, F.sqrt(1 / a11 - 1 / n11 + 1 / a_m - 1 / n_m))
    return (c.withColumn("ror_comb", ror)
             .withColumn("ror_comb_lower", F.exp(F.log(ror) - z * se))
             .withColumn("comb_signal", F.coalesce((F.col("a11") >= 5) & (F.col("ror_comb_lower") > 1), F.lit(False)))
             .withColumn("rate_both", a11 / n11)
             .withColumn("rate_d1_only", r10)
             .withColumn("rate_d2_only", r01)
             .withColumn("crr", crr)
             .withColumn("crr_lower", F.exp(F.log(crr) - z * se2))
             .withColumn("crr_upper", F.exp(F.log(crr) + z * se2))
             .withColumn("interaction_signal",
                         F.col("comb_signal") & F.coalesce(F.col("crr_lower") > 1, F.lit(False))))
