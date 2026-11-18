"""Drug x AE counts at very low support (Paper 2 uses just 5 reports), and
the disproportionality measures computed from them.

Why not SON here: SON's pass 1 prunes only if the scaled threshold p * s is
at least ~2 reports. With s = 5 and 16 partitions, p * s = 0.3, so every pair
that occurs anywhere becomes a candidate and pass 1 saves nothing. Instead we
use the two passes of A-priori directly, written as Spark joins:
  pass 1: count every single item; keep drugs and AEs with >= s reports
          (monotonicity: a pair can't reach s if one of its items doesn't)
  pass 2: count only (frequent drug, frequent AE) pairs, keep those >= s.
The constraint (one drug, one AE) means we never build drug-drug or AE-AE pairs.

From the counts of {drug}, {AE}, {drug, AE} and N we get the full 2x2 table:
    a = n(drug & AE)        b = n(drug) - a
    c = n(AE) - a           d = N - n(drug) - n(AE) + a
"""
from pyspark.sql import functions as F


def split_items(tx):
    return tx.select(F.filter("items", lambda i: i.startswith("DRUG=")).alias("drugs"),
                     F.filter("items", lambda i: i.startswith("AE=")).alias("aes"))


def drug_ae_counts(tx, min_count=5):
    t = split_items(tx).cache()
    n = t.count()
    # pass 1: single items
    drugs = (t.select(F.explode("drugs").alias("drug")).groupBy("drug").count()
              .where(F.col("count") >= min_count).withColumnRenamed("count", "n_drug"))
    aes = (t.select(F.explode("aes").alias("ae")).groupBy("ae").count()
            .where(F.col("count") >= min_count).withColumnRenamed("count", "n_ae"))
    drugs, aes = drugs.cache(), aes.cache()
    # pass 2: only frequent drug x frequent AE pairs
    pairs = (t.select(F.explode("drugs").alias("drug"), "aes")
              .join(F.broadcast(drugs.select("drug")), "drug")
              .select("drug", F.explode("aes").alias("ae"))
              .join(F.broadcast(aes.select("ae")), "ae")
              .groupBy("drug", "ae").count()
              .where(F.col("count") >= min_count).withColumnRenamed("count", "a"))
    out = (pairs.join(drugs, "drug").join(aes, "ae")
                .withColumn("N", F.lit(n).cast("long"))
                .select("drug", "ae", "a", "n_drug", "n_ae", "N"))
    return out, {"transactions": n, "frequent_drugs": drugs.count(), "frequent_aes": aes.count()}


def add_measures(counts, z=1.96):
    """ROR and PRR with 95% CIs; signal = (a >= 5) and lower ROR CI > 1 (Paper 2's rule)."""
    a, n_d, n_e, N = (F.col(c).cast("double") for c in ("a", "n_drug", "n_ae", "N"))
    b, c, d = n_d - a, n_e - a, N - n_d - n_e + a
    # a zero cell (e.g. an AE reported only with this drug) makes ROR undefined.
    # Every division sits inside when(ok, ...): Spark 4 runs in ANSI mode and
    # raises DIVIDE_BY_ZERO instead of quietly returning null.
    ok = (b > 0) & (c > 0) & (d > 0)
    ror = F.when(ok, (a * d) / (b * c))
    se_ror = F.when(ok, F.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    prr = F.when(ok, (a / (a + b)) / (c / (c + d)))
    se_prr = F.when(ok, F.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)))
    return (counts
            .withColumn("ror", ror)
            .withColumn("ror_lower", F.exp(F.log(ror) - z * se_ror))
            .withColumn("ror_upper", F.exp(F.log(ror) + z * se_ror))
            .withColumn("prr", prr)
            .withColumn("prr_lower", F.exp(F.log(prr) - z * se_prr))
            .withColumn("prr_upper", F.exp(F.log(prr) + z * se_prr))
            .withColumn("signal", F.coalesce((F.col("a") >= 5) & (F.col("ror_lower") > 1), F.lit(False))))
