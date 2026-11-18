"""Time windows: replay FAERS quarter by quarter and watch each drug -> AE
signal appear (and sometimes disappear).

Every report has a window: the quarter the FDA first received the case
(init_fda_dt). Cases first received before 2014Q3 (old cases that were
updated later) are merged into one "pre-2014Q3" window, the starting state.

For every drug-AE pair that has >= min_count reports overall, we compute the
2x2 table on the CUMULATIVE data up to each quarter q:
    a(q) = reports with drug and AE received up to q
    n_drug(q), n_ae(q), N(q) likewise
and the signal rule a >= 5 and lower ROR CI > 1 at that point in time.
A pair is evaluated at the quarters in which new reports of that pair arrive.

Per pair we then get:
    first_signal      first quarter at which the rule fires (time to signal)
    quarters_eval     quarters at which it was evaluated
    lost_after_first  how many of the later evaluations no longer pass the rule
    stable            signal at the end AND never lost after first firing
"""
from pyspark.sql import Window
from pyspark.sql import functions as F

FIRST_YEAR, FIRST_Q = 2014, 3
BASE = FIRST_YEAR * 4 + FIRST_Q - 2          # index of the merged pre-2014Q3 window


def quarter_index(col):
    """'2019Q2' -> year*4 + q - 1, with everything before 2014Q3 merged into BASE"""
    qi = F.substring(col, 1, 4).cast("int") * 4 + F.substring(col, 6, 1).cast("int") - 1
    return F.greatest(qi, F.lit(BASE))


def quarter_label(qi):
    return "pre-2014Q3" if qi == BASE else f"{qi // 4}Q{qi % 4 + 1}"


def _cumulative_dense(counts, key, keys_df, quarters):
    """per (key, quarter) running total, with a row for every quarter (also quarters with 0 new reports)"""
    grid = keys_df.crossJoin(F.broadcast(quarters))
    w = Window.partitionBy(key).orderBy("qi").rowsBetween(Window.unboundedPreceding, 0)
    return (grid.join(counts, [key, "qi"], "left").fillna(0, ["new"])
                .withColumn("cum", F.sum("new").over(w)).drop("new"))


def replay(tx, pairs, z=1.96):
    """
    tx    : transactions (items, window)
    pairs : DataFrame(drug, ae) of the pairs to follow (e.g. a >= 5 overall)
    Returns a DataFrame with one row per (drug, ae, qi) at which the pair was evaluated.
    """
    t = tx.select(quarter_index(F.col("window")).alias("qi"),
                  F.filter("items", lambda i: i.startswith("DRUG=")).alias("drugs"),
                  F.filter("items", lambda i: i.startswith("AE=")).alias("aes")).cache()

    quarters = t.select("qi").distinct()
    n_q = t.groupBy("qi").count()
    w_all = Window.orderBy("qi").rowsBetween(Window.unboundedPreceding, 0)
    n_cum = n_q.withColumn("N", F.sum("count").over(w_all)).select("qi", "N")

    drug_keys = pairs.select("drug").distinct()
    ae_keys = pairs.select("ae").distinct()
    drug_new = (t.select("qi", F.explode("drugs").alias("drug")).groupBy("drug", "qi").count()
                 .withColumnRenamed("count", "new"))
    ae_new = (t.select("qi", F.explode("aes").alias("ae")).groupBy("ae", "qi").count()
               .withColumnRenamed("count", "new"))
    drug_cum = _cumulative_dense(drug_new.join(drug_keys, "drug", "left_semi"), "drug", drug_keys, quarters) \
        .withColumnRenamed("cum", "n_drug")
    ae_cum = _cumulative_dense(ae_new.join(ae_keys, "ae", "left_semi"), "ae", ae_keys, quarters) \
        .withColumnRenamed("cum", "n_ae")

    pair_new = (t.select("qi", F.explode("drugs").alias("drug"), "aes")
                 .select("qi", "drug", F.explode("aes").alias("ae"))
                 .join(pairs, ["drug", "ae"], "left_semi")
                 .groupBy("drug", "ae", "qi").count())
    w_pair = Window.partitionBy("drug", "ae").orderBy("qi").rowsBetween(Window.unboundedPreceding, 0)
    pair_cum = pair_new.withColumn("a", F.sum("count").over(w_pair)).drop("count")

    a, n_d, n_e, N = (F.col(c).cast("double") for c in ("a", "n_drug", "n_ae", "N"))
    b, c, d = n_d - a, n_e - a, N - n_d - n_e + a
    ok = (b > 0) & (c > 0) & (d > 0)
    ror = F.when(ok, (a * d) / (b * c))
    se = F.when(ok, F.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    return (pair_cum.join(drug_cum, ["drug", "qi"]).join(ae_cum, ["ae", "qi"]).join(F.broadcast(n_cum), "qi")
            .withColumn("ror", ror)
            .withColumn("ror_lower", F.exp(F.log(ror) - z * se))
            .withColumn("signal", F.coalesce((F.col("a") >= 5) & (F.col("ror_lower") > 1), F.lit(False))))


def summarise_pairs(ev):
    """one row per pair: time to signal and stability"""
    first = ev.where("signal").groupBy("drug", "ae").agg(F.min("qi").alias("first_signal"))
    w_last = Window.partitionBy("drug", "ae").orderBy(F.desc("qi"))
    last = (ev.withColumn("r", F.row_number().over(w_last)).where("r = 1")
              .select("drug", "ae", F.col("signal").alias("signal_at_end"), F.col("a").alias("a_end"),
                      F.col("ror").alias("ror_end"), F.col("ror_lower").alias("ror_lower_end")))
    counts = ev.groupBy("drug", "ae").agg(F.count("*").alias("quarters_eval"),
                                          F.sum(F.col("signal").cast("int")).alias("quarters_signal"))
    s = (last.join(counts, ["drug", "ae"]).join(first, ["drug", "ae"], "left"))
    lost = (ev.join(first, ["drug", "ae"]).where((F.col("qi") > F.col("first_signal")) & ~F.col("signal"))
              .groupBy("drug", "ae").count().withColumnRenamed("count", "lost_after_first"))
    return (s.join(lost, ["drug", "ae"], "left").fillna(0, ["lost_after_first"])
             .withColumn("stable", F.col("signal_at_end") & (F.col("lost_after_first") == 0)))
