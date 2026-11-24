"""Near-duplicate reports with MinHash + LSH banding (b bands of r rows).

Exact deduplication (Paper 2's rule) only catches reports whose fields match
exactly. The same patient event reported twice (by a doctor and by the
company, say) often differs a little: one extra drug, an age off by a year.
We look for pairs of reports whose token sets have Jaccard similarity >= t.

Tokens of a report: DRUG=, AE=, IND= (indication), SEX=, AGE= (whole years),
CTRY=, EVT= (event date). Only reports with both an age and an event date are
used: without them, two different patients with the same drug and reaction
look identical, and we would call them duplicates for no good reason.

Steps
 1. reports with exactly the same token set are grouped first (an exact
    content duplicate is trivially a near-duplicate, and it would otherwise
    put thousands of identical sets in one LSH bucket);
 2. MinHash: n = b*r hash functions h_i(x) = (a_i x + c_i) mod p, signature
    entry i = min over the set's tokens of h_i. P[sig_i(A) = sig_i(B)] = J(A,B);
 3. banding: two sets become candidates if all r entries of at least one band
    agree. P(candidate) = 1 - (1 - J^r)^b, an S-curve with threshold about
    (1/b)^(1/r);
 4. every candidate pair is checked with the exact Jaccard (false positives go).
Buckets larger than `max_bucket` are skipped (and counted): a bucket of
size s would produce s(s-1)/2 pairs.
"""
import itertools

import numpy as np
from pyspark.sql import functions as F

PRIME = (1 << 31) - 1          # a < 2^31 and x < 2^31, so a*x + c fits in a signed 64-bit long


def candidate_probability(j, b, r):
    return 1 - (1 - j ** r) ** b


def report_tokens(reports, with_ids=True):
    """with_ids=True: only reports that have an age and an event date (used for LSH).
    with_ids=False: only reports MISSING one of them (used to flag mass reporting)."""
    def tag(col, prefix):
        return F.transform(F.coalesce(col, F.array().cast("array<string>")), lambda x: F.concat(F.lit(prefix), x))
    extra = F.array(
        F.concat(F.lit("SEX="), F.col("sex")),
        F.concat(F.lit("AGE="), F.round("age_years").cast("int").cast("string")),
        F.concat(F.lit("CTRY="), F.col("country")),
        F.concat(F.lit("EVT="), F.col("event_dt")))
    toks = F.concat(tag(F.col("drugs"), "DRUG="), tag(F.col("reactions"), "AE="),
                    tag(F.col("indications"), "IND="), F.filter(extra, lambda x: x.isNotNull()))
    has_ids = F.col("age_years").isNotNull() & F.col("event_dt").isNotNull()
    return (reports.where(has_ids if with_ids else ~has_ids)
                   .select("primaryid", F.array_sort(F.array_distinct(toks)).alias("tokens")))


def hash_params(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(1, PRIME, n, dtype=np.int64), rng.integers(0, PRIME, n, dtype=np.int64)


def band_rows(b, r, seed=42):
    """returns a function for mapPartitions: (set_id, tokens) -> (band, band_hash, set_id) rows"""
    a, c = hash_params(b * r, seed)

    def fn(rows):
        import hashlib
        import zlib
        for set_id, tokens in rows:
            x = np.fromiter((zlib.crc32(t.encode()) & 0x7FFFFFFF for t in tokens), dtype=np.int64, count=len(tokens))
            sig = ((a[:, None] * x[None, :] + c[:, None]) % PRIME).min(axis=1)       # n MinHash values
            for band in range(b):
                # Python's hash() is randomised per process, so use a fixed hash:
                # the same band must land in the same bucket on every worker
                key = hashlib.blake2b(sig[band * r:(band + 1) * r].tobytes(), digest_size=8).digest()
                yield band, int.from_bytes(key, "little", signed=True), set_id
    return fn


def find_near_duplicates(spark, reports, b=25, r=8, threshold=0.8, max_bucket=500, partitions=64):
    toks = report_tokens(reports).cache()
    n = toks.count()
    verified, groups, stats = near_duplicates(spark, toks, b, r, threshold, max_bucket, partitions)
    return verified, groups, toks, {"reports_with_age_and_event_date": n, **stats}


def near_duplicates(spark, toks, b=25, r=8, threshold=0.8, max_bucket=500, partitions=64):
    """the LSH steps 1-4 on any table of (primaryid, tokens); returns (verified pairs, groups, stats)"""
    stats = {}

    # 1. identical token sets
    groups = (toks.groupBy("tokens").agg(F.min("primaryid").alias("set_id"), F.count("*").alias("n_reports"))
                  .cache())
    stats["distinct_token_sets"] = groups.count()
    stats["reports_in_identical_groups"] = groups.where("n_reports > 1").agg(F.sum("n_reports")).first()[0] or 0
    stats["identical_groups"] = groups.where("n_reports > 1").count()

    # 2-3. MinHash signatures and banding
    sets = groups.select("set_id", "tokens")
    bands = spark.createDataFrame(
        sets.rdd.map(tuple).repartition(partitions).mapPartitions(band_rows(b, r)),
        "band int, bhash long, set_id long")
    buckets = bands.groupBy("band", "bhash").agg(F.collect_list("set_id").alias("ids")) \
                   .withColumn("size", F.size("ids")).where("size > 1").cache()
    stats["buckets_with_2_or_more"] = buckets.count()
    big = buckets.where(F.col("size") > max_bucket)
    stats["buckets_skipped_too_big"] = big.count()
    stats["largest_bucket"] = buckets.agg(F.max("size")).first()[0] or 0

    def pairs_of(ids):
        return [(min(p), max(p)) for p in itertools.combinations(sorted(set(ids)), 2)]
    cand = (buckets.where(F.col("size") <= max_bucket).select("ids").rdd
                   .flatMap(lambda row: pairs_of(row[0])).distinct()
                   .toDF("id1 long, id2 long").cache())
    stats["candidate_pairs"] = cand.count()

    # 4. exact Jaccard on candidates only
    t1 = sets.withColumnRenamed("set_id", "id1").withColumnRenamed("tokens", "t1")
    t2 = sets.withColumnRenamed("set_id", "id2").withColumnRenamed("tokens", "t2")
    verified = (cand.join(t1, "id1").join(t2, "id2")
                    .withColumn("jaccard", F.size(F.array_intersect("t1", "t2")) / F.size(F.array_union("t1", "t2")))
                    .where(F.col("jaccard") >= threshold).select("id1", "id2", "jaccard").cache())
    stats["near_duplicate_pairs"] = verified.count()
    stats["false_positive_candidates"] = stats["candidate_pairs"] - stats["near_duplicate_pairs"]
    stats["sets_in_a_near_duplicate_pair"] = (verified.select(F.col("id1").alias("i"))
                                              .union(verified.select(F.col("id2").alias("i"))).distinct().count())
    return verified, groups, stats


def same_case_filter(spark, reports, verified, age_tolerance=1):
    """Keep only near-duplicate pairs that can be the same case: the identifying fields
    must agree, not just the overall token sets.
      sex and country equal; age within age_tolerance years (age_years is rounded, and
      two reporters may compute it differently); event dates equal or one a prefix of
      the other (FAERS dates can be partial: 2019, 201904, 20190415).
    Returns (pairs that pass, counts of pairs failing each check)."""
    toks = report_tokens(reports)
    sets = toks.groupBy("tokens").agg(F.min("primaryid").alias("set_id"))
    field = lambda prefix: F.get(F.filter("tokens", lambda x: x.startswith(prefix)), 0).substr(len(prefix) + 1, 100)   # get: NULL if absent, also in ANSI mode
    f = sets.select("set_id", field("EVT=").alias("evt"), field("AGE=").cast("int").alias("age"),
                    field("SEX=").alias("sex"), field("CTRY=").alias("ctry"))
    side = lambda i: f.select(F.col("set_id").alias(f"id{i}"), *[F.col(c).alias(f"{c}{i}") for c in ("evt", "age", "sex", "ctry")])
    j = verified.join(side(1), "id1").join(side(2), "id2")
    ok_sex = F.col("sex1").eqNullSafe(F.col("sex2"))
    ok_ctry = F.col("ctry1").eqNullSafe(F.col("ctry2"))
    ok_age = F.coalesce(F.abs(F.col("age1") - F.col("age2")) <= age_tolerance, F.lit(False))
    ok_evt = F.coalesce((F.col("evt1") == F.col("evt2")) | F.col("evt1").startswith(F.col("evt2"))
                        | F.col("evt2").startswith(F.col("evt1")), F.lit(False))
    j = j.withColumn("ok_sex", ok_sex).withColumn("ok_ctry", ok_ctry).withColumn("ok_age", ok_age) \
         .withColumn("ok_evt", ok_evt).cache()
    r = j.agg(F.count("*").alias("pairs"),
              *[F.sum((~F.col(c)).cast("int")).alias("fail_" + c[3:]) for c in ("ok_sex", "ok_ctry", "ok_age", "ok_evt")],
              F.sum((F.col("ok_sex") & F.col("ok_ctry") & F.col("ok_age") & F.col("ok_evt")).cast("int")).alias("same_case")
              ).first().asDict()
    keep = j.where(F.col("ok_sex") & F.col("ok_ctry") & F.col("ok_age") & F.col("ok_evt")).select("id1", "id2", "jaccard")
    return keep, {k: int(v or 0) for k, v in r.items()}


def brute_force_pairs(token_sets, threshold):
    """all pairs with Jaccard >= threshold, by brute force (for a few thousand sets).
    Sorted by size: a pair can only reach the threshold if |small| >= threshold * |big|."""
    items = sorted(((len(t), sid, frozenset(t)) for sid, t in token_sets.items()), key=lambda x: x[0])
    out = set()
    for i, (n1, s1, a) in enumerate(items):
        for n2, s2, bset in items[i + 1:]:
            if n1 < threshold * n2:
                break
            inter = len(a & bset)
            if inter / (n1 + n2 - inter) >= threshold:
                out.add((min(s1, s2), max(s1, s2)))
    return out


def components(pairs):
    """union-find over (id1, id2) edges -> {id: smallest id in its connected component}"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]          # path halving
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in parent}


def reports_to_drop(spark, reports, verified):
    """One report is kept per group of near-duplicates (identical token sets plus
    LSH pairs, joined transitively); returns a DataFrame of primaryids to drop."""
    return ids_to_drop(spark, report_tokens(reports), verified)


def ids_to_drop(spark, toks, verified):
    """same as reports_to_drop, for any table of (primaryid, tokens)"""
    set_of = toks.groupBy("tokens").agg(F.min("primaryid").alias("set_id"), F.collect_list("primaryid").alias("ids")) \
                 .select("set_id", F.explode("ids").alias("primaryid"))
    roots = components((r["id1"], r["id2"]) for r in verified.select("id1", "id2").toLocalIterator())
    root_df = spark.createDataFrame(list(roots.items()), "set_id long, root long")
    mapped = (set_of.join(F.broadcast(root_df) if len(roots) < 2_000_000 else root_df, "set_id", "left")
                    .withColumn("root", F.coalesce("root", "set_id")))
    # keep exactly one report per component: the one whose id is the component's smallest id
    return mapped.where(F.col("primaryid") != F.col("root")).select("primaryid")
