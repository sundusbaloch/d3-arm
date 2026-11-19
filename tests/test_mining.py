"""SON and the drug x AE counter must give exactly the brute-force answer."""
import itertools
import random
from collections import Counter

import pytest

from d3arm.mining.apriori import apriori, drug_ae_constraint

ITEMS = [f"DRUG={c}" for c in "ABCDEFG"] + [f"AE={c}" for c in "uvwxyz"]


def make_baskets(n=600, seed=7):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        b = set(r.sample(ITEMS, r.randint(1, 6)))
        if r.random() < 0.3:
            b |= {"DRUG=A", "AE=u"}              # a planted drug -> AE association
        out.append(tuple(sorted(b)))
    out.append(tuple(sorted(ITEMS)))              # one very long basket
    return out


def brute(baskets, min_count, allowed=None):
    res = {}
    for k in range(1, len(ITEMS) + 1):
        cnt = Counter(s for b in baskets for s in itertools.combinations(b, k))
        res.update({s: v for s, v in cnt.items() if v >= min_count and (allowed is None or allowed(s))})
    return res


@pytest.mark.parametrize("min_count", [30, 60.5, 150])
def test_apriori_matches_brute_force(min_count):
    b = make_baskets()
    assert apriori(b, min_count) == brute(b, min_count)
    c = drug_ae_constraint(2)
    assert apriori(b, min_count, c) == brute(b, min_count, c)


@pytest.fixture(scope="module")
def spark():
    from d3arm.spark import get_spark
    s = get_spark("test-mining", "2", "1g", shuffle_partitions=4)
    yield s
    s.stop()


@pytest.mark.parametrize("parts", [1, 3, 7])
def test_son_is_exact(spark, parts):
    from d3arm.mining.son import son
    b = make_baskets()
    rdd = spark.sparkContext.parallelize(b, parts).cache()
    for m in (40, 90):
        freq, stats = son(rdd, len(b), m)
        assert freq == brute(b, m)
        assert stats["candidates"] >= stats["frequent"]          # pass 1 never misses anything
        freq_c, _ = son(rdd, len(b), m, drug_ae_constraint(2))
        assert freq_c == brute(b, m, drug_ae_constraint(2))


def test_pair_counts_match_son(spark):
    from d3arm.mining.pairs import add_measures, drug_ae_counts
    b = make_baskets()
    tx = spark.createDataFrame([(list(x),) for x in b], "items array<string>")
    counts, info = drug_ae_counts(tx, 20)
    got = {(r["drug"], r["ae"]): (r["a"], r["n_drug"], r["n_ae"], r["N"]) for r in counts.collect()}
    ref = brute(b, 20)
    want = {}
    for s, v in ref.items():
        if len(s) == 2 and s[0].startswith("AE=") and s[1].startswith("DRUG="):
            want[(s[1], s[0])] = (v, ref[(s[1],)], ref[(s[0],)], len(b))
    assert got == want
    row = add_measures(counts).where("drug = 'DRUG=A' AND ae = 'AE=u'").collect()[0]
    assert row["signal"] and row["ror"] > 1                     # the planted association is found


def test_triples_match_brute_force_and_find_planted_interaction(spark):
    import math
    from d3arm.mining.triples import add_interaction_measures, triple_counts
    r = random.Random(3)
    drugs, aes = [f"DRUG={c}" for c in "ABCDEFGH"], [f"AE={c}" for c in "uvwxyz"]
    baskets = []
    for _ in range(3000):
        b = set(r.sample(drugs, r.randint(1, 4))) | set(r.sample(aes, r.randint(1, 3)))
        if {"DRUG=A", "DRUG=B"} <= b and r.random() < 0.6:
            b.add("AE=z")                                     # planted: A and B together -> z
        baskets.append(sorted(b))
    baskets.append(drugs + aes)                               # one very long report
    tx = spark.createDataFrame([(i, b) for i, b in enumerate(baskets)], "primaryid long, items array<string>")
    c, stats = triple_counts(tx, 5)
    got = {(x["d1"], x["d2"], x["ae"]): x.asDict() for x in add_interaction_measures(c).collect()}

    cnt = Counter()
    for b in baskets:
        ds = sorted(i for i in b if i.startswith("DRUG="))
        for d1, d2 in itertools.combinations(ds, 2):
            for e in (i for i in b if i.startswith("AE=")):
                cnt[(d1, d2, e)] += 1
    assert set(got) == {k for k, v in cnt.items() if v >= 5}
    n = lambda *items: sum(1 for b in baskets if set(items) <= set(b))
    row = got[("DRUG=A", "DRUG=B", "AE=z")]
    assert (row["a11"], row["n11"], row["n_d1_ae"], row["n_d1"], row["N"]) == (
        n("DRUG=A", "DRUG=B", "AE=z"), n("DRUG=A", "DRUG=B"), n("DRUG=A", "AE=z"), n("DRUG=A"), len(baskets))
    r11 = row["a11"] / row["n11"]
    r10 = (row["n_d1_ae"] - row["a11"]) / (row["n_d1"] - row["n11"])
    r01 = (row["n_d2_ae"] - row["a11"]) / (row["n_d2"] - row["n11"])
    assert math.isclose(row["crr"], r11 / max(r10, r01))
    assert row["interaction_signal"]                          # the planted interaction is found
    others = [v for k, v in got.items() if k != ("DRUG=A", "DRUG=B", "AE=z")]
    assert sum(v["interaction_signal"] for v in others) <= 0.05 * len(others)   # few chance findings


@pytest.mark.parametrize("buckets", [7, 101, 4096])
def test_pcy_gives_the_same_itemsets_with_fewer_pairs(buckets):
    b = make_baskets()
    for m in (30, 90):
        plain, st_plain = {}, {}
        plain = apriori(b, m, stats=st_plain)
        st = {}
        assert apriori(b, m, pcy_buckets=buckets, stats=st) == plain == brute(b, m)
        assert st["pairs_counted"] <= st_plain["pairs_counted"]


def test_son_with_pcy_is_exact(spark):
    from d3arm.mining.son import son
    b = make_baskets()
    rdd = spark.sparkContext.parallelize(b, 3).cache()
    freq, stats = son(rdd, len(b), 60, pcy_buckets=64)
    assert freq == brute(b, 60)
    assert stats["pcy_frequent_buckets"] > 0 and stats["pass1_pairs_counted"] > 0


def test_bonferroni_lower_bound_rescaling(spark):
    import math
    from pyspark.sql import functions as F
    from d3arm.mining.triples import lower_at, z_bonferroni
    assert abs(z_bonferroni(1, 0.05) - 1.6449) < 1e-3
    assert abs(z_bonferroni(31761534) - 5.9237) < 1e-3
    se = 0.25
    x, lo = 4.0, math.exp(math.log(4.0) - 1.96 * se)
    got = spark.range(1).select(lower_at(F.lit(x), F.lit(lo), 5.0).alias("v")).first()["v"]
    assert abs(got - math.exp(math.log(4.0) - 5.0 * se)) < 1e-9
