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
