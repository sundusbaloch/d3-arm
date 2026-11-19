import itertools
import random

import pandas as pd
import pytest

from d3arm.crash.arm import frequent_itemsets, make_rules, mlxtend_rules, prune_redundant, to_baskets


@pytest.fixture(scope="module")
def spark():
    from d3arm.spark import get_spark
    s = get_spark("test-crash", "2", "1g", shuffle_partitions=4)
    yield s
    s.stop()


def fake_crashes(n=600, seed=1):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        light = r.choice(["Daylight", "Dark, not lighted", "Dark, lighted", "Unknown"])
        curve = r.choice(["Straight", "Curve"])
        ovt = "Yes" if (curve == "Curve" and r.random() < 0.6) or r.random() < 0.2 else "No"
        ka = "KA" if (ovt == "Yes" and light.startswith("Dark") and r.random() < 0.8) or r.random() < 0.3 else "BC"
        rows.append({"Lighting": light, "Road": curve, "Overturn": ovt, "Severity": ka,
                     "Veh": r.choice(["SUV", "Pickup", "Car", None])})
    return pd.DataFrame(rows)


def test_baskets_skip_missing():
    b = to_baskets(fake_crashes(50), ["Lighting", "Veh"])
    assert all(not any(i.endswith("=Unknown") or i.endswith("=None") for i in t) for t in b)


@pytest.mark.parametrize("method", ["son", "apriori"])
def test_target_rules_match_brute_force(spark, method):
    b = to_baskets(fake_crashes(), ["Lighting", "Road", "Overturn", "Severity", "Veh"])
    n = len(b)
    freq, mc = frequent_itemsets(spark, b, 0.02, method=method)
    rules = make_rules(freq, n, min_conf=0.5, min_lift=1.0, lift_strict=True, target="Severity=KA")
    items = sorted({i for t in b for i in t if i != "Severity=KA"})
    cnt = lambda s: sum(1 for t in b if set(s) <= set(t))
    expect = set()
    for k in range(0, 5):
        for a in itertools.combinations(items, k):
            f = cnt(a + ("Severity=KA",))
            if f < mc:
                continue
            conf = f / (cnt(a) if a else n)
            lift = conf / (cnt(("Severity=KA",)) / n)
            if conf >= 0.5 and lift > 1:
                expect.add(tuple(sorted(a)))
    assert set(rules["antecedent"]) == expect
    assert (rules["consequent"] == ("Severity=KA",)).all()
    pruned = prune_redundant(rules)
    assert 0 < len(pruned) < len(rules)
    top = pruned.iloc[0]["antecedent"]
    assert "Overturn=Yes" in top and any(i.startswith("Lighting=Dark") for i in top)   # the planted pattern


def test_rules_equal_mlxtend(spark):
    b = to_baskets(fake_crashes(), ["Lighting", "Road", "Overturn", "Severity", "Veh"])
    freq, _ = frequent_itemsets(spark, b, 0.05)
    ours = make_rules(freq, len(b), min_conf=0.6, min_lift=1.0, max_consequent=3)
    fi, mx = mlxtend_rules(b, 0.05, 0.6, 1.0)
    assert len(fi) == len(freq)
    key = lambda a, c: (tuple(sorted(a)), tuple(sorted(c)))
    assert {key(a, c) for a, c in zip(ours["antecedent"], ours["consequent"])} == \
           {key(a, c) for a, c in zip(mx["antecedents"], mx["consequents"])}


def test_paper4_table2_reproduced_from_their_data():
    """the authors' own 1,171 AV crashes give their Table 2 exactly (all 20 rows)"""
    from d3arm import config
    from d3arm.crash.av import baskets, load
    from d3arm.mining.apriori import apriori
    df = load()
    assert len(df) == 1171 and (df["Roadway surface"] == "wet").sum() == 68
    b = baskets(df)
    freq = apriori(b, 12, max_k=2)                        # support 0.01 of 1,171 -> 12 crashes
    r = make_rules(freq, len(b), min_conf=0.7, min_lift=1.0)
    ref = pd.read_csv(config.REFERENCE_DIR / "paper4_tables.csv").query("table == '2'")
    ours = r.head(20)
    assert list(ours["lift"].round(3)) == list(ref["lift"])
    assert list(ours["confidence"].round(3)) == list(ref["confidence"])
    assert ours.iloc[0]["antecedent"] == ("Weather=rain",) and ours.iloc[0]["consequent"] == ("Roadway surface=wet",)
