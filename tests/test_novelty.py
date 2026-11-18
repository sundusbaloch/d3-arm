"""Bloom filter and the quarter-by-quarter replay on hand-checkable data."""
import numpy as np
import pytest

from d3arm.novelty.bloom import Bloom, bloom_size, expected_fp


def test_bloom_no_false_negatives_and_fp_close_to_theory():
    n = 50_000
    m, k = bloom_size(n, 0.01)
    bf = Bloom(m, k)
    rng = np.random.default_rng(0)
    h = rng.integers(-2**63, 2**63 - 1, (2, n), dtype=np.int64)
    bf.add(h[0], h[1])
    assert bf.contains(h[0], h[1]).all()                       # never forgets an inserted item
    q = rng.integers(-2**63, 2**63 - 1, (2, 100_000), dtype=np.int64)
    fp = bf.contains(q[0], q[1]).mean()
    assert abs(fp - expected_fp(n, m, k)) < 0.003


@pytest.fixture(scope="module")
def spark():
    from d3arm.spark import get_spark
    s = get_spark("test-novelty", "2", "1g", shuffle_partitions=4)
    yield s
    s.stop()


def test_replay_finds_the_quarter_a_signal_appears(spark):
    from d3arm.novelty.windows import quarter_label, replay, summarise_pairs
    rows = []
    # background: 200 reports per quarter of unrelated drugs/AEs, 2020Q1..2020Q4
    for q in ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]:
        rows += [(q, [f"DRUG=X{i % 7}", f"AE=E{i % 5}"]) for i in range(200)]
    # drug D with AE Z: 2 reports in Q1 (too few), 4 more in Q3 -> signal from 2020Q3
    rows += [("2020Q1", ["DRUG=D", "AE=Z"])] * 2 + [("2020Q3", ["DRUG=D", "AE=Z"])] * 4
    rows += [("2020Q2", ["DRUG=D", "AE=E1"])] * 3
    rows += [("2020Q1", ["DRUG=X1", "AE=Z"])] * 10          # Z also happens without D, so c > 0
    tx = spark.createDataFrame(rows, "window string, items array<string>")
    pairs = spark.createDataFrame([("DRUG=D", "AE=Z")], "drug string, ae string")
    ev = replay(tx, pairs)
    got = {quarter_label(r["qi"]): (r["a"], r["signal"]) for r in ev.collect()}
    assert got == {"2020Q1": (2, False), "2020Q3": (6, True)}
    s = summarise_pairs(ev).collect()[0]
    assert quarter_label(s["first_signal"]) == "2020Q3" and s["lost_after_first"] == 0


def test_old_windows_are_merged(spark):
    from d3arm.novelty.windows import BASE, quarter_index, quarter_label
    df = spark.createDataFrame([("1999Q1",), ("2014Q2",), ("2014Q3",)], "w string")
    got = [quarter_label(r[0]) for r in df.select(quarter_index(df.w)).collect()]
    assert got == ["pre-2014Q3", "pre-2014Q3", "2014Q3"]


def test_lsh_finds_planted_near_duplicates(spark):
    import random
    from d3arm.novelty.lsh import brute_force_pairs, candidate_probability, find_near_duplicates
    r = random.Random(5)
    drugs = [f"D{i}" for i in range(40)]
    aes = [f"E{i}" for i in range(60)]
    rows, pid = [], 1
    for _ in range(300):                                   # unrelated reports
        rows.append((pid, r.choice("FM"), float(r.randint(18, 90)), "US", f"2020{r.randint(1, 12):02d}01",
                     r.sample(drugs, 3), r.sample(aes, 5), ["X"]))
        pid += 1
    for _ in range(40):                                    # 40 planted near-duplicate pairs
        d, e = r.sample(drugs, 4), r.sample(aes, 6)
        base = (r.choice("FM"), float(r.randint(18, 90)), "GB", "20210315")
        rows.append((pid, *base, d, e, ["Y"]))
        rows.append((pid + 1, *base, d, e + ["EXTRA"], ["Y"]))  # one extra AE: J = 14/15 = 0.93
        pid += 2
    rows.append((pid, "F", 50.0, "US", "20200101", ["D1"], ["E1"], ["X"]))   # an exact copy of the next one
    rows.append((pid + 1, "F", 50.0, "US", "20200101", ["D1"], ["E1"], ["X"]))
    rows.append((pid + 2, "F", None, "US", "20200101", ["D1"], ["E2"], ["X"]))  # no age: not used
    schema = ("primaryid long, sex string, age_years double, country string, event_dt string, "
              "drugs array<string>, reactions array<string>, indications array<string>")
    reports = spark.createDataFrame(rows, schema)
    verified, groups, toks, s = find_near_duplicates(spark, reports, b=25, r=8, threshold=0.8, partitions=4)
    assert s["reports_with_age_and_event_date"] == len(rows) - 1
    assert s["identical_groups"] == 1 and s["reports_in_identical_groups"] == 2
    found = {(x["id1"], x["id2"]) for x in verified.collect()}
    sets = {x["set_id"]: x["tokens"] for x in groups.collect()}
    truth = brute_force_pairs(sets, 0.8)
    assert found <= truth                                   # verification removes every false positive
    assert len(found) >= 0.9 * len(truth) and len(truth) >= 40
    assert candidate_probability(0.93, 25, 8) > 0.99


def test_components_joins_chains():
    from d3arm.novelty.lsh import components
    roots = components([(5, 3), (3, 9), (20, 21)])
    assert roots == {3: 3, 5: 3, 9: 3, 20: 20, 21: 20}
