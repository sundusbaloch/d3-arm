"""Paper 1: AUC, the Apriori counts and the VAERS pipeline on tiny data."""
import itertools

import numpy as np

from d3arm.paper1.simulation import apriori_scores, auc, simulate_ae


def test_auc_basic_and_ties():
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5
    assert abs(auc([1, 0, 1, 0], [3, 1, 2, 2]) - 0.875) < 1e-12


def test_apriori_counts_match_brute_force():
    rng = np.random.default_rng(0)
    v = rng.random((400, 5)) < 0.3
    y = rng.random((400, 4)) < 0.2
    main, ddi, pairs = apriori_scores(v, y, minsup=3)
    i, j = 1, 3
    a = int((v[:, i] & y[:, j]).sum())
    b, c = int(v[:, i].sum()) - a, int(y[:, j].sum()) - a
    d = 400 - a - b - c
    assert np.isclose(main["ror"][i, j], a * d / (b * c))
    r = pairs.index((0, 2))
    w = v[:, 0] & v[:, 2]
    a3 = int((w & y[:, 1]).sum())
    if a3 >= 3:
        assert np.isclose(ddi["conf"][r, 1], a3 / w.sum())


def test_prevalence_is_about_ten_percent():
    rng = np.random.default_rng(1)
    x = (rng.random((2000, 6)) < 0.2).astype(float)
    beta = np.zeros((6, 4)); beta[0, 0] = 1
    y = simulate_ae(x, beta, "random_effect", rng)
    assert abs(y.mean() - 0.1) < 0.02


def test_vaers_pipeline(tmp_path):
    import zipfile
    from d3arm.paper1.vaers import counts, load_transactions, modified_apriori, extract, same_vaccine
    from d3arm.spark import get_spark
    d = tmp_path / "csv"
    d.mkdir()
    (d / "2024VAERSDATA.csv").write_text('VAERS_ID,SYMPTOM_TEXT\n1,"fainted\nafter shot"\n2,x\n3,y\n4,z\n')
    (d / "2024VAERSSYMPTOMS.csv").write_text(
        "VAERS_ID,SYMPTOM1,SYMPTOMVERSION1,SYMPTOM2,SYMPTOMVERSION2,SYMPTOM3,SYMPTOMVERSION3,SYMPTOM4,SYMPTOMVERSION4,SYMPTOM5,SYMPTOMVERSION5\n"
        "1,Syncope,27,Dizziness,27,,,,,,\n2,Syncope,27,,,,,,,,\n3,Pain,27,,,,,,,,\n4,Syncope,27,Pain,27,,,,,,\n")
    (d / "2024VAERSVAX.csv").write_text(
        "VAERS_ID,VAX_TYPE,VAX_MANU,VAX_LOT,VAX_DOSE_SERIES,VAX_ROUTE,VAX_SITE,VAX_NAME\n"
        "1,HPV9,MERCK,,1,IM,LA,HPV (GARDASIL 9)\n1,TDAP,GSK,,1,IM,LA,TDAP (BOOSTRIX)\n"
        "2,HPV9,MERCK,,1,IM,LA,HPV (GARDASIL 9)\n3,FLU,SANOFI,,1,IM,LA,INFLUENZA (SEASONAL)\n"
        "4,HPV9,MERCK,,1,IM,LA,HPV (GARDASIL 9)\n4,TDAP,GSK,,1,IM,LA,TDAP (BOOSTRIX)\n")
    z = tmp_path / "v.zip"
    with zipfile.ZipFile(z, "w") as f:
        for p in d.iterdir():
            f.write(p, p.name)
    spark = get_spark("test-vaers", "2", "1g", shuffle_partitions=4)
    try:
        tx, info = load_transactions(spark, extract(z, tmp_path / "x"))
        assert info["reports_in_data_file"] == 4 and info["reports_with_vaccine_and_symptom"] == 4
        n, nv, nae, va, vv, vva = counts(tx)
        main, ddi = modified_apriori(n, nv, nae, va, vv, vva, minsup=2)
        row = main[(main.v == "HPV (GARDASIL 9)") & (main.ae == "Syncope")].iloc[0]
        assert row["a"] == 3 and np.isinf(row["ror"])          # Syncope only ever with HPV: c = 0
        assert len(ddi) == 1 and ddi.iloc[0]["a"] == 2          # HPV + TDAP -> Syncope in reports 1 and 4
    finally:
        spark.stop()
    assert same_vaccine("COVID19 (PFIZER-BIONTECH)", "COVID19 (COVID19 (PFIZER-BIONTECH))")
    assert not same_vaccine("HPV (GARDASIL 9)", "TDAP (BOOSTRIX)")
