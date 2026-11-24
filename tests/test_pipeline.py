"""End-to-end check of steps 1-4 on the fake FAERS sample.

    pytest -q tests/
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def make_fake_reference(path):
    """A tiny reference standard in Harpaz et al.'s layout, using the fake data's drugs."""
    import pandas as pd
    ref = pd.DataFrame({"EVENT_CONCEPT_NAME": ["Dizziness disorder", "Dizziness disorder", "Agitation", "Agitation"],
                        "DRUG_CONCEPT_NAME": ["Esketamine", "Sertraline", "Bupropion", "Venlafaxine"],
                        "GROUND_TRUTH": [1, 0, 1, 0]})
    ev = pd.DataFrame({"EVENT_CONCEPT_NAME": ["Dizziness disorder", None, "Agitation"],
                       "MEDDRA_PT": ["Dizziness", "Vertigo", "Agitation"],
                       "DEFINITION_LEVEL": ["narrow", "broad", "narrow"]})
    with pd.ExcelWriter(path) as w:
        ref.to_excel(w, sheet_name="Reference Standard", index=False)
        ev.to_excel(w, sheet_name="Event Definitions", index=False)


def make_fake_crescenddi(folder):
    """The three CRESCENDDI data records we use, in the published layout, with the fake data's drugs."""
    import pandas as pd
    folder.mkdir(parents=True, exist_ok=True)
    cols = ["DRUG_1_CONCEPT_NAME", "DRUG_2_CONCEPT_NAME", "EVENT_CONCEPT_NAME", "MDR_CODE"]
    pd.DataFrame([["Esketamine", "venlafaxine", "Dizziness", 1], ["sertraline", "Bupropion", "Nausea", 2]],
                 columns=cols).to_excel(folder / "Data Record 1 - Positive Controls.xlsx", index=False)
    pd.DataFrame([["Esketamine", "Risperidone", "Headache", 3], ["Aspirin", "Bupropion", "Vertigo", 4],
                  ["Esketamine", "Lisinopril", "Headache", 5]],
                 columns=cols).to_excel(folder / "Data Record 2 - Negative Controls.xlsx", index=False)
    with pd.ExcelWriter(folder / "Data Record 3 - Single-drug ADRs, indications and negative controls.xlsx") as w:
        pd.DataFrame({"DRUG_CONCEPT_NAME": ["esketamine", "bupropion", "esketamine"],
                      "EVENT_CONCEPT_NAME": ["dissociation", "agitation", "depression"],
                      "EVENT_TYPE": ["Adverse event", "Adverse event", "Indication"],
                      "SOURCE": ["SIDER", "BNF", "SIDER"]}).to_excel(w, sheet_name="Tab1 - Positive", index=False)
        pd.DataFrame({"DRUG_CONCEPT_NAME": ["sertraline", "risperidone"],
                      "EVENT_CONCEPT_NAME": ["vertigo", "nausea"]}).to_excel(w, sheet_name="Tab2 - Negative", index=False)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("d3arm_data")
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_faers.py"), str(d / "raw" / "faers")], check=True)
    env = dict(os.environ, D3ARM_DATA=str(d))
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, env=env, cwd=ROOT)
    run("scripts/01_download_faers.py", "--first", "2023Q1", "--last", "2023Q2", "--skip-download")
    run("scripts/02_to_parquet.py", "--first", "2023Q1", "--last", "2023Q2", "--cores", "2", "--mem", "2g")
    run("scripts/03_clean.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4", "--dedup", "exact")
    run("scripts/03_clean.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")   # paper2, used below
    run("scripts/04_transactions.py", "--cores", "2", "--mem", "2g")
    run("scripts/10_paper2_ror.py", "--cores", "2", "--mem", "2g", "--min-a", "1")
    run("scripts/20_fpgrowth_baseline.py", "--cores", "2", "--mem", "2g", "--min-support", "0.2", "--partitions", "2")
    run("scripts/30_son.py", "--cores", "2", "--mem", "2g", "--min-support", "0.2", "--partitions", "3")
    run("scripts/31_drug_ae_signals.py", "--cores", "2", "--mem", "2g", "--min-count", "1", "--shuffle-partitions", "4")
    run("scripts/32_son_pcy.py", "--cores", "2", "--mem", "2g", "--min-support", "0.2", "--partitions", "3", "--buckets", "64")
    run("scripts/33_ddi_triples.py", "--cores", "2", "--mem", "2g", "--min-count", "1", "--shuffle-partitions", "4")
    run("scripts/34_ddi_filter.py", "--cores", "2", "--mem", "2g")
    run("scripts/40_time_windows.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")
    run("scripts/50_lsh_near_duplicates.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4", "--partitions", "2")
    run("scripts/50_lsh_near_duplicates.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4", "--partitions", "2",
        "--bands", "20", "--rows", "10", "--out-name", "lsh_b20_r10")
    run("scripts/51_mass_reporting.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")
    run("scripts/60_ablation.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4", "--min-count", "1")
    run("scripts/52_lsh_same_case.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")
    run("scripts/60_ablation.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4", "--min-count", "1",
        "--lsh-pairs", "near_duplicate_pairs_strict")
    run("scripts/35_son_after_lsh.py", "--cores", "2", "--mem", "2g", "--min-support", "0.2,0.1", "--partitions", "3")
    ref = d / "harpaz_fake.xlsx"
    make_fake_reference(ref)
    run("scripts/36_ground_truth_harpaz.py", "--ref", str(ref), "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")
    cres = d / "crescenddi_fake"
    make_fake_crescenddi(cres)
    run("scripts/37_ground_truth_crescenddi.py", "--ref", str(cres), "--m", "1000",
        "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4")
    run("scripts/90_compare_methods.py", "--cores", "2", "--mem", "2g", "--shuffle-partitions", "4",
        "--itemsets", "itemsets_s0.2")
    run("scripts/70_scaling.py", "--mem", "2g", "--cores-list", "1,2", "--fractions", "0.5", "--max-cores", "2",
        "--support", "0.2", "--son-partitions", "2", "--shuffle-partitions", "4")
    return d


def test_dedup_counts(data_dir):
    for method in ("exact", "paper2"):
        c = json.loads((data_dir / "logs" / f"dedup_counts_{method}.json").read_text())
        assert c["raw_reports"] == 29              # 9 planted rows + 20 ordinary
        assert c["after_latest_version"] == 28     # case 100: v1 dropped
        assert c["after_deleted_cases"] == 27      # case 200 removed
    exact = json.loads((data_dir / "logs" / "dedup_counts_exact.json").read_text())
    p2 = json.loads((data_dir / "logs" / "dedup_counts_paper2.json").read_text())
    assert exact["after_exact_field_match"] == 26  # 300/301 collapse; 600 is not identical to 601
    assert p2["after_paper2_row_match"] == 25      # ...but every row of 600 is inside 601, so it goes too
    # near-duplicates 400/401 must still both be there (LSH's job, not exact dedup)


def test_transactions(data_dir):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    tx = spark.read.parquet(str(data_dir / "clean" / "transactions"))
    rows = {r["caseid"]: r for r in tx.collect()}
    assert 400 in rows and 401 in rows
    assert 601 in rows and 600 not in rows     # paper2 rule kept the newer, fuller report
    assert 301 in rows and 300 not in rows     # newest copy of an exact duplicate is the one kept
    items = set(rows[100]["items"])
    assert "DRUG=ESKETAMINE" in items          # SPRAVATO mapped via drug_map.csv... or prod_ai
    assert "DRUG=VENLAFAXINE" in items         # salt word HYDROCHLORIDE stripped
    assert "AE=DIZZINESS" in items and "SEX=F" in items and "AGE=18-44" in items
    assert rows[100]["window"] == "2023Q1"     # window = when the FDA first got the case


def test_paper2_ror_runs(data_dir):
    s = json.loads((data_dir / "results" / "paper2" / "summary.json").read_text())["ours"]
    # every clean report is either in the esketamine group, excluded, or a control
    assert s["control_cases"] + s["reports_mentioning_esketamine"] == s["all_clean_reports"]
    assert 0 < s["esketamine_cases"] <= s["esketamine_ps_or_ss"] <= s["reports_mentioning_esketamine"]
    assert (data_dir / "results" / "paper2" / "ror_main.csv").exists()


def test_fpgrowth_baseline_runs(data_dir):
    s = json.loads((data_dir / "results" / "fpgrowth" / "summary.json").read_text())
    r = s["runs"][0]
    assert s["transactions"] == 25 and r["itemsets"] > 0 and r["min_count"] == 5


def test_son_matches_fpgrowth_on_fake_data(data_dir):
    s = json.loads((data_dir / "results" / "son" / "summary.json").read_text())
    assert s["runs"][0]["vs_fpgrowth"]["identical"]


def test_drug_ae_signals_run(data_dir):
    s = json.loads((data_dir / "results" / "signals" / "summary.json").read_text())
    assert s["transactions"] == 25 and s["drug_ae_pairs"] > 0


def test_time_windows_run(data_dir):
    s = json.loads((data_dir / "results" / "windows" / "summary.json").read_text())
    assert s["pairs_followed"] > 0
    assert s["stable_signals"] <= s["signals_full_data"] <= s["pairs_followed"]
    for b in s["bloom"]:
        assert b["alerts_raised"] >= b["truly_new_signals"] - b["missed_new_signals"]


def test_lsh_runs(data_dir):
    s = json.loads((data_dir / "results" / "lsh" / "summary.json").read_text())
    assert s["reports_with_age_and_event_date"] > 0
    assert s["near_duplicate_pairs"] <= s["candidate_pairs"]


def test_son_after_lsh(data_dir):
    s = json.loads((data_dir / "results" / "son_after_lsh" / "summary.json").read_text())
    a = json.loads((data_dir / "results" / "ablation" / "summary.json").read_text())
    assert s["removed"] == a["step2_reports_dropped"] >= 1      # same reports dropped as in the ablation
    assert s["transactions_after"] == s["transactions_before"] - s["removed"]
    for r in s["runs"]:
        c = r["same_count"]
        assert c["gained"] == 0                     # counts can only fall when reports are removed
        assert c["kept"] + c["lost"] == c["itemsets_before"]


def test_ground_truth_harpaz(data_dir):
    import pandas as pd
    s = json.loads((data_dir / "results" / "ground_truth" / "harpaz_summary.json").read_text())
    assert s["reference"]["pairs"] == 4 and s["reference"]["positive"] == 2
    assert s["drugs_not_found"] == []
    for label, terms in [("narrow", {"DIZZINESS"}), ("narrow_broad", {"DIZZINESS", "VERTIGO"})]:
        d = s["definitions"][label]
        for m in d["methods"].values():
            assert m["tp"] + m["fn"] == 2 and m["fp"] + m["tn"] == 2
        # a for esketamine-dizziness must equal a direct count of reports from the transactions
        tx = pd.read_parquet(data_dir / "clean" / "transactions")
        direct = sum(1 for items in tx["items"]
                     if "DRUG=ESKETAMINE" in items and any(i[3:] in terms for i in items if i.startswith("AE=")))
        t = pd.read_csv(data_dir / "results" / "ground_truth" / f"harpaz_pairs_{label}.csv")
        row = t[(t.drug == "ESKETAMINE") & (t.event == "DIZZINESS DISORDER")].iloc[0]
        assert row["a"] == direct > 0
        assert row["a_lsh"] <= row["a"]


def test_ground_truth_crescenddi(data_dir):
    import pandas as pd
    s = json.loads((data_dir / "results" / "ground_truth" / "crescenddi_summary.json").read_text())
    assert s["triplets"]["total"] == 5 and s["triplets"]["positive"] == 2
    assert s["single"]["total"] == 4 and s["single"]["positive"] == 2      # the "Indication" row is left out
    assert s["ddi_drugs_not_found"] == ["LISINOPRIL"]
    for part, npos, nneg in [("triplets", 2, 3), ("single", 2, 2)]:
        for m in s[part]["methods"].values():
            assert m["tp"] + m["fn"] == npos and m["fp"] + m["tn"] == nneg
    # a11 for esketamine + venlafaxine -> dizziness must equal a direct count from the transactions
    tx = pd.read_parquet(data_dir / "clean" / "transactions")
    direct = sum(1 for items in tx["items"]
                 if {"DRUG=ESKETAMINE", "DRUG=VENLAFAXINE", "AE=DIZZINESS"} <= set(items))
    t = pd.read_csv(data_dir / "results" / "ground_truth" / "crescenddi_triplets.csv")
    row = t[(t.d1 == "ESKETAMINE") & (t.d2 == "VENLAFAXINE") & (t.ae == "DIZZINESS")].iloc[0]
    assert row["a11"] == direct > 0 and row["a11_lsh"] <= row["a11"]
    single = pd.read_csv(data_dir / "results" / "ground_truth" / "crescenddi_single.csv")
    direct1 = sum(1 for items in tx["items"] if {"DRUG=ESKETAMINE", "AE=DISSOCIATION"} <= set(items))
    assert single[(single.drug == "ESKETAMINE") & (single.event == "DISSOCIATION")]["a_end"].iloc[0] == direct1


def test_lsh_same_case(data_dir):
    s = json.loads((data_dir / "results" / "lsh" / "same_case_summary.json").read_text())
    lsh = json.loads((data_dir / "results" / "lsh" / "summary.json").read_text())
    assert s["pairs"] == lsh["near_duplicate_pairs"]
    assert s["same_case"] <= s["pairs"] and s["same_case"] >= 1     # 400/401 differ only by age 45 vs 46
    assert s["reports_dropped_same_case_pairs"] <= s["reports_dropped_all_pairs"]
    a = json.loads((data_dir / "results" / "ablation_strict" / "summary.json").read_text())
    assert a["step2_reports_dropped"] == s["reports_dropped_same_case_pairs"]


def test_ablation_and_scaling_run(data_dir):
    a = json.loads((data_dir / "results" / "ablation" / "summary.json").read_text())
    assert a["step2_transactions"] == a["step1_transactions"] - a["step2_reports_dropped"]
    assert a["step3_stable_signals"] <= a["step2_signals"]
    rows = (data_dir / "results" / "scaling" / "runs.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2 * 3                     # header + 2 workloads x (2 core settings + 1 fraction)


def test_lsh_tuning_and_mass_reporting_run(data_dir):
    t = json.loads((data_dir / "results" / "lsh_b20_r10" / "summary.json").read_text())
    assert t["bands"] == 20 and t["rows"] == 10
    m = json.loads((data_dir / "results" / "mass_reporting" / "summary.json").read_text())
    assert m["reports_in_those_groups"] <= m["reports_without_age_or_event_date"]


def test_method_comparison_runs(data_dir):
    s = json.loads((data_dir / "results" / "comparison" / "summary.json").read_text())
    assert len(s["methods"]) == 7
    assert all(m["drug_ae_results"] >= 0 for m in s["methods"])


def test_son_pcy_and_triples_run(data_dir):
    s = json.loads((data_dir / "results" / "son_pcy" / "summary.json").read_text())
    assert all(r["identical"] for r in s["runs"])
    t = json.loads((data_dir / "results" / "triples" / "summary.json").read_text())
    assert t["triples"] > 0 and t["interaction_signals"] <= t["combination_signals"] <= t["triples"]
    f = json.loads((data_dir / "results" / "triples" / "filter_summary.json").read_text())
    counts = [x["signals"] for x in f["steps"]]
    assert counts == sorted(counts, reverse=True) and counts[0] == t["interaction_signals"]
