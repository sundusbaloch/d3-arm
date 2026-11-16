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
