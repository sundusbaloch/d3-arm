"""Paths and default settings used across the project.

Everything can be overridden from the command line of each script, these are
just the defaults we used.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# On Colab we point this at Google Drive: export D3ARM_DATA=/content/drive/MyDrive/d3arm_data
DATA_DIR = Path(os.environ.get("D3ARM_DATA", ROOT / "data"))

RAW_DIR = DATA_DIR / "raw" / "faers"          # zip files as downloaded from the FDA
UNZIP_DIR = DATA_DIR / "raw" / "faers_unzipped"
PARQUET_DIR = DATA_DIR / "parquet"            # raw tables converted to parquet
CLEAN_DIR = DATA_DIR / "clean"                # after dedup + normalisation
LOG_DIR = DATA_DIR / "logs"
RESULTS_DIR = DATA_DIR / "results"          # tables and figures that go in the report
REFERENCE_DIR = ROOT / "data" / "reference"     # small hand-made files, kept in git

# Same window as Pisanu et al. (2026): 2014 Q3 to 2024 Q4
FIRST_QUARTER = "2014Q3"
LAST_QUARTER = "2024Q4"

# FAERS tables we actually need
TABLES = ["DEMO", "DRUG", "REAC", "INDI"]


def quarter_range(first=FIRST_QUARTER, last=LAST_QUARTER):
    """'2014Q3', '2015Q1' -> ['2014Q3', '2014Q4', '2015Q1']"""
    y, q = int(first[:4]), int(first[-1])
    y_end, q_end = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (y_end, q_end):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out
