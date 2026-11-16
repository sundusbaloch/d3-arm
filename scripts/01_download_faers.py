"""Step 1: download + unzip the FAERS quarters we need.

    python scripts/01_download_faers.py                       # 2014Q3..2024Q4
    python scripts/01_download_faers.py --first 2023Q1 --last 2023Q4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.faers.download import download_quarter, unzip_quarter

p = argparse.ArgumentParser()
p.add_argument("--first", default=config.FIRST_QUARTER)
p.add_argument("--last", default=config.LAST_QUARTER)
p.add_argument("--skip-download", action="store_true", help="only unzip zips already in data/raw/faers")
args = p.parse_args()

missing = []
for q in config.quarter_range(args.first, args.last):
    zip_path = None
    if not args.skip_download:
        zip_path = download_quarter(q, config.RAW_DIR)
    if zip_path is None:
        found = list(config.RAW_DIR.glob(f"faers_ascii_{q[:4]}[qQ]{q[-1]}.zip"))
        zip_path = found[0] if found else None
    if zip_path is None:
        missing.append(q)
        continue
    unzip_quarter(zip_path, config.UNZIP_DIR, q)
    print(f"{q}: unzipped")

if missing:
    print("\nMissing quarters (download by hand into data/raw/faers/):", ", ".join(missing))
