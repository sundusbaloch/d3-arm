"""Steps 1 + 2 together, one quarter at a time, keeping disk use low.

For each quarter: download zip -> unzip the 4 tables -> write parquet ->
delete the .TXT files (and the zip, if --delete-zip). Re-running skips
quarters whose parquet already exists, so it is safe to stop and restart.

    python scripts/prepare_all.py --delete-zip
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.faers.download import download_quarter, unzip_quarter
from d3arm.faers.load import build_parquet
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--first", default=config.FIRST_QUARTER)
p.add_argument("--last", default=config.LAST_QUARTER)
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--delete-zip", action="store_true")
args = p.parse_args()

spark = get_spark("faers-prepare", args.cores, args.mem)
t0 = time.time()
for q in config.quarter_range(args.first, args.last):
    # a quarter counts as done only if every table finished writing (_SUCCESS),
    # so a disconnect halfway through a quarter gets redone on the next run
    if all((config.PARQUET_DIR / t.lower() / f"quarter={q}" / "_SUCCESS").exists() for t in config.TABLES):
        print(f"{q}: already done")
        continue
    zip_path = download_quarter(q, config.RAW_DIR)
    if zip_path is None:
        continue
    unzip_quarter(zip_path, config.UNZIP_DIR, q)
    build_parquet(spark, config.UNZIP_DIR, [q], config.PARQUET_DIR, delete_text=True)
    if args.delete_zip:
        zip_path.unlink()
print(f"all done in {(time.time() - t0) / 60:.1f} min")
shutil.rmtree(config.UNZIP_DIR, ignore_errors=True)
spark.stop()
