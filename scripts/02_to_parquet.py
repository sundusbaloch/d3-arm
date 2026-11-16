"""Step 2: raw FAERS text files -> parquet (one folder per table, partitioned by quarter).

    python scripts/02_to_parquet.py --cores 8 --mem 8g
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3arm import config
from d3arm.faers.load import build_parquet
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--first", default=config.FIRST_QUARTER)
p.add_argument("--last", default=config.LAST_QUARTER)
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--delete-text", action="store_true",
               help="delete each quarter's unzipped .TXT files once its parquet is written (saves disk)")
args = p.parse_args()

spark = get_spark("faers-to-parquet", args.cores, args.mem)
t0 = time.time()
build_parquet(spark, config.UNZIP_DIR, config.quarter_range(args.first, args.last), config.PARQUET_DIR,
              delete_text=args.delete_text)
print(f"done in {time.time() - t0:.1f}s")
spark.stop()
