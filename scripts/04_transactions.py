"""Step 4: clean reports -> transactions (one basket of items per report).

    python scripts/04_transactions.py
    python scripts/04_transactions.py --no-demographics     # drugs and AEs only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F

from d3arm import config
from d3arm.faers.transactions import build_transactions
from d3arm.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--cores", default="*")
p.add_argument("--mem", default="8g")
p.add_argument("--no-demographics", action="store_true")
args = p.parse_args()

spark = get_spark("faers-transactions", args.cores, args.mem)
reports = spark.read.parquet(str(config.CLEAN_DIR / "reports"))
tx = build_transactions(reports, with_demographics=not args.no_demographics)
tx.repartition("window").write.mode("overwrite").partitionBy("window").parquet(str(config.CLEAN_DIR / "transactions"))

tx = spark.read.parquet(str(config.CLEAN_DIR / "transactions"))
print("transactions:", f"{tx.count():,}")
tx.select(F.avg(F.size("items")).alias("avg_items"), F.max(F.size("items")).alias("max_items")).show()
tx.groupBy("window").count().orderBy("window").show(60, truncate=False)
spark.stop()
