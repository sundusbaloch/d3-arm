"""Read the raw FAERS $-delimited text files into Spark and save them as parquet.

Every quarter has its own DEMOyyQq.TXT, DRUGyyQq.TXT, REACyyQq.TXT and
INDIyyQq.TXT. From 2014 Q3 the layout is stable, but header case and a few
optional columns change between quarters, so we lower-case the headers and
only keep the columns we need (missing ones become null).
"""
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

COLUMNS = {
    "DEMO": ["primaryid", "caseid", "caseversion", "init_fda_dt", "fda_dt", "event_dt", "age", "age_cod",
             "sex", "occr_country", "reporter_country", "occp_cod"],
    "DRUG": ["primaryid", "caseid", "drug_seq", "role_cod", "drugname", "prod_ai"],
    "REAC": ["primaryid", "caseid", "pt"],
    "INDI": ["primaryid", "caseid", "indi_drug_seq", "indi_pt"],
}


def _find(quarter_dir: Path, table: str):
    hits = [p for p in quarter_dir.glob("*.TXT") if p.name.startswith(table)]
    return hits[0] if hits else None


def read_table(spark: SparkSession, quarter_dir: Path, table: str, quarter: str):
    path = _find(quarter_dir, table)
    if path is None:
        print(f"  {quarter}: no {table} file, skipping")
        return None
    df = (
        spark.read.option("sep", "$")
        .option("header", True)
        .option("quote", "\u0000")          # FAERS has no quoting; stray " must not be treated as quotes
        .option("encoding", "ISO-8859-1")
        .option("mode", "PERMISSIVE")
        .csv(str(path))
    )
    df = df.toDF(*[c.strip().lower() for c in df.columns])
    cols = [F.col(c) if c in df.columns else F.lit(None).cast("string").alias(c) for c in COLUMNS[table]]
    return df.select(*cols).withColumn("quarter", F.lit(quarter))


def read_deleted(spark: SparkSession, quarter_dir: Path, quarter: str):
    """FDA lists deleted case ids (from 2019 onwards). One id per line."""
    files = [str(p) for p in quarter_dir.glob("*.TXT") if "DELETE" in p.name]
    if not files:
        return None
    return (
        spark.read.text(files)
        .select(F.regexp_extract("value", r"(\d+)", 1).alias("caseid"))
        .where(F.col("caseid") != "")
        .withColumn("quarter", F.lit(quarter))
    )


def build_parquet(spark: SparkSession, unzip_root: Path, quarters, out_dir: Path, delete_text=False):
    """One quarter at a time, so the unzipped text never has to sit on disk all at once.

    Output layout is out_dir/<table>/quarter=2023Q1/..., which Spark reads back
    as one table with a 'quarter' column.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in quarters:
        qdir = unzip_root / q
        if not qdir.exists():
            print(f"{q}: not unzipped, skipping")
            continue
        for table in COLUMNS:
            df = read_table(spark, qdir, table, q)
            if df is not None:
                df.drop("quarter").write.mode("overwrite").parquet(str(out_dir / table.lower() / f"quarter={q}"))
        deleted = read_deleted(spark, qdir, q)
        if deleted is not None:
            deleted.drop("quarter").distinct().write.mode("overwrite") \
                   .parquet(str(out_dir / "deleted" / f"quarter={q}"))
        print(f"{q}: parquet written")
        if delete_text:
            shutil.rmtree(qdir)
