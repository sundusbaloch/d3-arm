"""Cleaning and exact deduplication of FAERS, following Pisanu et al. (2026).

Steps (counts after each one are written to data/logs/dedup_counts.json):
  1. keep only the latest version of each case (highest fda_dt, then primaryid)
  2. drop cases the FDA lists as deleted
  3. drop duplicate reports, two ways (pick with dedup=...):
     - "paper2" (default): exactly what Pisanu et al. do in their public R script
       (github.com/claudiapis/faers_esketamine). Every report is exploded into
       rows of (age, sex, PT, drugname, event date, occurrence country, indication);
       only the first copy of each row is kept (newest fda_dt first). A report
       is dropped when none of its rows survive, i.e. everything in it was
       already reported in newer reports.
     - "exact": a stricter report-level rule - two reports are duplicates only
       if sex, age, country, event date AND the full sets of drugs, reactions
       and indications are all identical.

Both are "exact" deduplication in the sense that values must match exactly. Near-duplicates (one field
slightly different) are NOT caught here - that is what our MinHash-LSH step
is for (d3arm/dedup/lsh.py).
"""
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# multiply FAERS age by this to get years
AGE_TO_YEARS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52, "DY": 1 / 365, "HR": 1 / 8760}

SALT_WORDS = ["HYDROCHLORIDE", "HCL", "HYDROBROMIDE", "SULFATE", "SULPHATE", "MALEATE",
              "MESYLATE", "TARTRATE", "CITRATE", "ACETATE", "SUCCINATE", "FUMARATE", "PHOSPHATE"]


def normalise_drug(col):
    """Rough clean-up of drug names. Not perfect - see data/reference/drug_map.csv
    for brand -> ingredient fixes we add by hand (e.g. SPRAVATO -> ESKETAMINE)."""
    c = F.upper(F.trim(col))
    c = F.regexp_replace(c, r"\\", " + ")                       # multi-ingredient prod_ai uses backslash
    c = F.regexp_replace(c, r"\(.*?\)", " ")                    # anything in brackets
    c = F.regexp_replace(c, r"\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|UNITS?|%)\b", " ")
    c = F.regexp_replace(c, r"\b(" + "|".join(SALT_WORDS) + r")\b", " ")
    c = F.regexp_replace(c, r"[^A-Z0-9+ ]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def age_years(age_col, cod_col):
    expr = F.lit(None).cast("double")
    for code, factor in AGE_TO_YEARS.items():
        expr = F.when(F.upper(cod_col) == code, age_col.cast("double") * factor).otherwise(expr)
    return F.when((expr >= 0) & (expr <= 120), expr)


def age_group(years_col):
    return (F.when(years_col.isNull(), "UNKNOWN")
             .when(years_col < 18, "0-17")
             .when(years_col < 45, "18-44")
             .when(years_col < 65, "45-64")
             .otherwise("65+"))


def fda_quarter(date_col):
    """yyyymmdd string -> '2023Q2' (quarter the FDA first received the case)."""
    d = F.to_date(date_col, "yyyyMMdd")
    return F.when(d.isNotNull(), F.concat(F.year(d).cast("string"), F.lit("Q"), F.quarter(d).cast("string")))


def paper2_surviving_ids(demo, drug_raw, reac_raw, indi_raw):
    """Row-level duplicate rule from the Paper 2 R script:

        data_reduced_sorted = rows[order(-fda_dt, primaryid), ]
        data_unique = distinct(age, sex, pt, drugname, event_dt, occr_country, indi_pt)

    dplyr's distinct() keeps the first row of each group in that order, so for
    every key we keep the row with the highest fda_dt (ties: lowest primaryid).
    We do that with a groupBy + min(struct) instead of sorting 200M+ rows, which
    is the same result but lets Spark pre-aggregate inside each partition.
    Returns the primaryids that still own at least one row.
    """
    d = demo.select("primaryid", "fda_dt_i", "age", "sex", "event_dt", "occr_country")
    dr = drug_raw.select(F.col("primaryid").cast("long").alias("primaryid"), "drug_seq", "drugname")
    rc = reac_raw.select(F.col("primaryid").cast("long").alias("primaryid"), "pt")
    ii = indi_raw.select(F.col("primaryid").cast("long").alias("primaryid"),
                         F.col("indi_drug_seq").alias("drug_seq"), "indi_pt")
    rows = (d.join(dr, "primaryid").join(rc, "primaryid")
             .join(ii, ["primaryid", "drug_seq"], "left"))      # indication belongs to one drug (drug_seq)
    key = ["age", "sex", "pt", "drugname", "event_dt", "occr_country", "indi_pt"]
    newest_first = F.struct(F.coalesce(-F.col("fda_dt_i"), F.lit(2 ** 31 - 1)).alias("o1"),   # NA sorts last, like R
                            F.col("primaryid").alias("o2"))
    return (rows.groupBy(*key).agg(F.min(newest_first).alias("w"))
                .select(F.col("w.o2").alias("primaryid")).distinct())


def clean_faers(spark: SparkSession, parquet_dir: Path, out_dir: Path, log_dir: Path, drug_map: Path = None,
                dedup: str = "paper2"):
    if dedup not in ("paper2", "exact"):
        raise ValueError("dedup must be 'paper2' or 'exact'")
    counts = {"dedup_method": dedup}
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"dedup_counts_{dedup}.json"

    def note(name, value):
        # print and save each count as soon as we have it, so a slow write at the end can't lose them
        counts[name] = value
        print(f"{name:32s} {value:>12,}", flush=True)
        log_file.write_text(json.dumps(counts, indent=2))
    demo = spark.read.parquet(str(parquet_dir / "demo"))
    drug = spark.read.parquet(str(parquet_dir / "drug"))
    reac = spark.read.parquet(str(parquet_dir / "reac"))
    indi = spark.read.parquet(str(parquet_dir / "indi"))

    note("raw_reports", demo.count())
    note("raw_ae_records", reac.count())

    demo = (demo
            .withColumn("primaryid", F.col("primaryid").cast("long"))
            .withColumn("caseid", F.col("caseid").cast("long"))
            .withColumn("fda_dt_i", F.col("fda_dt").cast("int"))
            .where(F.col("primaryid").isNotNull() & F.col("caseid").isNotNull()))

    # 1. latest version of each case
    w = Window.partitionBy("caseid").orderBy(F.col("fda_dt_i").desc_nulls_last(), F.col("primaryid").desc())
    demo = demo.withColumn("rn", F.row_number().over(w)).where("rn = 1").drop("rn")
    note("after_latest_version", demo.count())

    # 2. FDA-deleted cases
    deleted_path = parquet_dir / "deleted"
    if deleted_path.exists():
        deleted = spark.read.parquet(str(deleted_path)).select(F.col("caseid").cast("long")).distinct()
        demo = demo.join(deleted, "caseid", "left_anti")
    note("after_deleted_cases", demo.count())

    drug_raw, reac_raw, indi_raw = drug, reac, indi       # paper2 dedup works on the raw names

    # per-report lists of drugs, reactions and indications
    drug = drug.withColumn("primaryid", F.col("primaryid").cast("long")) \
               .withColumn("drug", normalise_drug(F.coalesce(F.col("prod_ai"), F.col("drugname"))))
    if drug_map is not None and Path(drug_map).exists():
        m = spark.read.option("header", True).csv(str(drug_map)) \
                 .select(F.upper(F.trim("raw")).alias("drug"), F.upper(F.trim("ingredient")).alias("mapped"))
        drug = drug.join(F.broadcast(m), "drug", "left") \
                   .withColumn("drug", F.coalesce("mapped", "drug")).drop("mapped")
    drug = drug.where(F.col("drug") != "")

    drugs = drug.groupBy("primaryid").agg(F.array_sort(F.collect_set("drug")).alias("drugs"))
    reacs = (reac.withColumn("primaryid", F.col("primaryid").cast("long"))
                 .withColumn("pt", F.upper(F.trim("pt"))).where(F.col("pt") != "")
                 .groupBy("primaryid").agg(F.array_sort(F.collect_set("pt")).alias("reactions")))
    indis = (indi.withColumn("primaryid", F.col("primaryid").cast("long"))
                 .withColumn("indi_pt", F.upper(F.trim("indi_pt")))
                 .where(F.col("indi_pt").isNotNull() & (F.col("indi_pt") != ""))
                 .groupBy("primaryid").agg(F.array_sort(F.collect_set("indi_pt")).alias("indications")))

    reports = (demo
               .join(drugs, "primaryid", "inner")      # a report with no drug or no reaction is useless here
               .join(reacs, "primaryid", "inner")
               .join(indis, "primaryid", "left")
               .withColumn("indications", F.coalesce("indications", F.array().cast("array<string>")))
               .withColumn("sex", F.upper(F.coalesce("sex", F.lit("UNK"))))
               .withColumn("age_years", F.round(age_years(F.col("age"), F.col("age_cod")), 1))
               .withColumn("age_group", age_group(F.col("age_years")))
               .withColumn("country", F.upper(F.coalesce("occr_country", "reporter_country")))
               .withColumn("window", F.coalesce(fda_quarter(F.col("init_fda_dt")), F.col("quarter"))))
    note("reports_with_drug_and_reaction", reports.count())

    # 3. duplicate reports
    if dedup == "paper2":
        keep = paper2_surviving_ids(demo, drug_raw, reac_raw, indi_raw)
        reports = reports.join(keep, "primaryid", "left_semi")
        step = "after_paper2_row_match"
    else:
        key = ["sex", "age_years", "country", "event_dt", "reactions", "drugs", "indications"]
        w2 = Window.partitionBy(*key).orderBy(F.col("fda_dt_i").desc_nulls_last(), F.col("primaryid").desc())
        reports = reports.withColumn("rn", F.row_number().over(w2)).where("rn = 1").drop("rn")
        step = "after_exact_field_match"
    reports = reports.select("primaryid", "caseid", "window", "quarter", "sex", "age_years", "age_group",
                             "country", "event_dt", "fda_dt", "drugs", "reactions", "indications").cache()
    note(step, reports.count())
    note("clean_ae_records", reports.select(F.size("reactions").alias("n")).agg(F.sum("n")).first()[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    # one shuffle so every window ends up in ONE file. Without this each of the
    # shuffle partitions writes its own small file per window - tens of thousands
    # of files, which Google Drive handles very badly.
    print("writing cleaned reports...", flush=True)
    (reports.repartition("window")
            .write.mode("overwrite").partitionBy("window").parquet(str(out_dir / "reports")))

    return counts
