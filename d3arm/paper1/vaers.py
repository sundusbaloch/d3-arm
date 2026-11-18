"""Paper 1 reproduction, part 2: the real-data example on VAERS 2024.

A report (VAERS_ID) is a basket of vaccines (VAX_NAME) and symptoms
(SYMPTOM1..5, MedDRA PTs). The "modified Apriori" of the paper:
  1. vaccines and AEs with support >= minsup
  2. vaccine-AE pairs with support >= minsup                 (main effects)
  3. vaccine-vaccine pairs >= minsup, then (v1, v2, AE) >= minsup   (DDIs)
then secondary indicators for each, and the top 10 by ROR (their Tables 8-9).
The paper does not say which minsup it used for VAERS, so we run several.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F


def extract(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.upper().endswith(".CSV"):
                target = out_dir / Path(name).name.upper()
                if not target.exists():
                    with z.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    return out_dir


def _read(spark, path):
    return (spark.read.option("header", True).option("encoding", "ISO-8859-1")
                 .option("multiLine", True).option("escape", '"').csv(str(path)))


def load_transactions(spark, csv_dir, year=2024):
    csv_dir = Path(csv_dir)
    data = _read(spark, csv_dir / f"{year}VAERSDATA.CSV").select("VAERS_ID")
    sym = _read(spark, csv_dir / f"{year}VAERSSYMPTOMS.CSV")
    vax = _read(spark, csv_dir / f"{year}VAERSVAX.CSV")
    aes = (sym.select("VAERS_ID", F.explode(F.array(*[F.col(f"SYMPTOM{i}") for i in range(1, 6)])).alias("ae"))
              .where(F.col("ae").isNotNull() & (F.trim("ae") != ""))
              .groupBy("VAERS_ID").agg(F.array_sort(F.collect_set(F.trim("ae"))).alias("aes")))
    vaxes = (vax.where(F.col("VAX_NAME").isNotNull())
                .groupBy("VAERS_ID").agg(F.array_sort(F.collect_set(F.trim("VAX_NAME"))).alias("vaxes")))
    tx = vaxes.join(aes, "VAERS_ID").cache()
    info = {"reports_in_data_file": data.select("VAERS_ID").distinct().count(),
            "reports_with_vaccine_and_symptom": tx.count(),
            "distinct_aes": tx.select(F.explode("aes")).distinct().count(),
            "distinct_vaccine_names": tx.select(F.explode("vaxes")).distinct().count()}
    return tx, info


def _metrics(df, n):
    """same formulas as the authors' R code (b, c, d from the counts)"""
    a = df["a"].astype(float)
    b, c = df["n_x"] - a, df["n_ae"] - a
    d = n - df["n_x"] - df["n_ae"] + a
    df = df.assign(conf=a / (a + b), prr=(a / (a + b)) / (c / (c + d)), ror=(a * d) / (c * b),
                   rr=n * a / ((a + c) * (a + b)))          # c = 0 gives inf, as in R
    df["lift"] = df["conf"] / (df["n_ae"] / n)
    return df


def counts(tx):
    """all the counts the method needs (small data, so collected to pandas)"""
    n = tx.count()
    v1 = tx.select("VAERS_ID", F.explode("vaxes").alias("v"))
    ae = tx.select("VAERS_ID", F.explode("aes").alias("ae"))
    nv = v1.groupBy("v").count().toPandas().set_index("v")["count"]
    nae = ae.groupBy("ae").count().toPandas().set_index("ae")["count"]
    va = v1.join(ae, "VAERS_ID").groupBy("v", "ae").count().toPandas()
    pairs = (v1.alias("x").join(v1.alias("y"), "VAERS_ID").where(F.col("x.v") < F.col("y.v"))
               .select("VAERS_ID", F.col("x.v").alias("v1"), F.col("y.v").alias("v2")))
    vv = pairs.groupBy("v1", "v2").count().toPandas()
    vva = pairs.join(ae, "VAERS_ID").groupBy("v1", "v2", "ae").count().toPandas()
    return n, nv, nae, va, vv, vva


def modified_apriori(n, nv, nae, va, vv, vva, minsup):
    fv, fae = set(nv[nv >= minsup].index), set(nae[nae >= minsup].index)
    main = va[(va["count"] >= minsup) & va["v"].isin(fv) & va["ae"].isin(fae)].rename(columns={"count": "a"})
    main = _metrics(main.assign(n_x=main["v"].map(nv), n_ae=main["ae"].map(nae)), n)
    fvv = vv[(vv["count"] >= minsup) & vv["v1"].isin(fv) & vv["v2"].isin(fv)]
    ddi = (vva.merge(fvv.rename(columns={"count": "n_x"}), on=["v1", "v2"])
              .rename(columns={"count": "a"}))
    ddi = ddi[(ddi["a"] >= minsup) & ddi["ae"].isin(fae)]
    ddi = _metrics(ddi.assign(n_ae=ddi["ae"].map(nae)), n)
    return main, ddi


def _norm(s):
    s = re.sub(r"[^A-Z0-9]+", " ", str(s).upper())
    return frozenset(s.split())


def same_vaccine(paper_name, our_name):
    """paper tables shorten some names, e.g. 'COVID19 (PFIZER-BIONTECH)' for
    'COVID19 (COVID19 (PFIZER-BIONTECH))': match if all the paper's words appear in ours"""
    return _norm(paper_name) <= _norm(our_name)


def overlap_main(top, ref):
    hits = 0
    for _, r in ref.iterrows():
        hits += any(same_vaccine(r["vaccine"], v) and r["ae"].upper() == a.upper() for v, a in zip(top["v"], top["ae"]))
    return hits


def overlap_ddi(top, ref):
    hits = 0
    for _, r in ref.iterrows():
        hits += any(((same_vaccine(r["vaccine1"], x) and same_vaccine(r["vaccine2"], y)) or
                     (same_vaccine(r["vaccine1"], y) and same_vaccine(r["vaccine2"], x))) and r["ae"].upper() == a.upper()
                    for x, y, a in zip(top["v1"], top["v2"], top["ae"]))
    return hits
