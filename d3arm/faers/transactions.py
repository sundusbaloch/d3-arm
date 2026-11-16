"""Turn each clean FAERS report into one transaction (a basket of items).

    report 123 -> ["DRUG=ESKETAMINE", "DRUG=VENLAFAXINE", "AE=DIZZINESS", "SEX=F", "AGE=45-64"]

Item prefixes keep drugs, reactions and demographics apart, so the miner can
tell a drug from a side effect (we need that for the drug -> AE constraint).
"""
from pyspark.sql import functions as F


def build_transactions(reports, with_demographics=True):
    drug_items = F.transform("drugs", lambda d: F.concat(F.lit("DRUG="), d))
    ae_items = F.transform("reactions", lambda r: F.concat(F.lit("AE="), r))
    items = F.concat(drug_items, ae_items)
    if with_demographics:
        demo_items = F.array(
            F.concat(F.lit("SEX="), F.col("sex")),
            F.concat(F.lit("AGE="), F.col("age_group")),
        )
        items = F.concat(items, demo_items)
    return reports.select(
        "primaryid", "caseid", "window",
        F.array_distinct(items).alias("items"),
        F.size("drugs").alias("n_drugs"),
        F.size("reactions").alias("n_aes"),
    )
