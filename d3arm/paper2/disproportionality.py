"""Reproduction of the disproportionality part of Paper 2 (Pisanu et al., 2026,
Eur J Clin Pharmacol 82:13) on our cleaned FAERS 2014Q3-2024Q4.

What the paper does (checked against their R script on GitHub):
  - case group: reports where esketamine is the primary (PS) or secondary (SS)
    suspect, minus reports where it was given for a non-psychiatric reason
    (anaesthesia, pain, ...)
  - control group: every other report that does not mention esketamine at all
  - for every reaction (PT) seen with esketamine, a 2x2 table:
        a = esketamine cases with the PT     b = esketamine cases without it
        c = control cases with the PT        d = control cases without it
    ROR = (a/b)/(c/d) = ad/bc,  95% CI = exp(ln ROR +- 1.96 sqrt(1/a+1/b+1/c+1/d))
  - signal: a >= 5 and lower CI > 1
  - sensitivity analysis: control = only reports with venlafaxine
  - sex differences: ROR in women and in men separately, then
        relative ROR women = ROR_women / ROR_men,
        CI = exp(ln rel +- 1.96 sqrt(sum of the 8 reciprocals)),
    for PTs with >= 5 esketamine cases in both sexes.

The Spark part only counts; the 2x2 arithmetic is done on small tables in pandas.
"""
import math

import pandas as pd
from pyspark.sql import functions as F

ESKETAMINE = r"ESKETAMINE|SPRAVATO"          # ingredient or brand name (DiAna maps SPRAVATO -> ESKETAMINE)
VENLAFAXINE = r"VENLAFAXINE|EFFEXOR"

# indications for which the paper drops esketamine cases (copied from their script)
NON_PSYCH_INDICATIONS = [
    "New daily persistent headache", "Status epilepticus", "Sedation", "Anaesthesia",
    "Induction of anaesthesia", "Pain", "Migraine", "General anaesthesia", "Analgesic therapy",
    "Respiratory symptom", "Cardiogenic shock", "Toxicity to various agents", "Essential hypertension",
    "Neuralgia", "Anaesthesia procedure", "Parkinson's disease", "Asthenia", "Neck pain",
    "Product use in unapproved indication", "Back pain", "Complex regional pain syndrome",
    "Dyskinesia", "Endometriosis", "Restlessness",
]


# ---------------------------------------------------------------- arithmetic

def ror_ci(a, b, c, d, z=1.96):
    """ROR and its 95% CI. Returns (nan, nan, nan) if any cell is 0."""
    if min(a, b, c, d) <= 0:
        return float("nan"), float("nan"), float("nan")
    ror = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return ror, math.exp(math.log(ror) - z * se), math.exp(math.log(ror) + z * se)


def ror_table(case_counts: dict, n_case: int, ctrl_counts: dict, n_ctrl: int, min_a: int = 5) -> pd.DataFrame:
    """One row per PT that appears in the case group."""
    rows = []
    for pt, a in case_counts.items():
        c = ctrl_counts.get(pt, 0)
        b, d = n_case - a, n_ctrl - c
        ror, lo, hi = ror_ci(a, b, c, d)
        rows.append({"pt": pt, "a": a, "b": b, "c": c, "d": d, "ror": ror, "lower": lo, "upper": hi})
    df = pd.DataFrame(rows, columns=["pt", "a", "b", "c", "d", "ror", "lower", "upper"])
    df["signal"] = (df["a"] >= min_a) & (df["lower"] > 1)
    return df.sort_values(["a", "ror"], ascending=False).reset_index(drop=True)


def sex_table(women: pd.DataFrame, men: pd.DataFrame, min_a: int = 5, z=1.96) -> pd.DataFrame:
    """Relative RORs (women vs men) for PTs with >= min_a cases in both sexes."""
    m = women.merge(men, on="pt", suffixes=("_w", "_m"))
    m = m[(m["a_w"] >= min_a) & (m["a_m"] >= min_a)].copy()
    cells = [f"{x}_{s}" for s in ("w", "m") for x in "abcd"]
    m["se"] = sum(1 / m[c] for c in cells) ** 0.5
    m["rel_ror_w"] = m["ror_w"] / m["ror_m"]
    m["rel_lower_w"] = (m["rel_ror_w"].apply(math.log) - z * m["se"]).apply(math.exp)
    m["rel_upper_w"] = (m["rel_ror_w"].apply(math.log) + z * m["se"]).apply(math.exp)
    # relative ROR for men is just the reciprocal (and so is its CI, flipped)
    m["rel_ror_m"], m["rel_lower_m"], m["rel_upper_m"] = 1 / m["rel_ror_w"], 1 / m["rel_upper_w"], 1 / m["rel_lower_w"]
    m["more_in_women"] = (m["lower_w"] > 1) & (m["rel_lower_w"] > 1)
    m["more_in_men"] = (m["lower_m"] > 1) & (m["rel_lower_m"] > 1)
    keep = ["pt", "a_w", "ror_w", "lower_w", "upper_w", "a_m", "ror_m", "lower_m", "upper_m",
            "rel_ror_w", "rel_lower_w", "rel_upper_w", "more_in_women", "more_in_men"]
    return m[keep].sort_values("rel_ror_w", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- spark side

def _reports_mentioning(drug, pattern):
    """drug rows whose ingredient (prod_ai) or written name matches the pattern"""
    return drug.where(F.upper(F.coalesce("prod_ai", F.lit(""))).rlike(pattern)
                      | F.upper(F.coalesce("drugname", F.lit(""))).rlike(pattern))


def _pt_counts(reports, by=None):
    """number of reports per PT (each report counts once per PT), optionally split by another column"""
    cols = ["pt"] + ([by] if by else [])
    return (reports.select(F.explode(F.array_distinct("reactions")).alias("pt"), *([by] if by else []))
                   .groupBy(*cols).count())


def _as_dict(df, key="pt"):
    return {r[key]: r["count"] for r in df.collect()}


def run(spark, clean_dir, parquet_dir, min_a=5):
    """Returns (summary dict, {name: DataFrame}). Nothing is written here."""
    reports = (spark.read.parquet(str(clean_dir / "reports"))
                    .select("primaryid", "caseid", "sex", "reactions").cache())
    ids = reports.select("primaryid")

    drug = (spark.read.parquet(str(parquet_dir / "drug"))
                 .withColumn("primaryid", F.col("primaryid").cast("long"))
                 .join(ids, "primaryid", "left_semi")
                 .withColumn("role_cod", F.upper(F.trim("role_cod"))))
    indi = (spark.read.parquet(str(parquet_dir / "indi"))
                 .select(F.col("primaryid").cast("long").alias("primaryid"),
                         F.col("indi_drug_seq").alias("drug_seq"),
                         F.upper(F.trim("indi_pt")).alias("indi_pt")))

    esk_rows = _reports_mentioning(drug, ESKETAMINE).select("primaryid", "drug_seq", "role_cod").cache()
    any_esk = esk_rows.select("primaryid").distinct()
    ps_ss = esk_rows.where(F.col("role_cod").isin("PS", "SS"))
    bad = [x.upper() for x in NON_PSYCH_INDICATIONS]
    # a PS/SS esketamine row is usable if its indication is missing or psychiatric;
    # the case stays if at least one such row is usable
    esk_ids = (ps_ss.join(indi, ["primaryid", "drug_seq"], "left")
                    .withColumn("ok", (F.col("indi_pt").isNull() | ~F.col("indi_pt").isin(bad)).cast("int"))
                    .groupBy("primaryid").agg(F.max("ok").alias("ok"))
                    .where("ok = 1").select("primaryid"))
    venla_ids = _reports_mentioning(drug, VENLAFAXINE).select("primaryid").distinct()

    case = reports.join(esk_ids, "primaryid", "left_semi").cache()
    ctrl = (reports.join(any_esk, "primaryid", "left_anti")
                   .join(venla_ids.withColumn("venla", F.lit(1)), "primaryid", "left")
                   .withColumn("venla", F.coalesce("venla", F.lit(0))))

    s = {
        "all_clean_reports": reports.count(),
        "reports_mentioning_esketamine": any_esk.count(),
        "esketamine_ps_or_ss": ps_ss.select("primaryid").distinct().count(),
        "esketamine_cases": case.count(),
    }
    s["control_cases"] = s["all_clean_reports"] - s["reports_mentioning_esketamine"]
    s["excluded_not_ps_ss_or_non_psych"] = s["reports_mentioning_esketamine"] - s["esketamine_cases"]
    s["esketamine_women"] = case.where("sex = 'F'").count()
    s["esketamine_men"] = case.where("sex = 'M'").count()

    # ---- counts (one pass over the control group, split by sex and venlafaxine)
    case_all = _as_dict(_pt_counts(case))
    case_sex = _pt_counts(case, "sex").collect()
    ctrl_cnt = _pt_counts(ctrl.withColumn("g", F.concat_ws("|", "sex", "venla")), "g").collect()
    ctrl_n = {r["g"]: r["count"] for r in
              ctrl.withColumn("g", F.concat_ws("|", "sex", "venla")).groupBy("g").count().collect()}

    def ctrl_counts(sex=None, venla=None):
        out = {}
        for r in ctrl_cnt:
            sx, vn = r["g"].split("|")
            if (sex is None or sx == sex) and (venla is None or vn == str(venla)):
                out[r["pt"]] = out.get(r["pt"], 0) + r["count"]
        return out

    def ctrl_total(sex=None, venla=None):
        return sum(n for g, n in ctrl_n.items()
                   if (sex is None or g.split("|")[0] == sex) and (venla is None or g.split("|")[1] == str(venla)))

    def case_counts(sex):
        return {r["pt"]: r["count"] for r in case_sex if r["sex"] == sex}

    s["venlafaxine_control_cases"] = ctrl_total(venla=1)

    t = {}
    t["ror_main"] = ror_table(case_all, s["esketamine_cases"], ctrl_counts(), s["control_cases"], min_a)
    t["ror_vs_venlafaxine"] = ror_table(case_all, s["esketamine_cases"], ctrl_counts(venla=1),
                                        s["venlafaxine_control_cases"], min_a)
    women = ror_table(case_counts("F"), s["esketamine_women"], ctrl_counts(sex="F"), ctrl_total(sex="F"), min_a)
    men = ror_table(case_counts("M"), s["esketamine_men"], ctrl_counts(sex="M"), ctrl_total(sex="M"), min_a)
    t["ror_sex"] = sex_table(women, men, min_a)

    s["distinct_pts_with_esketamine"] = len(case_all)
    s["signals_main"] = int(t["ror_main"]["signal"].sum())
    s["signals_vs_venlafaxine"] = int(t["ror_vs_venlafaxine"]["signal"].sum())
    s["sex_pts_tested"] = len(t["ror_sex"])
    s["sex_more_in_women"] = int(t["ror_sex"]["more_in_women"].sum())
    s["sex_more_in_men"] = int(t["ror_sex"]["more_in_men"].sum())
    return s, t


# what the paper reports, for the side-by-side table in the report
PAPER2 = {
    "all_clean_reports": 10_126_485,
    "esketamine_cases": 7_790,
    "control_cases": 10_118_302,
    "excluded_not_ps_ss_or_non_psych": 393,
    "esketamine_women": 4_557,
    "esketamine_men": 2_530,
    "distinct_pts_with_esketamine": 1_536,
    "signals_main": 173,
    "signals_vs_venlafaxine": 72,
    "sex_more_in_women": 7,
    "sex_more_in_men": 16,
}


def compare_top30(ours: pd.DataFrame, paper_csv) -> pd.DataFrame:
    """Our numbers next to the paper's Table 2 (top 30 PTs)."""
    p = pd.read_csv(paper_csv)
    p["key"] = p["pt"].str.upper()
    o = ours.assign(key=ours["pt"].str.upper())[["key", "a", "c", "ror", "lower", "upper", "signal"]]
    m = p.merge(o, on="key", how="left", suffixes=("_paper", "_ours"))
    m["a_diff_%"] = 100 * (m["a_ours"] - m["a_paper"]) / m["a_paper"]
    m["ror_diff_%"] = 100 * (m["ror_ours"] - m["ror_paper"]) / m["ror_paper"]
    return m[["pt", "a_paper", "a_ours", "a_diff_%", "c_paper", "c_ours", "ror_paper", "ror_ours",
              "ror_diff_%", "lower_ours", "upper_ours", "signal"]]
