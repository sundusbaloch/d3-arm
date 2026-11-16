"""Builds two tiny FAKE FAERS quarters (same file layout as the real zips) so the
pipeline can be tested without downloading gigabytes. None of this is real data.

Planted cases, so we know what dedup should do:
  case 100: two versions (v1 in 2023Q1, v2 in 2023Q2)      -> keep only v2
  case 200: listed in the deleted-cases file                -> removed
  cases 300 / 301: identical on every field Pisanu et al. match on -> keep one
  cases 400 / 401: same except age 45 vs 46 (near-duplicate) -> both survive exact dedup
  cases 600 / 601: 600 is an older, shorter copy of 601 (one drug instead of two)
                   -> survives the strict "exact" rule, dropped by Paper 2's row rule
  cases 500-519: ordinary single reports

    python tests/make_fake_faers.py <out_dir>
"""
import random
import sys
import zipfile
from pathlib import Path

DEMO_H = "primaryid$caseid$caseversion$i_f_code$event_dt$mfr_dt$init_fda_dt$fda_dt$rept_cod$auth_num$mfr_num$mfr_sndr$lit_ref$age$age_cod$age_grp$sex$e_sub$wt$wt_cod$rept_dt$to_mfr$occp_cod$reporter_country$occr_country"
DRUG_H = "primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route$dose_vbm$cum_dose_chr$cum_dose_unit$dechal$rechal$lot_num$exp_dt$nda_num$dose_amt$dose_unit$dose_form$dose_freq"
REAC_H = "primaryid$caseid$pt$drug_rec_act"
INDI_H = "primaryid$caseid$indi_drug_seq$indi_pt"

DRUGS = [("SPRAVATO", "ESKETAMINE"), ("EFFEXOR XR", "VENLAFAXINE HYDROCHLORIDE"), ("WELLBUTRIN", "BUPROPION"),
         ("ZOLOFT", "SERTRALINE"), ("ASPIRIN 81 MG", ""), ("RISPERDAL", "RISPERIDONE")]
AES = ["Dizziness", "Nausea", "Headache", "Agitation", "Vertigo", "Dissociation"]


def demo_row(pid, cid, ver, init_dt, fda_dt, age, sex, country, event_dt):
    f = [pid, cid, ver, "I", event_dt, "", init_dt, fda_dt, "EXP", "", "", "", "", age, "YR", "", sex, "Y", "", "",
         fda_dt, "", "MD", country, country]
    return "$".join(map(str, f))


def drug_row(pid, cid, seq, role, name, ai):
    return "$".join(map(str, [pid, cid, seq, role, name, ai] + [""] * 14))


def build(out_dir: Path):
    rnd = random.Random(7)
    q = {"2023Q1": {"DEMO": [], "DRUG": [], "REAC": [], "INDI": []},
         "2023Q2": {"DEMO": [], "DRUG": [], "REAC": [], "INDI": []}}

    def add(quarter, pid, cid, ver, init_dt, fda_dt, age, sex, country, event_dt, drugs, aes, indi):
        t = q[quarter]
        t["DEMO"].append(demo_row(pid, cid, ver, init_dt, fda_dt, age, sex, country, event_dt))
        for i, (name, ai) in enumerate(drugs, 1):
            t["DRUG"].append(drug_row(pid, cid, i, "PS" if i == 1 else "C", name, ai))
        for ae in aes:
            t["REAC"].append(f"{pid}${cid}${ae}$")
        t["INDI"].append(f"{pid}${cid}$1${indi}")

    d = DRUGS
    add("2023Q1", 1001, 100, 1, "20230110", "20230110", 40, "F", "US", "20230101", [d[0]], ["Dizziness"], "Depression")
    add("2023Q2", 1002, 100, 2, "20230110", "20230420", 40, "F", "US", "20230101", [d[0], d[1]], ["Dizziness"], "Depression")
    add("2023Q1", 2001, 200, 1, "20230115", "20230115", 30, "M", "GB", "20230105", [d[2]], ["Nausea"], "Smoking cessation")
    add("2023Q1", 3001, 300, 1, "20230201", "20230201", 52, "F", "US", "20230120", [d[0], d[2]], ["Agitation"], "Depression")
    add("2023Q1", 3011, 301, 1, "20230203", "20230203", 52, "F", "US", "20230120", [d[0], d[2]], ["Agitation"], "Depression")
    add("2023Q2", 4001, 400, 1, "20230410", "20230410", 45, "F", "US", "20230401", [d[0], d[1]], ["Dizziness", "Nausea"], "Depression")
    add("2023Q2", 4011, 401, 1, "20230412", "20230412", 46, "F", "US", "20230401", [d[0], d[1]], ["Dizziness", "Nausea"], "Depression")
    add("2023Q1", 6001, 600, 1, "20230301", "20230301", 63, "M", "CA", "20230215", [d[3]], ["Headache"], "Depression")
    add("2023Q2", 6011, 601, 1, "20230402", "20230402", 63, "M", "CA", "20230215", [d[3], d[5]], ["Headache"], "Depression")
    for i in range(20):
        cid = 500 + i
        quarter = "2023Q1" if i < 10 else "2023Q2"
        month = "02" if i < 10 else "05"
        drugs = rnd.sample(d, rnd.randint(1, 3))
        aes = rnd.sample(AES, rnd.randint(1, 2))
        add(quarter, cid * 10 + 1, cid, 1, f"2023{month}{10 + i % 15}", f"2023{month}{10 + i % 15}",
            rnd.randint(18, 80), rnd.choice("MF"), "US", f"2023{month}01", drugs, aes, "Depression")

    out_dir.mkdir(parents=True, exist_ok=True)
    for quarter, tables in q.items():
        yy, qq = quarter[2:4], quarter[-1]
        zpath = out_dir / f"faers_ascii_{quarter[:4]}q{qq}.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for name, header in [("DEMO", DEMO_H), ("DRUG", DRUG_H), ("REAC", REAC_H), ("INDI", INDI_H)]:
                z.writestr(f"ASCII/{name}{yy}Q{qq}.txt", "\n".join([header] + tables[name]) + "\n")
            if quarter == "2023Q2":
                z.writestr(f"Deleted/DELETED_CASES_{yy}Q{qq}.txt", "200\n")
        print("wrote", zpath)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
