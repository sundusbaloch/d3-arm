"""Paper 1, VAERS 2024: which settings would reproduce their Tables 8 and 9?

The paper says only "top 10 by ROR" and does not give the minimum support or
any other filter for the real data. Ranking ALL frequent pairs by ROR puts
rare, very specific pairs on top (e.g. Pneumovax -> pneumococcal bacteraemia),
while their tables list common pairs (syncope, pain, dizziness). Their
simulation code also uses Confidence thresholds (0.1-0.4), so here we try
minsup x minimum-confidence combinations on the files script 81 saved, and
report the overlap with their top 10 for each. No Spark needed (small files).

    python scripts/82_paper1_vaers_grid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from d3arm import config
from d3arm.paper1.vaers import overlap_ddi, overlap_main

res = config.RESULTS_DIR / "paper1_vaers"
ref8 = pd.read_csv(config.REFERENCE_DIR / "paper1_vaers_table8.csv")
ref9 = pd.read_csv(config.REFERENCE_DIR / "paper1_vaers_table9.csv")
rows = []
for f in sorted(res.glob("main_minsup*.csv"), key=lambda p: int(p.stem.split("minsup")[1])):
    ms = int(f.stem.split("minsup")[1])
    main = pd.read_csv(f)
    ddi = pd.read_csv(res / f"ddi_minsup{ms}.csv")
    for conf in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4):
        tm = main[main["conf"] >= conf].sort_values(["ror", "a"], ascending=False).head(10)
        td = ddi[ddi["conf"] >= conf].sort_values(["ror", "a"], ascending=False).head(10)
        rows.append({"minsup": ms, "min_conf": conf, "pairs_left": int((main["conf"] >= conf).sum()),
                     "triples_left": int((ddi["conf"] >= conf).sum()),
                     "table8_overlap": overlap_main(tm, ref8), "table9_overlap": overlap_ddi(td, ref9)})
grid = pd.DataFrame(rows)
grid["total"] = grid["table8_overlap"] + grid["table9_overlap"]
grid.to_csv(res / "settings_grid.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(grid.to_string(index=False))
best = grid.sort_values(["total", "minsup"], ascending=[False, True]).iloc[0]
print(f"\nbest: minsup {int(best['minsup'])}, confidence >= {best['min_conf']}: "
      f"{int(best['table8_overlap'])}/10 of Table 8 and {int(best['table9_overlap'])}/10 of Table 9")
ms, conf = int(best["minsup"]), float(best["min_conf"])
main = pd.read_csv(res / f"main_minsup{ms}.csv")
print(f"\ntop 10 pairs at minsup {ms}, confidence >= {conf}:")
print(main[main["conf"] >= conf].sort_values(["ror", "a"], ascending=False).head(10)[["v", "ae", "a", "conf", "ror"]]
      .round(3).to_string(index=False))
print("\nCOVID-19 vaccine -> COVID-19 (the paper's example of confounding), all minsup files:")
for f in sorted(res.glob("ddi_minsup*.csv")):
    d = pd.read_csv(f)
    hit = d[(d["ae"] == "COVID-19") & d["v1"].str.contains("COVID19") & d["v2"].str.contains("COVID19")]
    if len(hit):
        print(f"  {f.stem}:")
        print(hit[["v1", "v2", "a", "ror", "conf"]].round(2).to_string(index=False))
